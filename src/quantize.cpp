#include "yue2/quantize.h"

#include "gguf_model.h"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

namespace yue2 {
namespace {

// Bumped whenever the tensor plan below changes, so a published file records
// which recipe produced it.
constexpr std::uint32_t kRecipeVersion = 1;

std::runtime_error quantize_error(const std::string & message) {
    return std::runtime_error("[yue2:quantize] " + message);
}

struct MixTypes {
    const char * name;
    ggml_type body;
    ggml_type critical;
    // general.file_type as enumerated by the GGUF specification.
    std::uint32_t file_type;
};

MixTypes mix_types(QuantizationMix mix) {
    switch (mix) {
        case QuantizationMix::q4_k_m: return {"Q4_K_M", GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, 15};
        case QuantizationMix::q5_k_m: return {"Q5_K_M", GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, 17};
        case QuantizationMix::q8_0: return {"Q8_0", GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 7};
        case QuantizationMix::f16: return {"F16", GGML_TYPE_F16, GGML_TYPE_F16, 1};
        case QuantizationMix::f32: return {"F32", GGML_TYPE_F32, GGML_TYPE_F32, 0};
    }
    throw std::invalid_argument("unknown YuE2 quantization mix");
}

bool starts_with(const std::string & value, const char * prefix) {
    return value.rfind(prefix, 0) == 0;
}

bool ends_with(const std::string & value, const char * suffix) {
    const std::size_t length = std::strlen(suffix);
    return value.size() >= length &&
        value.compare(value.size() - length, length, suffix) == 0;
}

// The matrices whose error reaches the output most directly take the
// higher-precision type in the K-quant mixes, following sa3-quantize and the
// llama.cpp *_K_M recipes: attention values and MLP down projections in both
// the AR and NAR branches, plus the 184,704-row embedding and LM head.
bool is_critical(const std::string & name) {
    return ends_with(name, ".v_proj.weight") || ends_with(name, ".down_proj.weight") ||
        name == "model.embed_tokens.weight" || name == "lm_head.weight";
}

// The flow branch's latent bridges, timestep MLP and position table sit
// directly on the continuous latent path and total about 55M parameters.
// latent_pos_embed.pe is also read through a view and an F32 cast rather than
// mul_mat or get_rows, which quantized storage cannot serve.
bool is_flow_boundary(const std::string & name) {
    return starts_with(name, "latent_pos_embed.") || starts_with(name, "vae2llm.") ||
        starts_with(name, "llm2vae.") || starts_with(name, "time_embedder.");
}

ggml_type planned_type(const std::string & name, const ggml_tensor * tensor, const MixTypes & mix) {
    if (ggml_n_dims(tensor) != 2) return tensor->type;
    const bool boundary = is_flow_boundary(name);
    if (!boundary && !ends_with(name, ".weight")) return tensor->type;
    const ggml_type target = is_critical(name) ? mix.critical : mix.body;
    if (boundary && ggml_is_quantized(target)) return tensor->type;
    // Rows must be whole blocks of the target type: 256 for the K-quants.
    if (tensor->ne[0] % ggml_blck_size(target) != 0) return tensor->type;
    return target;
}

struct GgufFile {
    GgufFile() = default;
    GgufFile(const GgufFile &) = delete;
    GgufFile & operator=(const GgufFile &) = delete;
    ~GgufFile() {
        if (gguf) gguf_free(gguf);
        if (tensors) ggml_free(tensors);
        if (file) std::fclose(file);
    }

    std::string path;
    gguf_context * gguf = nullptr;
    ggml_context * tensors = nullptr;
    FILE * file = nullptr;
    std::uint64_t bytes = 0;
};

std::unique_ptr<GgufFile> open_gguf(const std::string & path) {
    auto result = std::make_unique<GgufFile>();
    result->path = path;
    const gguf_init_params params = {true, &result->tensors};
    result->gguf = gguf_init_from_file(path.c_str(), params);
    if (!result->gguf) throw quantize_error("failed to open GGUF: " + path);
    result->file = std::fopen(path.c_str(), "rb");
    if (!result->file) throw quantize_error("cannot read " + path);
    result->bytes = detail::file_size(result->file);
    return result;
}

std::string string_key(const gguf_context * gguf, const char * key) {
    const auto index = gguf_find_key(gguf, key);
    if (index < 0 || gguf_get_kv_type(gguf, index) != GGUF_TYPE_STRING) return {};
    return gguf_get_val_str(gguf, index);
}

void require_generation(const GgufFile & input) {
    const auto architecture = string_key(input.gguf, "general.architecture");
    const auto component = string_key(input.gguf, "yue2.component");
    if (architecture == "yue2" && component == "generation") return;
    if (architecture == "yue2_vae") {
        throw quantize_error(
            "the YuE2 VAE is distributed as F16/F32; quantize the generation model instead: " +
            input.path);
    }
    if (component == "transcription") {
        throw quantize_error(
            "SheetSage2/MERT2 transcription GGUFs are not quantized: " + input.path);
    }
    throw quantize_error("not a yue2.cpp YuE2 generation GGUF: " + input.path);
}

std::vector<std::uint8_t> read_tensor(const GgufFile & source, std::int64_t index) {
    const std::uint64_t offset = static_cast<std::uint64_t>(gguf_get_data_offset(source.gguf)) +
        gguf_get_tensor_offset(source.gguf, index);
    const std::size_t size = gguf_get_tensor_size(source.gguf, index);
    const std::string name = gguf_get_tensor_name(source.gguf, index);
    if (offset > source.bytes || size > source.bytes - offset) {
        throw quantize_error("truncated tensor " + name + " in " + source.path);
    }
    std::vector<std::uint8_t> bytes(size);
    if (!detail::file_seek(source.file, offset) ||
        std::fread(bytes.data(), 1, size, source.file) != size) {
        throw quantize_error("short read for tensor " + name + " in " + source.path);
    }
    return bytes;
}

// Runs work(begin_row, end_row) over row ranges. Tensors below about a
// million elements stay on the calling thread.
template <typename Work>
void parallel_rows(std::int64_t rows, std::int64_t n_per_row, unsigned threads, const Work & work) {
    constexpr std::int64_t kMinimumElementsPerWorker = 1 << 20;
    std::int64_t workers = std::min<std::int64_t>(std::max(threads, 1U), rows);
    workers = std::min(workers, std::max<std::int64_t>(1, rows * n_per_row / kMinimumElementsPerWorker));
    if (workers <= 1) {
        work(0, rows);
        return;
    }
    const std::int64_t chunk = (rows + workers - 1) / workers;
    std::vector<std::thread> pool;
    std::vector<std::exception_ptr> errors(static_cast<std::size_t>(workers));
    for (std::int64_t worker = 0; worker < workers; ++worker) {
        const std::int64_t begin = worker * chunk;
        const std::int64_t end = std::min(rows, begin + chunk);
        if (begin >= end) break;
        pool.emplace_back([&work, &errors, worker, begin, end]() {
            try {
                work(begin, end);
            } catch (...) {
                errors[static_cast<std::size_t>(worker)] = std::current_exception();
            }
        });
    }
    for (auto & thread : pool) thread.join();
    for (const auto & error : errors) {
        if (error) std::rethrow_exception(error);
    }
}

std::vector<float> dequantize(
    ggml_type type,
    const std::vector<std::uint8_t> & raw,
    std::int64_t n_per_row,
    std::int64_t rows,
    unsigned threads,
    const std::string & name) {
    const std::size_t row_bytes = ggml_row_size(type, n_per_row);
    if (raw.size() != row_bytes * static_cast<std::size_t>(rows)) {
        throw quantize_error("unexpected storage size for tensor " + name);
    }
    std::vector<float> values(static_cast<std::size_t>(n_per_row * rows));
    if (type == GGML_TYPE_F32) {
        std::memcpy(values.data(), raw.data(), raw.size());
        return values;
    }
    const auto * traits = ggml_get_type_traits(type);
    if (!traits || !traits->to_float) {
        throw quantize_error(std::string("no dequantizer for ") + ggml_type_name(type) +
            " (tensor " + name + ")");
    }
    parallel_rows(rows, n_per_row, threads, [&](std::int64_t begin, std::int64_t end) {
        traits->to_float(
            raw.data() + static_cast<std::size_t>(begin) * row_bytes,
            values.data() + begin * n_per_row,
            (end - begin) * n_per_row);
    });
    return values;
}

std::vector<std::uint8_t> encode(
    ggml_type type,
    const std::vector<float> & values,
    std::int64_t n_per_row,
    std::int64_t rows,
    unsigned threads) {
    const std::size_t row_bytes = ggml_row_size(type, n_per_row);
    std::vector<std::uint8_t> output(row_bytes * static_cast<std::size_t>(rows));
    if (type == GGML_TYPE_F32) {
        std::memcpy(output.data(), values.data(), output.size());
    } else if (type == GGML_TYPE_F16) {
        parallel_rows(rows, n_per_row, threads, [&](std::int64_t begin, std::int64_t end) {
            ggml_fp32_to_fp16_row(
                values.data() + begin * n_per_row,
                static_cast<ggml_fp16_t *>(static_cast<void *>(
                    output.data() + static_cast<std::size_t>(begin) * row_bytes)),
                (end - begin) * n_per_row);
        });
    } else {
        ggml_quantize_init(type);
        parallel_rows(rows, n_per_row, threads, [&](std::int64_t begin, std::int64_t end) {
            // start is an element index; src and dst are the whole buffers.
            ggml_quantize_chunk(
                type, values.data(), output.data(), begin * n_per_row, end - begin,
                n_per_row, nullptr);
        });
    }
    return output;
}

void require_finite(const std::vector<float> & values, const std::string & name) {
    for (const float value : values) {
        if (!std::isfinite(value)) throw quantize_error("non-finite weight in tensor " + name);
    }
}

void write_all(FILE * file, const std::uint8_t * data, std::size_t size, const std::string & path) {
    if (size != 0 && std::fwrite(data, 1, size, file) != size) {
        throw quantize_error("failed to write " + path);
    }
}

unsigned resolve_threads(unsigned requested) {
    if (requested != 0) return requested;
    return std::max(1U, std::thread::hardware_concurrency());
}

} // namespace

QuantizationMix parse_quantization_mix(const std::string & name) {
    std::string key = name;
    for (char & c : key) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (key == "q4_k_m" || key == "q4_km") return QuantizationMix::q4_k_m;
    if (key == "q5_k_m" || key == "q5_km") return QuantizationMix::q5_k_m;
    if (key == "q8_0") return QuantizationMix::q8_0;
    if (key == "f16") return QuantizationMix::f16;
    if (key == "f32") return QuantizationMix::f32;
    throw std::invalid_argument(
        "unknown YuE2 quantization mix '" + name + "' (expected q4_k_m, q5_k_m, q8_0, f16, or f32)");
}

const char * quantization_mix_name(QuantizationMix mix) noexcept {
    switch (mix) {
        case QuantizationMix::q4_k_m: return "Q4_K_M";
        case QuantizationMix::q5_k_m: return "Q5_K_M";
        case QuantizationMix::q8_0: return "Q8_0";
        case QuantizationMix::f16: return "F16";
        case QuantizationMix::f32: return "F32";
    }
    return "unknown";
}

QuantizeReport quantize_generation_gguf(
    const std::string & input_path,
    const std::string & output_path,
    const QuantizeOptions & options) {
    namespace fs = std::filesystem;
    const auto mix = mix_types(options.mix);
    const unsigned threads = resolve_threads(options.threads);

    std::error_code error;
    if (fs::exists(output_path, error)) {
        if (fs::equivalent(input_path, output_path, error)) {
            throw quantize_error("input and output are the same file: " + output_path);
        }
        if (!options.overwrite) {
            throw quantize_error("output already exists (pass overwrite): " + output_path);
        }
    }

    const auto input = open_gguf(input_path);
    require_generation(*input);

    std::unique_ptr<gguf_context, decltype(&gguf_free)> output(gguf_init_empty(), gguf_free);
    if (!output) throw quantize_error("could not create GGUF context");
    gguf_set_kv(output.get(), input->gguf);
    gguf_set_val_u32(output.get(), "general.file_type", mix.file_type);
    gguf_set_val_str(output.get(), "yue2.quantization.encoding", mix.name);
    gguf_set_val_u32(output.get(), "yue2.quantization.version", kRecipeVersion);
    if (gguf_get_alignment(input->gguf) != gguf_get_alignment(output.get())) {
        throw quantize_error("unsupported GGUF alignment in " + input_path);
    }

    const std::int64_t count = gguf_get_n_tensors(input->gguf);
    std::vector<ggml_tensor *> tensors(static_cast<std::size_t>(count));
    std::vector<ggml_type> targets(static_cast<std::size_t>(count));
    for (std::int64_t index = 0; index < count; ++index) {
        const std::string name = gguf_get_tensor_name(input->gguf, index);
        auto * tensor = ggml_get_tensor(input->tensors, name.c_str());
        if (!tensor) throw quantize_error("missing tensor descriptor: " + name);
        tensors[static_cast<std::size_t>(index)] = tensor;
        targets[static_cast<std::size_t>(index)] = planned_type(name, tensor, mix);
        gguf_add_tensor(output.get(), tensor);
    }
    for (std::int64_t index = 0; index < count; ++index) {
        const auto * tensor = tensors[static_cast<std::size_t>(index)];
        const auto target = targets[static_cast<std::size_t>(index)];
        if (target != tensor->type) gguf_set_tensor_type(output.get(), tensor->name, target);
    }

    QuantizeReport report;
    report.tensors.reserve(static_cast<std::size_t>(count));
    report.input_bytes = input->bytes;

    const std::string work_path = output_path + ".incomplete";
    fs::remove(work_path, error);
    try {
        FILE * file = std::fopen(work_path.c_str(), "wb");
        if (!file) throw quantize_error("cannot create " + work_path);
        std::unique_ptr<FILE, int (*)(FILE *)> guard(file, std::fclose);

        std::vector<std::uint8_t> meta(gguf_get_meta_size(output.get()));
        gguf_get_meta_data(output.get(), meta.data());
        write_all(file, meta.data(), meta.size(), work_path);

        const std::size_t alignment = gguf_get_alignment(output.get());
        const std::vector<std::uint8_t> padding(alignment, 0);
        std::uint64_t data_written = 0;
        for (std::int64_t index = 0; index < count; ++index) {
            const auto * tensor = tensors[static_cast<std::size_t>(index)];
            const auto target = targets[static_cast<std::size_t>(index)];
            const std::string name = tensor->name;
            if (data_written != gguf_get_tensor_offset(output.get(), index)) {
                throw quantize_error("internal offset mismatch at tensor " + name);
            }

            std::vector<std::uint8_t> bytes = read_tensor(*input, index);
            if (target != tensor->type) {
                const std::int64_t n_per_row = tensor->ne[0];
                const std::int64_t rows = ggml_nrows(tensor);
                auto values = dequantize(tensor->type, bytes, n_per_row, rows, threads, name);
                std::vector<std::uint8_t>().swap(bytes);
                require_finite(values, name);
                bytes = encode(target, values, n_per_row, rows, threads);
                ++report.converted_count;
            } else {
                ++report.kept_count;
            }
            const std::size_t size = gguf_get_tensor_size(output.get(), index);
            if (bytes.size() != size) {
                throw quantize_error("internal size mismatch at tensor " + name);
            }
            write_all(file, bytes.data(), bytes.size(), work_path);
            const std::size_t padded = GGML_PAD(size, alignment);
            write_all(file, padding.data(), padded - size, work_path);
            data_written += padded;

            report.tensors.push_back({name, ggml_type_name(tensor->type), ggml_type_name(target)});
            if (options.on_tensor) {
                options.on_tensor(
                    report.tensors.back(), static_cast<std::size_t>(index + 1),
                    static_cast<std::size_t>(count));
            }
        }
        if (std::fclose(guard.release()) != 0) throw quantize_error("failed to finish " + work_path);

        if (options.overwrite) fs::remove(output_path, error);
        fs::rename(work_path, output_path);
    } catch (...) {
        fs::remove(work_path, error);
        throw;
    }
    report.output_bytes = fs::file_size(output_path);
    return report;
}

QuantCheckReport check_quantized_gguf(
    const std::string & reference_path,
    const std::string & candidate_path,
    double threshold,
    unsigned threads) {
    threads = resolve_threads(threads);
    const auto reference = open_gguf(reference_path);
    const auto candidate = open_gguf(candidate_path);
    const std::int64_t count = gguf_get_n_tensors(candidate->gguf);
    if (count != gguf_get_n_tensors(reference->gguf)) {
        throw quantize_error("tensor count differs between " + reference_path + " and " + candidate_path);
    }

    QuantCheckReport report;
    for (std::int64_t index = 0; index < count; ++index) {
        const std::string name = gguf_get_tensor_name(candidate->gguf, index);
        const auto reference_index = gguf_find_tensor(reference->gguf, name.c_str());
        if (reference_index < 0) throw quantize_error("tensor missing from reference: " + name);
        const auto * expected = ggml_get_tensor(reference->tensors, name.c_str());
        const auto * actual = ggml_get_tensor(candidate->tensors, name.c_str());
        if (!expected || !actual || !ggml_are_same_shape(expected, actual)) {
            throw quantize_error("tensor shape differs: " + name);
        }
        if (expected->type == actual->type) continue;

        const std::int64_t n_per_row = actual->ne[0];
        const std::int64_t rows = ggml_nrows(actual);
        const auto a = dequantize(
            expected->type, read_tensor(*reference, reference_index), n_per_row, rows, threads, name);
        const auto b = dequantize(
            actual->type, read_tensor(*candidate, index), n_per_row, rows, threads, name);
        double dot = 0.0;
        double norm_a = 0.0;
        double norm_b = 0.0;
        for (std::size_t element = 0; element < a.size(); ++element) {
            dot += static_cast<double>(a[element]) * b[element];
            norm_a += static_cast<double>(a[element]) * a[element];
            norm_b += static_cast<double>(b[element]) * b[element];
        }
        const double denominator = std::sqrt(norm_a * norm_b);
        const double cosine = denominator > 0.0 ? dot / denominator : (norm_a == norm_b ? 1.0 : 0.0);

        report.compared.push_back({name, ggml_type_name(expected->type), ggml_type_name(actual->type), cosine});
        report.minimum_cosine = std::min(report.minimum_cosine, cosine);
        if (!(cosine >= threshold)) ++report.below_threshold;
    }
    return report;
}

} // namespace yue2
