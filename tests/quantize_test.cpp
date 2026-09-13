// Synthetic YuE2 quantizer test: tensor plan, streamed GGUF layout, metadata,
// dequantization quality, refusals, and backend execution of the quantized
// get_rows/mul_mat paths the generation graph uses.
//
// yue2-quantize-test [DEVICE]
#include "gguf_model.h"
#include "yue2/quantize.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

constexpr const char * kHash =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error("quantize test failed: " + message);
}

struct TemporaryDirectory {
    fs::path path = fs::temp_directory_path() /
        ("yue2-quantize-test-" + std::to_string(std::random_device{}()));
    TemporaryDirectory() { fs::create_directories(path); }
    ~TemporaryDirectory() {
        std::error_code error;
        fs::remove_all(path, error);
    }
};

struct SourceTensor {
    const char * name;
    std::int64_t ne0;
    std::int64_t ne1; // 0 for a vector
};

// YuE2 names at reduced width. 256 keeps the K-quant block constraint
// satisfiable; the 96-wide gate projection is eligible only for Q8_0.
constexpr SourceTensor kTensors[] = {
    {"model.embed_tokens.weight", 256, 32},
    {"model.layers.0.self_attn.q_proj.weight", 256, 4},
    {"model.layers.0.self_attn.v_proj.weight", 256, 4},
    {"model.layers.0.nar_self_attn.v_proj.weight", 256, 4},
    {"model.layers.0.mlp.down_proj.weight", 256, 4},
    {"model.layers.0.mlp.gate_proj.weight", 96, 4},
    {"model.norm.weight", 256, 0},
    {"latent_pos_embed.pe", 256, 16},
    {"time_embedder.mlp.2.weight", 256, 256},
    {"llm2vae.weight", 256, 64},
    {"lm_head.weight", 256, 32},
};

void write_source(const fs::path & path, const char * architecture, const char * component) {
    std::size_t bytes = 1 << 20;
    for (const auto & spec : kTensors) {
        bytes += ggml_tensor_overhead() +
            ggml_row_size(GGML_TYPE_BF16, spec.ne0) * static_cast<std::size_t>(spec.ne1 ? spec.ne1 : 1);
    }
    auto * context = ggml_init({bytes, nullptr, false});
    require(context != nullptr, "ggml_init");
    auto * file = gguf_init_empty();
    gguf_set_val_str(file, "general.architecture", architecture);
    gguf_set_val_str(file, "general.name", "synthetic YuE2");
    gguf_set_val_str(file, "yue2.component", component);
    gguf_set_val_str(file, "yue2.checkpoint.sha256", kHash);

    std::mt19937 generator(1234);
    std::normal_distribution<float> normal(0.0F, 0.02F);
    for (const auto & spec : kTensors) {
        auto * tensor = spec.ne1
            ? ggml_new_tensor_2d(context, GGML_TYPE_BF16, spec.ne0, spec.ne1)
            : ggml_new_tensor_1d(context, GGML_TYPE_BF16, spec.ne0);
        ggml_set_name(tensor, spec.name);
        std::vector<float> values(static_cast<std::size_t>(ggml_nelements(tensor)));
        for (auto & value : values) value = normal(generator);
        ggml_fp32_to_bf16_row(
            values.data(), static_cast<ggml_bf16_t *>(tensor->data),
            static_cast<std::int64_t>(values.size()));
        gguf_add_tensor(file, tensor);
    }
    require(gguf_write_to_file(file, path.string().c_str(), false), "write source GGUF");
    gguf_free(file);
    ggml_free(context);
}

struct Loaded {
    gguf_context * gguf = nullptr;
    ggml_context * tensors = nullptr;
    explicit Loaded(const fs::path & path) {
        gguf = gguf_init_from_file(path.string().c_str(), {true, &tensors});
        require(gguf != nullptr, "open " + path.string());
    }
    ~Loaded() {
        gguf_free(gguf);
        ggml_free(tensors);
    }
    ggml_type type(const char * name) const {
        const auto index = gguf_find_tensor(gguf, name);
        require(index >= 0, std::string("missing tensor ") + name);
        return gguf_get_tensor_type(gguf, index);
    }
    std::string string(const char * key) const {
        const auto index = gguf_find_key(gguf, key);
        require(index >= 0 && gguf_get_kv_type(gguf, index) == GGUF_TYPE_STRING, key);
        return gguf_get_val_str(gguf, index);
    }
    std::uint32_t u32(const char * key) const {
        const auto index = gguf_find_key(gguf, key);
        require(index >= 0 && gguf_get_kv_type(gguf, index) == GGUF_TYPE_UINT32, key);
        return gguf_get_val_u32(gguf, index);
    }
};

std::vector<std::uint8_t> raw_tensor(const fs::path & path, const char * name) {
    const Loaded file(path);
    const auto index = gguf_find_tensor(file.gguf, name);
    require(index >= 0, std::string("missing tensor ") + name);
    std::ifstream input(path, std::ios::binary);
    input.seekg(static_cast<std::streamoff>(
        gguf_get_data_offset(file.gguf) + gguf_get_tensor_offset(file.gguf, index)));
    std::vector<std::uint8_t> bytes(gguf_get_tensor_size(file.gguf, index));
    input.read(static_cast<char *>(static_cast<void *>(bytes.data())),
        static_cast<std::streamsize>(bytes.size()));
    require(static_cast<bool>(input), std::string("read tensor ") + name);
    return bytes;
}

std::vector<float> host_float(const fs::path & path, const char * name, ggml_type type) {
    const auto raw = raw_tensor(path, name);
    const std::size_t elements = raw.size() / ggml_type_size(type) *
        static_cast<std::size_t>(ggml_blck_size(type));
    std::vector<float> values(elements);
    if (type == GGML_TYPE_F32) {
        std::memcpy(values.data(), raw.data(), raw.size());
    } else {
        ggml_get_type_traits(type)->to_float(
            raw.data(), values.data(), static_cast<std::int64_t>(values.size()));
    }
    return values;
}

void expect_types(const fs::path & path, const std::map<std::string, ggml_type> & expected) {
    const Loaded file(path);
    for (const auto & [name, type] : expected) {
        const auto actual = file.type(name.c_str());
        require(actual == type,
            name + " is " + ggml_type_name(actual) + ", expected " + ggml_type_name(type));
    }
}

bool throws_containing(const std::function<void()> & action, const char * needle) {
    try {
        action();
    } catch (const std::exception & error) {
        return std::string(error.what()).find(needle) != std::string::npos;
    }
    return false;
}

// Execute the quantized paths the AR graph depends on and compare with host
// dequantized math. mul_mat backends quantize activations for K-quant dot
// products, so matrix products are compared by relative error, not bitwise.
void check_backend_execution(const fs::path & path, const char * device) {
    auto model = yue2::detail::load_gguf_raw(path.string().c_str(), device);
    auto * context = ggml_init({64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true});
    auto * ids = ggml_new_tensor_1d(context, GGML_TYPE_I32, 3);
    auto * x = ggml_new_tensor_2d(context, GGML_TYPE_F32, 256, 2);
    ggml_set_input(ids);
    ggml_set_input(x);
    auto * embedded = ggml_get_rows(context, model.get("model.embed_tokens.weight"), ids);
    auto * projected = ggml_mul_mat(context, model.get("model.layers.0.self_attn.q_proj.weight"), x);
    auto * logits = ggml_mul_mat(context, model.get("lm_head.weight"), x);
    auto * graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, embedded);
    ggml_build_forward_expand(graph, projected);
    ggml_build_forward_expand(graph, logits);
    auto * allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend()));
    require(ggml_gallocr_alloc_graph(allocator, graph), "allocate graph");

    const std::vector<std::int32_t> id_values = {31, 0, 7};
    std::vector<float> x_values(512);
    std::mt19937 generator(99);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    for (auto & value : x_values) value = normal(generator);
    ggml_backend_tensor_set(ids, id_values.data(), 0, sizeof(std::int32_t) * id_values.size());
    ggml_backend_tensor_set(x, x_values.data(), 0, sizeof(float) * x_values.size());
    require(ggml_backend_graph_compute(model.backend(), graph) == GGML_STATUS_SUCCESS, "compute");

    auto fetch = [](ggml_tensor * tensor) {
        std::vector<float> values(static_cast<std::size_t>(ggml_nelements(tensor)));
        ggml_backend_tensor_get(tensor, values.data(), 0, ggml_nbytes(tensor));
        return values;
    };

    const auto embedding = host_float(path, "model.embed_tokens.weight", GGML_TYPE_Q6_K);
    const auto actual_rows = fetch(embedded);
    for (std::size_t row = 0; row < id_values.size(); ++row) {
        for (std::size_t column = 0; column < 256; ++column) {
            const float expected = embedding[static_cast<std::size_t>(id_values[row]) * 256 + column];
            require(std::abs(actual_rows[row * 256 + column] - expected) <= 1.0e-5F,
                "quantized get_rows differs from host dequantization");
        }
    }

    auto check_product = [&](ggml_tensor * output, const char * name, ggml_type type, std::int64_t rows) {
        const auto weight = host_float(path, name, type);
        const auto actual = fetch(output);
        double error = 0.0;
        double reference = 0.0;
        for (std::int64_t column = 0; column < 2; ++column) {
            for (std::int64_t row = 0; row < rows; ++row) {
                double expected = 0.0;
                for (std::int64_t input = 0; input < 256; ++input) {
                    expected += static_cast<double>(weight[static_cast<std::size_t>(row * 256 + input)]) *
                        x_values[static_cast<std::size_t>(column * 256 + input)];
                }
                const double difference = actual[static_cast<std::size_t>(column * rows + row)] - expected;
                error += difference * difference;
                reference += expected * expected;
            }
        }
        const double relative = std::sqrt(error / reference);
        std::cout << "  " << name << " " << ggml_type_name(type) << " mul_mat relative_rms=" << relative << "\n";
        require(relative < 0.02, std::string("quantized mul_mat diverges for ") + name);
    };
    check_product(projected, "model.layers.0.self_attn.q_proj.weight", GGML_TYPE_Q4_K, 4);
    check_product(logits, "lm_head.weight", GGML_TYPE_Q6_K, 32);

    ggml_gallocr_free(allocator);
    ggml_free(context);
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const char * device = argc > 1 ? argv[1] : "cpu";
        TemporaryDirectory temporary;
        const auto source = temporary.path / "yue2-synthetic-v1.0-BF16.gguf";
        const auto q4 = temporary.path / "yue2-synthetic-v1.0-Q4_K_M.gguf";
        const auto q8 = temporary.path / "yue2-synthetic-v1.0-Q8_0.gguf";
        const auto f16 = temporary.path / "yue2-synthetic-v1.0-F16.gguf";
        write_source(source, "yue2", "generation");

        require(yue2::parse_quantization_mix("Q4_K_M") == yue2::QuantizationMix::q4_k_m, "Q4_K_M spelling");
        require(yue2::parse_quantization_mix("q4_km") == yue2::QuantizationMix::q4_k_m, "q4_km alias");
        require(yue2::parse_quantization_mix("q8_0") == yue2::QuantizationMix::q8_0, "q8_0");
        require(throws_containing([] { yue2::parse_quantization_mix("q3_k"); }, "unknown"), "unknown mix");

        yue2::QuantizeOptions options;
        options.threads = 3;
        std::size_t callbacks = 0;
        options.on_tensor = [&](const yue2::QuantizedTensor &, std::size_t index, std::size_t count) {
            require(index == ++callbacks && count == std::size(kTensors), "progress order");
        };
        const auto report = yue2::quantize_generation_gguf(source.string(), q4.string(), options);
        require(callbacks == std::size(kTensors), "progress count");
        require(report.converted_count == 6 && report.kept_count == 5, "Q4_K_M plan counts");
        require(report.output_bytes < report.input_bytes, "Q4_K_M shrinks the file");
        require(!fs::exists(q4.string() + ".incomplete"), "work file removed");

        expect_types(q4, {
            {"model.embed_tokens.weight", GGML_TYPE_Q6_K},
            {"model.layers.0.self_attn.q_proj.weight", GGML_TYPE_Q4_K},
            {"model.layers.0.self_attn.v_proj.weight", GGML_TYPE_Q6_K},
            {"model.layers.0.nar_self_attn.v_proj.weight", GGML_TYPE_Q6_K},
            {"model.layers.0.mlp.down_proj.weight", GGML_TYPE_Q6_K},
            {"model.layers.0.mlp.gate_proj.weight", GGML_TYPE_BF16},
            {"model.norm.weight", GGML_TYPE_BF16},
            {"latent_pos_embed.pe", GGML_TYPE_BF16},
            {"time_embedder.mlp.2.weight", GGML_TYPE_BF16},
            {"llm2vae.weight", GGML_TYPE_BF16},
            {"lm_head.weight", GGML_TYPE_Q6_K},
        });
        for (const char * kept : {"model.layers.0.mlp.gate_proj.weight", "model.norm.weight",
                                  "latent_pos_embed.pe", "time_embedder.mlp.2.weight", "llm2vae.weight"}) {
            require(raw_tensor(source, kept) == raw_tensor(q4, kept), std::string("kept bytes for ") + kept);
        }
        {
            const Loaded file(q4);
            require(file.string("general.architecture") == "yue2", "architecture preserved");
            require(file.string("yue2.component") == "generation", "component preserved");
            require(file.string("yue2.checkpoint.sha256") == kHash, "LoRA fingerprint preserved");
            require(file.string("yue2.quantization.encoding") == "Q4_K_M", "encoding recorded");
            require(file.u32("yue2.quantization.version") == 1, "recipe version recorded");
            require(file.u32("general.file_type") == 15, "general.file_type is MOSTLY_Q4_K_M");
        }

        const auto q4_check = yue2::check_quantized_gguf(source.string(), q4.string(), 0.99, 2);
        require(q4_check.compared.size() == 6 && q4_check.below_threshold == 0, "Q4_K_M cosine check");
        for (const auto & result : q4_check.compared) {
            std::cout << "  " << result.name << " " << result.candidate_type << " cos=" << result.cosine << "\n";
            if (result.candidate_type == "q6_K") require(result.cosine > 0.999, "Q6_K cosine for " + result.name);
        }
        require(yue2::check_quantized_gguf(source.string(), q4.string(), 1.0).below_threshold == 6,
            "the check reports tensors below an unattainable threshold");

        require(throws_containing([&] { yue2::quantize_generation_gguf(source.string(), q4.string(), options); },
            "already exists"), "existing output refused");
        options.on_tensor = nullptr;
        options.overwrite = true;
        yue2::quantize_generation_gguf(source.string(), q4.string(), options);
        require(throws_containing([&] { yue2::quantize_generation_gguf(q4.string(), q4.string(), options); },
            "same file"), "in-place quantization refused");

        options.mix = yue2::QuantizationMix::q8_0;
        options.overwrite = false;
        yue2::quantize_generation_gguf(source.string(), q8.string(), options);
        expect_types(q8, {
            {"model.layers.0.self_attn.q_proj.weight", GGML_TYPE_Q8_0},
            {"model.layers.0.mlp.gate_proj.weight", GGML_TYPE_Q8_0},
            {"latent_pos_embed.pe", GGML_TYPE_BF16},
            {"lm_head.weight", GGML_TYPE_Q8_0},
        });
        require(yue2::check_quantized_gguf(source.string(), q8.string(), 0.9999).below_threshold == 0,
            "Q8_0 cosine check");

        // Re-encoding a quantized file dequantizes it; float mixes also cover
        // the flow boundary tensors.
        options.mix = yue2::QuantizationMix::f16;
        yue2::quantize_generation_gguf(q4.string(), f16.string(), options);
        expect_types(f16, {
            {"model.layers.0.self_attn.q_proj.weight", GGML_TYPE_F16},
            {"latent_pos_embed.pe", GGML_TYPE_F16},
            {"model.norm.weight", GGML_TYPE_BF16},
        });
        require(Loaded(f16).string("yue2.quantization.encoding") == "F16", "F16 encoding recorded");
        const auto f16_check = yue2::check_quantized_gguf(source.string(), f16.string(), 0.99);
        require(f16_check.compared.size() == 10 && f16_check.below_threshold == 0, "F16 from Q4_K_M check");

        const auto vae = temporary.path / "yue2-vae-v1.0-F16.gguf";
        write_source(vae, "yue2_vae", "vae");
        require(throws_containing([&] {
            yue2::quantize_generation_gguf(vae.string(), (temporary.path / "vae-q4.gguf").string());
        }, "VAE"), "VAE refused");
        const auto transcription = temporary.path / "sheetsage2.gguf";
        write_source(transcription, "yue2-sheetsage2", "transcription");
        require(throws_containing([&] {
            yue2::quantize_generation_gguf(transcription.string(), (temporary.path / "t-q4.gguf").string());
        }, "transcription"), "transcription refused");

        check_backend_execution(q4, device);
        std::cout << "yue2-quantize-test: ok\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
