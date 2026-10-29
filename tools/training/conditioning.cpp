// VAE targets and frozen-AR conditioning: prepare_nar_targets.py and
// prepare_nar_dataset.py, product path only.
#include "training/preparation.h"

#include "training/gguf_file.h"
#include "training/preparation_detail.h"
#include "training/process.h"
#include "training/sha256.h"
#include "training/workflow.h"
#include "yue2/audio.h"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>

namespace yue2::training {
namespace fs = std::filesystem;
using namespace detail;
namespace {

std::string with_frames(const std::string & sha, long long start, long long frames) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "-%07lld-%lld", start, frames);
    return sha.substr(0, 12) + buffer;
}

bool real_and_eligible(const json::Value & row) {
    return truthy(get(row, "eligible", B(true))) && get(row, "kind", S("real")).text == "real";
}

// numpy's pairwise float64 summation, which np.mean uses on a contiguous array.
double pairwise_sum(const double * values, std::size_t n) {
    if (n < 8) {
        double sum = 0;
        for (std::size_t i = 0; i < n; ++i) sum += values[i];
        return sum;
    }
    if (n <= 128) {
        double r[8];
        for (int j = 0; j < 8; ++j) r[j] = values[j];
        std::size_t i = 8;
        for (; i < n - n % 8; i += 8) for (int j = 0; j < 8; ++j) r[j] += values[i + j];
        double sum = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) sum += values[i];
        return sum;
    }
    std::size_t half = n / 2;
    half -= half % 8;
    return pairwise_sum(values, half) + pairwise_sum(values + half, n - half);
}

// float(np.sqrt(np.mean(values.astype(np.float64) ** 2)))
double latent_rms(const std::vector<float> & values) {
    std::vector<double> squares(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) squares[i] = static_cast<double>(values[i]) * values[i];
    return std::sqrt(pairwise_sum(squares.data(), squares.size()) / static_cast<double>(squares.size()));
}

std::string key_of(const json::Value & expected) { return sha256_hex(json::python_dumps_sorted(expected)); }

json::Value native_facts(const json::Value & event) {
    return make({{"cache_bytes", member(event, "bytes")}, {"cache_fnv", member(event, "fnv")}, {"cache_tokens", member(event, "tokens")},
                 {"cache_frames", member(event, "frames")}, {"cache_half", member(event, "half")}});
}

json::Value elapsed(std::chrono::steady_clock::time_point start) {
    return F(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
}

bool finite_payload(const std::string & bytes, bool half) {
    if (half) {
        for (std::size_t i = 0; i + 1 < bytes.size(); i += 2) {
            const auto bits = static_cast<std::uint16_t>(static_cast<unsigned char>(bytes[i]) | (static_cast<unsigned char>(bytes[i + 1]) << 8));
            if ((bits & 0x7c00) == 0x7c00) return false;
        }
        return true;
    }
    for (std::size_t i = 0; i + 3 < bytes.size(); i += 4) {
        float value;
        std::memcpy(&value, bytes.data() + i, 4);
        if (!std::isfinite(value)) return false;
    }
    return true;
}

// prepare_nar_dataset.verify_cache(): native output provenance and its inputs.
void verify_cache(const fs::path & cache, const std::string & key, const json::Value & row, const json::Value & target,
                  const fs::path & base, const fs::path & common, const fs::path & adapter, const std::string & kv_type) {
    const GgufFile file(cache);
    const auto field = [&](const char * name) {
        auto value = file.field(name);
        if (!value) throw std::invalid_argument(std::string("Cache metadata missing: ") + name);
        return *value;
    };
    if (field("general.architecture").text != "yue2_nar_sample" || field("yue2.training.cache.identity").text != key) {
        throw std::invalid_argument("Frozen NAR cache identity differs");
    }
    if (resolved_path(from_utf8(field("yue2.training.model").text)) != resolved_path(base)) throw std::invalid_argument("Cache base differs");
    const auto expected = resolved_path(common) + "=1.000000\n" + resolved_path(adapter) + "=1.000000\n";
    if (field("yue2.training.conditioning_adapters").text != expected || !field("yue2.training.fixed_adapters").text.empty()) {
        throw std::invalid_argument("Cache frozen adapters differ");
    }
    auto names = file.tensors();
    std::sort(names.begin(), names.end());
    if (names != std::vector<std::string>{"clean", "codes", "keys", "prefix", "values"}) throw std::invalid_argument("Unexpected cache payload");
    for (const auto & [name, path] : std::vector<std::pair<std::string, std::string>>{
             {"prefix", text(row, "prefix")}, {"codes", text(row, "codes")}, {"clean", text(target, "means")}}) {
        const auto type = name == "clean" ? GGML_TYPE_F32 : GGML_TYPE_I32;
        if (file.tensor(name).type != type || file.payload(name) != read_bytes(from_utf8(path))) {
            throw std::invalid_argument("Cache input payload differs: " + name);
        }
    }
    const long long tokens = static_cast<long long>(fs::file_size(from_utf8(text(row, "prefix"))) / 4) + integer(member(row, "frames")) + 1;
    if (integer(field("yue2.training.ar_tokens")) != tokens) throw std::invalid_argument("Cache conditioning length differs");
    for (const char * name : {"keys", "values"}) {
        const auto & tensor = file.tensor(name);
        const auto type = kv_type == "f16" ? GGML_TYPE_F16 : GGML_TYPE_F32;
        if (tensor.type != type || tensor.ne[0] != 128 || tensor.ne[1] != tokens || tensor.ne[2] != 8 || tensor.ne[3] != 28) {
            throw std::invalid_argument("Cache conditioning shape/type differs");
        }
        if (!finite_payload(file.payload(name), kv_type == "f16")) throw std::invalid_argument("Nonfinite conditioning");
    }
}

std::vector<std::string> batch_command(const std::string & binary, const std::string & plan, const std::string & base,
                                       const std::string & common, const std::string & adapter, const std::string & device,
                                       const std::string & threads, const std::string & kv_type, const std::string & batch,
                                       const std::optional<std::string> & stop_file) {
    std::vector<std::string> command{binary, "--prepare-batch", plan, "--conditioning-only", "--model", base, "--conditioning-lora",
        common, "--conditioning-lora", adapter, "--device", device, "--threads", threads, "--cache-kv-type", kv_type,
        "--conditioning-batch", batch};
    if (stop_file) { command.push_back("--stop-file"); command.push_back(*stop_file); }
    return command;
}

// prepare_nar_dataset.run_batch(): hand each CACHE_EVENT to on_event.
void run_batch(const std::vector<std::string> & command, const fs::path & cwd, const fs::path & log,
               const std::function<void(const json::Value &)> & on_event) {
    const std::vector<std::string> arguments(command.begin() + 1, command.end());
    const int code = run_streaming(from_utf8(command[0]), arguments, cwd, log, [&](const std::string & line) {
        if (line.rfind("CACHE_EVENT ", 0) == 0) {
            on_event(json::parse(line.substr(12)));
            std::cout << line << std::endl;
        }
    });
    if (code) throw std::runtime_error(log_tail(log, 4000));
}

std::string count_text(long long value) { return std::to_string(value); }

} // namespace

void prepare_targets(const TargetOptions & o) {
    if (o.threads < 1 || o.core_frames < 1 || o.halo_frames < 0) throw std::invalid_argument("Invalid VAE preparation profile");
    set_environment("GGML_CUDA_TF32", "0");
    const auto data = read_json(o.dataset);
    fs::create_directories(o.out);
    if (!truthy(member(data, "completed"))) throw std::invalid_argument("Dataset preparation is incomplete");
    std::vector<json::Value> rows;
    for (const auto & row : member(data, "rows").array) if (real_and_eligible(row)) rows.push_back(row);
    if (rows.empty()) throw std::invalid_argument("No eligible VAE source rows");
    const auto identity = make({{"format", S("native-vae-target-v1")}, {"vae_sha256", S(digest(o.vae))},
        {"native_runtime", runtime_identity(from_utf8(resolved_path(o.binary)))}, {"device", S(o.device)}, {"threads", I(o.threads)},
        {"core_frames", I(o.core_frames)}, {"encoder_halo_frames", I(16)}, {"posterior", S("mean")},
        {"precision", S("GGML F32 weights/activations from folded F16/F32 VAE")}, {"halo_frames", I(o.halo_frames)},
        {"resampler", S("native per-channel centered polyphase Kaiser5 FIR; unchanged amplitude; zero-extended context")},
        {"boundary_policy", S("Real source context up to halo; native encoder padding at actual recording edges")},
        {"precision_environment", make({{"GGML_CUDA_TF32", S("0")}})}});
    const auto index_path = o.out / "index.json";
    if (fs::exists(index_path) && !python_equal(member(read_json(index_path), "identity"), identity)) {
        throw std::invalid_argument("VAE target profile changed; use a new output folder. Existing targets are preserved.");
    }
    struct Entry { json::Value row; fs::path target, metadata; json::Value expected; };
    std::vector<Entry> entries;
    auto pending = json::array();
    std::vector<std::pair<std::string, fs::path>> checked;
    fs::path read_source;
    std::string bytes;
    for (const auto & row : rows) {
        const auto source = from_utf8(text(row, "source"));
        // Windows of one recording are adjacent; read each recording once.
        if (source != read_source) { bytes = read_bytes(source); read_source = source; }
        const auto info = audio::audio_info(reinterpret_cast<const std::uint8_t *>(bytes.data()), bytes.size());
        const auto sha = text(row, "source_sha256");
        if (std::none_of(checked.begin(), checked.end(), [&](const auto & c) { return c.first == sha; })) {
            if (sha256_hex(bytes) != sha) throw std::invalid_argument("Source changed");
            checked.emplace_back(sha, source);
        }
        if (integer(get(row, "native_decoder_trim_samples", I(0)))) throw std::invalid_argument("Invalid native decoder edge declaration");
        const long long start = integer(member(row, "start_frame")), frames = integer(member(row, "frames"));
        const auto total = static_cast<long long>(std::floor(static_cast<double>(info.frames) / info.sample_rate * 25));
        if (start < 0 || frames < 1 || start + frames > total || o.halo_frames < 0) throw std::invalid_argument("Invalid target core");
        const long long begin = std::max(0LL, start - o.halo_frames), end = std::min(total, start + frames + o.halo_frames);
        const auto folder = o.out / with_frames(sha, start, frames);
        fs::create_directory(folder);
        const auto target = folder / "means.f32", metadata = folder / "target.json";
        auto expected = make({{"source_sha256", S(sha)}, {"start_frame", I(start)}, {"frames", I(frames)},
                              {"context_begin_frame", I(begin)}, {"context_end_frame", I(end)}});
        for (const auto & [key, value] : identity.object) expected.object[key] = value;
        if (fs::exists(target) && fs::exists(metadata)) {
            const auto previous = read_json(metadata);
            if (!python_equal(member(previous, "identity"), expected) || digest(target) != text(previous, "means_sha256")) {
                throw std::invalid_argument("VAE target identity changed; use a new output folder");
            }
        } else {
            if (fs::exists(target)) {
                int number = 1;
                while (fs::exists(folder / ("means.unregistered-" + std::to_string(number) + ".f32"))) ++number;
                fs::rename(target, folder / ("means.unregistered-" + std::to_string(number) + ".f32"));
            }
            pending.array.push_back(make({{"audio", S(resolved_path(source))}, {"begin_frame", I(begin)}, {"end_frame", I(end)},
                {"core_offset", I(start - begin)}, {"frames", I(frames)}, {"out", S(resolved_path(target))}, {"identity", S(key_of(expected))},
                {"decoder_trim_samples", I(0)}}));
        }
        entries.push_back({row, target, metadata, expected});
    }
    const auto record = [&](const Entry & entry) {
        const auto values = read_f32(entry.target);
        const auto frames = integer(member(entry.row, "frames"));
        bool finite = static_cast<long long>(values.size()) == frames * 64;
        for (const auto value : values) finite = finite && std::isfinite(value);
        if (!finite) throw std::invalid_argument("Invalid native VAE target payload");
        if (!fs::exists(entry.metadata)) {
            auto shape = json::array();
            shape.array = {I(frames), I(64)};
            save(entry.metadata, make({{"identity", entry.expected}, {"means_sha256", S(digest(entry.target))}, {"shape", shape},
                                       {"latent_rms", F(latent_rms(values))}}));
        }
        const auto previous = read_json(entry.metadata);
        if (!python_equal(member(previous, "identity"), entry.expected) || digest(entry.target) != text(previous, "means_sha256")) {
            throw std::invalid_argument("VAE target changed");
        }
        return make({{"source_sha256", member(entry.row, "source_sha256")}, {"track", member(entry.row, "track")},
                     {"start_frame", member(entry.row, "start_frame")}, {"frames", member(entry.row, "frames")},
                     {"split", member(entry.row, "split")}, {"means", S(resolved_path(entry.target))},
                     {"means_sha256", member(previous, "means_sha256")}});
    };
    // Write a manifest even before the first event; partial data is never ready.
    save(index_path, make({{"completed", B(false)}, {"identity", identity}, {"rows", json::array()}}));
    const auto total = I(static_cast<long long>(rows.size()));
    if (!pending.array.empty()) {
        const auto plan = o.out / "pending.json";
        save(plan, pending);
        std::vector<std::string> command{resolved_path(o.binary), "--model", resolved_path(o.vae), "--plan", resolved_path(plan),
                                         "--device", o.device, "--threads", count_text(o.threads), "--core-frames", count_text(o.core_frames)};
        if (o.stop_file) { command.push_back("--stop-file"); command.push_back(resolved_path(*o.stop_file)); }
        long long completed = static_cast<long long>(entries.size() - pending.array.size());
        const auto start = std::chrono::steady_clock::now();
        save(o.out / "progress.json", make({{"stage", S("vae-targets-native")}, {"backend", S(o.device)}, {"current", I(completed)}, {"total", total}}));
        std::map<std::string, const Entry *> by_path;
        for (const auto & entry : entries) by_path[resolved_path(entry.target)] = &entry;
        const std::vector<std::string> arguments(command.begin() + 1, command.end());
        const int code = run_streaming(from_utf8(command[0]), arguments, {}, o.out / "encode.log", [&](const std::string & line) {
            if (line.rfind("VAE_EVENT ", 0) != 0) return;
            const auto event = json::parse(line.substr(10));
            const auto & item = pending.array.at(static_cast<std::size_t>(integer(member(event, "index"))));
            if (text(event, "identity") != text(item, "identity") || !python_equal(member(event, "frames"), member(item, "frames"))) {
                throw std::invalid_argument("Native VAE event identity differs");
            }
            const auto & entry = *by_path.at(text(item, "out"));
            record(entry);
            ++completed;
            save(o.out / "progress.json", make({{"stage", S("vae-targets-native")}, {"backend", S(o.device)}, {"current", I(completed)},
                                                {"total", total}, {"track", member(entry.row, "track")}}));
            std::cout << "VAE " << completed << " " << rows.size() << " " << text(entry.row, "track") << " "
                      << integer(member(entry.row, "start_frame")) << std::endl;
        });
        save(o.out / "command.json", make({{"command", strings(command)}, {"returncode", I(code)}, {"seconds", elapsed(start)}}));
        if (code) throw std::runtime_error(log_tail(o.out / "encode.log", 5000));
    }
    auto prepared = json::array();
    for (const auto & entry : entries) {
        if (!fs::exists(entry.target)) {
            if (stop_requested(o.stop_file)) return;
            throw std::invalid_argument("Native VAE output missing");
        }
        if (!fs::exists(entry.metadata)) throw std::invalid_argument("Native VAE output has no committed provenance");
        prepared.array.push_back(record(entry));
        save(index_path, make({{"completed", B(false)}, {"identity", identity}, {"rows", prepared}}));
    }
    for (const auto & [sha, source] : checked) if (digest(source) != sha) throw std::invalid_argument("Source changed during VAE encoding");
    save(index_path, make({{"completed", B(true)}, {"identity", identity}, {"rows", prepared}}));
    save(o.out / "progress.json", make({{"stage", S("vae-targets-native-complete")}, {"backend", S(o.device)}, {"current", total}, {"total", total}}));
}

void prepare_conditioning(const ConditioningOptions & o) {
    if (o.threads < 1) throw std::invalid_argument("threads must be positive");
    if (o.device != "cpu" && o.device != "cuda") throw std::invalid_argument("conditioning device must be cpu or cuda");
    if (o.kv_type != "f16" && o.kv_type != "f32") throw std::invalid_argument("cache K/V type must be f16 or f32");
    if (o.storage != "cache" && o.storage != "rolling" && o.storage != "resident") throw std::invalid_argument("storage must be cache, rolling or resident");
    set_environment("GGML_CUDA_TF32", "0");
    const auto config = read_json(o.config);
    const auto dataset_path = from_utf8(text(config, "training_dataset"));
    const auto data = read_json(dataset_path);
    const auto targets = read_json(o.targets);
    if (!truthy(member(data, "completed")) || !truthy(member(targets, "completed"))) {
        throw std::invalid_argument("AR preparation and VAE targets must be complete");
    }
    std::vector<json::Value> rows;
    for (const auto & row : member(data, "rows").array) if (real_and_eligible(row)) rows.push_back(row);
    if (rows.empty()) throw std::invalid_argument("Dataset has no eligible real examples");
    // match_targets(): a one-to-one join; never a nearby offset or another split.
    const auto core = [](const json::Value & row) {
        return text(row, "source_sha256") + "/" + std::to_string(integer(member(row, "start_frame"))) + "/" +
               std::to_string(integer(member(row, "frames")));
    };
    std::map<std::string, json::Value> by_key;
    for (const auto & row : member(targets, "rows").array) {
        if (!by_key.emplace(core(row), row).second) throw std::invalid_argument("Duplicate VAE target core");
    }
    std::vector<std::pair<json::Value, json::Value>> pairs;
    for (const auto & row : rows) {
        const auto found = by_key.find(core(row));
        if (found == by_key.end() || text(found->second, "split") != text(row, "split")) {
            throw std::invalid_argument("VAE target missing or split differs");
        }
        pairs.emplace_back(row, found->second);
        by_key.erase(found);
    }
    if (!by_key.empty()) throw std::invalid_argument("Unexpected VAE target cores");
    fs::create_directories(o.out);
    const auto & models = member(config, "models");
    const auto base = from_utf8(resolved_path(from_utf8(text(models, "base"))));
    const auto common = from_utf8(resolved_path(from_utf8(text(models, "companion"))));
    const auto adapter = from_utf8(resolved_path(o.ar_adapter));
    const auto binary = from_utf8(resolved_path(o.binary));
    auto identity = make({{"format", S("yue2-nar-cache-job-v1")}, {"ar_dataset_sha256", S(digest(dataset_path))},
        {"target_index_sha256", S(digest(o.targets))}, {"vae_identity", member(targets, "identity")}, {"base_sha256", S(digest(base))},
        {"companion_sha256", S(digest(common))}, {"ar_adapter_sha256", S(digest(adapter))}, {"runtime", runtime_identity(binary)},
        {"device", S(o.device)}, {"threads", I(o.threads)}, {"tf32", B(false)}, {"conditioning_only", B(true)}, {"kv_type", S(o.kv_type)},
        {"prefill_batch", I(o.batch)}, {"flash_attention", B(false)}, {"preparation_source_sha256", S(conditioning_sha256)},
        {"prefix_policy", S("Exact AR training prefix and semantic IDs")}, {"replay_policy", S("Real targets only")},
        {"auxiliary", json::Value{}}});
    if (o.storage != "cache") identity.object["storage"] = S(o.storage);
    if (o.storage != "cache" && o.kv_type != "f16") throw std::invalid_argument("Rolling and resident conditioning use F16 K/V");
    const auto job_path = o.out / "identity.json";
    if (fs::exists(job_path) && !python_equal(read_json(job_path), identity)) {
        throw std::invalid_argument("NAR cache identity changed; use another output directory");
    }
    save(job_path, identity);

    struct Entry { json::Value row, target; fs::path cache, metadata; json::Value expected; std::string key; };
    std::vector<Entry> entries;
    auto pending = json::array();
    const auto pending_item = [](const Entry & entry) {
        return make({{"prefix", member(entry.row, "prefix")}, {"tokens", member(entry.row, "codes")}, {"latents", member(entry.target, "means")},
                     {"cache-out", S(resolved_path(entry.cache))}, {"cache-identity", S(entry.key)}});
    };
    for (const auto & [row, target] : pairs) {
        for (const auto & [path, hash] : std::vector<std::pair<std::string, std::string>>{
                 {text(row, "prefix"), text(row, "prefix_sha256")}, {text(row, "codes"), text(row, "codes_sha256")},
                 {text(target, "means"), text(target, "means_sha256")}}) {
            if (digest(from_utf8(path)) != hash) throw std::invalid_argument("NAR input bytes changed");
        }
        if (static_cast<long long>(fs::file_size(from_utf8(text(target, "means")))) != integer(member(row, "frames")) * 64 * 4) {
            throw std::invalid_argument("VAE target frame count differs");
        }
        const auto folder = o.out / with_frames(text(row, "source_sha256"), integer(member(row, "start_frame")), integer(member(row, "frames")));
        fs::create_directory(folder);
        Entry entry{row, target, folder / "sample.gguf", folder / "cache.json",
                    make({{"job", identity}, {"prefix_sha256", member(row, "prefix_sha256")}, {"codes_sha256", member(row, "codes_sha256")},
                          {"means_sha256", member(target, "means_sha256")}, {"frames", member(row, "frames")}}), {}};
        entry.key = key_of(entry.expected);
        entries.push_back(entry);
        if (o.storage == "resident") continue;
        if (o.storage == "rolling" && !fs::exists(entry.cache) && fs::exists(entry.metadata)) {
            const auto previous = read_json(entry.metadata);
            if (!python_equal(get(previous, "identity", json::Value{}), entry.expected) || !previous.find("native")) {
                throw std::invalid_argument("Rolling NAR cache record changed");
            }
            continue;
        }
        if (!fs::exists(entry.cache)) {
            pending.array.push_back(pending_item(entry));
            continue;
        }
        verify_cache(entry.cache, entry.key, row, target, base, common, adapter, o.kv_type);
        if (fs::exists(entry.metadata)) {
            const auto previous = read_json(entry.metadata);
            if (!python_equal(member(previous, "identity"), entry.expected) || digest(entry.cache) != text(previous, "cache_sha256")) {
                throw std::invalid_argument("Frozen NAR cache changed");
            }
        } else {
            // After a coordinator crash there is no trusted payload checksum.
            // Preserve the unregistered file and recompute instead of blessing
            // potentially corrupted K/V bytes with a newly invented checksum.
            int number = 1;
            while (fs::exists(folder / ("sample.unregistered-" + std::to_string(number) + ".gguf"))) ++number;
            const auto quarantine = folder / ("sample.unregistered-" + std::to_string(number) + ".gguf");
            fs::rename(entry.cache, quarantine);
            save(with_suffix(quarantine, ".json"), make({{"reason", S("Missing committed cache checksum")}, {"identity", entry.expected},
                                                         {"sha256", S(digest(quarantine))}}));
            pending.array.push_back(pending_item(entry));
        }
    }
    const auto total = I(static_cast<long long>(entries.size()));
    if (!pending.array.empty()) {
        if (o.device == "cuda" && truthy(get(config, "inference_urls", json::array()))) {
            throw StageError("Waiting for inference_urls is not supported by yue2-train-run");
        }
        const auto plan = o.out / "pending.json";
        save(plan, pending);
        const auto command = batch_command(python_path(binary), python_path(plan), python_path(base), python_path(common), python_path(adapter),
            o.device, count_text(o.threads), o.kv_type, count_text(o.batch),
            o.stop_file ? std::optional<std::string>(python_path(*o.stop_file)) : std::nullopt);
        const auto start = std::chrono::steady_clock::now();
        long long completed = static_cast<long long>(entries.size() - pending.array.size());
        const auto committed = [&](const json::Value & event) {
            const auto & item = pending.array.at(static_cast<std::size_t>(integer(member(event, "index"))));
            const auto out = text(item, "cache-out");
            const auto found = std::find_if(entries.begin(), entries.end(), [&](const Entry & e) { return resolved_path(e.cache) == resolved_path(from_utf8(out)); });
            const auto & entry = *found;
            verify_cache(entry.cache, entry.key, entry.row, entry.target, base, common, adapter, o.kv_type);
            auto record = make({{"identity", entry.expected}, {"cache_sha256", S(digest(entry.cache))}});
            if (o.storage == "rolling") {
                // Keep only what proves a later regeneration is byte-identical.
                record.object["native"] = native_facts(event);
                save(entry.metadata, record);
                fs::remove(entry.cache);
            } else {
                save(entry.metadata, record);
            }
            ++completed;
            save(o.out / "progress.json", make({{"status", S("running")}, {"stage", S("frozen-ar-conditioning")}, {"current", I(completed)}, {"total", total}}));
        };
        try {
            run_batch(command, from_utf8(text(config, "workspace")), o.out / "prepare.log", committed);
        } catch (...) {
            save(o.out / "command.json", make({{"command", strings(command)}, {"seconds", elapsed(start)}}));
            throw;
        }
        save(o.out / "command.json", make({{"command", strings(command)}, {"seconds", elapsed(start)}}));
    }
    const auto stopped = [&](std::size_t current) {
        save(o.out / "progress.json", make({{"status", S("stopped")}, {"stage", S("frozen-ar-conditioning")},
                                            {"current", I(static_cast<long long>(current))}, {"total", total}}));
    };
    auto prepared = json::array();
    for (const auto & entry : entries) {
        auto row = entry.row;
        if (o.storage == "resident") {
            // No cache file exists; the path only names the window for the fitter.
            row.object["cache"] = S(resolved_path(entry.cache));
            row.object["means"] = member(entry.target, "means");
            row.object["means_sha256"] = member(entry.target, "means_sha256");
            prepared.array.push_back(row);
            continue;
        }
        if (o.storage == "rolling") {
            if (!fs::exists(entry.metadata)) {
                if (stop_requested(o.stop_file)) { stopped(prepared.array.size()); return; }
                throw std::invalid_argument("Native preparation did not record all caches");
            }
            const auto record = read_json(entry.metadata);
            if (!python_equal(get(record, "identity", json::Value{}), entry.expected) || !record.find("native")) {
                throw std::invalid_argument("Rolling NAR cache record changed");
            }
            row.object["cache"] = S(resolved_path(entry.cache));
            row.object["cache_sha256"] = member(record, "cache_sha256");
            row.object["cache_identity"] = S(entry.key);
            row.object["means"] = member(entry.target, "means");
            row.object["means_sha256"] = member(entry.target, "means_sha256");
            for (const auto & [key, value] : member(record, "native").object) row.object[key] = value;
            prepared.array.push_back(row);
            continue;
        }
        if (!fs::exists(entry.cache)) {
            if (stop_requested(o.stop_file)) { stopped(prepared.array.size()); return; }
            throw std::invalid_argument("Native preparation did not produce all caches");
        }
        const auto cache_hash = digest(entry.cache);
        if (!fs::exists(entry.metadata) ||
            !python_equal(read_json(entry.metadata), make({{"identity", entry.expected}, {"cache_sha256", S(cache_hash)}}))) {
            throw std::invalid_argument("Cache has no matching committed checksum");
        }
        row.object["cache"] = S(resolved_path(entry.cache));
        row.object["cache_sha256"] = S(cache_hash);
        row.object["means"] = member(entry.target, "means");
        row.object["means_sha256"] = member(entry.target, "means_sha256");
        prepared.array.push_back(row);
    }
    // Recheck every cache before marking the dataset ready for the trainer.
    if (digest(dataset_path) != text(identity, "ar_dataset_sha256") || digest(o.targets) != text(identity, "target_index_sha256")) {
        throw std::invalid_argument("Input manifest changed during preparation");
    }
    for (const auto & row : prepared.array) {
        for (const char * key : {"cache", "prefix", "codes", "means"}) {
            if (std::string(key) == "cache" && o.storage != "cache") continue;
            if (digest(from_utf8(text(row, key))) != text(row, (std::string(key) + "_sha256").c_str())) {
                throw std::invalid_argument("Input or cache changed during preparation");
            }
        }
    }
    save(o.out / "dataset.json", make({{"format", S("yue2-nar-prepared-dataset-v1")}, {"completed", B(true)}, {"frames", member(data, "frames")},
                                       {"identity", identity}, {"rows", prepared}}));
    const auto count = I(static_cast<long long>(pairs.size()));
    save(o.out / "progress.json", make({{"status", S("completed")}, {"stage", S("frozen-ar-conditioning")}, {"current", count}, {"total", count}}));
}

long long materialize(const fs::path & manifest, const std::vector<json::Value> & rows, const fs::path & binary_path, const fs::path & base,
                      const fs::path & common, const fs::path & adapter, const std::optional<fs::path> & stop_file) {
    const auto identity = member(read_json(manifest), "identity");
    // Conditioning identities record TF32 off; the Python left that to its caller.
    set_environment("GGML_CUDA_TF32", "0");
    // A segment may draw the same window more than once; generate it once.
    // Like the Python dict, the first draw keeps its place and the last its row.
    std::vector<std::string> order;
    std::map<std::string, json::Value> unique;
    for (const auto & row : rows) {
        if (fs::exists(from_utf8(text(row, "cache")))) continue;
        const auto path = resolved_path(from_utf8(text(row, "cache")));
        if (!unique.count(path)) order.push_back(path);
        unique[path] = row;
    }
    if (order.empty()) return 0;
    // The native tool runs from its own folder, so every path it reads is absolute.
    const auto out = from_utf8(resolved_path(manifest)).parent_path();
    const auto plan = out / "materialize.json";
    const auto absolute = [](const json::Value & row, const char * key) { return S(resolved_path(from_utf8(text(row, key)))); };
    auto items = json::array();
    for (const auto & path : order) {
        const auto & row = unique.at(path);
        items.array.push_back(make({{"prefix", absolute(row, "prefix")}, {"tokens", absolute(row, "codes")}, {"latents", absolute(row, "means")},
                                    {"cache-out", absolute(row, "cache")}, {"cache-identity", member(row, "cache_identity")}}));
    }
    save(plan, items);
    const auto binary = from_utf8(resolved_path(binary_path));
    const auto check = [&](const json::Value & event) {
        const auto & row = unique.at(resolved_path(from_utf8(text(event, "path"))));
        const auto facts = native_facts(event);
        bool same = true;
        for (const auto & [key, value] : facts.object) same = same && python_equal(value, member(row, key.c_str()));
        if (!same || digest(from_utf8(text(row, "cache"))) != text(row, "cache_sha256")) {
            fs::remove(from_utf8(text(row, "cache")));
            throw std::invalid_argument("Regenerated NAR cache differs from its recorded bytes: " + text(row, "cache"));
        }
    };
    run_batch(batch_command(python_path(binary), python_path(plan), resolved_path(base), resolved_path(common), resolved_path(adapter),
                            text(identity, "device"), std::to_string(integer(member(identity, "threads"))), text(identity, "kv_type"),
                            std::to_string(integer(member(identity, "prefill_batch"))),
                            stop_file ? std::optional<std::string>(resolved_path(*stop_file)) : std::nullopt),
              binary.parent_path(), out / "materialize.log", check);
    std::optional<std::string> missing;
    long long made = 0;
    for (const auto & path : order) {
        const auto cache = text(unique.at(path), "cache");
        if (fs::exists(from_utf8(cache))) ++made;
        else if (!missing) missing = cache;
    }
    if (missing && !stop_requested(stop_file)) throw std::invalid_argument("Native preparation did not regenerate " + *missing);
    return made;
}

} // namespace yue2::training
