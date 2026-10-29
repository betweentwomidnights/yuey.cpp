#include "training/preparation.h"

#include "training/dataset.h"
#include "training/preparation_detail.h"
#include "training/process.h"
#include "training/sha256.h"
#include "training/unicode.h"
#include "training/workflow.h"
#include "yue2/audio.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <set>

namespace yue2::training {
namespace fs = std::filesystem;
using namespace detail;

const char * const normalizer_sha256 = "be8943f22fd37ee0f0ce4bef32a17568b60a416da490eb12a296a028988f6fb0";
const char * const conditioning_sha256 = "d649fed4a7f9ea1d59dff161214ecbf9b7cded296b4227f2a7dc5eaf5e0528ea";
// The same scripts with CRLF line endings, as a Windows checkout ran them.
const char * const normalizer_sha256_crlf = "10a5670155cb6abb5bce1c72c30abb0def92bafc63b17a873661e20dd5e2b4a3";
const char * const conditioning_sha256_crlf = "324fddf538ff39267c10b4a2afcc8e3255f4c4be7c403dde0118a41dd4db188b";

std::string recorded_digest(const std::optional<std::string> & recorded, const char * committed, const char * crlf) {
    return recorded && (*recorded == committed || *recorded == crlf) ? *recorded : committed;
}

bool python_equal(const json::Value & a, const json::Value & b) {
    const auto numeric = [](const json::Value & v) { return v.type == json::Type::number || v.type == json::Type::boolean; };
    if (numeric(a) && numeric(b)) {
        if (a.type == json::Type::number && b.type == json::Type::number && json::is_integer(a) && json::is_integer(b)) {
            return std::strtoll(a.text.c_str(), nullptr, 10) == std::strtoll(b.text.c_str(), nullptr, 10);
        }
        const auto value = [](const json::Value & v) { return v.type == json::Type::boolean ? (v.boolean ? 1.0 : 0.0) : std::strtod(v.text.c_str(), nullptr); };
        return value(a) == value(b);
    }
    if (a.type != b.type) return false;
    switch (a.type) {
        case json::Type::null: return true;
        case json::Type::string: return a.text == b.text;
        case json::Type::array:
            if (a.array.size() != b.array.size()) return false;
            for (std::size_t i = 0; i < a.array.size(); ++i) if (!python_equal(a.array[i], b.array[i])) return false;
            return true;
        case json::Type::object:
            if (a.object.size() != b.object.size()) return false;
            for (const auto & [key, value] : a.object) {
                const auto found = b.object.find(key);
                if (found == b.object.end() || !python_equal(value, found->second)) return false;
            }
            return true;
        default: return false;
    }
}

namespace detail {

const json::Value & member(const json::Value & object, const char * key) {
    const auto * value = object.find(key);
    // A missing key is a KeyError in the Python, which no stage catches.
    if (!value) throw StageError(std::string("Missing key: ") + key);
    return *value;
}

std::string text(const json::Value & object, const char * key) {
    const auto & value = member(object, key);
    if (value.type != json::Type::string) throw StageError(std::string(key) + " must be a string");
    return value.text;
}

json::Value get(const json::Value & object, const char * key, json::Value fallback) {
    const auto * value = object.find(key);
    return value ? *value : fallback;
}

double number(const json::Value & value) {
    if (value.type == json::Type::boolean) return value.boolean;
    if (value.type != json::Type::number) throw StageError("Expected a number");
    return std::strtod(value.text.c_str(), nullptr);
}

long long integer(const json::Value & value) {
    if (value.type == json::Type::number && json::is_integer(value)) return std::strtoll(value.text.c_str(), nullptr, 10);
    return static_cast<long long>(number(value));
}

bool truthy(const json::Value & value) {
    switch (value.type) {
        case json::Type::null: return false;
        case json::Type::boolean: return value.boolean;
        case json::Type::number: return std::strtod(value.text.c_str(), nullptr) != 0;
        case json::Type::string: return !value.text.empty();
        case json::Type::array: return !value.array.empty();
        case json::Type::object: return !value.object.empty();
    }
    return false;
}

bool stop_requested(const std::optional<fs::path> & stop_file) { return stop_file && fs::exists(*stop_file); }

std::optional<fs::path> config_stop_file(const json::Value & config) {
    const auto * value = config.find("stop_file");
    if (!value || !truthy(*value)) return std::nullopt;
    return from_utf8(value->text);
}

std::string digest(const fs::path & path) { return sha256_file(path); }

template <class T> std::vector<T> read_array(const fs::path & path) {
    const auto bytes = read_bytes(path);
    std::vector<T> output(bytes.size() / sizeof(T));
    std::memcpy(output.data(), bytes.data(), output.size() * sizeof(T));
    return output;
}

std::vector<std::int32_t> read_i32(const fs::path & path) { return read_array<std::int32_t>(path); }
std::vector<float> read_f32(const fs::path & path) { return read_array<float>(path); }

std::string executable(const char * name) {
#ifdef _WIN32
    return std::string(name) + ".exe";
#else
    return name;
#endif
}

std::string log_tail(const fs::path & log, std::size_t characters) {
    const auto codes = unicode::decode(read_text(log));
    return unicode::encode(codes, codes.size() > characters ? codes.size() - characters : 0);
}

void execute(const std::vector<std::string> & command, const fs::path & log, const fs::path & cwd, const json::Value & inference_urls) {
    if (fs::exists(log)) throw StageError("Log exists: " + python_path(log));
    if (truthy(inference_urls)) throw StageError("Waiting for inference_urls is not supported by yue2-train-run");
    fs::create_directories(log.parent_path());
    const auto start = std::chrono::steady_clock::now();
    const std::vector<std::string> arguments(command.begin() + 1, command.end());
    const int code = run_to_log(from_utf8(command[0]), arguments, cwd, log);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    save(with_suffix(log, ".command.json"), make({{"command", strings(command)}, {"returncode", I(code)}, {"seconds", F(seconds)}}));
    if (code) throw std::runtime_error(log_tail(log, 5000));
}

void reject_research_option(const json::Value & object, const char * key) {
    if (const auto * value = object.find(key); value && truthy(*value)) {
        throw StageError(std::string(key) + " is a research option that yue2-train-run does not support");
    }
}

} // namespace detail

namespace {

json::Value load_config(const fs::path & path) {
    auto config = read_json(path);
    if (json::string(config, "format") != "yue2-ar-training-v1") throw std::invalid_argument("Unsupported job format");
    set_environment("GGML_CUDA_TF32", "0");
    return config;
}

std::string device_of(const json::Value & config) {
    const auto value = get(config, "device", S("cuda"));
    return value.type == json::Type::string ? value.text : "cuda";
}

std::string with_frames(const std::string & sha, long long start, long long frames) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "-%07lld-%lld", start, frames);
    return sha.substr(0, 12) + buffer;
}

// One decoded recording at a time: windows of a source are planned together.
class SourceAudio {
public:
    const audio::PcmAudio & get(const fs::path & path) {
        if (path != path_) { audio_ = audio::read_audio_pcm(path); path_ = path; }
        return audio_;
    }
private:
    fs::path path_;
    audio::PcmAudio audio_;
};

} // namespace

void prepare_semantic(const fs::path & config_path) {
    const auto config = load_config(config_path);
    const auto root = from_utf8(text(config, "workspace"));
    const auto out = from_utf8(text(config, "output"));
    fs::create_directories(out);
    reject_research_option(config, "reuse_semantic_manifests");
    const auto binary = from_utf8(text(config, "bin")) / executable("yue2-semantic-tokenize");
    const auto model = from_utf8(text(member(config, "models"), "semantic"));
    const auto device = device_of(config);
    const auto fingerprint = make({{"model_sha256", S(digest(model))}, {"runtime", runtime_identity(binary)}, {"device", S(device)},
        {"preprocessing", S("native full-track MERT/v9; 25Hz; audio reader/resampler pinned by binary")}});
    const auto stop_file = config_stop_file(config);
    const auto & sources = member(config, "sources").array;
    auto rows = json::array();
    for (std::size_t index = 0; index < sources.size(); ++index) {
        if (stop_requested(stop_file)) return;
        const auto & source = sources[index];
        const auto path = from_utf8(text(source, "path"));
        const auto name = path.filename().u8string();
        const auto bytes = read_bytes(path);
        const auto sha = sha256_hex(bytes);
        if (sha != text(source, "sha256")) throw std::invalid_argument("Source changed: " + name);
        const auto folder = out / "semantic" / sha;
        fs::create_directories(folder);
        const auto target = folder / "codes.i32", metadata = folder / "cache.json";
        auto identity = make({{"source_sha256", S(sha)}});
        for (const auto & [key, value] : fingerprint.object) identity.object[key] = value;
        save(out / "preparation-progress.json", make({{"stage", S("semantic")}, {"current", I(static_cast<long long>(index))},
                                                      {"total", I(static_cast<long long>(sources.size()))}, {"track", S(name)}}));
        json::Value provenance;
        if (fs::exists(target)) {
            if (!fs::exists(metadata)) throw std::invalid_argument("Unfinished cache requires inspection: " + python_path(folder));
            const auto previous = read_json(metadata);
            if (!python_equal(member(previous, "identity"), identity) || digest(target) != text(previous, "codes_sha256")) {
                throw std::invalid_argument("Cached semantic identity differs");
            }
            provenance = member(previous, "provenance");
        } else {
            const auto temporary = with_suffix(target, ".i32.partial");
            execute({python_path(binary), "--model", python_path(model), "--audio", python_path(path), "--out", python_path(temporary),
                     "--device", device}, folder / "encode.log", root, get(config, "inference_urls", json::array()));
            fs::rename(temporary, target);
            provenance = make({{"encoded", B(true)}});
        }
        const auto codes = read_i32(target);
        const auto info = audio::audio_info(reinterpret_cast<const std::uint8_t *>(bytes.data()), bytes.size());
        bool valid = !codes.empty();
        for (const auto code : codes) valid = valid && code >= 0 && code < 32768;
        const double duration = static_cast<double>(info.frames) / info.sample_rate;
        if (!valid || std::abs(static_cast<double>(codes.size()) - duration * 25) > 2) {
            throw std::invalid_argument("Invalid frame count or codec IDs for " + name);
        }
        const auto record = make({{"identity", identity}, {"codes", P(target)}, {"codes_sha256", S(digest(target))},
                                  {"frames", I(static_cast<long long>(codes.size()))}, {"provenance", provenance}});
        if (!fs::exists(metadata)) save(metadata, record);
        auto row = source;
        for (const char * key : {"codes", "codes_sha256", "frames", "provenance"}) row.object[key] = record.object.at(key);
        rows.array.push_back(row);
        save(out / "semantic-index.json", make({{"format", S("yue2-ar-semantic-index-v1")}, {"completed", B(index + 1 == sources.size())},
                                                {"fingerprint", fingerprint}, {"rows", rows}}));
        std::cout << "SEMANTIC " << index + 1 << " " << sources.size() << " " << name << std::endl;
    }
    const auto count = I(static_cast<long long>(rows.array.size()));
    save(out / "preparation-progress.json", make({{"stage", S("semantic-complete")}, {"current", count}, {"total", count}}));
}

void prepare_scores(const fs::path & config_path) {
    const auto config = load_config(config_path);
    const auto out = from_utf8(text(config, "output"));
    fs::create_directories(out);
    const auto index = read_json(out / "semantic-index.json");
    if (!truthy(member(index, "completed"))) throw std::invalid_argument("Semantic preparation is incomplete");
    const auto & preparation = member(config, "preparation");
    const auto seconds_value = member(preparation, "window_seconds");
    const double seconds = number(seconds_value);
    const long long frames = round_half_even(seconds * 25);
    const auto root = from_utf8(text(config, "workspace"));
    const auto binary = from_utf8(text(config, "bin"));
    const auto & models = member(config, "models");
    const auto score_tool = from_utf8(text(config, "score_binary"));
    const auto transcriber = binary / executable("yue2-transcribe");
    const auto prefix_tool = binary / executable("yue2-ar-train");
    // A run the Python started keeps the digest it recorded for these rules.
    const auto dataset_output = get(preparation, "dataset_output", S("dataset.json")).text;
    std::optional<std::string> recorded;
    if (fs::exists(out / from_utf8(dataset_output))) {
        const auto previous = read_json(out / from_utf8(dataset_output));
        if (const auto * tools = previous.find("tools")) {
            if (const auto * value = tools->find("normalizer_sha256"); value && value->type == json::Type::string) recorded = value->text;
        }
    }
    const auto normalizer = recorded_digest(recorded, normalizer_sha256, normalizer_sha256_crlf);
    const auto tool_identity = make({{"normalizer_sha256", S(normalizer)}, {"score_runtime", runtime_identity(score_tool)},
        {"transcription_runtime", runtime_identity(transcriber)}, {"prefix_runtime", runtime_identity(prefix_tool)},
        {"transcription_model_sha256", S(digest(from_utf8(text(models, "transcriber"))))},
        {"text_tokenizer_sha256", S(digest(from_utf8(text(models, "tokenizer"))))},
        {"precision_environment", make({{"GGML_CUDA_TF32", S("0")}})}});
    const auto example_directory = get(preparation, "example_directory", S("examples")).text;
    const auto dataset_name = get(preparation, "dataset_output", S("dataset.json")).text;
    if (from_utf8(example_directory).filename().u8string() != example_directory || from_utf8(dataset_name).filename().u8string() != dataset_name) {
        throw std::invalid_argument("Prepared names must be local basenames");
    }
    reject_research_option(preparation, "reuse_examples_manifest");
    const auto trigger = text(config, "trigger");
    const auto device = device_of(config);
    const auto urls = get(config, "inference_urls", json::array());
    const auto stop_file = config_stop_file(config);

    struct Planned { const json::Value * source; CaptionMetadata metadata; long long start; };
    std::vector<Planned> planned;
    for (const auto & source : member(index, "rows").array) {
        const auto metadata = caption_metadata(text(source, "caption"));
        const auto real_frames = std::min(integer(member(source, "frames")),
                                          static_cast<long long>(std::floor(number(member(source, "seconds")) * 25)));
        for (const auto start : window_starts(real_frames, frames, metadata.bpm)) planned.push_back({&source, metadata, start});
    }
    SourceAudio decoded;
    auto rows = json::array();
    for (std::size_t number_index = 0; number_index < planned.size(); ++number_index) {
        if (stop_requested(stop_file)) return;
        const auto & [source_pointer, metadata, start] = planned[number_index];
        const auto & source = *source_pointer;
        const auto path = from_utf8(text(source, "path"));
        const auto name = path.filename().u8string();
        if (digest(with_suffix(path, ".txt")) != text(source, "caption_sha256")) throw std::invalid_argument("Caption changed: " + name);
        const auto folder = out / from_utf8(example_directory) / with_frames(text(source, "sha256"), start, frames);
        fs::create_directories(folder);
        const auto record_path = folder / "example.json";
        const auto identity = make({{"source_sha256", member(source, "sha256")}, {"caption_sha256", member(source, "caption_sha256")},
            {"semantic_sha256", member(source, "codes_sha256")}, {"start_frame", I(start)}, {"frames", I(frames)},
            {"tools", tool_identity}, {"trigger", S(trigger)}});
        save(out / "preparation-progress.json", make({{"stage", S("scores")}, {"current", I(static_cast<long long>(number_index))},
            {"total", I(static_cast<long long>(planned.size()))}, {"track", S(name)}, {"start_frame", I(start)}}));
        json::Value record;
        if (fs::exists(record_path)) {
            record = read_json(record_path);
            if (!python_equal(member(record, "identity"), identity)) {
                throw std::invalid_argument("Prepared example identity differs: " + python_path(folder));
            }
            if (truthy(member(record, "eligible"))) {
                for (const std::string key : {"prefix", "codes", "score"}) {
                    if (digest(from_utf8(text(record, key.c_str()))) != text(record, (key + "_sha256").c_str())) {
                        throw std::invalid_argument("Prepared example changed");
                    }
                }
            }
        } else {
            auto style = metadata.style;
            while (!style.empty() && style.back() == '.') style.pop_back();
            record = make({{"identity", identity}, {"source", member(source, "path")}, {"track", S(name)},
                {"source_sha256", member(source, "sha256")}, {"split", member(source, "split")}, {"frames", I(frames)},
                {"start_frame", I(start)}, {"directory", P(folder)}, {"bpm", F(metadata.bpm)}, {"key", S(metadata.key)},
                {"caption", S(trigger + ". Instrumental music. " + style + ".")},
                {"caption_policy", S("Sidecar style preserved; BPM/key encoded in score from absolute note timing and pitches")},
                {"predict_end", B(false)}, {"eligible", B(false)}});
            const auto all_codes = read_i32(from_utf8(text(source, "codes")));
            if (start + frames > static_cast<long long>(all_codes.size())) throw std::invalid_argument("Source code window out of bounds");
            const std::vector<std::int32_t> codes(all_codes.begin() + start, all_codes.begin() + start + frames);
            const auto code_path = folder / "codes.i32";
            const std::string code_bytes(reinterpret_cast<const char *>(codes.data()), codes.size() * sizeof(std::int32_t));
            if (fs::exists(code_path)) {
                if (read_bytes(code_path) != code_bytes) throw std::invalid_argument("Prepared code window changed");
            } else {
                std::ofstream(code_path, std::ios::binary).write(code_bytes.data(), static_cast<std::streamsize>(code_bytes.size()));
            }
            const auto audio_path = folder / "source.wav", events_path = folder / "events.json";
            if (!fs::exists(audio_path)) {
                const auto & pcm = decoded.get(path);
                const auto rate = pcm.sample_rate, channels = pcm.channels;
                const long long offset = round_half_even(static_cast<double>(start) / 25 * rate);
                const long long wanted = round_half_even(seconds * rate);
                const long long available = static_cast<long long>(pcm.interleaved_samples.size()) / channels;
                const long long got = std::max(0LL, std::min(wanted, available - offset));
                if (got != wanted) throw std::invalid_argument("Insufficient source audio");
                const std::vector<float> window(pcm.interleaved_samples.begin() + offset * channels,
                                                pcm.interleaved_samples.begin() + (offset + got) * channels);
                const auto wav = audio::encode_wav_float(window, rate, channels);
                std::ofstream(audio_path, std::ios::binary).write(reinterpret_cast<const char *>(wav.data()), static_cast<std::streamsize>(wav.size()));
            }
            if (!fs::exists(events_path)) {
                execute({python_path(transcriber), "--model", text(models, "transcriber"), "--audio", python_path(audio_path),
                         "--output", python_path(folder / "transcribed.abc"), "--events", python_path(events_path), "--full",
                         "--device", device}, folder / "transcribe.log", root, urls);
            }
            try {
                const auto events = read_json(events_path).object.at("events");
                auto [abc, stats] = score_from_events(events, seconds_value, metadata.bpm, metadata.key);
                if (number(stats.object.at("max_rounding_seconds")) > number(stats.object.at("max_expected_grid_error_seconds")) + 1e-5) {
                    throw std::invalid_argument("Timing quantization exceeds one grid unit");
                }
                const auto normalized = folder / "normalized.abc";
                if (fs::exists(normalized) && read_text(normalized) != abc) throw std::invalid_argument("Normalized score changed");
                write_text(normalized, abc);
                const auto used = folder / "used.abc";
                if (!fs::exists(used)) {
                    execute({python_path(score_tool), "--input", python_path(normalized), "--output", python_path(used), "--instrumental",
                             "--midi-dir", python_path(folder / "midi")}, folder / "score-validation.log", root, json::array());
                }
                const auto prefix = folder / "prefix.i32";
                if (!fs::exists(prefix)) {
                    execute({python_path(prefix_tool), "--prefix-out", python_path(prefix), "--tokenizer", text(models, "tokenizer"),
                             "--abc", python_path(used), "--instrumental", "--prompt", text(record, "caption")},
                            folder / "prefix.log", root, json::array());
                }
                const bool sparse = integer(stats.object.at("retained_monophonic_notes")) < 8;
                record.object["eligible"] = B(true);
                record.object["score"] = P(used);
                record.object["score_sha256"] = S(digest(used));
                record.object["prefix"] = P(prefix);
                record.object["prefix_sha256"] = S(digest(prefix));
                record.object["codes"] = P(code_path);
                record.object["codes_sha256"] = S(digest(code_path));
                record.object["prefix_tokens"] = I(static_cast<long long>(fs::file_size(prefix) / 4));
                record.object["score_stats"] = stats;
                record.object["sparse_score"] = B(sparse);
            } catch (const std::invalid_argument & error) {
                record.object["error"] = S(error.what());
            } catch (const std::runtime_error & error) {
                record.object["error"] = S(error.what());
            }
            save(record_path, record);
        }
        rows.array.push_back(record);
        save(out / from_utf8(dataset_name), make({{"format", S("yue2-ar-prepared-dataset-v1")},
            {"completed", B(number_index + 1 == planned.size())}, {"frames", I(frames)}, {"window_seconds", seconds_value},
            {"tools", tool_identity}, {"rows", rows}, {"manual_note_accuracy_verified", B(false)}}));
        const auto * tokens = record.find("prefix_tokens");
        std::cout << "SCORE " << number_index + 1 << " " << planned.size() << " " << name << " " << start << " "
                  << (truthy(member(record, "eligible")) ? "True" : "False") << " "
                  << (tokens ? tokens->text : get(record, "error", S("None")).text) << std::endl;
    }
    for (const char * split : {"train", "validation"}) {
        std::set<std::string> expected, actual;
        for (const auto & source : member(config, "sources").array) if (text(source, "split") == split) expected.insert(text(source, "sha256"));
        for (const auto & row : rows.array) {
            if (text(row, "split") == split && truthy(member(row, "eligible"))) actual.insert(text(row, "source_sha256"));
        }
        if (actual != expected) throw std::invalid_argument(std::string("At least one whole recording has no eligible score in ") + split);
    }
    const auto count = I(static_cast<long long>(rows.array.size()));
    save(out / "preparation-progress.json", make({{"stage", S("scores-complete")}, {"current", count}, {"total", count}}));
}

} // namespace yue2::training
