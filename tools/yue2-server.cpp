// yue2-server: gary4local-style HTTP service for YuE2 generation, covers, and
// audio-to-score transcription. Every compute request returns a session id at
// once; a single worker thread runs jobs in order and clients poll
// /poll_status/<id>, the same contract sa3-server and the gary4local Python
// services expose to gary4juce. See docs/server.md.
#include "yue2/audio.h"
#include "yue2/generation_pipeline.h"
#include "yue2/mert2_encoder.h"
#include "yue2/runtime_info.h"
#include "yue2/transcription.h"

#include "server/base64.h"
#include "server/http.h"
#include "server/json.h"
#include "server/policy.h"
#include "server/prompts.h"
#include "yue2_ui_html.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <locale>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

namespace fs = std::filesystem;
namespace json = yue2::server::json;
using yue2::server::HttpRequest;
using yue2::server::HttpResponse;

constexpr int kDefaultPort = 8007;
// The VAE emits one 64-channel latent frame per 1920 samples at 48 kHz, and
// each semantic codec token becomes one frame.
constexpr double kSemanticTokensPerSecond = 25.0;
constexpr auto kFinishedJobLifetime = std::chrono::minutes(5);
// Automatic tier selection prefers precision; small GPUs pass --encoding.
const std::vector<std::string> kEncodingPreference = {"BF16", "F16", "Q8_0", "Q5_K_M", "Q4_K_M", "F32"};

std::atomic<bool> stop_requested{false};

void stop_signal(int) { stop_requested.store(true); }

bool starts_with(std::string_view value, std::string_view prefix) {
    return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

bool ends_with(std::string_view value, std::string_view suffix) {
    return value.size() >= suffix.size() &&
        value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string upper(std::string value) {
    for (auto & c : value) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return value;
}

std::string environment(const char * name) {
    const char * value = std::getenv(name);
    return value ? std::string(value) : std::string();
}

std::string real(double value) {
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output.precision(17);
    output << value;
    return output.str();
}

std::string json_bool(bool value) { return value ? "true" : "false"; }

std::string json_path(const fs::path & path) {
    return path.empty() ? "null" : json::quote(path.string());
}

std::uint64_t random_u64() {
    static std::mutex mutex;
    static std::mt19937_64 generator(
        std::random_device{}() ^
        static_cast<std::uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::lock_guard<std::mutex> lock(mutex);
    return generator();
}

std::string random_session_id() {
    char buffer[33];
    std::snprintf(buffer, sizeof buffer, "%016llx%016llx",
        static_cast<unsigned long long>(random_u64()), static_cast<unsigned long long>(random_u64()));
    return buffer;
}

// ---------------------------------------------------------------------------
// Configuration

struct Configuration {
    yue2::server::HttpServerOptions http;
    fs::path models_dir = "models";
    fs::path adapters_dir;
    std::string encoding = "auto";
    fs::path generation_model;
    fs::path vae;
    fs::path tokenizer;
    fs::path transcription_model;
    fs::path semantic_tokenizer_model;
    std::string device;
    int threads = 0;
    bool keep_models = false;
    bool force_unload = false;
    // Ceiling on planner-chosen length. Zero lets a request run to the model's
    // own ending, which suits a local install and not a shared backend.
    double natural_max_seconds = 180.0;
    double planning_overrun = 2.0;
    std::uint32_t planning_loop_bars = 16;
    std::vector<yue2::LoraAdapterSpec> loras;
    std::vector<yue2::LoraAdapterSpec> instrumental_loras;
    std::vector<yue2::LoraAdapterSpec> continuation_loras;
};

bool has(int argc, char ** argv, std::string_view name) {
    for (int index = 1; index < argc; ++index) {
        if (std::string_view(argv[index]) == name) return true;
    }
    return false;
}

std::string option(int argc, char ** argv, std::string_view name) {
    for (int index = 1; index < argc; ++index) {
        if (std::string_view(argv[index]) != name) continue;
        if (index + 1 == argc || std::string_view(argv[index + 1]).rfind("--", 0) == 0) {
            throw std::invalid_argument(std::string(name) + " requires a value");
        }
        return argv[index + 1];
    }
    return {};
}

std::vector<std::string> options(int argc, char ** argv, std::string_view name) {
    std::vector<std::string> output;
    for (int index = 1; index < argc; ++index) {
        if (std::string_view(argv[index]) != name) continue;
        if (index + 1 == argc || std::string_view(argv[index + 1]).rfind("--", 0) == 0) {
            throw std::invalid_argument(std::string(name) + " requires a value");
        }
        output.emplace_back(argv[++index]);
    }
    return output;
}

yue2::LoraAdapterSpec lora_spec(const std::string & value) {
    yue2::LoraAdapterSpec output;
    const auto separator = value.rfind('=');
    if (separator == std::string::npos) {
        output.path = value;
        return output;
    }
    output.path = value.substr(0, separator);
    std::size_t used = 0;
    output.strength = std::stof(value.substr(separator + 1), &used);
    if (output.path.empty() || used != value.size() - separator - 1 ||
        !std::isfinite(output.strength)) {
        throw std::invalid_argument("--lora expects PATH or PATH=SCALE");
    }
    return output;
}

void usage(const char * executable) {
    std::cout
        << "Usage: " << executable << " [options]\n\n"
        << "Models (resolved by the GGUF naming convention; loaded on first use):\n"
        << "  --models-dir DIR             Default ./models (YUE2_MODELS_DIR)\n"
        << "  --encoding ENC               auto, BF16, F16, F32, Q8_0, Q5_K_M, Q4_K_M (YUE2_ENCODING)\n"
        << "  --model PATH --vae PATH --tokenizer PATH --transcription-model PATH\n"
        << "                               Explicit files instead of resolution\n"
        << "  --semantic-tokenizer-model PATH\n"
        << "                               Real-audio continuation tokenizer GGUF\n"
        << "  --adapters-dir DIR           LoRA discovery, default models dir (YUE2_ADAPTERS_DIR)\n"
        << "  --lora PATH[=SCALE]          Default adapter for requests without \"loras\"; repeatable\n"
        << "  --instrumental-lora REF[=SCALE]\n"
        << "                               Adapter added whenever instrumental=true; repeatable\n"
        << "  --continuation-lora REF[=SCALE]\n"
        << "                               Matching real-audio NAR adapter for /continue; repeatable\n"
        << "  --keep-models                Keep models resident between jobs by default\n"
        << "  --force-unload               Ignore per-request keep_models and unload after every job\n"
        << "  --device NAME                cpu, cuda, or another GGML backend (YUE2_DEVICE)\n"
        << "  --threads N                  CPU worker threads\n\n"
        << "Server:\n"
        << "  --host IPV4                  Bind address (default 127.0.0.1)\n"
        << "  --port N                     Port (default 8007, YUE2_PORT)\n"
        << "  --max-body-mb N              Upload limit (default 512)\n\n"
        << "One-shot:\n"
        << "  --props                      Print the GET /props document and exit without serving\n"
        << "  --version                    Print the engine version and exit\n\n"
        << "Routes: GET /, GET /health, GET /props, GET /loras, POST /plan, POST /generate, POST /cover, POST /continue, POST /transcribe,\n"
        << "        GET /poll_status/<id>[?consume=1], POST /cancel/<id>, POST /unload\n";
}

Configuration parse_configuration(int argc, char ** argv) {
    Configuration result;
    const auto pick = [&](std::string_view flag, const char * variable) {
        auto value = option(argc, argv, flag);
        return value.empty() ? environment(variable) : value;
    };
    result.http.port = kDefaultPort;
    if (const auto value = option(argc, argv, "--host"); !value.empty()) result.http.host = value;
    if (const auto value = pick("--port", "YUE2_PORT"); !value.empty()) result.http.port = std::stoi(value);
    if (const auto value = option(argc, argv, "--max-body-mb"); !value.empty()) {
        const auto megabytes = std::stoull(value);
        if (megabytes == 0 || megabytes > std::numeric_limits<std::uint64_t>::max() / (1024 * 1024)) {
            throw std::invalid_argument("invalid --max-body-mb");
        }
        result.http.max_request_body_bytes = megabytes * 1024 * 1024;
    }
    if (const auto value = pick("--models-dir", "YUE2_MODELS_DIR"); !value.empty()) result.models_dir = value;
    const auto adapters = pick("--adapters-dir", "YUE2_ADAPTERS_DIR");
    result.adapters_dir = adapters.empty() ? result.models_dir : fs::path(adapters);
    if (const auto value = pick("--encoding", "YUE2_ENCODING"); !value.empty()) {
        result.encoding = upper(value) == "AUTO" ? "auto" : upper(value);
        if (result.encoding != "auto" &&
            std::find(kEncodingPreference.begin(), kEncodingPreference.end(), result.encoding) ==
                kEncodingPreference.end()) {
            throw std::invalid_argument("--encoding must be auto, BF16, F16, F32, Q8_0, Q5_K_M, or Q4_K_M");
        }
    }
    result.generation_model = option(argc, argv, "--model");
    result.vae = option(argc, argv, "--vae");
    result.tokenizer = option(argc, argv, "--tokenizer");
    result.transcription_model = option(argc, argv, "--transcription-model");
    result.semantic_tokenizer_model = pick(
        "--semantic-tokenizer-model", "YUE2_SEMANTIC_TOKENIZER_MODEL");
    result.device = option(argc, argv, "--device");
    if (const auto value = option(argc, argv, "--threads"); !value.empty()) result.threads = std::stoi(value);
    if (result.threads < 0) throw std::invalid_argument("--threads cannot be negative");
    result.keep_models = has(argc, argv, "--keep-models");
    const auto force_unload = upper(environment("YUE2_FORCE_UNLOAD"));
    result.force_unload = has(argc, argv, "--force-unload") ||
        force_unload == "1" || force_unload == "TRUE";
    if (result.force_unload) result.keep_models = false;
    {
        auto value = option(argc, argv, "--natural-max-seconds");
        if (value.empty()) value = environment("YUE2_NATURAL_MAX_SECONDS");
        if (!value.empty()) {
            result.natural_max_seconds = std::stod(value);
            if (!(result.natural_max_seconds >= 0.0) ||
                result.natural_max_seconds > 900.0) {
                throw std::invalid_argument(
                    "--natural-max-seconds must be in [0, 900] seconds");
            }
        }
    }
    {
        auto value = option(argc, argv, "--planning-loop-bars");
        if (value.empty()) value = environment("YUE2_PLANNING_LOOP_BARS");
        if (!value.empty()) {
            const auto parsed = std::stoul(value);
            if (parsed > 1024) {
                throw std::invalid_argument("--planning-loop-bars must be at most 1024; 0 disables it");
            }
            result.planning_loop_bars = static_cast<std::uint32_t>(parsed);
        }
    }
    {
        auto value = option(argc, argv, "--planning-overrun");
        if (value.empty()) value = environment("YUE2_PLANNING_OVERRUN");
        if (!value.empty()) {
            result.planning_overrun = std::stod(value);
            if (!(result.planning_overrun >= 0.0) || result.planning_overrun > 100.0) {
                throw std::invalid_argument(
                    "--planning-overrun must be in [0, 100]; 0 disables the stop");
            }
        }
    }
    for (const auto & value : options(argc, argv, "--lora")) result.loras.push_back(lora_spec(value));
    for (const auto & value : options(argc, argv, "--instrumental-lora")) {
        result.instrumental_loras.push_back(lora_spec(value));
    }
    if (const auto value = environment("YUE2_INSTRUMENTAL_LORA"); !value.empty()) {
        result.instrumental_loras.push_back(lora_spec(value));
    }
    for (const auto & value : options(argc, argv, "--continuation-lora")) {
        result.continuation_loras.push_back(lora_spec(value));
    }
    if (const auto value = environment("YUE2_CONTINUATION_LORA"); !value.empty()) {
        result.continuation_loras.push_back(lora_spec(value));
    }
    return result;
}

// ---------------------------------------------------------------------------
// Model and adapter resolution

std::vector<fs::path> gguf_files(const fs::path & directory) {
    std::vector<fs::path> output;
    std::error_code error;
    if (!fs::is_directory(directory, error)) return output;
    const auto collect = [&](const fs::path & folder) {
        for (const auto & entry : fs::directory_iterator(folder, error)) {
            if (entry.is_regular_file(error) && entry.path().extension() == ".gguf") {
                output.push_back(entry.path());
            }
        }
    };
    collect(directory);
    // One level deeper covers the converter's models/YuE2-3B-GGUF/ layout.
    for (const auto & entry : fs::directory_iterator(directory, error)) {
        if (entry.is_directory(error)) collect(entry.path());
    }
    std::sort(output.begin(), output.end());
    return output;
}

std::optional<fs::path> find_component(
    const fs::path & directory,
    const std::string & component,
    const std::vector<std::string> & encodings) {
    const auto found = yue2::find_model_file(
        yue2::inspect_model_files(directory.string()), component, encodings);
    return found ? std::optional<fs::path>(found->path) : std::nullopt;
}

struct GenerationPaths {
    fs::path model;
    fs::path vae;
    fs::path tokenizer;
};

GenerationPaths resolve_generation(
    const Configuration & configuration,
    const std::string & requested_encoding = {}) {
    GenerationPaths paths;
    if (!configuration.generation_model.empty()) {
        paths.model = configuration.generation_model;
    } else {
        const auto & encoding = requested_encoding.empty() ? configuration.encoding : requested_encoding;
        const auto encodings = encoding == "auto"
            ? kEncodingPreference
            : std::vector<std::string>{encoding};
        const auto found = find_component(configuration.models_dir, "generation", encodings);
        if (!found) {
            throw std::runtime_error(
                "no YuE2 generation GGUF (" +
                (encoding == "auto" ? std::string("any encoding") : encoding) +
                ") under " + configuration.models_dir.string());
        }
        paths.model = *found;
    }
    if (!configuration.vae.empty()) {
        paths.vae = configuration.vae;
    } else if (const auto found = find_component(configuration.models_dir, "vae", {"F16", "F32"})) {
        paths.vae = *found;
    } else {
        throw std::runtime_error("no yue2-vae GGUF under " + configuration.models_dir.string());
    }
    if (!configuration.tokenizer.empty()) {
        paths.tokenizer = configuration.tokenizer;
    } else {
        const auto folder = paths.model.parent_path();
        for (const fs::path & candidate : {
                 folder / "sidecars" / "yue2-qwen.tiktoken", folder / "qwen.tiktoken",
                 configuration.models_dir / "sidecars" / "yue2-qwen.tiktoken",
                 configuration.models_dir / "qwen.tiktoken"}) {
            std::error_code error;
            if (fs::is_regular_file(candidate, error)) {
                paths.tokenizer = candidate;
                break;
            }
        }
        if (paths.tokenizer.empty()) {
            throw std::runtime_error("no qwen.tiktoken beside " + paths.model.string());
        }
    }
    return paths;
}

fs::path resolve_transcription(const Configuration & configuration) {
    if (!configuration.transcription_model.empty()) return configuration.transcription_model;
    if (const auto found = find_component(configuration.models_dir, "transcription", {"F16", "F32"})) {
        return *found;
    }
    throw std::runtime_error("no sheetsage2-mert2 GGUF under " + configuration.models_dir.string());
}

fs::path resolve_semantic_tokenizer(const Configuration & configuration) {
    if (!configuration.semantic_tokenizer_model.empty()) {
        return configuration.semantic_tokenizer_model;
    }
    if (const auto found = find_component(
            configuration.models_dir, "semantic-tokenizer", {"F16", "F32"})) {
        return *found;
    }
    throw std::runtime_error(
        "no yue2 semantic-tokenizer GGUF under " + configuration.models_dir.string());
}

// <name>-v1.0-F16-LoRA.gguf -> <name>
std::string adapter_name(const std::string & filename) {
    constexpr std::string_view kSuffix = "-LoRA.gguf";
    if (!ends_with(filename, kSuffix)) return {};
    std::string stem = filename.substr(0, filename.size() - kSuffix.size());
    auto dash = stem.rfind('-');
    if (dash == std::string::npos) return stem;
    stem.resize(dash); // Encoding
    dash = stem.rfind('-');
    if (dash != std::string::npos && dash + 1 < stem.size() && stem[dash + 1] == 'v') stem.resize(dash);
    return stem;
}

std::vector<std::pair<std::string, fs::path>> list_adapters(const fs::path & directory) {
    std::vector<std::pair<std::string, fs::path>> output;
    for (const auto & file : gguf_files(directory)) {
        const auto name = adapter_name(file.filename().string());
        if (!name.empty()) output.emplace_back(name, file);
    }
    return output;
}

fs::path resolve_adapter(const Configuration & configuration, const std::string & reference) {
    std::error_code error;
    const fs::path candidate(reference);
    if (candidate.extension() == ".gguf") {
        if (fs::is_regular_file(candidate, error)) return candidate;
        if (fs::is_regular_file(configuration.adapters_dir / candidate, error)) {
            return configuration.adapters_dir / candidate;
        }
        throw std::invalid_argument("LoRA adapter not found: " + reference);
    }
    for (const auto & [name, path] : list_adapters(configuration.adapters_dir)) {
        if (name == reference) return path;
    }
    throw std::invalid_argument("unknown LoRA adapter: " + reference);
}

void discover_published_adapters(Configuration & configuration) {
    for (const auto & [name, path] : list_adapters(configuration.adapters_dir)) {
        if (name == "yue2-instrumental-cot-full" &&
            configuration.instrumental_loras.empty()) {
            configuration.instrumental_loras.push_back({path.string(), 1.0F});
            std::cerr << "[yue2-server] discovered instrumental adapter: "
                      << path.string() << '\n';
        } else if (name == "yue2-realaudio-nar-v9" &&
                   configuration.continuation_loras.empty()) {
            configuration.continuation_loras.push_back({path.string(), 1.0F});
            std::cerr << "[yue2-server] discovered continuation adapter: "
                      << path.string() << '\n';
        }
    }
}

// ---------------------------------------------------------------------------
// Request parsing helpers

bool present(const json::Value & object, std::string_view key) {
    const auto * value = object.find(key);
    return value && value->type != json::Type::null;
}

std::string first_string(const json::Value & object, std::initializer_list<std::string_view> keys) {
    for (const auto key : keys) {
        if (present(object, key)) return json::string(object, key);
    }
    return {};
}

std::string header(const HttpRequest & request, const std::string & name) {
    const auto found = request.headers.find(name);
    return found == request.headers.end() ? std::string{} : found->second;
}

bool query_flag(const std::string & query, std::string_view name) {
    std::size_t start = 0;
    while (start <= query.size()) {
        const auto end = std::min(query.find('&', start), query.size());
        const std::string_view pair(query.data() + start, end - start);
        if (pair == name || pair == std::string(name) + "=1" || pair == std::string(name) + "=true") {
            return true;
        }
        start = end + 1;
    }
    return false;
}

json::Value parse_object(const std::string & body) {
    json::Value root;
    try {
        root = json::parse(body);
    } catch (const std::exception & error) {
        throw std::invalid_argument(std::string("invalid JSON: ") + error.what());
    }
    if (root.type != json::Type::object) throw std::invalid_argument("request body must be a JSON object");
    return root;
}

yue2::SymbolicMode symbolic_mode(const std::string & value) {
    if (value == "off") return yue2::SymbolicMode::off;
    if (value == "melody") return yue2::SymbolicMode::melody;
    if (value == "full") return yue2::SymbolicMode::full;
    throw std::invalid_argument("symbolic_mode must be off, melody, or full");
}

bool melody_only(const std::string & value) {
    if (value.empty() || value == "melody") return true;
    if (value == "full") return false;
    throw std::invalid_argument("transcription mode must be melody or full");
}

// 16-bit PCM is what the gary4local services return and gary4juce expects.
std::vector<std::uint8_t> encode_wav_pcm16(
    const std::vector<float> & samples,
    std::int32_t sample_rate,
    std::int32_t channels) {
    const std::uint64_t data_bytes = static_cast<std::uint64_t>(samples.size()) * 2;
    if (data_bytes > std::numeric_limits<std::uint32_t>::max() - 36) {
        throw std::runtime_error("audio is too long for a WAV file");
    }
    std::vector<std::uint8_t> output;
    output.reserve(44 + static_cast<std::size_t>(data_bytes));
    const auto text = [&](const char * value) { output.insert(output.end(), value, value + 4); };
    const auto u16 = [&](std::uint32_t value) {
        output.push_back(static_cast<std::uint8_t>(value & 255));
        output.push_back(static_cast<std::uint8_t>((value >> 8) & 255));
    };
    const auto u32 = [&](std::uint32_t value) {
        u16(value & 0xffff);
        u16(value >> 16);
    };
    text("RIFF");
    u32(static_cast<std::uint32_t>(36 + data_bytes));
    text("WAVE");
    text("fmt ");
    u32(16);
    u16(1);
    u16(static_cast<std::uint32_t>(channels));
    u32(static_cast<std::uint32_t>(sample_rate));
    u32(static_cast<std::uint32_t>(sample_rate * channels * 2));
    u16(static_cast<std::uint32_t>(channels * 2));
    u16(16);
    text("data");
    u32(static_cast<std::uint32_t>(data_bytes));
    for (const float sample : samples) {
        const float clamped = std::isfinite(sample) ? std::clamp(sample, -1.0F, 1.0F) : 0.0F;
        const auto value = static_cast<std::int16_t>(std::lrint(clamped * 32767.0F));
        u16(static_cast<std::uint16_t>(value));
    }
    return output;
}

HttpResponse failure(int status, const std::string & message) {
    return yue2::server::json_response(
        "{\"success\":false,\"error\":" + json::quote(message) + "}", status);
}

// ---------------------------------------------------------------------------
// Jobs

enum class JobKind { plan, generate, cover, continue_audio, transcribe };

struct Job {
    std::string id;
    JobKind kind = JobKind::generate;

    // Request, fixed at submission.
    yue2::SongRequest song;
    yue2::GenerationRunOptions run;
    std::vector<yue2::LoraAdapterSpec> loras;
    yue2::audio::MonoAudio input;
    yue2::TranscriptionOptions transcription;
    std::uint32_t continuation_bars = 0;
    bool keep_models = false;
    bool float_wav = false;
    std::string encoding = "auto";
    std::atomic<bool> cancel{false};

    // Progress and result, guarded by ServerState::jobs_mutex_.
    std::string status = "queued";
    std::string stage = "queued";
    int progress = 0;
    std::uint32_t step = 0;
    std::uint32_t total_steps = 0;
    std::string audio_data;
    std::string abc;
    std::string midi_data;
    std::string melody_midi_data;
    std::string vocal_midi_data;
    std::string instrumental_midi_data;
    std::string chords_midi_data;
    std::string events_json;
    std::string error;
    bool cancelled = false;
    std::size_t semantic_frames = 0;
    std::size_t semantic_prefix_frames = 0;
    bool instrumental_adapter = false;
    bool continuation_adapter = false;
    bool abc_truncated = false;
    bool abc_repaired = false;
    bool semantic_truncated = false;
    std::uint32_t score_bars = 0;
    double score_duration_seconds = 0.0;
    std::uint32_t semantic_budget = 0;
    double duration_seconds = 0.0;
    std::chrono::steady_clock::time_point finished;

    bool done() const { return status == "completed" || status == "failed"; }
};

void set_midi_exports(Job & job, const yue2::TranscriptionMidiExports & exports) {
    const auto encode = [](const std::vector<std::uint8_t> & data) {
        return data.empty()
            ? std::string{}
            : yue2::server::base64_encode(data.data(), data.size());
    };
    job.midi_data = encode(exports.transcription);
    job.melody_midi_data = encode(exports.melody);
    job.vocal_midi_data = encode(exports.vocal);
    job.instrumental_midi_data = encode(exports.instrumental);
    job.chords_midi_data = encode(exports.chords);
}

void append_midi_json(std::string & body, const Job & job) {
    if (job.midi_data.empty()) return;
    body += ",\"midi_data\":\"" + job.midi_data +
        "\",\"midi_files\":{\"transcription.mid\":\"" + job.midi_data +
        "\",\"melody.mid\":\"" + job.melody_midi_data +
        "\",\"melody_vocal.mid\":\"" + job.vocal_midi_data +
        "\",\"melody_instrumental.mid\":\"" + job.instrumental_midi_data + "\"";
    if (!job.chords_midi_data.empty()) {
        body += ",\"chords.mid\":\"" + job.chords_midi_data + "\"";
    }
    body += "}";
}

class ServerState {
public:
    explicit ServerState(Configuration configuration)
        : configuration_(std::move(configuration)), worker_([this]() { work(); }) {}

    ~ServerState() {
        {
            std::lock_guard<std::mutex> lock(jobs_mutex_);
            stopping_ = true;
            if (running_) running_->cancel.store(true);
        }
        jobs_ready_.notify_all();
        worker_.join();
    }

    ServerState(const ServerState &) = delete;
    ServerState & operator=(const ServerState &) = delete;

    HttpResponse handle(const HttpRequest & request) {
        try {
            const auto & path = request.path;
            const bool get = request.method == "GET";
            const bool post = request.method == "POST";
            if (path == "/" || path == "/index.html") return get ? ui() : not_allowed();
            if (path == "/health") return get ? health() : not_allowed();
            if (path == "/props") return get ? props() : not_allowed();
            if (path == "/loras") return get ? loras() : not_allowed();
            if (path == "/plan") return post ? submit(request, JobKind::plan) : not_allowed();
            if (path == "/generate") return post ? submit(request, JobKind::generate) : not_allowed();
            if (path == "/cover") return post ? submit(request, JobKind::cover) : not_allowed();
            if (path == "/continue") return post ? submit(request, JobKind::continue_audio) : not_allowed();
            // Style prompts for a client's dice button. Static, so no job and no
            // model load: a client may call it before anything is warm.
            if (path == "/prompts") {
                if (!get) return not_allowed();
                return yue2::server::json_response(yue2::server::dice_prompts_json());
            }

            if (path == "/transcribe") return post ? submit(request, JobKind::transcribe) : not_allowed();
            if (path == "/unload") return post ? unload() : not_allowed();
            if (starts_with(path, "/poll_status/")) {
                return get ? poll(path.substr(std::strlen("/poll_status/")), request.query) : not_allowed();
            }
            if (starts_with(path, "/cancel/")) {
                return post ? cancel(path.substr(std::strlen("/cancel/"))) : not_allowed();
            }
            return failure(404, "route not found");
        } catch (const std::invalid_argument & error) {
            return failure(400, error.what());
        } catch (const std::exception & error) {
            return failure(500, error.what());
        }
    }

private:
    static HttpResponse not_allowed() { return failure(405, "method is not allowed for this route"); }

    // --- HTTP handlers ------------------------------------------------------

    HttpResponse ui() {
        HttpResponse response;
        response.content_type = "text/html; charset=utf-8";
        response.body.assign(
            reinterpret_cast<const char *>(yue2::ui::index_html), yue2::ui::index_html_size);
        response.headers.emplace("Cache-Control", "no-cache");
        response.headers.emplace("X-Content-Type-Options", "nosniff");
        return response;
    }

    HttpResponse health() {
        std::string generation;
        try {
            const auto paths = resolve_generation(configuration_);
            generation = "{\"available\":true,\"loaded\":" + json_bool(generator_loaded_.load()) +
                ",\"model\":" + json_path(paths.model) + ",\"vae\":" + json_path(paths.vae) +
                ",\"tokenizer\":" + json_path(paths.tokenizer) + "}";
        } catch (const std::exception & error) {
            generation = "{\"available\":false,\"loaded\":false,\"error\":" + json::quote(error.what()) + "}";
        }
        std::string transcription;
        try {
            transcription = "{\"available\":true,\"loaded\":" + json_bool(transcriber_loaded_.load()) +
                ",\"model\":" + json_path(resolve_transcription(configuration_)) + "}";
        } catch (const std::exception & error) {
            transcription = "{\"available\":false,\"loaded\":false,\"error\":" + json::quote(error.what()) + "}";
        }
        std::string continuation;
        try {
            continuation = "{\"available\":true,\"model\":" +
                json_path(resolve_semantic_tokenizer(configuration_)) +
                ",\"adapter_configured\":" +
                json_bool(!configuration_.continuation_loras.empty()) + "}";
        } catch (const std::exception & error) {
            continuation = "{\"available\":false,\"error\":" +
                json::quote(error.what()) + "}";
        }
        bool busy = false;
        std::size_t queued = 0;
        {
            std::lock_guard<std::mutex> lock(jobs_mutex_);
            busy = static_cast<bool>(running_);
            queued = queue_.size();
        }
        const auto device = configuration_.device.empty() ? environment("YUE2_DEVICE") : configuration_.device;
        return yue2::server::json_response(
            "{\"status\":\"ok\",\"service\":\"yue2\",\"api_version\":1,\"version\":" + json::quote(yue2::version()) +
            ",\"device\":" + json::quote(device.empty() ? "auto" : device) +
            ",\"encoding\":" + json::quote(configuration_.encoding) +
            ",\"models_dir\":" + json_path(configuration_.models_dir) +
            ",\"keep_models\":" + json_bool(configuration_.keep_models) +
            ",\"force_unload\":" + json_bool(configuration_.force_unload) +
            "," + yue2::server::generation_policy_json(generation_policy()) +
            ",\"busy\":" + json_bool(busy) + ",\"queued\":" + std::to_string(queued) +
            ",\"generation\":" + generation + ",\"transcription\":" + transcription +
            ",\"continuation\":" + continuation + "}");
    }

    HttpResponse props() {
        const auto devices = yue2::available_devices();
        std::uint64_t accelerator_total = 0;
        std::uint64_t accelerator_free = 0;
        bool have_accelerator = false;
        bool have_discrete = false;
        for (const auto & device : devices) {
            const bool discrete = device.type == yue2::DeviceType::gpu;
            const bool integrated = device.type == yue2::DeviceType::integrated_gpu;
            if (!discrete && (!integrated || have_discrete)) continue;
            have_accelerator = true;
            if ((discrete && !have_discrete) || device.memory_total_bytes > accelerator_total) {
                have_discrete = discrete;
                accelerator_total = device.memory_total_bytes;
                accelerator_free = device.memory_free_bytes;
            }
        }

        const auto files = yue2::inspect_model_files(configuration_.models_dir.string());
        const auto tiers = yue2::quantization_tiers(files, accelerator_total, accelerator_free);
        std::string body =
            "{\"success\":true,\"service\":\"yue2\",\"api_version\":1,\"version\":" +
            json::quote(yue2::version()) +
            ",\"capabilities\":{\"plan\":true,\"generate\":true,\"transcribe\":true,\"cover\":true,\"continue\":true,"
            "\"planning_controls\":true,\"instrumental\":true,"
            "\"instrumental_adapter\":" + json_bool(!configuration_.instrumental_loras.empty()) + ","
            "\"continuation_adapter\":" + json_bool(!configuration_.continuation_loras.empty()) + ","
            "\"instrumental_best_effort\":" + json_bool(configuration_.instrumental_loras.empty()) + ","
            "\"vocal_rest_experiment\":true,"
            "\"score_editing\":true,\"score_aligned_generation\":true,"
            "\"model_downloads\":false},\"devices\":[";
        for (std::size_t index = 0; index < devices.size(); ++index) {
            const auto & device = devices[index];
            if (index) body.push_back(',');
            body += "{\"index\":" + std::to_string(device.index) +
                ",\"name\":" + json::quote(device.name) +
                ",\"description\":" + json::quote(device.description) +
                ",\"backend\":" + json::quote(device.backend) +
                ",\"id\":" + json::quote(device.id) +
                ",\"type\":" + json::quote(yue2::device_type_name(device.type)) +
                ",\"memory_free_bytes\":" + std::to_string(device.memory_free_bytes) +
                ",\"memory_total_bytes\":" + std::to_string(device.memory_total_bytes) + "}";
        }
        body += "],\"hardware\":{\"accelerator_available\":" + json_bool(have_accelerator) +
            ",\"memory_free_bytes\":" + std::to_string(accelerator_free) +
            ",\"memory_total_bytes\":" + std::to_string(accelerator_total) +
            ",\"recommended_encoding\":" +
            json::quote(yue2::recommended_quantization(accelerator_total)) + "},\"models\":{\"directory\":" +
            json_path(configuration_.models_dir) + ",\"tiers\":[";
        for (std::size_t index = 0; index < tiers.size(); ++index) {
            const auto & tier = tiers[index];
            if (index) body.push_back(',');
            body += "{\"encoding\":" + json::quote(tier.encoding) +
                ",\"label\":" + json::quote(tier.label) +
                ",\"estimated_model_bytes\":" + std::to_string(tier.estimated_model_bytes) +
                ",\"recommended_vram_bytes\":" + std::to_string(tier.recommended_vram_bytes) +
                ",\"installed\":" + json_bool(tier.installed) +
                ",\"model\":" + (tier.model_path.empty() ? std::string("null") : json::quote(tier.model_path)) +
                ",\"installed_bytes\":" + std::to_string(tier.installed_bytes) +
                ",\"fits_total_memory\":" + json_bool(tier.fits_total_memory) +
                ",\"fits_free_memory\":" + json_bool(tier.fits_free_memory) +
                ",\"recommended\":" + json_bool(tier.recommended) + "}";
        }
        body += "],\"files\":[";
        for (std::size_t index = 0; index < files.size(); ++index) {
            const auto & file = files[index];
            if (index) body.push_back(',');
            body += "{\"name\":" + json::quote(file.name) +
                ",\"path\":" + json::quote(file.path) +
                ",\"component\":" + json::quote(file.component) +
                ",\"encoding\":" + json::quote(file.encoding) +
                ",\"size_bytes\":" + std::to_string(file.size_bytes) +
                ",\"metadata_classified\":" + json_bool(file.metadata_classified) + "}";
        }
        body += "]},\"defaults\":{\"encoding\":" + json::quote(configuration_.encoding) +
            ",\"device\":" + json::quote(configuration_.device.empty() ? "auto" : configuration_.device) +
            ",\"keep_models\":" + json_bool(configuration_.keep_models) +
            ",\"force_unload\":" + json_bool(configuration_.force_unload) +
            "," + yue2::server::generation_policy_json(generation_policy()) + "}}";
        return yue2::server::json_response(std::move(body));
    }

    HttpResponse loras() {
        std::string body = "{\"success\":true,\"adapters_dir\":" + json_path(configuration_.adapters_dir) +
            ",\"loras\":[";
        std::size_t index = 0;
        for (const auto & [name, path] : list_adapters(configuration_.adapters_dir)) {
            if (index) body.push_back(',');
            body += "{\"index\":" + std::to_string(index++) + ",\"name\":" + json::quote(name) +
                ",\"path\":" + json_path(path) + "}";
        }
        body += "]}";
        return yue2::server::json_response(std::move(body));
    }

    HttpResponse submit(const HttpRequest & request, JobKind kind) {
        if (header(request, "content-type").rfind("application/json", 0) != 0) {
            return failure(415, "send application/json");
        }
        const auto root = parse_object(request.body);
        auto job = std::make_shared<Job>();
        job->kind = kind;
        job->keep_models = configuration_.force_unload
            ? false : json::boolean(root, "keep_models", configuration_.keep_models);

        if ((kind == JobKind::cover || kind == JobKind::continue_audio) &&
            (present(root, "abc") || present(root, "abc_prefix") || present(root, "planning"))) {
            throw std::invalid_argument(
                kind == JobKind::cover
                    ? "/cover scores audio_data itself; use /generate to supply abc"
                    : "/continue transcribes and extends audio_data itself; abc, abc_prefix, and planning are not accepted");
        }
        if (kind == JobKind::cover || kind == JobKind::continue_audio ||
            kind == JobKind::transcribe) {
            const auto encoded = json::string(root, "audio_data");
            if (encoded.empty()) throw std::invalid_argument("audio_data (base64 WAV) is required");
            const auto wav = yue2::server::base64_decode(encoded);
            try {
                job->input = yue2::audio::decode_wav_mono(wav.data(), wav.size());
            } catch (const std::exception & error) {
                throw std::invalid_argument(std::string("audio_data is not a readable WAV: ") + error.what());
            }
            if (job->input.samples.empty()) throw std::invalid_argument("audio_data contains no samples");
            job->transcription.melody_only = melody_only(first_string(root, {"transcription_mode", "mode"}));
        }
        if (kind != JobKind::transcribe) parse_song(root, *job);

        job->id = random_session_id();
        {
            std::lock_guard<std::mutex> lock(jobs_mutex_);
            prune_locked();
            jobs_[job->id] = job;
            queue_.push_back(job);
        }
        jobs_ready_.notify_one();
        std::string body = "{\"success\":true,\"session_id\":" + json::quote(job->id);
        if (kind != JobKind::transcribe) body += ",\"seed\":" + std::to_string(job->song.seed);
        body += ",\"status\":\"queued\"}";
        return yue2::server::json_response(std::move(body));
    }

    void parse_song(const json::Value & root, Job & job) {
        auto & song = job.song;
        job.encoding = upper(json::string(root, "encoding", configuration_.encoding));
        if (job.encoding.empty()) job.encoding = "auto";
        if (job.encoding != "AUTO" &&
            std::find(kEncodingPreference.begin(), kEncodingPreference.end(), job.encoding) ==
                kEncodingPreference.end()) {
            throw std::invalid_argument(
                "encoding must be auto, BF16, F16, F32, Q8_0, Q5_K_M, or Q4_K_M");
        }
        if (job.encoding == "AUTO") job.encoding = "auto";
        song.style = first_string(root, {"style", "caption", "prompt", "tags"});
        song.lyrics = json::string(root, "lyrics");
        song.instrumental = json::boolean(root, "instrumental", false);
        song.experimental_vocal_rest = json::boolean(root, "experimental_vocal_rest", false);
        song.target_bars = json::u32(root, "target_bars", 0);
        if (job.kind == JobKind::continue_audio) {
            job.continuation_bars = json::u32(root, "continuation_bars", 0);
            if (job.continuation_bars > 256) {
                throw std::invalid_argument("continuation_bars must be in [1,256], or zero for natural length");
            }
            if (job.continuation_bars != 0 && song.target_bars != 0) {
                throw std::invalid_argument(
                    "continuation_bars and target_bars are mutually exclusive");
            }
            if (job.continuation_bars != 0) {
                song.ending_mode = yue2::EndingMode::natural;
                song.target_bars = 0;
            }
        }
        song.outro_bars = json::u32(root, "outro_bars", 4);
        // The operator ceiling is a maximum, not a default: a request may ask
        // for less, or for none when the operator allows none, but it can
        // never buy itself more than the backend is willing to render.
        song.planning_loop_bars = configuration_.planning_loop_bars;
        if (present(root, "planning_loop_bars")) {
            song.planning_loop_bars = json::u32(root, "planning_loop_bars", 0);
            if (song.planning_loop_bars > 1024) {
                throw std::invalid_argument("planning_loop_bars must be at most 1024; 0 disables it");
            }
        }
        song.planning_overrun = configuration_.planning_overrun;
        if (present(root, "planning_overrun")) {
            const double requested = json::number(root, "planning_overrun", 0.0);
            if (!(requested >= 0.0) || requested > 100.0) {
                throw std::invalid_argument(
                    "planning_overrun must be in [0, 100]; 0 disables the stop");
            }
            song.planning_overrun = requested;
        }
        song.natural_max_seconds = configuration_.natural_max_seconds;
        if (present(root, "natural_max_seconds")) {
            const double requested = json::number(root, "natural_max_seconds", 0.0);
            if (!(requested >= 0.0) || requested > 900.0) {
                throw std::invalid_argument(
                    "natural_max_seconds must be in [0, 900] seconds");
            }
            if (configuration_.natural_max_seconds <= 0.0) {
                song.natural_max_seconds = requested;
            } else if (requested <= 0.0) {
                song.natural_max_seconds = configuration_.natural_max_seconds;
            } else {
                song.natural_max_seconds =
                    std::min(requested, configuration_.natural_max_seconds);
            }
        }
        const auto ending = json::string(
            root, "ending", song.target_bars == 0 ? "natural" : "outro");
        if (ending == "natural") song.ending_mode = yue2::EndingMode::natural;
        else if (ending == "outro") song.ending_mode = yue2::EndingMode::outro;
        else throw std::invalid_argument("ending must be natural or outro");
        if (song.target_bars != 0 && song.ending_mode != yue2::EndingMode::outro) {
            throw std::invalid_argument("target_bars requires ending outro");
        }
        if (song.ending_mode == yue2::EndingMode::outro && song.target_bars == 0) {
            throw std::invalid_argument("ending outro requires target_bars");
        }
        if (song.target_bars != 0 &&
            (song.outro_bars == 0 || song.outro_bars > song.target_bars)) {
            throw std::invalid_argument("outro_bars must be within target_bars");
        }
        if (song.instrumental && !song.lyrics.empty()) {
            throw std::invalid_argument("instrumental is mutually exclusive with lyrics");
        }
        if (song.instrumental && song.experimental_vocal_rest) {
            throw std::invalid_argument(
                "instrumental already includes the vocal-rest intervention");
        }
        const auto abc = json::string(root, "abc");
        if (job.kind == JobKind::cover && !abc.empty()) {
            throw std::invalid_argument("/cover scores audio_data itself; use /generate to supply abc");
        }
        if (!abc.empty()) song.abc = abc;
        const auto abc_prefix = json::string(root, "abc_prefix");
        if (job.kind == JobKind::cover && !abc_prefix.empty()) {
            throw std::invalid_argument("/cover transcribes its own planning score; abc_prefix is not accepted");
        }
        const auto * planning = root.find("planning");
        if (planning && planning->type != json::Type::null) {
            if (job.kind == JobKind::cover) {
                throw std::invalid_argument("/cover transcribes its own score; planning controls are not accepted");
            }
            if (planning->type != json::Type::object) {
                throw std::invalid_argument("planning must be an object");
            }
            if (!abc.empty() || !abc_prefix.empty()) {
                throw std::invalid_argument("planning is mutually exclusive with abc and abc_prefix");
            }
            yue2::PlanningHeader header;
            header.bpm = json::u32(*planning, "bpm", 0);
            header.meter_numerator = json::u32(*planning, "meter_numerator", 4);
            header.meter_denominator = json::u32(*planning, "meter_denominator", 4);
            header.key = json::string(*planning, "key");
            song.abc_prefix = yue2::make_planning_abc_prefix(header);
        } else if (!abc.empty() && !abc_prefix.empty()) {
            throw std::invalid_argument("abc and abc_prefix are mutually exclusive");
        }
        if (!abc_prefix.empty() && !song.abc_prefix) song.abc_prefix = abc_prefix;
        const bool scored = job.kind == JobKind::cover ||
            job.kind == JobKind::continue_audio || song.abc.has_value();
        auto mode = first_string(root, {"symbolic_mode", "cot"});
        if (mode.empty()) {
            // A supplied or transcribed score conditions melody by default, as
            // the upstream cover workflow recommends; without one YuE2 plans a
            // full score first.
            mode = (job.kind == JobKind::cover || job.kind == JobKind::continue_audio)
                ? (job.transcription.melody_only ? "melody" : "full")
                : (scored ? "melody" : "full");
        }
        song.symbolic_mode = symbolic_mode(mode);
        if ((scored || song.abc_prefix.has_value()) &&
            song.symbolic_mode == yue2::SymbolicMode::off) {
            throw std::invalid_argument(
                "a score or planning prefix requires symbolic_mode melody or full");
        }

        const auto * seed = root.find("seed");
        const bool random = !seed || seed->type == json::Type::null ||
            (!seed->text.empty() && seed->text.front() == '-');
        song.seed = random ? (random_u64() & 0x7fffffffULL) : json::u64(root, "seed", 0);
        if (present(root, "guidance_scale")) {
            song.guidance_scale = static_cast<float>(json::number(root, "guidance_scale", 1.0));
        }

        auto & run = job.run;
        if (present(root, "duration") || present(root, "max_seconds")) {
            const double seconds = present(root, "max_seconds")
                ? json::number(root, "max_seconds", 0.0)
                : json::number(root, "duration", 0.0);
            if (!(seconds > 0.0) || seconds > 900.0) {
                throw std::invalid_argument("max_seconds must be in (0, 900] seconds");
            }
            run.generation.semantic.max_tokens =
                static_cast<std::uint32_t>(std::ceil(seconds * kSemanticTokensPerSecond));
            run.semantic_budget_explicit = true;
        }
        auto & semantic = run.generation.semantic;
        if (present(root, "semantic_max_tokens")) {
            semantic.max_tokens = json::u32(root, "semantic_max_tokens", semantic.max_tokens);
            run.semantic_budget_explicit = true;
        }
        semantic.min_tokens = std::min(json::u32(root, "semantic_min_tokens", semantic.min_tokens), semantic.max_tokens);
        semantic.temperature = static_cast<float>(json::number(root, "temperature", semantic.temperature));
        semantic.top_k = json::u32(root, "top_k", semantic.top_k);
        semantic.top_p = static_cast<float>(json::number(root, "top_p", semantic.top_p));
        semantic.repetition_penalty = static_cast<float>(
            json::number(root, "repetition_penalty", semantic.repetition_penalty));
        run.generation.abc.max_tokens = json::u32(root, "abc_max_tokens", run.generation.abc.max_tokens);
        run.flow.ode_steps = json::u32(root, "ode_steps", run.flow.ode_steps);

        const auto format = json::string(root, "audio_format", "wav");
        if (format != "wav" && format != "wav_float") {
            throw std::invalid_argument("audio_format must be wav or wav_float");
        }
        job.float_wav = format == "wav_float";
        job.loras = parse_loras(root);
        if (job.kind == JobKind::continue_audio) {
            if (!json::boolean(root, "use_continuation_adapter", true)) {
                throw std::invalid_argument(
                    "/continue cannot run with use_continuation_adapter=false; use score continuation instead");
            }
            if (configuration_.continuation_loras.empty()) {
                throw std::invalid_argument(
                    "/continue requires a configured --continuation-lora matching the semantic tokenizer");
            }
            for (auto configured : configuration_.continuation_loras) {
                configured.path = resolve_adapter(configuration_, configured.path).string();
                const auto duplicate = std::find_if(
                    job.loras.begin(), job.loras.end(),
                    [&configured](const yue2::LoraAdapterSpec & existing) {
                        return existing.path == configured.path;
                    });
                if (duplicate == job.loras.end()) job.loras.push_back(std::move(configured));
            }
            job.continuation_adapter = true;
        }
        if (song.instrumental &&
            json::boolean(root, "use_instrumental_adapter", true) &&
            !configuration_.instrumental_loras.empty()) {
            // The instrumental AR adapters are trained for score-first COT and
            // chord-annotated full SheetSage2 plans. Keep this policy in the
            // server so every client gets the same reliable toggle behavior.
            song.symbolic_mode = yue2::SymbolicMode::full;
            if (job.kind == JobKind::cover || job.kind == JobKind::continue_audio) {
                job.transcription.melody_only = false;
            }
            for (auto configured : configuration_.instrumental_loras) {
                configured.path = resolve_adapter(configuration_, configured.path).string();
                const auto duplicate = std::find_if(
                    job.loras.begin(), job.loras.end(),
                    [&configured](const yue2::LoraAdapterSpec & existing) {
                        return existing.path == configured.path;
                    });
                if (duplicate == job.loras.end()) job.loras.push_back(std::move(configured));
            }
            job.instrumental_adapter = true;
        }
    }

    std::vector<yue2::LoraAdapterSpec> parse_loras(const json::Value & root) {
        const auto * value = root.find("loras");
        if (!value || value->type == json::Type::null) return configuration_.loras;
        if (value->type != json::Type::array) throw std::invalid_argument("loras must be an array");
        std::vector<yue2::LoraAdapterSpec> output;
        for (const auto & item : value->array) {
            if (item.type != json::Type::object) throw std::invalid_argument("each lora must be an object");
            auto reference = json::string(item, "path");
            if (reference.empty()) reference = json::string(item, "name");
            if (reference.empty()) throw std::invalid_argument("each lora needs a name or path");
            yue2::LoraAdapterSpec spec;
            spec.path = resolve_adapter(configuration_, reference).string();
            spec.strength = static_cast<float>(
                json::number(item, "strength", json::number(item, "scale", 1.0)));
            if (!std::isfinite(spec.strength)) throw std::invalid_argument("lora strength must be finite");
            output.push_back(std::move(spec));
        }
        return output;
    }

    HttpResponse poll(const std::string & id, const std::string & query) {
        const bool consume = query_flag(query, "consume");
        std::string body;
        std::lock_guard<std::mutex> lock(jobs_mutex_);
        prune_locked();
        const auto found = jobs_.find(id);
        if (found == jobs_.end()) return failure(404, "unknown session: " + id);
        const Job & job = *found->second;
        const bool in_progress = !job.done();
        body = "{\"success\":" + json_bool(job.status != "failed") +
            ",\"session_id\":" + json::quote(job.id) +
            ",\"generation_in_progress\":" + json_bool(in_progress) +
            ",\"transform_in_progress\":false" +
            ",\"progress\":" + std::to_string(job.progress) +
            ",\"step\":" + std::to_string(job.step) +
            ",\"total_steps\":" + std::to_string(job.total_steps) +
            ",\"status\":" + json::quote(job.status) +
            ",\"stage\":" + json::quote(job.stage) +
            ",\"queue_status\":" + queue_status_locked(job);
        if (job.kind != JobKind::transcribe) {
            body += ",\"seed\":" + std::to_string(job.song.seed) +
                ",\"encoding\":" + json::quote(job.encoding);
            std::vector<std::string> warnings;
            if (job.song.instrumental) {
                warnings.emplace_back(
                    "instrumental mode is best-effort; occasional vocal material may occur");
            }
            if (job.status == "completed" && job.semantic_truncated) {
                warnings.emplace_back(
                    "semantic safety budget was exhausted before MUSIC_END");
            }
            if (job.status == "completed" && job.abc_truncated) {
                warnings.emplace_back(
                    "planning reached its token limit, so the score may be "
                    "shorter than the bars requested");
            }
            if (job.status == "completed" && job.abc_repaired) {
                warnings.emplace_back(
                    "the planned score had a bar that would not render and was "
                    "shortened to the last complete section");
            }
            if (!warnings.empty()) {
                body += ",\"warnings\":[";
                for (std::size_t index = 0; index < warnings.size(); ++index) {
                    if (index) body.push_back(',');
                    body += json::quote(warnings[index]);
                }
                body.push_back(']');
            }
        }
        if (job.status == "completed") {
            if (job.kind == JobKind::transcribe) {
                body += ",\"abc\":" + json::quote(job.abc);
                append_midi_json(body, job);
                body += ",\"duration\":" + real(job.duration_seconds) +
                    ",\"events\":" + job.events_json;
            } else if (job.kind == JobKind::plan) {
                body += ",\"abc\":" + json::quote(job.abc) +
                    ",\"meta\":{\"seed\":" + std::to_string(job.song.seed) +
                    ",\"score_bars\":" + std::to_string(job.score_bars) +
                    ",\"score_duration\":" + real(job.score_duration_seconds) +
                    ",\"encoding\":" + json::quote(job.encoding) +
                    ",\"instrumental_adapter\":" + json_bool(job.instrumental_adapter) +
                    ",\"continuation_adapter\":" + json_bool(job.continuation_adapter) +
                    ",\"abc_truncated\":" + json_bool(job.abc_truncated) + "}";
                append_midi_json(body, job);
            } else {
                // Base64 needs no JSON escaping.
                body += ",\"audio_data\":\"" + job.audio_data + "\",\"abc\":" + json::quote(job.abc) +
                    ",\"meta\":{\"seed\":" + std::to_string(job.song.seed) +
                    ",\"duration\":" + real(job.duration_seconds) +
                    ",\"sample_rate\":48000,\"channels\":2" +
                    ",\"semantic_frames\":" + std::to_string(job.semantic_frames) +
                    ",\"semantic_prefix_frames\":" +
                        std::to_string(job.semantic_prefix_frames) +
                    ",\"semantic_prefix_duration\":" +
                        real(job.semantic_prefix_frames / kSemanticTokensPerSecond) +
                    ",\"semantic_budget\":" + std::to_string(job.semantic_budget) +
                    ",\"score_bars\":" + std::to_string(job.score_bars) +
                    ",\"score_duration\":" + real(job.score_duration_seconds) +
                    ",\"encoding\":" + json::quote(job.encoding) +
                    ",\"instrumental_adapter\":" + json_bool(job.instrumental_adapter) +
                    ",\"continuation_adapter\":" + json_bool(job.continuation_adapter) +
                    ",\"abc_truncated\":" + json_bool(job.abc_truncated) +
                    ",\"semantic_truncated\":" + json_bool(job.semantic_truncated) + "}";
                append_midi_json(body, job);
            }
        }
        if (job.status == "failed") {
            body += ",\"error\":" + json::quote(job.error) + ",\"cancelled\":" + json_bool(job.cancelled);
        }
        body += "}";
        if (consume && job.done()) jobs_.erase(found);
        return yue2::server::json_response(std::move(body));
    }

    std::string queue_status_locked(const Job & job) const {
        if (job.status == "queued") {
            std::size_t position = 1;
            for (const auto & entry : queue_) {
                if (entry.get() == &job) break;
                ++position;
            }
            return "{\"status\":\"queued\",\"position\":" + std::to_string(position) +
                ",\"total_queued\":" + std::to_string(queue_.size()) + ",\"message\":\"queued locally\"}";
        }
        if (!job.done()) return "{\"status\":\"ready\",\"message\":" + json::quote(job.stage) + "}";
        return "{}";
    }

    HttpResponse cancel(const std::string & id) {
        std::lock_guard<std::mutex> lock(jobs_mutex_);
        const auto found = jobs_.find(id);
        if (found == jobs_.end()) return failure(404, "unknown session: " + id);
        auto & job = *found->second;
        if (job.done()) return failure(409, "session already finished");
        job.cancel.store(true);
        const auto queued = std::find(queue_.begin(), queue_.end(), found->second);
        if (queued != queue_.end()) {
            queue_.erase(queued);
            fail_locked(job, "cancelled");
        }
        return yue2::server::json_response(
            "{\"success\":true,\"session_id\":" + json::quote(id) + ",\"status\":" + json::quote(job.status) + "}");
    }

    HttpResponse unload() {
        std::unique_lock<std::mutex> lock(models_mutex_, std::try_to_lock);
        if (!lock.owns_lock()) return failure(409, "a job is running; unload after it finishes");
        release_models();
        return yue2::server::json_response("{\"success\":true,\"status\":\"unloaded\"}");
    }

    // --- Worker -------------------------------------------------------------

    void work() {
        for (;;) {
            std::shared_ptr<Job> job;
            {
                std::unique_lock<std::mutex> lock(jobs_mutex_);
                jobs_ready_.wait(lock, [this]() { return stopping_ || !queue_.empty(); });
                if (stopping_) return;
                job = queue_.front();
                queue_.pop_front();
                running_ = job;
            }
            {
                std::lock_guard<std::mutex> models(models_mutex_);
                execute(*job);
            }
            std::lock_guard<std::mutex> lock(jobs_mutex_);
            running_.reset();
        }
    }

    void execute(Job & job) {
        try {
            if (job.cancel.load()) throw std::runtime_error("cancelled");
            if (job.kind == JobKind::cover || job.kind == JobKind::continue_audio ||
                job.kind == JobKind::transcribe) {
                transcribe(job);
            }
            if (job.kind == JobKind::continue_audio) tokenize_continuation(job);
            if (job.kind == JobKind::plan) plan(job);
            if (job.kind == JobKind::generate || job.kind == JobKind::cover ||
                job.kind == JobKind::continue_audio) {
                generate(job);
            }
        } catch (const std::exception & error) {
            // Whatever failed may have been an allocation; hand the memory back.
            if (!job.keep_models) release_models();
            std::lock_guard<std::mutex> lock(jobs_mutex_);
            fail_locked(job, job.cancel.load() ? "cancelled" : error.what());
        }
    }

    void update(Job & job, const char * status, const char * stage, int progress,
        std::uint32_t step, std::uint32_t total_steps) {
        std::lock_guard<std::mutex> lock(jobs_mutex_);
        job.status = status;
        job.stage = stage;
        job.progress = std::clamp(std::max(job.progress, progress), 0, 99);
        job.step = step;
        job.total_steps = total_steps;
    }

    void transcribe(Job & job) {
        const bool cover = job.kind == JobKind::cover;
        const bool continuation = job.kind == JobKind::continue_audio;
        update(job, "transcribing", "load", 0, 0, 0);
        if (!transcriber_) {
            yue2::TranscriberRuntimeOptions options;
            options.device = configuration_.device;
            options.threads = configuration_.threads;
            transcriber_ = std::make_unique<yue2::Transcriber>(resolve_transcription(configuration_), options);
            transcriber_loaded_.store(true);
        }
        yue2::TranscriptionControl control;
        control.should_cancel = [&job]() { return job.cancel.load(); };
        control.on_progress = [&](std::size_t current, std::size_t total) {
            const int span = cover ? 15 : (continuation ? 12 : 99);
            update(job, "transcribing", "transcription",
                static_cast<int>(span * current / std::max<std::size_t>(1, total)),
                static_cast<std::uint32_t>(current), static_cast<std::uint32_t>(total));
        };
        auto result = transcriber_->transcribe_mono(
            job.input.samples.data(), job.input.samples.size(), job.input.sample_rate,
            job.transcription, control);
        if (!job.keep_models || continuation) {
            transcriber_.reset();
            transcriber_loaded_.store(false);
        }
        if (job.cancel.load()) throw std::runtime_error("cancelled");

        std::lock_guard<std::mutex> lock(jobs_mutex_);
        job.abc = result.abc;
        if (cover || continuation) {
            if (result.abc.empty()) throw std::runtime_error("transcription produced no score");
            if (cover) {
                job.song.abc = result.abc;
                job.input = {};
            } else {
                set_midi_exports(job, result.midi_exports);
                job.events_json = yue2::serialize_transcription_json(result);
                if (result.abc.back() != '\n') result.abc.push_back('\n');
                job.song.abc_prefix = result.abc;
                if (job.continuation_bars != 0) {
                    const auto source = yue2::inspect_abc_score(result.abc);
                    if (source.bars == 0 ||
                        source.bars > std::numeric_limits<std::uint32_t>::max() -
                            job.continuation_bars) {
                        throw std::runtime_error(
                            "could not derive a safe continuation bar target from transcription");
                    }
                    job.song.target_bars = source.bars + job.continuation_bars;
                    job.song.ending_mode = yue2::EndingMode::outro;
                    job.song.outro_bars = std::min(4U, job.continuation_bars);
                }
            }
            return;
        }
        set_midi_exports(job, result.midi_exports);
        job.events_json = yue2::serialize_transcription_json(result);
        job.duration_seconds = result.duration_seconds;
        complete_locked(job);
    }

    void tokenize_continuation(Job & job) {
        update(job, "tokenizing", "semantic-tokenizer", 12, 0, 0);
        auto samples = job.input.sample_rate == yue2::audio::transcription_sample_rate
            ? job.input.samples
            : yue2::audio::resample_sinc(
                  job.input.samples, job.input.sample_rate,
                  yue2::audio::transcription_sample_rate);
        yue2::mert2::EncoderOptions options;
        options.device = configuration_.device;
        options.threads = configuration_.threads;
        {
            // The tokenizer contains its own unmodified MERT parent. Keep it
            // request-local and destroy it before loading the 3B generator.
            yue2::mert2::Encoder tokenizer(
                resolve_semantic_tokenizer(configuration_).string(), options);
            job.song.semantic_prefix = tokenizer.semantic_tokens_24k(samples);
        }
        if (job.song.semantic_prefix.empty()) {
            throw std::runtime_error("real-audio tokenizer produced no continuation frames");
        }
        job.semantic_prefix_frames = job.song.semantic_prefix.size();
        update(
            job, "tokenizing", "semantic-prefix", 25,
            static_cast<std::uint32_t>(job.semantic_prefix_frames),
            static_cast<std::uint32_t>(job.semantic_prefix_frames));
    }

    void generate(Job & job) {
        const int base = job.kind == JobKind::cover ? 15 :
            (job.kind == JobKind::continue_audio ? 25 : 0);
        update(job, "generating", "load", base, 0, 0);
        load_generator(job.loras, job.encoding);

        yue2::GenerationControl control;
        control.should_cancel = [&job]() { return job.cancel.load(); };
        control.on_progress = [&, base](yue2::GenerationStage stage, std::uint32_t current, std::uint32_t total) {
            const double fraction = total ? static_cast<double>(current) / total : 1.0;
            const auto at = [&](double from, double to) {
                return static_cast<int>(base + (100 - base) * (from + (to - from) * fraction));
            };
            switch (stage) {
                case yue2::GenerationStage::abc:
                    update(job, "generating", "abc", at(0.0, 0.10), current, total);
                    break;
                case yue2::GenerationStage::semantic:
                    update(job, "generating", "semantic", at(0.10, 0.80), current, total);
                    break;
                case yue2::GenerationStage::flow:
                    update(job, "generating", "flow", at(0.80, 0.92), current, total);
                    break;
                case yue2::GenerationStage::decode:
                case yue2::GenerationStage::complete:
                    update(job, "decoding", "decode", at(0.92, 0.99), current, total);
                    break;
            }
        };
        auto song = generator_->generate(job.song, job.run, control);
        if (!job.keep_models) {
            generator_.reset();
            generator_loaded_.store(false);
        }
        const auto wav = job.float_wav
            ? yue2::audio::encode_wav_float(song.audio.interleaved_samples, song.audio.sample_rate, song.audio.channels)
            : encode_wav_pcm16(song.audio.interleaved_samples, song.audio.sample_rate, song.audio.channels);
        auto encoded = yue2::server::base64_encode(wav.data(), wav.size());

        std::lock_guard<std::mutex> lock(jobs_mutex_);
        job.audio_data = std::move(encoded);
        job.abc = song.abc;
        set_midi_exports(job, song.midi_exports);
        job.semantic_frames = song.semantic_codec_ids.size();
        job.semantic_prefix_frames = song.semantic_prefix_frames;
        job.abc_truncated = song.abc_truncated;
        job.abc_repaired = song.abc_repaired;
        job.semantic_truncated = song.semantic_truncated;
        job.score_bars = song.score_bars;
        job.score_duration_seconds = song.score_duration_seconds;
        job.semantic_budget = song.semantic_budget;
        job.duration_seconds = static_cast<double>(song.audio.interleaved_samples.size()) /
            (static_cast<double>(song.audio.sample_rate) * song.audio.channels);
        job.input = {};
        complete_locked(job);
    }

    void plan(Job & job) {
        update(job, "planning", "load", 0, 0, 0);
        load_generator(job.loras, job.encoding);

        yue2::GenerationControl control;
        control.should_cancel = [&job]() { return job.cancel.load(); };
        control.on_progress = [&](yue2::GenerationStage stage, std::uint32_t current,
                                  std::uint32_t total) {
            if (stage != yue2::GenerationStage::abc) return;
            const auto progress = total
                ? static_cast<int>(99ULL * current / total)
                : 0;
            update(job, "planning", "abc", progress, current, total);
        };
        auto result = generator_->plan(job.song, job.run, control);
        if (!job.keep_models) {
            generator_.reset();
            generator_loaded_.store(false);
        }

        std::lock_guard<std::mutex> lock(jobs_mutex_);
        job.abc = std::move(result.abc);
        set_midi_exports(job, result.midi_exports);
        job.abc_truncated = result.abc_truncated;
        job.abc_repaired = result.abc_repaired;
        job.score_bars = result.score_bars;
        job.score_duration_seconds = result.score_duration_seconds;
        complete_locked(job);
    }

    void load_generator(
        const std::vector<yue2::LoraAdapterSpec> & loras,
        const std::string & encoding) {
        const auto same = generator_ && loras.size() == generator_loras_.size() &&
            encoding == generator_encoding_ &&
            std::equal(loras.begin(), loras.end(), generator_loras_.begin(),
                [](const yue2::LoraAdapterSpec & a, const yue2::LoraAdapterSpec & b) {
                    return a.path == b.path && a.strength == b.strength;
                });
        if (same) return;
        // Adapters are bound at construction, so a different set reloads.
        generator_.reset();
        generator_loaded_.store(false);
        const auto paths = resolve_generation(configuration_, encoding);
        yue2::GenerationPipelineOptions options;
        options.autoregressive.device = configuration_.device;
        options.autoregressive.threads = configuration_.threads;
        options.autoregressive.lora_adapters = loras;
        generator_ = std::make_unique<yue2::GenerationPipeline>(
            paths.model.string(), paths.vae.string(), paths.tokenizer.string(), options);
        generator_loras_ = loras;
        generator_encoding_ = encoding;
        generator_loaded_.store(true);
    }

    void release_models() {
        generator_.reset();
        generator_loaded_.store(false);
        generator_encoding_.clear();
        transcriber_.reset();
        transcriber_loaded_.store(false);
    }

    void complete_locked(Job & job) {
        job.status = "completed";
        job.stage = "complete";
        job.progress = 100;
        job.finished = std::chrono::steady_clock::now();
    }

    void fail_locked(Job & job, const std::string & message) {
        job.status = "failed";
        job.stage = "failed";
        job.error = message;
        job.cancelled = job.cancel.load();
        job.input = {};
        job.finished = std::chrono::steady_clock::now();
    }

    void prune_locked() {
        const auto now = std::chrono::steady_clock::now();
        for (auto entry = jobs_.begin(); entry != jobs_.end();) {
            if (entry->second->done() && now - entry->second->finished > kFinishedJobLifetime) {
                entry = jobs_.erase(entry);
            } else {
                ++entry;
            }
        }
    }


    yue2::server::GenerationPolicy generation_policy() const {
        yue2::server::GenerationPolicy policy;
        policy.natural_max_seconds = configuration_.natural_max_seconds;
        policy.planning_overrun = configuration_.planning_overrun;
        policy.planning_loop_bars = configuration_.planning_loop_bars;
        return policy;
    }

    Configuration configuration_;

    std::mutex jobs_mutex_;
    std::condition_variable jobs_ready_;
    std::map<std::string, std::shared_ptr<Job>> jobs_;
    std::deque<std::shared_ptr<Job>> queue_;
    std::shared_ptr<Job> running_;
    bool stopping_ = false;

    // Models are used only by the worker, which holds models_mutex_ for a
    // whole job; /unload takes it without waiting.
    std::mutex models_mutex_;
    std::unique_ptr<yue2::Transcriber> transcriber_;
    std::unique_ptr<yue2::GenerationPipeline> generator_;
    std::vector<yue2::LoraAdapterSpec> generator_loras_;
    std::string generator_encoding_;
    std::atomic<bool> transcriber_loaded_{false};
    std::atomic<bool> generator_loaded_{false};

    std::thread worker_;
};

} // namespace

int main(int argc, char ** argv) {
    try {
        if (has(argc, argv, "--help") || has(argc, argv, "-h")) {
            usage(argv[0]);
            return 0;
        }
        if (has(argc, argv, "--version")) {
            std::cout << yue2::version() << '\n';
            return 0;
        }
        auto configuration = parse_configuration(argc, argv);
        discover_published_adapters(configuration);
        if (has(argc, argv, "--props")) {
            // An installer asks this before it ever starts the service: which
            // backends actually came up, how much memory they report, and which
            // tier fits. It goes through the real route so the two cannot drift.
            ServerState state(std::move(configuration));
            HttpRequest request;
            request.method = "GET";
            request.path = "/props";
            const auto response = state.handle(request);
            std::cout << response.body << '\n';
            return response.status == 200 ? 0 : 1;
        }
        const auto host = configuration.http.host;
        const auto http = configuration.http;
        if (host != "127.0.0.1") {
            std::cerr << "[yue2-server] WARNING: bound to " << host
                      << "; anyone who can reach this port can submit jobs\n";
        }
        ServerState state(std::move(configuration));
        std::signal(SIGINT, stop_signal);
        std::signal(SIGTERM, stop_signal);
        yue2::server::serve_http(
            http,
            [&state](const HttpRequest & request) { return state.handle(request); },
            stop_requested);
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "yue2-server: " << error.what() << '\n';
        return 2;
    }
}
