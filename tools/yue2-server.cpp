#include "yue2/audio.h"
#include "yue2/generation_pipeline.h"
#include "yue2/transcription.h"

#include "server/http.h"
#include "server/json.h"
#include "server/multipart.h"

#include <atomic>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <iostream>
#include <locale>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

std::atomic<bool> stop_requested{false};

void stop_signal(int) { stop_requested.store(true); }

struct Configuration {
    yue2::server::HttpServerOptions http;
    std::string transcription_model;
    std::string generation_model;
    std::string vae;
    std::string tokenizer;
    std::string device;
    int threads = 0;
    std::string transcription_id = "yue2-transcription";
    std::string generation_id = "yue2-generation";
    std::vector<yue2::LoraAdapterSpec> loras;
};

bool has(int argc, char ** argv, std::string_view name) {
    for (int index = 1; index < argc; ++index) {
        if (std::string_view(argv[index]) == name) return true;
    }
    return false;
}

std::string option(
    int argc, char ** argv, std::string_view name, bool required = false) {
    for (int index = 1; index < argc; ++index) {
        if (std::string_view(argv[index]) != name) continue;
        if (index + 1 == argc || std::string_view(argv[index + 1]).rfind("--", 0) == 0) {
            throw std::invalid_argument(std::string(name) + " requires a value");
        }
        return argv[index + 1];
    }
    if (required) throw std::invalid_argument("missing required option " + std::string(name));
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
        << "Usage: " << executable << " [model options] [server options]\n\n"
        << "Model options:\n"
        << "  --transcription-model PATH   Enable audio-to-ABC transcription\n"
        << "  --model PATH --vae PATH --tokenizer PATH   Enable music generation\n"
        << "  --lora PATH[=SCALE]          Resident generation adapter; repeatable\n"
        << "  --device NAME                cpu, cuda, or another GGML backend\n"
        << "  --threads N                  CPU worker threads\n\n"
        << "Server options:\n"
        << "  --host IPV4                  Bind address (default 127.0.0.1)\n"
        << "  --port N                     Port (default 8080)\n"
        << "  --max-body-mb N              Upload limit (default 512)\n"
        << "  --transcription-id ID        Model id advertised by /v1/models\n"
        << "  --generation-id ID           Model id advertised by /v1/models\n\n"
        << "Routes: GET /health, GET /v1/models, POST /v1/audio/transcriptions,\n"
        << "        POST /v1/music/generations, POST /v1/tasks/run\n";
}

Configuration parse_configuration(int argc, char ** argv) {
    Configuration result;
    if (const auto value = option(argc, argv, "--host"); !value.empty()) result.http.host = value;
    if (const auto value = option(argc, argv, "--port"); !value.empty()) result.http.port = std::stoi(value);
    if (const auto value = option(argc, argv, "--max-body-mb"); !value.empty()) {
        const auto megabytes = std::stoull(value);
        if (megabytes == 0 || megabytes > std::numeric_limits<std::uint64_t>::max() / (1024 * 1024)) {
            throw std::invalid_argument("invalid --max-body-mb");
        }
        result.http.max_request_body_bytes = megabytes * 1024 * 1024;
    }
    result.transcription_model = option(argc, argv, "--transcription-model");
    result.generation_model = option(argc, argv, "--model");
    result.vae = option(argc, argv, "--vae");
    result.tokenizer = option(argc, argv, "--tokenizer");
    result.device = option(argc, argv, "--device");
    if (const auto value = option(argc, argv, "--threads"); !value.empty()) result.threads = std::stoi(value);
    if (const auto value = option(argc, argv, "--transcription-id"); !value.empty()) {
        result.transcription_id = value;
    }
    if (const auto value = option(argc, argv, "--generation-id"); !value.empty()) {
        result.generation_id = value;
    }
    for (const auto & value : options(argc, argv, "--lora")) result.loras.push_back(lora_spec(value));
    const bool any_generation = !result.generation_model.empty() || !result.vae.empty() || !result.tokenizer.empty();
    const bool full_generation = !result.generation_model.empty() && !result.vae.empty() && !result.tokenizer.empty();
    if (any_generation != full_generation) {
        throw std::invalid_argument("generation requires --model, --vae, and --tokenizer together");
    }
    if (result.transcription_model.empty() && !full_generation) {
        throw std::invalid_argument("enable transcription, generation, or both");
    }
    if (!full_generation && !result.loras.empty()) throw std::invalid_argument("--lora requires generation");
    if (result.threads < 0) throw std::invalid_argument("--threads cannot be negative");
    return result;
}

std::string header(
    const yue2::server::HttpRequest & request,
    const std::string & name) {
    const auto found = request.headers.find(name);
    return found == request.headers.end() ? std::string{} : found->second;
}

std::string base64(const std::vector<std::uint8_t> & input) {
    static constexpr char alphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string output;
    output.reserve((input.size() + 2) / 3 * 4);
    for (std::size_t offset = 0; offset < input.size(); offset += 3) {
        const auto remaining = input.size() - offset;
        const std::uint32_t value = static_cast<std::uint32_t>(input[offset]) << 16 |
            (remaining > 1 ? static_cast<std::uint32_t>(input[offset + 1]) << 8 : 0) |
            (remaining > 2 ? input[offset + 2] : 0);
        output.push_back(alphabet[(value >> 18) & 63]);
        output.push_back(alphabet[(value >> 12) & 63]);
        output.push_back(remaining > 1 ? alphabet[(value >> 6) & 63] : '=');
        output.push_back(remaining > 2 ? alphabet[value & 63] : '=');
    }
    return output;
}

std::string real(double value) {
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output.precision(17);
    output << value;
    return output.str();
}

yue2::SymbolicMode symbolic_mode(const std::string & value) {
    if (value == "off") return yue2::SymbolicMode::off;
    if (value == "melody") return yue2::SymbolicMode::melody;
    if (value == "full" || value.empty()) return yue2::SymbolicMode::full;
    throw std::invalid_argument("symbolic_mode must be off, melody, or full");
}

class ServerState {
public:
    explicit ServerState(const Configuration & configuration)
        : configuration_(configuration) {
        if (!configuration.transcription_model.empty()) {
            yue2::TranscriberRuntimeOptions options;
            options.device = configuration.device;
            options.threads = configuration.threads;
            transcriber_ = std::make_unique<yue2::Transcriber>(
                configuration.transcription_model, options);
        }
        if (!configuration.generation_model.empty()) {
            yue2::GenerationPipelineOptions options;
            options.autoregressive.device = configuration.device;
            options.autoregressive.threads = configuration.threads;
            options.autoregressive.lora_adapters = configuration.loras;
            generation_defaults_ = {options.generation, options.flow};
            generator_ = std::make_unique<yue2::GenerationPipeline>(
                configuration.generation_model, configuration.vae,
                configuration.tokenizer, options);
        }
    }

    yue2::server::HttpResponse handle(const yue2::server::HttpRequest & request) {
        try {
            if (request.method == "GET" && request.path == "/health") return health();
            if (request.method == "GET" && request.path == "/v1/models") return models();
            if (request.method == "POST" && request.path == "/v1/audio/transcriptions") {
                return transcribe(request);
            }
            if (request.method == "POST" &&
                (request.path == "/v1/music/generations" || request.path == "/v1/tasks/run")) {
                return generate(request);
            }
            if (request.path == "/health" || request.path == "/v1/models" ||
                request.path == "/v1/audio/transcriptions" ||
                request.path == "/v1/music/generations" || request.path == "/v1/tasks/run") {
                return yue2::server::error_response(
                    405, "method is not allowed for this route", "method_not_allowed");
            }
            return yue2::server::error_response(404, "route not found", "not_found");
        } catch (const std::invalid_argument & error) {
            return yue2::server::error_response(400, error.what(), "invalid_request_error");
        } catch (const std::exception & error) {
            return yue2::server::error_response(500, error.what(), "inference_error");
        }
    }

private:
    yue2::server::HttpResponse health() const {
        return yue2::server::json_response(
            "{\"status\":\"ok\",\"version\":" +
            yue2::server::json::quote(yue2::version()) +
            ",\"transcription\":" + (transcriber_ ? "true" : "false") +
            ",\"generation\":" + (generator_ ? "true" : "false") + "}");
    }

    yue2::server::HttpResponse models() const {
        std::string body = "{\"object\":\"list\",\"data\":[";
        bool comma = false;
        if (transcriber_) {
            body += "{\"id\":" + yue2::server::json::quote(configuration_.transcription_id) +
                ",\"object\":\"model\",\"owned_by\":\"yue2.cpp\",\"task\":\"transcription\"}";
            comma = true;
        }
        if (generator_) {
            if (comma) body.push_back(',');
            body += "{\"id\":" + yue2::server::json::quote(configuration_.generation_id) +
                ",\"object\":\"model\",\"owned_by\":\"yue2.cpp\",\"task\":\"music-generation\"}";
        }
        body += "]}";
        return yue2::server::json_response(std::move(body));
    }

    yue2::server::HttpResponse transcribe(const yue2::server::HttpRequest & request) {
        if (!transcriber_) return yue2::server::error_response(503, "transcription model is disabled", "model_unavailable");
        std::string audio;
        std::map<std::string, std::string> fields;
        const auto content_type = header(request, "content-type");
        if (const auto boundary = yue2::server::multipart_boundary(content_type)) {
            for (auto & part : yue2::server::parse_multipart(request.body, *boundary)) {
                if (part.name == "file") {
                    if (!audio.empty()) throw std::invalid_argument("multipart request has multiple file fields");
                    audio = std::move(part.data);
                } else {
                    fields[part.name] = std::move(part.data);
                }
            }
        } else if (content_type.rfind("audio/wav", 0) == 0 ||
                   content_type.rfind("audio/x-wav", 0) == 0 ||
                   content_type.rfind("application/octet-stream", 0) == 0) {
            audio = request.body;
        } else {
            return yue2::server::error_response(415, "send a WAV body or multipart/form-data", "unsupported_media_type");
        }
        if (audio.empty()) throw std::invalid_argument("transcription audio is empty");
        if (const auto found = fields.find("model"); found != fields.end() &&
            found->second != configuration_.transcription_id) {
            throw std::invalid_argument("unknown transcription model: " + found->second);
        }
        yue2::TranscriptionOptions options;
        if (const auto found = fields.find("mode"); found != fields.end()) {
            if (found->second == "full") options.melody_only = false;
            else if (found->second == "melody") options.melody_only = true;
            else throw std::invalid_argument("transcription mode must be melody or full");
        }
        const auto decoded = yue2::audio::decode_wav_mono(
            reinterpret_cast<const std::uint8_t *>(audio.data()), audio.size());
        yue2::TranscriptionResult result;
        {
            std::lock_guard<std::mutex> lock(transcription_mutex_);
            result = transcriber_->transcribe_mono(
                decoded.samples.data(), decoded.samples.size(), decoded.sample_rate, options);
        }
        const auto format = fields.count("response_format") ? fields["response_format"] : "json";
        if (format == "abc" || format == "text") {
            return {200, "text/plain; charset=utf-8", std::move(result.abc), {}};
        }
        const auto native = yue2::serialize_transcription_json(result);
        if (format == "verbose_json") return yue2::server::json_response(native);
        if (format != "json") throw std::invalid_argument("response_format must be json, verbose_json, or abc");
        return yue2::server::json_response(
            "{\"text\":" + yue2::server::json::quote(result.abc) +
            ",\"abc\":" + yue2::server::json::quote(result.abc) +
            ",\"duration\":" + real(result.duration_seconds) +
            ",\"model\":" + yue2::server::json::quote(configuration_.transcription_id) +
            ",\"result\":" + native + "}");
    }

    yue2::server::HttpResponse generate(const yue2::server::HttpRequest & request) {
        if (!generator_) return yue2::server::error_response(503, "generation model is disabled", "model_unavailable");
        const auto content_type = header(request, "content-type");
        if (content_type.rfind("application/json", 0) != 0) {
            return yue2::server::error_response(415, "music generation requires application/json", "unsupported_media_type");
        }
        const auto root = yue2::server::json::parse(request.body);
        if (root.type != yue2::server::json::Type::object) throw std::invalid_argument("request body must be a JSON object");
        const auto model = yue2::server::json::string(root, "model", configuration_.generation_id);
        if (model != configuration_.generation_id) throw std::invalid_argument("unknown generation model: " + model);
        const auto * payload = &root;
        if (request.path == "/v1/tasks/run") {
            if (const auto * nested = root.find("request")) {
                if (nested->type != yue2::server::json::Type::object) {
                    throw std::invalid_argument("request must be a JSON object");
                }
                payload = nested;
            }
        }

        yue2::SongRequest song;
        song.style = yue2::server::json::string(
            *payload, "style", yue2::server::json::string(*payload, "instructions", {}));
        song.lyrics = yue2::server::json::string(
            *payload, "lyrics", yue2::server::json::string(*payload, "input", {}));
        const auto abc = yue2::server::json::string(*payload, "abc", {});
        if (!abc.empty()) song.abc = abc;
        song.symbolic_mode = symbolic_mode(yue2::server::json::string(*payload, "symbolic_mode", "full"));
        song.seed = yue2::server::json::u64(*payload, "seed", 831001);
        song.guidance_scale = static_cast<float>(
            yue2::server::json::number(*payload, "guidance_scale", 1.5));

        auto run = generation_defaults_;
        run.generation.abc.min_tokens = yue2::server::json::u32(
            *payload, "abc_min_tokens", run.generation.abc.min_tokens);
        run.generation.abc.max_tokens = yue2::server::json::u32(
            *payload, "abc_max_tokens", run.generation.abc.max_tokens);
        run.generation.semantic.min_tokens = yue2::server::json::u32(
            *payload, "semantic_min_tokens", run.generation.semantic.min_tokens);
        run.generation.semantic.max_tokens = yue2::server::json::u32(
            *payload, "semantic_max_tokens", run.generation.semantic.max_tokens);
        run.generation.semantic.temperature = static_cast<float>(
            yue2::server::json::number(*payload, "temperature", run.generation.semantic.temperature));
        run.generation.semantic.top_k = yue2::server::json::u32(
            *payload, "top_k", run.generation.semantic.top_k);
        run.generation.semantic.top_p = static_cast<float>(
            yue2::server::json::number(*payload, "top_p", run.generation.semantic.top_p));
        run.generation.semantic.repetition_penalty = static_cast<float>(
            yue2::server::json::number(
                *payload, "repetition_penalty", run.generation.semantic.repetition_penalty));
        run.generation.semantic.penalty_window = yue2::server::json::u32(
            *payload, "penalty_window", run.generation.semantic.penalty_window);
        run.flow.ode_steps = yue2::server::json::u32(*payload, "ode_steps", run.flow.ode_steps);

        yue2::GeneratedSong generated;
        {
            std::lock_guard<std::mutex> lock(generation_mutex_);
            generated = generator_->generate(song, run);
        }
        const auto wav = yue2::audio::encode_wav_float(
            generated.audio.interleaved_samples,
            generated.audio.sample_rate,
            generated.audio.channels);
        const auto format = yue2::server::json::string(*payload, "response_format", "wav");
        if (format == "wav") {
            yue2::server::HttpResponse response;
            response.content_type = "audio/wav";
            response.body.assign(reinterpret_cast<const char *>(wav.data()), wav.size());
            response.headers["X-YuE2-Seed"] = std::to_string(song.seed);
            response.headers["X-YuE2-Semantic-Frames"] =
                std::to_string(generated.semantic_codec_ids.size());
            return response;
        }
        if (format != "json") throw std::invalid_argument("response_format must be wav or json");
        std::string body = "{\"model\":" + yue2::server::json::quote(configuration_.generation_id) +
            ",\"seed\":\"" + std::to_string(song.seed) + "\",\"abc\":" +
            yue2::server::json::quote(generated.abc) +
            ",\"semantic_count\":" + std::to_string(generated.semantic_codec_ids.size()) +
            ",\"audio\":{\"format\":\"wav\",\"encoding\":\"base64\",\"sample_rate\":" +
            std::to_string(generated.audio.sample_rate) + ",\"channels\":" +
            std::to_string(generated.audio.channels) + ",\"data\":" +
            yue2::server::json::quote(base64(wav)) + "}}";
        return yue2::server::json_response(std::move(body));
    }

    Configuration configuration_;
    std::unique_ptr<yue2::Transcriber> transcriber_;
    std::unique_ptr<yue2::GenerationPipeline> generator_;
    yue2::GenerationRunOptions generation_defaults_;
    std::mutex transcription_mutex_;
    std::mutex generation_mutex_;
};

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc == 1 || has(argc, argv, "--help") || has(argc, argv, "-h")) {
            usage(argv[0]);
            return 0;
        }
        const auto configuration = parse_configuration(argc, argv);
        ServerState state(configuration);
        std::signal(SIGINT, stop_signal);
        std::signal(SIGTERM, stop_signal);
        yue2::server::serve_http(
            configuration.http,
            [&state](const yue2::server::HttpRequest & request) {
                return state.handle(request);
            },
            stop_requested);
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "yue2-server: " << error.what() << '\n';
        return 2;
    }
}
