#include "yue2/audio.h"
#include "yue2/generation_pipeline.h"
#include "yue2/runtime_info.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

void usage(const char * argv0) {
    std::cout
        << "Usage: " << argv0 << " --prompt TEXT --out song.wav [options]\n\n"
        << "Models resolve from $YUE2_MODELS_DIR or ./models by default.\n\n"
        << "Options:\n"
        << "  --models-dir DIR      Model root (default $YUE2_MODELS_DIR or ./models)\n"
        << "  --encoding TYPE       auto, BF16, F16, Q8_0, Q5_K_M, Q4_K_M, or F32\n"
        << "  --model/--vae/--tokenizer PATH  Override resolved component paths\n"
        << "  --prompt, --style TEXT          Production/style description\n"
        << "  --out, --output PATH            Output WAV\n"
        << "  --duration, --seconds N         Approximate output length in seconds\n"
        << "  --lyrics TEXT | --lyrics-file PATH  Optional vocal lyrics\n"
        << "  --instrumental        Generate without vocals using score-aligned sections\n"
        << "  --experimental-vocal-rest  Rest Vocal without changing Ins; not an instrumental mode\n"
        << "  --symbolic MODE       off, melody, or full (default full)\n"
        << "  --abc PATH            Use an external ABC score instead of planning one\n"
        << "  --abc-prefix PATH     Seed symbolic planning with an exact ABC text prefix\n"
        << "  --bpm N --key KEY     Build a trusted planning prefix (for example --key \"C# minor\")\n"
        << "  --meter N/D           Planning meter when --bpm/--key are used (default 4/4)\n"
        << "  --score-output PATH   Write the used or generated ABC score\n"
        << "  --seed N              Sampling and flow-noise seed (default 831001)\n"
        << "  --guidance N          Classifier-free guidance scale\n"
        << "  --abc-min-tokens N    ABC minimum before stop (default 32)\n"
        << "  --abc-max-tokens N    ABC planning limit (default 4096)\n"
        << "  --semantic-min-tokens N  Audio-code minimum before stop (default 200)\n"
        << "  --semantic-max-tokens N  Audio-code limit (default 9000)\n"
        << "  --ode-steps N         Midpoint flow steps (default 32)\n"
        << "  --temperature N       Semantic sampling temperature (default 1.0)\n"
        << "  --top-k N             Semantic top-k (default 100)\n"
        << "  --top-p N             Semantic top-p (default 0.95)\n"
        << "  --lora PATH[=SCALE]   Load a generation LoRA; repeat to stack adapters\n"
        << "  --device NAME         Require a backend such as cuda or cpu\n"
        << "  --threads N           CPU worker threads\n";
}

bool has(int argc, char ** argv, std::string_view name) {
    for (int index = 1; index < argc; ++index) {
        if (std::string_view(argv[index]) == name) return true;
    }
    return false;
}

std::string value_after(int argc, char ** argv, std::string_view name, bool required = true) {
    for (int index = 1; index < argc; ++index) {
        if (std::string_view(argv[index]) != name) continue;
        if (index + 1 >= argc || std::string_view(argv[index + 1]).rfind("--", 0) == 0) {
            throw std::runtime_error("option requires a value: " + std::string(name));
        }
        return argv[index + 1];
    }
    if (required) throw std::runtime_error("missing required option " + std::string(name));
    return {};
}

std::string first_value_after(
    int argc, char ** argv, std::initializer_list<std::string_view> names,
    bool required = true) {
    for (const auto name : names) {
        const auto value = value_after(argc, argv, name, false);
        if (!value.empty() || has(argc, argv, name)) return value;
    }
    if (required) {
        std::string message = "missing required option ";
        bool first = true;
        for (const auto name : names) {
            if (!first) message += " or ";
            message += std::string(name);
            first = false;
        }
        throw std::runtime_error(message);
    }
    return {};
}

std::string environment(const char * name) {
    const char * value = std::getenv(name);
    return value ? std::string(value) : std::string();
}

std::string upper(std::string value) {
    for (auto & character : value) {
        character = static_cast<char>(std::toupper(static_cast<unsigned char>(character)));
    }
    return value;
}

struct ModelPaths {
    std::string model;
    std::string vae;
    std::string tokenizer;
    std::string encoding;
};

ModelPaths resolve_models(
    const std::string & models_directory,
    const std::string & requested_encoding,
    std::string model,
    std::string vae,
    std::string tokenizer) {
    const auto files = yue2::inspect_model_files(models_directory);
    auto encoding = upper(requested_encoding);
    if (encoding.empty() || encoding == "AUTO") encoding = "auto";
    static const std::vector<std::string> preference = {
        "BF16", "F16", "Q8_0", "Q5_K_M", "Q4_K_M", "F32"};
    if (encoding != "auto" &&
        std::find(preference.begin(), preference.end(), encoding) == preference.end()) {
        throw std::runtime_error(
            "--encoding must be auto, BF16, F16, Q8_0, Q5_K_M, Q4_K_M, or F32");
    }
    if (model.empty()) {
        const auto encodings = encoding == "auto"
            ? preference
            : std::vector<std::string>{encoding};
        const auto found = yue2::find_model_file(files, "generation", encodings);
        if (!found) {
            throw std::runtime_error(
                "no YuE2 generation GGUF for " + encoding + " under " + models_directory);
        }
        model = found->path;
        encoding = found->encoding;
    }
    if (vae.empty()) {
        const auto found = yue2::find_model_file(files, "vae", {"F16", "F32"});
        if (!found) throw std::runtime_error("no YuE2 VAE GGUF under " + models_directory);
        vae = found->path;
    }
    if (tokenizer.empty()) {
        const auto found = yue2::find_model_file(files, "tokenizer");
        if (!found) throw std::runtime_error("no yue2-qwen.tiktoken under " + models_directory);
        tokenizer = found->path;
    }
    return {std::move(model), std::move(vae), std::move(tokenizer), std::move(encoding)};
}

std::vector<std::string> values_after(
    int argc, char ** argv, std::string_view name) {
    std::vector<std::string> values;
    for (int index = 1; index < argc; ++index) {
        if (std::string_view(argv[index]) != name) continue;
        if (index + 1 >= argc || std::string_view(argv[index + 1]).rfind("--", 0) == 0) {
            throw std::runtime_error("option requires a value: " + std::string(name));
        }
        values.emplace_back(argv[++index]);
    }
    return values;
}

yue2::LoraAdapterSpec parse_lora(const std::string & value) {
    yue2::LoraAdapterSpec result;
    const auto separator = value.rfind('=');
    if (separator == std::string::npos) {
        result.path = value;
        return result;
    }
    result.path = value.substr(0, separator);
    if (result.path.empty() || separator + 1 == value.size()) {
        throw std::runtime_error("--lora expects PATH or PATH=SCALE");
    }
    std::size_t used = 0;
    result.strength = std::stof(value.substr(separator + 1), &used);
    if (used != value.size() - separator - 1) {
        throw std::runtime_error("invalid --lora scale: " + value);
    }
    return result;
}

std::string read_text(const std::filesystem::path & path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open " + path.string());
    std::ostringstream contents;
    contents << stream.rdbuf();
    if (!stream.good() && !stream.eof()) throw std::runtime_error("cannot read " + path.string());
    return contents.str();
}

void write_text(const std::filesystem::path & path, const std::string & text) {
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    std::ofstream stream(path, std::ios::binary);
    if (!stream || !stream.write(text.data(), static_cast<std::streamsize>(text.size()))) {
        throw std::runtime_error("cannot write " + path.string());
    }
}

yue2::SymbolicMode parse_symbolic(const std::string & value) {
    if (value == "off") return yue2::SymbolicMode::off;
    if (value == "melody") return yue2::SymbolicMode::melody;
    if (value == "full") return yue2::SymbolicMode::full;
    throw std::runtime_error("--symbolic must be off, melody, or full");
}

std::pair<std::uint32_t, std::uint32_t> parse_meter(const std::string & value) {
    const auto separator = value.find('/');
    if (separator == std::string::npos || value.find('/', separator + 1) != std::string::npos) {
        throw std::runtime_error("--meter expects N/D, for example 4/4");
    }
    std::size_t used_numerator = 0;
    std::size_t used_denominator = 0;
    const auto numerator = std::stoull(value.substr(0, separator), &used_numerator);
    const auto denominator = std::stoull(value.substr(separator + 1), &used_denominator);
    if (used_numerator != separator || used_denominator != value.size() - separator - 1) {
        throw std::runtime_error("--meter expects integer N/D values");
    }
    if (numerator > std::numeric_limits<std::uint32_t>::max() ||
        denominator > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("--meter is outside the uint32 range");
    }
    return {static_cast<std::uint32_t>(numerator), static_cast<std::uint32_t>(denominator)};
}

std::uint32_t parse_u32(const std::string & value, const char * option) {
    std::size_t used = 0;
    const auto parsed = std::stoull(value, &used);
    if (used != value.size() || parsed > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error(std::string(option) + " is outside the uint32 range");
    }
    return static_cast<std::uint32_t>(parsed);
}

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc == 1 || has(argc, argv, "--help") || has(argc, argv, "-h")) {
            usage(argv[0]);
            return 0;
        }
        yue2::GenerationPipelineOptions options;
        auto models_directory = value_after(argc, argv, "--models-dir", false);
        if (models_directory.empty()) models_directory = environment("YUE2_MODELS_DIR");
        if (models_directory.empty()) models_directory = "models";
        auto encoding = value_after(argc, argv, "--encoding", false);
        if (encoding.empty()) encoding = environment("YUE2_ENCODING");
        if (encoding.empty()) encoding = "auto";
        const auto model_paths = resolve_models(
            models_directory, encoding,
            value_after(argc, argv, "--model", false),
            value_after(argc, argv, "--vae", false),
            value_after(argc, argv, "--tokenizer", false));
        const auto output = std::filesystem::path(
            first_value_after(argc, argv, {"--out", "--output"}));
        options.autoregressive.device = value_after(argc, argv, "--device", false);
        if (options.autoregressive.device.empty()) {
            options.autoregressive.device = environment("YUE2_DEVICE");
        }
        const auto threads = value_after(argc, argv, "--threads", false);
        if (!threads.empty()) options.autoregressive.threads = std::stoi(threads);
        const auto abc_minimum = value_after(argc, argv, "--abc-min-tokens", false);
        if (!abc_minimum.empty()) options.generation.abc.min_tokens = std::stoul(abc_minimum);
        const auto abc_limit = value_after(argc, argv, "--abc-max-tokens", false);
        if (!abc_limit.empty()) options.generation.abc.max_tokens = std::stoul(abc_limit);
        const auto semantic_minimum = value_after(argc, argv, "--semantic-min-tokens", false);
        if (!semantic_minimum.empty()) {
            options.generation.semantic.min_tokens = std::stoul(semantic_minimum);
        }
        const auto semantic_limit = value_after(argc, argv, "--semantic-max-tokens", false);
        const auto duration = first_value_after(
            argc, argv, {"--duration", "--seconds"}, false);
        if (!duration.empty()) {
            std::size_t used = 0;
            const double seconds = std::stod(duration, &used);
            if (used != duration.size() || !(seconds > 0.0) || seconds > 900.0) {
                throw std::runtime_error("--duration must be in (0, 900] seconds");
            }
            options.generation.semantic.max_tokens =
                static_cast<std::uint32_t>(std::ceil(seconds * 25.0));
            options.generation.semantic.min_tokens = std::min(
                options.generation.semantic.min_tokens,
                options.generation.semantic.max_tokens);
        }
        if (!semantic_limit.empty()) options.generation.semantic.max_tokens = std::stoul(semantic_limit);
        const auto steps = value_after(argc, argv, "--ode-steps", false);
        if (!steps.empty()) options.flow.ode_steps = std::stoul(steps);
        const auto temperature = value_after(argc, argv, "--temperature", false);
        if (!temperature.empty()) options.generation.semantic.temperature = std::stof(temperature);
        const auto top_k = value_after(argc, argv, "--top-k", false);
        if (!top_k.empty()) options.generation.semantic.top_k = std::stoul(top_k);
        const auto top_p = value_after(argc, argv, "--top-p", false);
        if (!top_p.empty()) options.generation.semantic.top_p = std::stof(top_p);
        for (const auto & adapter : values_after(argc, argv, "--lora")) {
            options.autoregressive.lora_adapters.push_back(parse_lora(adapter));
        }

        yue2::SongRequest request;
        request.style = first_value_after(argc, argv, {"--prompt", "--style"});
        const bool has_lyrics = has(argc, argv, "--lyrics");
        const bool has_lyrics_file = has(argc, argv, "--lyrics-file");
        if (has_lyrics && has_lyrics_file) {
            throw std::runtime_error("--lyrics and --lyrics-file are mutually exclusive");
        }
        request.instrumental = has(argc, argv, "--instrumental");
        if (request.instrumental && (has_lyrics || has_lyrics_file)) {
            throw std::runtime_error("--instrumental is mutually exclusive with lyrics");
        }
        if (has_lyrics) request.lyrics = value_after(argc, argv, "--lyrics");
        if (has_lyrics_file) request.lyrics = read_text(value_after(argc, argv, "--lyrics-file"));
        request.experimental_vocal_rest = has(argc, argv, "--experimental-vocal-rest");
        if (request.instrumental && request.experimental_vocal_rest) {
            throw std::runtime_error(
                "--instrumental already includes the vocal-rest intervention");
        }
        const auto symbolic = value_after(argc, argv, "--symbolic", false);
        if (!symbolic.empty()) request.symbolic_mode = parse_symbolic(symbolic);
        const auto abc_path = value_after(argc, argv, "--abc", false);
        if (!abc_path.empty()) request.abc = read_text(abc_path);
        const auto abc_prefix_path = value_after(argc, argv, "--abc-prefix", false);
        if (!abc_prefix_path.empty()) request.abc_prefix = read_text(abc_prefix_path);
        const auto bpm = value_after(argc, argv, "--bpm", false);
        const auto key = value_after(argc, argv, "--key", false);
        const auto meter = value_after(argc, argv, "--meter", false);
        if (!bpm.empty() || !key.empty() || !meter.empty()) {
            if (bpm.empty() || key.empty()) {
                throw std::runtime_error("--bpm and --key must be supplied together");
            }
            if (request.abc || request.abc_prefix) {
                throw std::runtime_error("--bpm/--key are mutually exclusive with --abc and --abc-prefix");
            }
            yue2::PlanningHeader planning;
            planning.bpm = parse_u32(bpm, "--bpm");
            planning.key = key;
            if (!meter.empty()) {
                const auto parsed = parse_meter(meter);
                planning.meter_numerator = parsed.first;
                planning.meter_denominator = parsed.second;
            }
            request.abc_prefix = yue2::make_planning_abc_prefix(planning);
        }
        const auto seed = value_after(argc, argv, "--seed", false);
        if (!seed.empty()) request.seed = std::stoull(seed);
        const auto guidance = value_after(argc, argv, "--guidance", false);
        if (!guidance.empty()) request.guidance_scale = std::stof(guidance);

        std::cerr << "[yue2] model: " << model_paths.encoding << " from "
                  << models_directory << '\n';
        yue2::GenerationPipeline pipeline(
            model_paths.model, model_paths.vae, model_paths.tokenizer, options);
        const auto result = pipeline.generate(request);
        yue2::audio::write_wav_float(
            output, result.audio.interleaved_samples,
            result.audio.sample_rate, result.audio.channels);
        const auto score_output = value_after(argc, argv, "--score-output", false);
        if (!score_output.empty()) write_text(score_output, result.abc);
        std::cout << "wrote " << output << " (" << result.semantic_codec_ids.size()
                  << " semantic frames, "
                  << result.audio.interleaved_samples.size() / result.audio.channels
                  << " audio frames)\n";
        if (result.abc_truncated) std::cerr << "warning: ABC generation reached its token limit\n";
        if (result.semantic_truncated) {
            std::cerr << "warning: semantic generation reached its token limit\n";
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "yue2-generate: " << error.what() << '\n';
        return 2;
    }
}
