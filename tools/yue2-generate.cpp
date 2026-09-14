#include "yue2/audio.h"
#include "yue2/generation_pipeline.h"

#include <filesystem>
#include <fstream>
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
        << "Usage: " << argv0 << " --model yue2.gguf --vae vae.gguf --tokenizer qwen.tiktoken\\\n\n"
        << "  --style TEXT [--lyrics TEXT | --lyrics-file PATH] --output song.wav [options]\n\n"
        << "Options:\n"
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
        const auto model_path = value_after(argc, argv, "--model");
        const auto vae_path = value_after(argc, argv, "--vae");
        const auto tokenizer_path = value_after(argc, argv, "--tokenizer");
        const auto output = std::filesystem::path(value_after(argc, argv, "--output"));
        options.autoregressive.device = value_after(argc, argv, "--device", false);
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
        request.style = value_after(argc, argv, "--style");
        const bool has_lyrics = has(argc, argv, "--lyrics");
        const bool has_lyrics_file = has(argc, argv, "--lyrics-file");
        if (has_lyrics && has_lyrics_file) {
            throw std::runtime_error("--lyrics and --lyrics-file are mutually exclusive");
        }
        if (has_lyrics) request.lyrics = value_after(argc, argv, "--lyrics");
        if (has_lyrics_file) request.lyrics = read_text(value_after(argc, argv, "--lyrics-file"));
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

        yue2::GenerationPipeline pipeline(model_path, vae_path, tokenizer_path, options);
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
