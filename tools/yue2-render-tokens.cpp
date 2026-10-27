// Render a fixed semantic stream without AR sampling. Used to evaluate real-audio
// supervision and to compare acoustic adapters with identical conditioning/noise.
#include "yue2/audio.h"
#include "yue2/autoregressive.h"
#include "yue2/tokenizer.h"
#include "yue2/vae.h"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>

int main(int argc, char **argv) {
    try {
        std::map<std::string, std::string> args;
        yue2::AutoregressiveOptions runtime;
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            if (key == "--help") {
                std::cout
                    << "Usage: yue2-render-tokens --model MODEL.gguf --vae VAE.gguf\n"
                       "  --tokenizer qwen.tiktoken --tokens codes.i32 --out render.wav\n"
                       "  [--prompt TEXT] [--abc score.abc] [--lyrics-file lyrics.txt]\n"
                       "  [--prefix prefix.i32]  Exact prefix instead of text/ABC/lyrics\n"
                       "  [--lora PATH (repeatable)] [--seed N] [--ode-steps N]\n"
                       "  [--device cuda|cpu] [--threads N] [--latents-out latents.f32]\n"
                       "  [--source-latents means.f32 --strength 0.35]  Source-latent remix "
                       "control\n"
                       "VAE-only control: --vae VAE.gguf --latents-in means.f32 --out vae.wav\n"
                       "Raw little-endian int32 codec IDs at 25 Hz; no planning or token "
                       "sampling.\n"
                       "Flow uses the native unguided midpoint solver (no CFG/shift controls).\n";
                return 0;
            }
            if (i + 1 == argc)
                throw std::runtime_error("missing value for " + key);
            const std::string value = argv[++i];
            if (key == "--lora")
                runtime.lora_adapters.push_back({value, 1.0F});
            else if (key == "--model" || key == "--vae" || key == "--tokenizer" ||
                     key == "--tokens" || key == "--out" || key == "--prompt" || key == "--abc" ||
                     key == "--lyrics-file" || key == "--prefix" || key == "--seed" || key == "--ode-steps" ||
                     key == "--device" || key == "--threads" || key == "--latents-out" ||
                     key == "--latents-in" || key == "--source-latents" || key == "--strength") {
                if (!args.emplace(key, value).second)
                    throw std::runtime_error("duplicate " + key);
            } else
                throw std::runtime_error("unknown option " + key);
        }
        auto required = [&](const char *key) {
            if (!args.count(key) || args.at(key).empty())
                throw std::runtime_error(std::string("missing ") + key);
            return args.at(key);
        };
        auto read_text = [](const std::string &path) {
            std::ifstream file(path, std::ios::binary);
            if (!file)
                throw std::runtime_error("cannot read " + path);
            return std::string(std::istreambuf_iterator<char>(file), {});
        };
        const auto vae_path = required("--vae");
        const auto output_path = required("--out");
        runtime.device = args["--device"];
        if (args.count("--threads"))
            runtime.threads = std::stoi(args.at("--threads"));
        std::vector<float> latents;
        std::size_t rendered_frames = 0;
        if (args.count("--latents-in")) {
            if (args.count("--tokens"))
                throw std::runtime_error("latents-in and tokens are mutually exclusive");
            const auto data = read_text(args.at("--latents-in"));
            if (data.empty() || data.size() % (64 * sizeof(float)))
                throw std::runtime_error("latents must have shape [frames,64]");
            latents.resize(data.size() / sizeof(float));
            std::memcpy(latents.data(), data.data(), data.size());
            for (const auto value : latents)
                if (!std::isfinite(value))
                    throw std::runtime_error("nonfinite latent");
            rendered_frames = latents.size() / 64;
        } else {
            const auto model_path = required("--model");
            const auto tokenizer_path = required("--tokenizer");
            const auto bytes = read_text(required("--tokens"));
            if (bytes.empty() || bytes.size() % 4)
                throw std::runtime_error("tokens must be a nonempty int32 stream");
            std::vector<std::int32_t> codes;
            for (std::size_t i = 0; i < bytes.size(); i += 4) {
                std::uint32_t value = 0;
                for (int b = 0; b < 4; ++b)
                    value |= std::uint32_t(static_cast<unsigned char>(bytes[i + b])) << (8 * b);
                if (value >= yue2::kCodecSize)
                    throw std::runtime_error("codec ID outside [0,32767]");
                codes.push_back(static_cast<std::int32_t>(value));
            }
            yue2::SongRequest request;
            request.style = args.count("--prompt") ? args.at("--prompt") : "";
            request.symbolic_mode = yue2::SymbolicMode::off;
            if (args.count("--lyrics-file"))
                request.lyrics = read_text(args.at("--lyrics-file"));
            std::vector<std::int32_t> prefix;
            if (args.count("--prefix")) {
                if (args.count("--prompt") || args.count("--abc") || args.count("--lyrics-file"))
                    throw std::runtime_error("prefix is mutually exclusive with prompt/ABC/lyrics");
                const auto raw = read_text(args.at("--prefix"));
                if (raw.empty() || raw.size() % sizeof(std::int32_t))
                    throw std::runtime_error("prefix must be a nonempty int32 stream");
                prefix.resize(raw.size() / sizeof(std::int32_t));
                std::memcpy(prefix.data(), raw.data(), raw.size());
            } else {
                yue2::TextTokenizer tokenizer(tokenizer_path);
                std::vector<std::int32_t> abc_ids;
                if (args.count("--abc")) {
                    request.symbolic_mode = yue2::SymbolicMode::full;
                    request.abc = read_text(args.at("--abc"));
                    abc_ids = tokenizer.encode(*request.abc);
                }
                prefix = yue2::make_positive_prefix(request, tokenizer, abc_ids);
            }
            yue2::FlowOptions flow;
            if (args.count("--ode-steps")) {
                const auto steps = std::stoi(args.at("--ode-steps"));
                if (steps < 1 || steps > 1024)
                    throw std::runtime_error("ode-steps must be in [1,1024]");
                flow.ode_steps = static_cast<std::uint32_t>(steps);
            }
            const auto seed = args.count("--seed") ? std::stoull(args.at("--seed")) : 831001ULL;
            { // Release the language model before allocating the VAE on small GPUs.
                yue2::AutoregressiveModel model(model_path, runtime);
                if (args.count("--source-latents")) {
                    const auto bytes = read_text(args.at("--source-latents"));
                    if (bytes.size() != codes.size() * 64 * sizeof(float))
                        throw std::runtime_error(
                            "source latents and tokens must have identical frame counts");
                    std::vector<float> clean(bytes.size() / sizeof(float));
                    std::memcpy(clean.data(), bytes.data(), bytes.size());
                    const float strength =
                        args.count("--strength") ? std::stof(args.at("--strength")) : .35F;
                    if (!std::isfinite(strength) || strength < 0 || strength > 1)
                        throw std::runtime_error("strength must be in [0,1]");
                    for (auto value : clean)
                        if (!std::isfinite(value))
                            throw std::runtime_error("nonfinite source latent");
                    if (strength == 0)
                        latents = clean;
                    else {
                        std::mt19937 rng(static_cast<uint32_t>(seed));
                        std::normal_distribution<float> normal;
                        for (auto &value : clean)
                            value = (1 - strength) * value + strength * normal(rng);
                        flow.start_noise_level = strength;
                        latents = model.synthesize_latents(prefix, codes, clean, flow);
                    }
                } else {
                    if (args.count("--strength"))
                        throw std::runtime_error("strength requires source-latents");
                    latents = model.synthesize_latents(prefix, codes, seed, flow);
                }
            }
            rendered_frames = codes.size();
        }
        if (args.count("--latents-out")) {
            std::ofstream file(args.at("--latents-out"), std::ios::binary);
            if (!file.write(reinterpret_cast<const char *>(latents.data()),
                            latents.size() * sizeof(float)))
                throw std::runtime_error("could not write latents");
        }
        yue2::VaeRuntimeOptions vae_runtime;
        vae_runtime.device = runtime.device;
        vae_runtime.threads = runtime.threads;
        yue2::VaeDecoder vae(vae_path, vae_runtime);
        const auto audio = vae.decode(latents);
        const auto parent = std::filesystem::path(output_path).parent_path();
        if (!parent.empty())
            std::filesystem::create_directories(parent);
        yue2::audio::write_wav_float(output_path, audio.interleaved_samples, audio.sample_rate,
                                     audio.channels);
        std::cout << "rendered " << rendered_frames << " frames (" << rendered_frames / 25.0
                  << " seconds)\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "yue2-render-tokens: " << error.what() << '\n';
        return 2;
    }
}
