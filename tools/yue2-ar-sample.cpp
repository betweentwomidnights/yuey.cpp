// Exact-prefix semantic sampling for controlled training-capacity auditions.
// No planning occurs here; all adapters apply to semantic generation and flow.
#include "yue2/audio.h"
#include "yue2/autoregressive.h"
#include "yue2/vae.h"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>

int main(int argc, char **argv) {
    try {
        std::map<std::string, std::string> args;
        yue2::AutoregressiveOptions options;
        const std::set<std::string> accepted{
            "--model",  "--vae",     "--prefix", "--frames",    "--tokens-out", "--out",
            "--device", "--threads", "--seed",   "--ode-steps", "--temperature"};
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            if (key == "--help") {
                std::cout << "Controlled fixed-duration semantic sampling, with no planning.\n"
                             "--model MODEL --vae VAE --prefix prefix.i32 --frames N\n"
                             "--tokens-out NEW.i32 --out NEW.wav [--lora PATH (repeatable)]\n"
                             "[--device cuda|cpu --threads N --seed 831001 --ode-steps 32]\n"
                             "[--temperature 1.0]  Positive finite semantic sampling temperature\n"
                             "No source audio or reference semantic stream is supplied.\n";
                return 0;
            }
            if (key != "--lora" && !accepted.count(key))
                throw std::runtime_error("unknown " + key);
            if (++i == argc)
                throw std::runtime_error("missing value for " + key);
            if (key == "--lora")
                options.lora_adapters.push_back({argv[i], 1});
            else if (!args.emplace(key, argv[i]).second)
                throw std::runtime_error("duplicate " + key);
        }
        auto required = [&](const char *key) -> const std::string & {
            if (!args.count(key) || args.at(key).empty())
                throw std::runtime_error(std::string("missing ") + key);
            return args.at(key);
        };
        const auto model_path = required("--model"), vae_path = required("--vae");
        const std::filesystem::path audio_path(required("--out")),
            code_path(required("--tokens-out"));
        if (std::filesystem::exists(audio_path) || std::filesystem::exists(code_path))
            throw std::runtime_error("output already exists");
        const auto frames = std::stoi(required("--frames"));
        if (frames < 1 || frames > 2048)
            throw std::runtime_error("frames must be in [1,2048]");
        const auto bytes = std::filesystem::file_size(required("--prefix"));
        if (!bytes || bytes % 4 || bytes / 4 + frames > 24576)
            throw std::runtime_error("invalid prefix length");
        std::vector<std::int32_t> prefix(bytes / 4);
        std::ifstream in(args.at("--prefix"), std::ios::binary);
        if (!in.read(reinterpret_cast<char *>(prefix.data()), bytes) ||
            prefix.back() != yue2::kMusicStartToken)
            throw std::runtime_error("invalid MUSIC_START prefix");
        if (args.count("--device"))
            options.device = args.at("--device");
        if (args.count("--threads"))
            options.threads = std::stoi(args.at("--threads"));
        const auto seed = args.count("--seed") ? std::stoull(args.at("--seed")) : 831001ULL;
        yue2::FlowOptions flow;
        if (args.count("--ode-steps"))
            flow.ode_steps = std::stoul(args.at("--ode-steps"));
        yue2::GenerationSampling sampling;
        sampling.min_tokens = sampling.max_tokens = frames;
        if (args.count("--temperature"))
            sampling.temperature = std::stof(args.at("--temperature"));
        if (!std::isfinite(sampling.temperature) || sampling.temperature <= 0)
            throw std::runtime_error("temperature must be finite and positive");
        std::vector<std::int32_t> codes;
        std::vector<float> latents;
        {
            yue2::AutoregressiveModel model(model_path, options);
            const auto result =
                model.generate(prefix, sampling, yue2::AutoregressivePhase::semantic, seed);
            for (auto token : result.tokens) {
                if (token < yue2::kCodecOffset || token >= yue2::kCodecOffset + yue2::kCodecSize)
                    throw std::runtime_error("non-codec sample");
                codes.push_back(token - yue2::kCodecOffset);
            }
            if (codes.size() != size_t(frames))
                throw std::runtime_error("unexpected sample count");
            latents = model.synthesize_latents(prefix, codes, seed, flow);
        }
        if (!code_path.parent_path().empty())
            std::filesystem::create_directories(code_path.parent_path());
        std::ofstream codes_file(code_path, std::ios::binary);
        if (!codes_file.write(reinterpret_cast<const char *>(codes.data()), codes.size() * 4))
            throw std::runtime_error("cannot write codes");
        yue2::VaeRuntimeOptions vae_options;
        vae_options.device = options.device;
        vae_options.threads = options.threads;
        yue2::VaeDecoder vae(vae_path, vae_options);
        const auto audio = vae.decode(latents);
        if (!audio_path.parent_path().empty())
            std::filesystem::create_directories(audio_path.parent_path());
        yue2::audio::write_wav_float(audio_path.string(), audio.interleaved_samples,
                                     audio.sample_rate, audio.channels);
        std::cout << "sampled and rendered " << codes.size() << " fresh frames\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "yue2-ar-sample: " << e.what() << '\n';
        return 2;
    }
}
