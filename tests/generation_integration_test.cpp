#include "yue2/generation_pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char ** argv) {
    if (argc < 4 || argc > 6) {
        std::cerr << "usage: yue2-generation-integration-test MODEL.gguf VAE.gguf "
                     "QWEN.TIKTOKEN [DEVICE] [LORA.gguf]\n";
        return 2;
    }
    yue2::GenerationPipelineOptions options;
    if (argc >= 5) options.autoregressive.device = argv[4];
    if (argc == 6) options.autoregressive.lora_adapters.push_back({argv[5], 1.0F});
    options.generation.semantic.temperature = 0.0F;
    options.generation.semantic.min_tokens = 2;
    options.generation.semantic.max_tokens = 2;
    options.semantic_budget_explicit = true;
    options.flow.ode_steps = 1;
    yue2::GenerationPipeline pipeline(argv[1], argv[2], argv[3], options);

    yue2::SongRequest request;
    request.style = "acoustic, intimate";
    request.lyrics = "[Verse]\nA quiet line";
    request.symbolic_mode = yue2::SymbolicMode::full;
    request.abc = "X:1\nM:4/4\nK:C\nC2 E2 G4|";
    request.seed = 831001;
    const auto first = pipeline.generate(request);
    if (first.abc != *request.abc || first.abc_token_ids.empty() ||
        first.semantic_codec_ids.size() != 2 || !first.semantic_truncated ||
        first.latents.size() != 128 || first.audio.sample_rate != 48000 ||
        first.audio.channels != 2 || first.audio.interleaved_samples.empty() ||
        !std::all_of(
            first.audio.interleaved_samples.begin(), first.audio.interleaved_samples.end(),
            [](float value) { return std::isfinite(value); })) {
        throw std::runtime_error("YuE2 complete generation pipeline result is invalid");
    }
    const auto second = pipeline.generate(request);
    if (first.semantic_codec_ids != second.semantic_codec_ids ||
        first.latents != second.latents ||
        first.audio.interleaved_samples != second.audio.interleaved_samples) {
        throw std::runtime_error("YuE2 generation pipeline is not seed-deterministic");
    }
    std::cout << "semantic=" << first.semantic_codec_ids[0] << ','
              << first.semantic_codec_ids[1]
              << " latents=" << first.latents.size()
              << " pcm=" << first.audio.interleaved_samples.size() << "\n";
    return 0;
}
