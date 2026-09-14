#pragma once

#include "yue2/autoregressive.h"
#include "yue2/generation.h"
#include "yue2/vae.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace yue2 {

struct GenerationPipelineOptions {
    AutoregressiveOptions autoregressive;
    VaeRuntimeOptions vae;
    GenerationDefaults generation;
    FlowOptions flow;
    bool semantic_budget_explicit = false;
};

// Per-call controls that do not affect model residency or backend allocation.
// A server can vary these without reconstructing GenerationPipeline.
struct GenerationRunOptions {
    GenerationDefaults generation;
    FlowOptions flow;
    // Explicit token/seconds ceilings are advanced escape hatches. Normal
    // score-based generation derives this budget after planning.
    bool semantic_budget_explicit = false;
};

enum class GenerationStage {
    abc,
    semantic,
    flow,
    decode,
    complete,
};

struct GenerationControl {
    std::function<void(
        GenerationStage stage,
        std::uint32_t current,
        std::uint32_t total)> on_progress;
    std::function<bool()> should_cancel;
};

struct GeneratedSong {
    std::string abc;
    std::vector<std::int32_t> abc_token_ids;
    bool abc_truncated = false;
    std::vector<std::int32_t> semantic_codec_ids;
    bool semantic_truncated = false;
    std::uint32_t score_bars = 0;
    double score_duration_seconds = 0.0;
    std::uint32_t semantic_budget = 0;
    std::vector<float> latents;
    DecodedAudio audio;
};

// Complete request -> optional ABC -> semantic codes -> flow latents -> PCM
// path. Models remain loaded across requests for an embedded server or plugin.
class GenerationPipeline {
public:
    GenerationPipeline(
        const std::string & model_gguf_path,
        const std::string & vae_gguf_path,
        const std::string & qwen_tiktoken_path,
        const GenerationPipelineOptions & options = {});
    ~GenerationPipeline();
    GenerationPipeline(GenerationPipeline &&) noexcept;
    GenerationPipeline & operator=(GenerationPipeline &&) noexcept;
    GenerationPipeline(const GenerationPipeline &) = delete;
    GenerationPipeline & operator=(const GenerationPipeline &) = delete;

    GeneratedSong generate(const SongRequest & request);
    GeneratedSong generate(
        const SongRequest & request,
        const GenerationRunOptions & options);
    GeneratedSong generate(
        const SongRequest & request,
        const GenerationRunOptions & options,
        const GenerationControl & control);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace yue2
