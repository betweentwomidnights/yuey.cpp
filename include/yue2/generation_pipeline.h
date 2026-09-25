#pragma once

#include "yue2/autoregressive.h"
#include "yue2/generation.h"
#include "yue2/transcription.h"
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

// Per-call controls. A server can vary these without reconstructing
// GenerationPipeline; keep_models is the one that affects residency, and a
// call after a frugal one simply reloads what it freed.
struct GenerationRunOptions {
    GenerationDefaults generation;
    FlowOptions flow;
    // Explicit token/seconds ceilings are advanced escape hatches. Normal
    // score-based generation prevents MUSIC_END before the final score bar and
    // derives a conservative post-score safety budget after planning.
    bool semantic_budget_explicit = false;
    // Residency, as in sa3.cpp. true keeps the generator loaded through the
    // VAE decode and into the next call. false frees it once flow synthesis
    // is done, so the decode does not share the card with a model nothing
    // will use again this call; the next call reloads it.
    bool keep_models = true;
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

struct GeneratedPlan {
    std::string abc;
    std::vector<std::int32_t> abc_token_ids;
    bool abc_truncated = false;
    // The planned score held a bar that would not render and was shortened to
    // the last block that does. Distinct from abc_truncated, which means
    // planning hit its token limit.
    bool abc_repaired = false;
    std::uint32_t score_bars = 0;
    double score_duration_seconds = 0.0;
    TranscriptionMidiExports midi_exports;
};

struct GeneratedSong {
    std::string abc;
    std::vector<std::int32_t> abc_token_ids;
    bool abc_truncated = false;
    bool abc_repaired = false;
    std::vector<std::int32_t> semantic_codec_ids;
    std::uint32_t semantic_prefix_frames = 0;
    bool semantic_truncated = false;
    std::uint32_t score_bars = 0;
    double score_duration_seconds = 0.0;
    std::uint32_t semantic_budget = 0;
    TranscriptionMidiExports midi_exports;
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

    // Symbolic planning only. The autoregressive model remains resident for a
    // later render, while semantic, flow, and VAE work is skipped entirely.
    GeneratedPlan plan(const SongRequest & request);
    GeneratedPlan plan(
        const SongRequest & request,
        const GenerationRunOptions & options,
        const GenerationControl & control = {});

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
