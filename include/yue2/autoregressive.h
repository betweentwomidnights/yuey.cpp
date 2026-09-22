#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "yue2/generation.h"

namespace yue2 {

class AutoregressiveState;

struct LoraAdapterSpec {
    std::string path;
    float strength = 1.0F;
};

struct AutoregressiveOptions {
    std::string device;
    int threads = 0;
    bool flash_attention = true;
    // Adapters are loaded once, remain unmerged, and compose additively in
    // this order. A zero strength is a validated no-op.
    std::vector<LoraAdapterSpec> lora_adapters;
};

enum class AutoregressivePhase {
    abc,
    semantic,
};

struct AutoregressiveResult {
    std::vector<std::int32_t> tokens;
    bool reached_end = false;
};

struct FlowOptions {
    std::uint32_t ode_steps = 32;
    std::uint32_t context_length = 24576;
};

struct AutoregressiveControl {
    std::function<void(std::uint32_t current, std::uint32_t total)> on_progress;
    std::function<bool()> should_cancel;
    // Called only when the model proposes the phase end token. Return false
    // to suppress that token and sample the next-best allowed token instead.
    // The history contains generated tokens, excluding the proposed end.
    std::function<bool(const std::vector<std::int32_t> & history)> allow_stop;
    // Called after each accepted token. Return true to end generation even
    // though the model did not propose the end token. The history includes the
    // token just accepted, so a caller that needs to stop on a structural
    // boundary can look for one. reached_end is set, because a forced stop at a
    // boundary leaves a complete result rather than a truncated one.
    std::function<bool(const std::vector<std::int32_t> & history)> force_stop;
};

class AutoregressiveSession {
public:
    ~AutoregressiveSession();
    AutoregressiveSession(AutoregressiveSession &&) noexcept;
    AutoregressiveSession & operator=(AutoregressiveSession &&) noexcept;
    AutoregressiveSession(const AutoregressiveSession &) = delete;
    AutoregressiveSession & operator=(const AutoregressiveSession &) = delete;

    // Appends tokens to the persistent KV cache and returns next-token logits.
    std::vector<float> append(const std::vector<std::int32_t> & token_ids);
    std::size_t token_count() const noexcept;
    std::size_t capacity() const noexcept;

private:
    friend class AutoregressiveModel;
    class Impl;
    explicit AutoregressiveSession(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

// Native YuE2 AR transformer with both exact full-prefix diagnostics and
// request-local persistent KV-cache sessions.
class AutoregressiveModel {
public:
    explicit AutoregressiveModel(
        const std::string & model_gguf_path,
        const AutoregressiveOptions & options = {});
    ~AutoregressiveModel();
    AutoregressiveModel(AutoregressiveModel &&) noexcept;
    AutoregressiveModel & operator=(AutoregressiveModel &&) noexcept;
    AutoregressiveModel(const AutoregressiveModel &) = delete;
    AutoregressiveModel & operator=(const AutoregressiveModel &) = delete;

    // Returns the 184704 logits predicting the token after token_ids.back().
    std::vector<float> logits(const std::vector<std::int32_t> & token_ids);

    std::unique_ptr<AutoregressiveSession> create_session(std::size_t capacity);

    AutoregressiveResult generate(
        const std::vector<std::int32_t> & prefix,
        const GenerationSampling & sampling,
        AutoregressivePhase phase,
        std::uint64_t seed,
        const AutoregressiveControl & control = {});

    AutoregressiveResult generate_cfg(
        const std::vector<std::int32_t> & positive_prefix,
        const std::vector<std::int32_t> & negative_prefix,
        const GenerationSampling & sampling,
        AutoregressivePhase phase,
        float guidance_scale,
        std::uint64_t seed,
        const AutoregressiveControl & control = {});

    // Converts raw semantic codec IDs [0,32767] and caller-supplied row-major
    // [frames,64] noise into VAE latents. Supplying noise explicitly makes the
    // flow solver independently reproducible and suitable for parity tests.
    std::vector<float> synthesize_latents(
        const std::vector<std::int32_t> & generation_prefix,
        const std::vector<std::int32_t> & semantic_codec_ids,
        const std::vector<float> & noise,
        const FlowOptions & options = {},
        const AutoregressiveControl & control = {});

    std::vector<float> synthesize_latents(
        const std::vector<std::int32_t> & generation_prefix,
        const std::vector<std::int32_t> & semantic_codec_ids,
        std::uint64_t seed,
        const FlowOptions & options = {},
        const AutoregressiveControl & control = {});

private:
    std::shared_ptr<AutoregressiveState> state_;
};

} // namespace yue2
