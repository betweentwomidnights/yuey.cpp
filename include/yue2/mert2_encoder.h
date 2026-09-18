#pragma once

#include "yue2/mert2_frontend.h"

#include <cstdint>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace yue2::mert2 {

struct HiddenFeatures {
    std::int64_t frames = 0;
    std::int64_t channels = 0;
    // Row-major [frames, channels].
    std::vector<float> values;
};

struct EncoderOptions {
    // Empty selects the best registered backend. "cpu" forces CPU.
    std::string device;
    // Zero uses GGML's default, or YUE2_THREADS when set.
    int threads = 0;
};

// Native GGML implementation of the released MERT2 audio encoder. The first
// milestone exposes the complete ConvNeXt subsampler; Conformer execution is
// added behind the same object so callers and model loading stay stable.
class Encoder {
public:
    explicit Encoder(const std::string & model_path, const EncoderOptions & options = {});
    ~Encoder();
    Encoder(Encoder &&) noexcept;
    Encoder & operator=(Encoder &&) noexcept;
    Encoder(const Encoder &) = delete;
    Encoder & operator=(const Encoder &) = delete;

    HiddenFeatures subsample(const LogMelFeatures & features);
    // Run the ConvNeXt subsampler followed by all 24 Conformer blocks.
    HiddenFeatures encode(const LogMelFeatures & features, int layer_count = 24);
    // Apply SheetSage2's learned mixture of the subsampler and all 24 encoder
    // layers, then its 1024 -> 512 encoder projection.
    HiddenFeatures sheetsage_memory(const LogMelFeatures & features);
    // Convert mono 24 kHz PCM into YuE2 semantic codec IDs at 25 Hz using
    // the dedicated real-audio tokenizer head. This requires a
    // yue2-semantic-tokenizer GGUF rather than the SheetSage2 model.
    std::vector<std::int32_t> semantic_tokens_24k(
        const std::vector<float> & mono_samples);
    // Validation hook: return layer zero after stage 1=FFN1, 2=attention,
    // 3=convolution, 4=FFN2 residual, 5=final LayerNorm, or 6=the
    // rotated query projection flattened to 1024 channels.
    HiddenFeatures layer0_stage(const LogMelFeatures & features, int stage);
    // Full-sequence SheetSage2 BART decoder logits retained as a parity oracle.
    std::vector<float> decode_logits(
        const HiddenFeatures & memory,
        const std::vector<std::int32_t> & decoder_token_ids);
    std::vector<std::int32_t> generate_tokens(
        const HiddenFeatures & memory,
        const std::vector<std::int32_t> & prefix,
        std::size_t max_tokens,
        double stop_time_seconds = -1.0);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace yue2::mert2
