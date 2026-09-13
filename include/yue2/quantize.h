#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace yue2 {

// Weight encodings written by yue2-quantize. The names are the Encoding field
// of the GGUF file-naming convention (`<BaseName>-<SizeLabel>-<Version>-
// <Encoding>.gguf`) shared with sa3.cpp and audiocraft.cpp.
//
//   Q4_K_M  V/down projections and both vocabulary matrices Q6_K, rest Q4_K
//   Q5_K_M  the same promotion over a Q5_K body
//   Q8_0    every eligible matrix Q8_0
//   F16/F32 re-encode; dequantizes an already quantized input
enum class QuantizationMix { q4_k_m, q5_k_m, q8_0, f16, f32 };

// Accepts the canonical lower-case names, their upper-case Encoding spelling,
// and the compact q4_km/q5_km aliases. Throws std::invalid_argument.
QuantizationMix parse_quantization_mix(const std::string & name);

// The Encoding spelling, e.g. "Q4_K_M".
const char * quantization_mix_name(QuantizationMix mix) noexcept;

struct QuantizedTensor {
    std::string name;
    std::string source_type;
    std::string output_type;
};

struct QuantizeOptions {
    QuantizationMix mix = QuantizationMix::q4_k_m;
    // Worker threads for dequantization and encoding; 0 uses every core.
    unsigned threads = 0;
    bool overwrite = false;
    // Called after each tensor has been written.
    std::function<void(const QuantizedTensor & tensor, std::size_t index, std::size_t count)>
        on_tensor;
};

struct QuantizeReport {
    std::vector<QuantizedTensor> tensors;
    std::size_t converted_count = 0;
    std::size_t kept_count = 0;
    std::uint64_t input_bytes = 0;
    std::uint64_t output_bytes = 0;
};

// Re-encodes a yue2.cpp YuE2 generation GGUF. Only 2-D projection and
// embedding matrices change type; norms, biases, the flow branch's latent
// bridges, timestep MLP and latent position table keep their source storage
// in the quantized mixes. Every metadata key is preserved, including the
// source checkpoint fingerprint LoRA adapters bind to, and the output records
// general.file_type plus yue2.quantization.{encoding,version}.
//
// Tensors are streamed one at a time and the result is written beside the
// destination before an atomic rename, so peak host memory is bounded by the
// largest tensor rather than the model. The VAE and transcription GGUFs are
// rejected: both are distributed unquantized.
QuantizeReport quantize_generation_gguf(
    const std::string & input_path,
    const std::string & output_path,
    const QuantizeOptions & options = {});

struct QuantCheckResult {
    std::string name;
    std::string reference_type;
    std::string candidate_type;
    double cosine = 0.0;
};

struct QuantCheckReport {
    std::vector<QuantCheckResult> compared;
    std::size_t below_threshold = 0;
    double minimum_cosine = 1.0;
};

// Dequantizes every tensor whose storage differs between two GGUFs with the
// same tensor set and shapes, and reports its cosine similarity against the
// reference. Streams like the quantizer and never allocates backend memory.
QuantCheckReport check_quantized_gguf(
    const std::string & reference_path,
    const std::string & candidate_path,
    double threshold = 0.99,
    unsigned threads = 0);

} // namespace yue2
