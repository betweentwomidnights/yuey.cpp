#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace yue2 {

constexpr std::int32_t kEodToken = 151643;
constexpr std::int32_t kAbcStartToken = 151847;
constexpr std::int32_t kAbcEndToken = 151848;
constexpr std::int32_t kMusicStartToken = 151851;
constexpr std::int32_t kMusicEndToken = 151852;
constexpr std::int32_t kCodecOffset = 151853;
constexpr std::int32_t kCodecSize = 32768;
constexpr std::int32_t kLatentStartToken = 184621;
constexpr std::int32_t kLatentEndToken = 184622;
constexpr std::int32_t kLatentPadToken = 184623;
constexpr std::int32_t kGenerationVocabSize = 184704;

class TextTokenizer;

enum class SymbolicMode {
    off,
    melody,
    full,
};

// Typed UI/host controls for the trusted planning-prefix builder. Supplying
// this header locks musical structure before YuE2 samples any score notes.
// A client should omit the whole object to retain automatic planning.
struct PlanningHeader {
    std::uint32_t bpm = 0;
    std::uint32_t meter_numerator = 4;
    std::uint32_t meter_denominator = 4;
    std::string key;
};

// Accepts UI-friendly spellings such as C# minor, C#:minor, C#m, and Db major,
// returning canonical ABC (C#m, Db). Throws std::invalid_argument on invalid
// or injection-prone input.
std::string normalize_abc_key(const std::string & key);

// Emits the YuE2 two-voice header validated by the locked-header listening
// test, including the initial section marker expected before score notes.
std::string make_planning_abc_prefix(const PlanningHeader & header);

// Converts YuE2's bounded two-voice ABC dialect into an instrumental
// experiment: Vocal notes become rests (retaining chord symbols and timing),
// while the Vocal lead is routed to Ins when that section contains notes.
// Throws when the score has no native Vocal/Ins pair.
std::string make_instrumental_abc(const std::string & abc);

struct SongRequest {
    std::string style;
    std::string lyrics;
    SymbolicMode symbolic_mode = SymbolicMode::full;
    std::optional<std::string> abc;
    std::uint64_t seed = 831001;
    std::optional<float> guidance_scale;
    // Optional exact text placed after ABC_START before symbolic sampling.
    // This is the low-level injection point for a validated tempo/key/meter
    // header. It is mutually exclusive with a complete external ABC score.
    std::optional<std::string> abc_prefix;
    // Experimental symbolic intervention. Requires melody/full mode and empty
    // lyrics; the planned/provided ABC is rewritten before semantic inference.
    bool instrumental = false;
};

struct GenerationSampling {
    float temperature = 1.0F;
    float top_p = 0.95F;
    std::uint32_t top_k = 100;
    float repetition_penalty = 1.2F;
    std::uint32_t penalty_window = 50;
    std::uint32_t min_tokens = 200;
    std::uint32_t max_tokens = 9000;
};

struct GenerationDefaults {
    GenerationSampling abc{0.7F, 0.9F, 30, 1.005F, 100, 32, 4096};
    GenerationSampling semantic{};
};

// These functions mirror upstream protocol.py exactly. Prefix construction
// keeps ABC IDs explicit so classifier-free guidance can reuse the exact
// positive-branch symbolic sequence rather than re-tokenizing text.
const char * generation_instruction(SymbolicMode mode) noexcept;
float generation_guidance(const SongRequest & request);
std::string generation_request_text(const SongRequest & request);
std::vector<std::int32_t> make_positive_prefix(
    const SongRequest & request,
    const TextTokenizer & tokenizer,
    const std::optional<std::vector<std::int32_t>> & abc_ids = std::nullopt);
std::vector<std::int32_t> make_negative_prefix(
    const SongRequest & request,
    const TextTokenizer & tokenizer,
    const std::optional<std::vector<std::int32_t>> & abc_ids = std::nullopt);

struct GenerationModelConfig {
    std::uint32_t vocab_size = kGenerationVocabSize;
    std::uint32_t context_length = 24576;
    std::uint32_t embedding_length = 2048;
    std::uint32_t block_count = 28;
    std::uint32_t attention_head_count = 16;
    std::uint32_t attention_head_count_kv = 8;
    std::uint32_t latent_dim = 64;
    float timestep_shift = 1.0F;
};

struct GenerationVaeConfig {
    std::uint32_t sample_rate = 48000;
    std::uint32_t audio_channels = 2;
    std::uint32_t downsampling_ratio = 1920;
    std::uint32_t latent_dim = 64;
    std::uint32_t decode_core_frames = 1024;
    std::uint32_t decode_halo_frames = 16;
};

struct GenerationPackageInfo {
    GenerationModelConfig model;
    GenerationVaeConfig vae;
    std::size_t model_tensor_count = 0;
    std::size_t vae_tensor_count = 0;
    std::uint64_t model_file_bytes = 0;
    std::uint64_t vae_file_bytes = 0;
    std::string model_storage_type;
    std::string vae_storage_type;
    std::string model_checkpoint_sha256;
    std::string vae_checkpoint_sha256;
};

// Reads only GGUF metadata and tensor descriptors.  Model payloads are not
// allocated or copied, so this remains cheap even for the 7 GB main model.
GenerationPackageInfo inspect_generation_package(
    const std::string & model_gguf_path,
    const std::string & vae_gguf_path);

} // namespace yue2
