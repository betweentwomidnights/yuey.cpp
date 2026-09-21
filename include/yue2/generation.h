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

enum class EndingMode {
    // Keep the complete score produced by planning or supplied by the host.
    natural,
    // Fit to target_bars by preserving the opening and the planner's final
    // bars, exposed as an explicit outro section.
    outro,
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

// Converts YuE2's bounded two-voice ABC dialect into a vocal-rest experiment:
// Vocal notes become rests (retaining chord symbols and timing), while every
// existing Ins block is preserved exactly. Throws when the score has no native
// Vocal/Ins pair.
std::string make_vocal_rest_abc(const std::string & abc);

// Builds the empty lyric-section sequence used by YuE2 for intentional
// instrumental rendering. Section markers are read from native ABC comments
// such as "% intro" and "% chorus" and retained in score order. A score
// without usable markers receives the conventional default song form.
std::string make_instrumental_lyrics(const std::string & abc);

struct AbcScoreInfo {
    std::uint32_t bars = 0;
    std::uint32_t bpm = 0;
    std::uint32_t meter_numerator = 4;
    std::uint32_t meter_denominator = 4;
    double duration_seconds = 0.0;
};

// Inspect YuE2's native ABC score without loading model weights. Bar count is
// per musical timeline rather than the sum of the Vocal and Ins lanes.
//
// Set allow_empty_score for a score that may legitimately carry no bars yet,
// such as the header-only planning prefix the typed controls build. It reports
// zero bars instead of failing for want of Vocal and Ins body blocks. Leave it
// false for a completed plan, where missing lanes are a real fault.
AbcScoreInfo inspect_abc_score(const std::string & abc, bool allow_empty_score = false);

// Fit a completed native two-voice plan to an exact bar count. The beginning
// is retained and the final outro_bars are taken from the planner's real tail,
// so a short render includes ending material rather than cutting mid-section.
std::string fit_abc_score_to_bars(
    const std::string & abc,
    std::uint32_t target_bars,
    std::uint32_t outro_bars = 4);

// Conservative semantic budget for allowing MUSIC_END after a complete score.
// It is a safety ceiling, not a requested audio duration.
std::uint32_t score_aligned_semantic_budget(const AbcScoreInfo & score);

// Recover the complete part of a score whose planning ran out of token budget.
// The unfinished trailing block is dropped so a plan that was cut off mid-bar
// still renders, rather than failing the job outright. A score that already
// parses, or that has no complete block at all, is returned unchanged.
std::string trim_to_complete_score(const std::string & abc);

// Hold a planner-chosen score to a wall-clock ceiling. The bar allowance comes
// from the score's own tempo and meter, so the ceiling means the same thing at
// any tempo. Returns the score unchanged when it already fits, when the
// ceiling is zero, or when the score carries no bars to measure.
// retained_bars are bars the caller already supplied, as a continuation
// prefix. They do not count against the ceiling: a continuation is measured by
// what it adds, and must never come back shorter than the source it extends.
std::string fit_natural_plan_to_ceiling(
    const std::string & abc,
    double max_seconds,
    std::uint32_t outro_bars = 4,
    std::uint32_t retained_bars = 0);

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
    // Optional real-audio continuation seed, expressed as raw codec IDs
    // [0,32767] at 25 Hz. The planner may still extend abc_prefix first; the
    // semantic model then continues from these frames instead of starting at
    // silence. Matching real-audio NAR weights are expected for reconstruction.
    std::vector<std::int32_t> semantic_prefix;
    // Experimental symbolic intervention. Requires melody/full mode and empty
    // lyrics; Vocal is rested without changing Ins before semantic inference.
    bool experimental_vocal_rest = false;
    // Best-effort instrumental control; vocal material may still occur.
    // Requires melody/full mode and empty lyrics. The pipeline supplies an
    // explicit empty section scaffold, adds no-vocal style conditioning, and
    // rests Vocal before semantic inference.
    bool instrumental = false;
    // Zero retains the complete planned/supplied score. With outro ending,
    // the completed plan is fitted to this exact number of bars before audio.
    std::uint32_t target_bars = 0;
    EndingMode ending_mode = EndingMode::natural;
    std::uint32_t outro_bars = 4;
    // Safety ceiling for planner-chosen length, in seconds of score. It only
    // applies when target_bars is zero and the planner wrote the score itself,
    // because that is the one case where nothing else bounds the result: the
    // score's duration becomes the semantic floor, so an overlong plan forces
    // an overlong render. An exceeded plan is fitted to the ceiling through
    // fit_abc_score_to_bars, so it still ends rather than stopping. Zero lifts
    // the ceiling and lets the model run to its own end, which is reasonable
    // locally and a poor idea on a shared backend.
    double natural_max_seconds = 180.0;
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
    // A full symbolic plan carries a chord annotation on every Vocal bar, so it
    // spends several times the tokens a melody plan does and can reach its budget
    // mid-bar. This is a cap, not a target: a plan that ends naturally costs the
    // same whatever the ceiling is, so a generous one only helps the dense case.
    GenerationSampling abc{0.7F, 0.9F, 30, 1.005F, 100, 32, 16384};
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
