#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace yue2 {

enum class TranscriptionPreset {
    standard,
    paper,
};

struct TranscriptionOptions {
    bool melody_only = true;
    TranscriptionPreset preset = TranscriptionPreset::standard;
    float window_seconds = 300.0F;
    float overlap_seconds = 200.0F;
    float lookahead_seconds = 100.0F;
    std::size_t max_tokens = 5120;
};

struct TranscriberRuntimeOptions {
    // Empty selects the best registered accelerator and falls back to CPU.
    // Explicit names such as "cuda" and "cpu" never silently change device.
    std::string device;
    // Zero uses the backend default, or YUE2_THREADS for a CPU backend.
    int threads = 0;
};

struct TranscriptionControl {
    std::function<void(std::size_t current_window, std::size_t total_windows)> on_progress;
    std::function<bool()> should_cancel;
};

struct NoteEvent {
    std::int32_t pitch = 0;
    std::int32_t track = 0;
    std::int32_t duration_bin = 0;
    std::int32_t duration_steps = 0;
    double end_time_seconds = 0.0;
};

struct ScoreEvent {
    std::int64_t subbeat = 0;
    std::int64_t source_subbeat = 0;
    std::size_t window_index = 0;
    double time_seconds = 0.0;
    bool has_timestamp = false;
    std::int32_t meter_numerator = 0;
    std::int32_t meter_denominator = 0;
    std::int32_t eighth_position = -1;
    std::string structure;
    std::string key;
    std::string chord;
    std::vector<std::int32_t> payload_tokens;
    std::vector<NoteEvent> notes;
};

struct TranscriptionWindow {
    double start_seconds = 0.0;
    double end_seconds = 0.0;
    double accept_start_seconds = 0.0;
    double accept_end_seconds = 0.0;
    double prefix_end_seconds = 0.0;
    // Negative for the final window; otherwise the local timestamp at which
    // generation is stopped so the right-hand lookahead remains context only.
    double generation_stop_seconds = -1.0;
};

struct TranscriptionWindowResult {
    TranscriptionWindow window;
    std::vector<std::int32_t> tokens;
    std::size_t prefix_tokens = 0;
    std::size_t decoded_events = 0;
    std::size_t accepted_events = 0;
};

struct TranscriptionMidiExports {
    // Full conductor + active melodies + optional chord-note arrangement.
    std::vector<std::uint8_t> transcription;
    // Official-style component files, each with its own conductor track.
    std::vector<std::uint8_t> melody;
    std::vector<std::uint8_t> vocal;
    std::vector<std::uint8_t> instrumental;
    std::vector<std::uint8_t> chords;
};

struct TranscriptionResult {
    std::string abc;
    // Compatibility alias for midi_exports.transcription.
    std::vector<std::uint8_t> midi;
    TranscriptionMidiExports midi_exports;
    // Populated for a one-window result. Whole-song callers should use the
    // lossless per-window sequences in windows instead.
    std::vector<std::int32_t> tokens;
    std::vector<ScoreEvent> events;
    std::vector<std::string> warnings;
    std::vector<TranscriptionWindowResult> windows;
    double duration_seconds = 0.0;
};

std::vector<TranscriptionWindow> make_transcription_window_plan(
    double duration_seconds,
    double window_seconds = 300.0,
    double overlap_seconds = 200.0,
    double lookahead_seconds = 100.0);

// Decode one grammar-constrained SheetSage2 v1 token sequence without loading
// model weights. Timestamp anchors are interpolated onto every returned event
// and note end, clamped to the supplied audio duration.
std::vector<ScoreEvent> decode_sheetsage2_tokens(
    const std::vector<std::int32_t> & tokens,
    double audio_duration_seconds);

// Serialize decoded events in the native two-voice SheetSage2/YuE2 ABC
// dialect. Melody-only omits chord symbols but retains both melody voices.
std::string serialize_sheetsage2_abc(
    const std::vector<ScoreEvent> & events,
    bool melody_only = true);

// Serialize a DAW-oriented format-1 Standard MIDI File. The conductor track
// carries inferred tempo, meter, key, and structure markers; active vocal and
// instrumental melody lanes get named tracks; full mode also adds SheetSage2's
// chord labels as voiced, downbeat-rearticulated chord notes.
std::vector<std::uint8_t> serialize_sheetsage2_midi(
    const std::vector<ScoreEvent> & events,
    bool melody_only = false,
    double duration_seconds = 0.0);

// Build the combined transcription MIDI and the same component MIDI set as
// the released SheetSage2 exporter.
TranscriptionMidiExports serialize_sheetsage2_midis(
    const std::vector<ScoreEvent> & events,
    bool melody_only = false,
    double duration_seconds = 0.0);

// Convert YuE2's constrained native two-lane ABC score into the same combined
// and component Standard MIDI files returned by transcription. This performs
// structural and rhythmic validation and does not require model weights.
TranscriptionMidiExports serialize_yue2_abc_midis(const std::string & abc);

// Lossless JSON form used by the CLI and C ABI. It includes raw per-window
// tokens, typed events, notes, timing provenance, and warnings.
std::string serialize_transcription_json(const TranscriptionResult & result);

class Transcriber {
public:
    explicit Transcriber(const std::filesystem::path & model_path);
    Transcriber(
        const std::filesystem::path & model_path,
        const TranscriberRuntimeOptions & runtime_options);
    ~Transcriber();

    Transcriber(const Transcriber &) = delete;
    Transcriber & operator=(const Transcriber &) = delete;
    Transcriber(Transcriber &&) noexcept;
    Transcriber & operator=(Transcriber &&) noexcept;

    TranscriptionResult transcribe(
        const std::filesystem::path & audio_path,
        const TranscriptionOptions & options = {});

    // Embedding boundary for JUCE and servers that already own decoded mono
    // PCM. Samples are amplitude-preserving floats at the supplied rate.
    TranscriptionResult transcribe_mono(
        const float * samples,
        std::size_t sample_count,
        std::int32_t sample_rate,
        const TranscriptionOptions & options = {},
        const TranscriptionControl & control = {});

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

const char * version() noexcept;
bool transcription_runtime_available() noexcept;

} // namespace yue2
