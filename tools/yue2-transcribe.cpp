#include "yue2/transcription.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

void usage(const char * argv0) {
    std::cout
        << "Usage: " << argv0 << " --model sheetsage2.gguf --audio input.wav --output score.abc [options]\n\n"
        << "Options:\n"
        << "  --midi PATH          Also write a Standard MIDI File\n"
        << "  --midi-dir DIR       Also write transcription, melody, vocal, instrumental, and chord MIDIs\n"
        << "  --events PATH        Also write decoded events and raw tokens as JSON\n"
        << "  --full               Also decode meter, structure, key, and chords\n"
        << "  --max-tokens N       Maximum sequence length (default 5120)\n"
        << "  --window-seconds N   Inference window, at most 300 (default 300)\n"
        << "  --overlap-seconds N  Whole-song overlap (default 200)\n"
        << "  --lookahead-seconds N  Right context excluded from nonfinal output (default 100)\n"
        << "  --device NAME        Require a backend such as cuda or cpu\n"
        << "  --threads N          CPU worker threads (default backend/YUE2_THREADS)\n"
        << "  --version            Print version\n\n"
        << "Backend selection uses YUE2_DEVICE=cpu and YUE2_GPU=<index-or-name>.\n";
}

std::string value_after(int argc, char ** argv, std::string_view name, bool required = true) {
    for (int index = 1; index < argc; ++index) {
        if (std::string_view(argv[index]) != name) continue;
        if (index + 1 >= argc || std::string_view(argv[index + 1]).rfind("--", 0) == 0) {
            throw std::runtime_error("option requires a value: " + std::string(name));
        }
        return argv[index + 1];
    }
    if (required) throw std::runtime_error("missing required option " + std::string(name));
    return {};
}

bool has(int argc, char ** argv, std::string_view name) {
    for (int index = 1; index < argc; ++index) if (std::string_view(argv[index]) == name) return true;
    return false;
}

void write_text(const std::filesystem::path & path, const std::string & text) {
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    if (!output || !output.write(text.data(), static_cast<std::streamsize>(text.size()))) {
        throw std::runtime_error("cannot write " + path.string());
    }
}

void write_bytes(const std::filesystem::path & path, const std::vector<std::uint8_t> & bytes) {
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    if (!output || !output.write(reinterpret_cast<const char *>(bytes.data()),
                                 static_cast<std::streamsize>(bytes.size()))) {
        throw std::runtime_error("cannot write " + path.string());
    }
}

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--version") {
            std::cout << "yue2-transcribe " << yue2::version() << '\n';
            return 0;
        }
        if (argc == 1 || has(argc, argv, "--help") || has(argc, argv, "-h")) {
            usage(argv[0]);
            return 0;
        }

        const auto model_path = std::filesystem::path(value_after(argc, argv, "--model"));
        const auto audio_path = std::filesystem::path(value_after(argc, argv, "--audio"));
        const auto output_path = std::filesystem::path(value_after(argc, argv, "--output"));
        const auto midi_value = value_after(argc, argv, "--midi", false);
        const auto midi_dir_value = value_after(argc, argv, "--midi-dir", false);
        const auto events_value = value_after(argc, argv, "--events", false);
        yue2::TranscriptionOptions options;
        options.melody_only = !has(argc, argv, "--full");
        const auto max_tokens = value_after(argc, argv, "--max-tokens", false);
        if (!max_tokens.empty()) options.max_tokens = static_cast<std::size_t>(std::stoull(max_tokens));
        const auto window = value_after(argc, argv, "--window-seconds", false);
        if (!window.empty()) options.window_seconds = std::stof(window);
        const auto overlap = value_after(argc, argv, "--overlap-seconds", false);
        if (!overlap.empty()) options.overlap_seconds = std::stof(overlap);
        const auto lookahead = value_after(argc, argv, "--lookahead-seconds", false);
        if (!lookahead.empty()) options.lookahead_seconds = std::stof(lookahead);

        yue2::TranscriberRuntimeOptions runtime;
        runtime.device = value_after(argc, argv, "--device", false);
        const auto threads = value_after(argc, argv, "--threads", false);
        if (!threads.empty()) runtime.threads = std::stoi(threads);
        yue2::Transcriber transcriber(model_path, runtime);
        const auto result = transcriber.transcribe(audio_path, options);
        write_text(output_path, result.abc);
        if (!midi_value.empty()) write_bytes(midi_value, result.midi);
        if (!midi_dir_value.empty()) {
            const auto directory = std::filesystem::path(midi_dir_value);
            write_bytes(directory / "transcription.mid", result.midi_exports.transcription);
            write_bytes(directory / "melody.mid", result.midi_exports.melody);
            write_bytes(directory / "melody_vocal.mid", result.midi_exports.vocal);
            write_bytes(directory / "melody_instrumental.mid", result.midi_exports.instrumental);
            if (!result.midi_exports.chords.empty()) {
                write_bytes(directory / "chords.mid", result.midi_exports.chords);
            }
        }
        if (!events_value.empty()) {
            write_text(events_value, yue2::serialize_transcription_json(result));
        }
        std::size_t note_count = 0;
        for (const auto & event : result.events) note_count += event.notes.size();
        std::cout << "wrote " << output_path << " (" << result.events.size()
                  << " events, " << note_count << " notes)\n";
        if (!midi_value.empty()) std::cout << "wrote " << midi_value << '\n';
        if (!midi_dir_value.empty()) std::cout << "wrote MIDI set under " << midi_dir_value << '\n';
        if (!events_value.empty()) std::cout << "wrote " << events_value << '\n';
        for (const auto & warning : result.warnings) std::cerr << "warning: " << warning << '\n';
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "yue2-transcribe: " << error.what() << '\n';
        return 2;
    }
}
