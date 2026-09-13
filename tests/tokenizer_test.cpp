#include "yue2/sheetsage2_tokens.h"
#include "yue2/transcription.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>

using yue2::sheetsage2::PromptGrammarState;
using yue2::sheetsage2::TokenLayout;

namespace {

bool any(const std::vector<std::uint8_t> & mask, std::int32_t begin, std::int32_t end) {
    for (auto i = begin; i < end; ++i) if (mask[static_cast<std::size_t>(i)]) return true;
    return false;
}

bool all(const std::vector<std::uint8_t> & mask, std::int32_t begin, std::int32_t end) {
    for (auto i = begin; i < end; ++i) if (!mask[static_cast<std::size_t>(i)]) return false;
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    static_assert(TokenLayout::vocab_size == 31678);
    static_assert(TokenLayout::duration_end == TokenLayout::vocab_size);
    static_assert(TokenLayout::time_end - TokenLayout::time_begin == 30000);
    static_assert(TokenLayout::chord_full_end - TokenLayout::chord_full_begin == 361);

    assert(TokenLayout::type(TokenLayout::time_begin) == TokenLayout::Type::time);
    assert(TokenLayout::type(TokenLayout::duration_end - 1) == TokenLayout::Type::duration);
    assert(TokenLayout::type(TokenLayout::vocab_size) == TokenLayout::Type::invalid);

    PromptGrammarState grammar;
    auto mask = grammar.allowed();
    assert(!mask[TokenLayout::eos]);
    assert(all(mask, TokenLayout::subbeat_shift_begin, TokenLayout::subbeat_shift_end));
    assert(all(mask, TokenLayout::time_begin, TokenLayout::time_end));

    grammar.update(TokenLayout::meter_begin);
    mask = grammar.allowed();
    assert(all(mask, TokenLayout::eighth_position_begin, TokenLayout::eighth_position_end));
    assert(!any(mask, TokenLayout::pitch_begin, TokenLayout::pitch_end));

    grammar.update(TokenLayout::eighth_position_begin);
    grammar.update(TokenLayout::pitch_begin);
    mask = grammar.allowed();
    assert(all(mask, TokenLayout::pitch_begin, TokenLayout::pitch_end));
    assert(all(mask, TokenLayout::duration_begin, TokenLayout::duration_end));
    assert(!any(mask, TokenLayout::time_begin, TokenLayout::time_end));

    grammar.update(TokenLayout::duration_begin);
    grammar.update(TokenLayout::subbeat_shift_begin + 1);
    assert(grammar.generated_events() == 1);
    mask = grammar.allowed();
    assert(all(mask, TokenLayout::time_begin, TokenLayout::time_end));
    assert(!mask[TokenLayout::eos]);

    grammar.update(TokenLayout::time_begin);
    assert(grammar.update(TokenLayout::eos));

    const std::vector<std::int32_t> full_sequence = {
        1, 4, 5, 6, 7, 9, 11, 3,
        260, 617, 30537, 30709, 30969, 30990, 31041,
        31458, 31657, 31590, 31655,
        264, 717, 31460, 31654, 2,
    };
    const auto events = yue2::decode_sheetsage2_tokens(full_sequence, 10.0);
    assert(events.size() == 2);
    assert(events[0].subbeat == 0 && events[0].time_seconds == 1.0);
    assert(events[0].has_timestamp);
    assert(events[0].meter_numerator == 4 && events[0].meter_denominator == 4);
    assert(events[0].eighth_position == 0);
    assert(events[0].structure == "chorus");
    assert(events[0].key == "D:major");
    assert(events[0].chord == "C:maj");
    assert(events[0].notes.size() == 2);
    assert(events[0].notes[0].pitch == 60 && events[0].notes[0].track == 0);
    assert(events[0].notes[0].duration_steps == 4);
    assert(events[0].notes[0].end_time_seconds == 2.0);
    assert(events[0].notes[1].pitch == 64 && events[0].notes[1].track == 1);
    assert(events[1].subbeat == 4 && events[1].time_seconds == 2.0);
    assert(events[1].notes.size() == 1 && events[1].notes[0].duration_steps == 1);
    assert(events[1].notes[0].end_time_seconds == 2.25);

    auto export_events = events;
    yue2::ScoreEvent later;
    later.subbeat = 8;
    later.key = "A:minor";
    later.chord = "N";
    later.structure = "bridge";
    yue2::NoteEvent tied_note;
    tied_note.pitch = 61;
    tied_note.track = 0;
    tied_note.duration_steps = 9;
    later.notes.push_back(tied_note);
    export_events.push_back(later);

    const auto full_abc = yue2::serialize_sheetsage2_abc(export_events, false);
    assert(full_abc.rfind("X:1\nT:\nM:4/4\nL:1/16\n", 0) == 0);
    assert(full_abc.find("V: Vocal clef=treble name=\"Vocal Melody\" snm=\"Vocal\"") != std::string::npos);
    assert(full_abc.find("V: Ins clef=treble name=\"Ins Melody\" snm=\"Inst.\"") != std::string::npos);
    assert(full_abc.find("\"C\"") != std::string::npos);
    assert(full_abc.find("[K:Am]") != std::string::npos);
    assert(full_abc.find("^C8-") != std::string::npos);
    assert(full_abc.find("|]") == std::string::npos);

    const auto melody_abc = yue2::serialize_sheetsage2_abc(export_events, true);
    assert(melody_abc.find("V: Vocal\n") != std::string::npos);
    assert(melody_abc.find("V: Ins\n") != std::string::npos);
    assert(melody_abc.find("\"C\"") == std::string::npos);
    const auto midi = yue2::serialize_sheetsage2_midi(export_events);
    assert(std::find(midi.begin(), midi.end(), static_cast<std::uint8_t>(0x91)) != midi.end());

    std::vector<yue2::ScoreEvent> mixed_events(4);
    mixed_events[0].subbeat = 0;
    mixed_events[0].meter_numerator = 4;
    mixed_events[0].meter_denominator = 4;
    mixed_events[0].eighth_position = 0;
    mixed_events[0].structure = "verse";
    mixed_events[0].key = "C:major";
    mixed_events[0].chord = "C:maj";
    yue2::NoteEvent long_vocal;
    long_vocal.pitch = 60;
    long_vocal.track = 0;
    long_vocal.duration_steps = 48;
    mixed_events[0].notes.push_back(long_vocal);
    mixed_events[1].subbeat = 16;
    mixed_events[1].meter_numerator = 3;
    mixed_events[1].meter_denominator = 4;
    mixed_events[1].eighth_position = 0;
    mixed_events[1].structure = "bridge";
    yue2::NoteEvent instrumental_note;
    instrumental_note.pitch = 64;
    instrumental_note.track = 1;
    instrumental_note.duration_steps = 12;
    mixed_events[1].notes.push_back(instrumental_note);
    mixed_events[2].subbeat = 28;
    mixed_events[2].meter_numerator = 5;
    mixed_events[2].meter_denominator = 8;
    mixed_events[2].eighth_position = 0;
    mixed_events[2].key = "G:major";
    mixed_events[2].chord = "G:maj";
    mixed_events[3].subbeat = 48;
    mixed_events[3].meter_numerator = 5;
    mixed_events[3].meter_denominator = 8;
    mixed_events[3].eighth_position = 0;
    const auto mixed_abc = yue2::serialize_sheetsage2_abc(mixed_events, false);
    assert(mixed_abc.rfind("X:1\nT:\nM:4/4\nL:1/32\n", 0) == 0);
    assert(mixed_abc.find("V: Vocal\nM:3/4\n") != std::string::npos);
    assert(mixed_abc.find("V: Ins\nM:3/4\n") != std::string::npos);
    assert(mixed_abc.find("V: Vocal\nM:5/8\n") != std::string::npos);
    assert(mixed_abc.find("V: Ins\nM:5/8\n") != std::string::npos);
    assert(mixed_abc.find("[K:G]") != std::string::npos);
    if (argc == 2) {
        std::ofstream output(argv[1], std::ios::binary);
        assert(output && output.write(full_abc.data(), static_cast<std::streamsize>(full_abc.size())));
    } else if (argc == 3) {
        std::ofstream first(argv[1], std::ios::binary);
        std::ofstream second(argv[2], std::ios::binary);
        assert(first && first.write(full_abc.data(), static_cast<std::streamsize>(full_abc.size())));
        assert(second && second.write(mixed_abc.data(), static_cast<std::streamsize>(mixed_abc.size())));
    }

    bool rejected = false;
    try {
        (void) yue2::decode_sheetsage2_tokens({1, 4, 11, 3, 517, 2}, 1.0);
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    assert(rejected);

    const auto one_window = yue2::make_transcription_window_plan(100.0);
    assert(one_window.size() == 1);
    assert(one_window[0].start_seconds == 0.0);
    assert(one_window[0].end_seconds == 100.0);
    assert(one_window[0].accept_end_seconds == 100.0);
    assert(one_window[0].generation_stop_seconds < 0.0);

    const auto two_windows = yue2::make_transcription_window_plan(400.0);
    assert(two_windows.size() == 2);
    assert(two_windows[0].start_seconds == 0.0);
    assert(two_windows[0].accept_start_seconds == 0.0);
    assert(two_windows[0].accept_end_seconds == 200.0);
    assert(two_windows[0].generation_stop_seconds == 200.0);
    assert(two_windows[1].start_seconds == 100.0);
    assert(two_windows[1].prefix_end_seconds == 200.0);
    assert(two_windows[1].accept_start_seconds == 200.0);
    assert(two_windows[1].accept_end_seconds == 400.0);

    const auto partial_tail = yue2::make_transcription_window_plan(305.0);
    assert(partial_tail.size() == 2);
    assert(partial_tail[1].start_seconds == 100.0);
    assert(partial_tail[1].end_seconds == 305.0);
    assert(partial_tail[1].prefix_end_seconds == 200.0);
    assert(partial_tail[1].accept_start_seconds == 200.0);
    assert(partial_tail[1].accept_end_seconds == 305.0);

    const auto narrow_tail = yue2::make_transcription_window_plan(401.0);
    assert(narrow_tail.size() == 3);
    assert(narrow_tail[2].start_seconds == 200.0);
    assert(narrow_tail[2].end_seconds == 401.0);
    assert(narrow_tail[2].accept_start_seconds == 300.0);
    assert(narrow_tail[2].accept_end_seconds == 401.0);

    rejected = false;
    try {
        (void) yue2::make_transcription_window_plan(400.0, 300.0, 50.0, 100.0);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    assert(rejected);

    std::cout << "SheetSage2 v1 token layout, grammar, typed decode, native export, and window plan: ok\n";
    return 0;
}
