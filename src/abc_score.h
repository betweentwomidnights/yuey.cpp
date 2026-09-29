// Internal: the native two-voice ABC dialect as bars and timed notes, for the
// rewrites that must keep every bar they do not touch byte for byte (melody
// transfer, and the score editor's transforms). Pitches are read with the same
// key table the MIDI export uses, so a rewrite and a render agree on every note.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace yue2::abc {

constexpr std::uint64_t kWholeTicks = 3840;  // 960 per quarter

std::string trim(const std::string & value);
std::vector<std::string> split_lines(const std::string & text);

// Key-signature alterations for C D E F G A B; all zero for a key the table
// does not know, exactly as score_midi.cpp reads it.
std::array<int, 7> key_accidentals(const std::string & key);

struct Span {
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
    int pitch = 0;
};

struct Mark {
    std::uint64_t tick = 0;
    std::string text;  // written as-is: "\"Cm\"" or "[K:G]"
};

// One bar as the text holds it. A multi-bar rest "Z3" is one segment that
// stands for three bars.
struct Segment {
    std::size_t line = 0;
    std::size_t index = 0;           // position among the line's segments
    std::string text;                // without its closing '|'
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
    std::vector<std::uint64_t> bar_ends;  // one per bar the segment covers
    std::string key;                 // key in force at the segment's start
    std::vector<Mark> marks;         // chords and inline fields, absolute ticks
    bool has_notes = false;
};

struct Voice {
    std::vector<Segment> segments;
    std::vector<Span> notes;  // ties joined, in time order
};

struct Parsed {
    std::uint32_t unit_denominator = 32;
    std::array<Voice, 2> voices;  // Vocal, Ins
    std::vector<std::vector<std::string>> line_segments;  // per line; empty for non-music

    std::uint64_t unit_ticks() const;
};

// Throws std::invalid_argument on anything outside the dialect.
Parsed parse(const std::string & abc);

// Explicit spelling, as SheetSage2 writes it: flats in flat keys, sharps
// otherwise, "=" on every natural.
std::string spell(int pitch, const std::string & key);

// Writes a segment's bars as note and rest runs from `notes`, with the
// segment's marks at their ticks. A note that runs past a bar or a mark is
// tied through it.
std::string write_bars(const Segment & segment, const std::vector<Span> & notes,
                       std::uint64_t unit_ticks, const std::string & key);

// Rebuilds the score: a changed line is rejoined from its segments, a line in
// `replaced` is taken as given, and every other line is kept exactly. Throws
// std::runtime_error, naming `what`, if the result would not render.
std::string assemble(const std::string & abc,
                     const std::vector<std::vector<std::string>> & line_segments,
                     const std::vector<bool> & line_changed,
                     const std::map<std::size_t, std::string> & replaced,
                     const char * what);

} // namespace yue2::abc
