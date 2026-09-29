#include "yue2/score_transform.h"

#include "abc_score.h"

#include <cmath>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace yue2 {
namespace {

using abc::Segment;
using abc::Span;

constexpr int kMinBpm = 20;
constexpr int kMaxBpm = 400;

// SheetSage2's spellings: keys as its abc_key writes them, chord roots and
// basses in sharps. The planning header writes keys the same way.
const char * const kMajorKeys[] = {"C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"};
const char * const kMinorKeys[] = {"Cm", "C#m", "Dm", "Ebm", "Em", "Fm", "F#m", "Gm", "G#m", "Am", "Bbm", "Bm"};
const char * const kSharpRoots[] = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};

int wrap(int pitch_class) { return ((pitch_class % 12) + 12) % 12; }

// Pitch class of a leading root ("C", "Eb", "F#"), or -1; `length` is how many
// characters it used.
int root_pitch_class(const std::string & text, std::size_t & length) {
    static const std::string naturals = "C D EF G A B";
    if (text.empty() || text[0] < 'A' || text[0] > 'G') return -1;
    int pitch_class = static_cast<int>(naturals.find(text[0]));
    length = 1;
    if (text.size() > 1 && (text[1] == '#' || text[1] == 'b')) {
        pitch_class += text[1] == '#' ? 1 : -1;
        length = 2;
    }
    return wrap(pitch_class);
}

std::string transpose_key(const std::string & key, int semitones) {
    std::size_t length = 0;
    const int root = root_pitch_class(key, length);
    const auto mode = key.substr(length);
    if (root < 0 || (!mode.empty() && mode != "m")) {
        throw std::invalid_argument("cannot transpose the key " + key +
                                    "; only major and minor keys are supported");
    }
    const int moved = wrap(root + semitones);
    return mode.empty() ? kMajorKeys[moved] : kMinorKeys[moved];
}

// A chord symbol: root, quality, optional /bass. Anything that does not start
// with a root (a no-chord mark) is left alone.
std::string transpose_chord(const std::string & label, int semitones) {
    std::size_t length = 0;
    const int root = root_pitch_class(label, length);
    if (root < 0) return label;
    const auto slash = label.find('/', length);
    std::string out = std::string(kSharpRoots[wrap(root + semitones)]) +
        label.substr(length, slash == std::string::npos ? std::string::npos : slash - length);
    if (slash != std::string::npos) {
        const auto bass_text = label.substr(slash + 1);
        std::size_t bass_length = 0;
        const int bass = root_pitch_class(bass_text, bass_length);
        out += '/';
        out += bass >= 0 && bass_length == bass_text.size()
            ? std::string(kSharpRoots[wrap(bass + semitones)]) : bass_text;
    }
    return out;
}

std::string transpose_mark(const std::string & mark, int semitones) {
    if (mark.size() >= 2 && mark.front() == '"') {
        return '"' + transpose_chord(mark.substr(1, mark.size() - 2), semitones) + '"';
    }
    if (mark.rfind("[K:", 0) == 0) {
        return "[K:" + transpose_key(abc::trim(mark.substr(3, mark.size() - 4)), semitones) + "]";
    }
    return mark;
}

// Rewrites every segment of a lane that holds a note or, when `with_marks`, a
// mark, from `notes`. `edit` adjusts the segment first (its marks and key).
template <typename Edit>
void rewrite_lane(const abc::Parsed & parsed, int lane, const std::vector<Span> & notes,
                  bool with_marks, Edit edit,
                  std::vector<std::vector<std::string>> & line_segments,
                  std::vector<bool> & line_changed) {
    const auto & voice = parsed.voices[static_cast<std::size_t>(lane)];
    for (const auto & segment : voice.segments) {
        bool reached = segment.has_notes || (with_marks && !segment.marks.empty());
        for (const auto & note : notes) {
            if (reached) break;
            reached = note.begin < segment.end && note.end > segment.begin;
        }
        if (!reached) continue;
        Segment copy = segment;
        edit(copy);
        line_segments[copy.line][copy.index] =
            abc::write_bars(copy, notes, parsed.unit_ticks(), copy.key);
        line_changed[copy.line] = true;
    }
}

} // namespace

std::string scale_abc_tempo(const std::string & abc_text, double factor) {
    if (!(factor > 0.0) || !std::isfinite(factor)) {
        throw std::invalid_argument("tempo factor must be a positive number");
    }
    const auto lines = abc::split_lines(abc_text);
    for (std::size_t index = 0; index < lines.size(); ++index) {
        const auto line = abc::trim(lines[index]);
        if (line == "V: Vocal") break;
        if (line.rfind("Q:", 0) != 0) continue;
        const auto equals = line.rfind('=');
        const auto value = equals == std::string::npos ? std::string() : abc::trim(line.substr(equals + 1));
        if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) {
            throw std::invalid_argument("the score's Q: field has no tempo to scale");
        }
        const auto bpm = std::stol(value);
        const auto scaled = std::lround(static_cast<double>(bpm) * factor);
        if (scaled < kMinBpm || scaled > kMaxBpm) {
            throw std::invalid_argument(
                "that would put the tempo at " + std::to_string(scaled) + " bpm; yuey takes " +
                std::to_string(kMinBpm) + " to " + std::to_string(kMaxBpm));
        }
        if (scaled == bpm) return abc_text;
        (void)abc::parse(abc_text);  // only a score that renders is edited
        std::map<std::size_t, std::string> replaced = {
            {index, line.substr(0, equals + 1) + std::to_string(scaled)}};
        return abc::assemble(abc_text, {}, {}, replaced, "the tempo change");
    }
    throw std::invalid_argument("the score has no Q: tempo");
}

std::string transpose_abc(const std::string & abc_text, int semitones) {
    if (semitones < -24 || semitones > 24) {
        throw std::invalid_argument("transpose by at most two octaves either way");
    }
    if (semitones == 0) return abc_text;
    const auto parsed = abc::parse(abc_text);
    const bool keys_move = semitones % 12 != 0;

    auto line_segments = parsed.line_segments;
    std::vector<bool> line_changed(line_segments.size(), false);
    for (int lane = 0; lane < 2; ++lane) {
        auto notes = parsed.voices[static_cast<std::size_t>(lane)].notes;
        for (auto & note : notes) {
            note.pitch += semitones;
            if (note.pitch < 0 || note.pitch > 127) {
                throw std::invalid_argument("that would move a note out of the MIDI range");
            }
        }
        rewrite_lane(parsed, lane, notes, keys_move, [&](Segment & segment) {
            if (!keys_move) return;
            for (auto & mark : segment.marks) mark.text = transpose_mark(mark.text, semitones);
            segment.key = transpose_key(segment.key, semitones);
        }, line_segments, line_changed);
    }

    // K: lines, in the header and in the voice blocks.
    std::map<std::size_t, std::string> replaced;
    if (keys_move) {
        const auto lines = abc::split_lines(abc_text);
        for (std::size_t index = 0; index < lines.size(); ++index) {
            const auto line = abc::trim(lines[index]);
            if (line.rfind("K:", 0) == 0) {
                replaced[index] = "K:" + transpose_key(abc::trim(line.substr(2)), semitones);
            }
        }
    }
    return abc::assemble(abc_text, line_segments, line_changed, replaced, "the transposition");
}

std::string swap_abc_lanes(const std::string & abc_text) {
    const auto parsed = abc::parse(abc_text);
    const auto & vocal = parsed.voices[0].notes;
    const auto & ins = parsed.voices[1].notes;
    if (vocal.empty() && ins.empty()) return abc_text;

    auto line_segments = parsed.line_segments;
    std::vector<bool> line_changed(line_segments.size(), false);
    const auto keep = [](Segment &) {};
    rewrite_lane(parsed, 0, ins, false, keep, line_segments, line_changed);
    rewrite_lane(parsed, 1, vocal, false, keep, line_segments, line_changed);
    return abc::assemble(abc_text, line_segments, line_changed, {}, "the lane swap");
}

std::string drop_abc_chords(const std::string & abc_text) {
    const auto parsed = abc::parse(abc_text);
    auto line_segments = parsed.line_segments;
    std::vector<bool> line_changed(line_segments.size(), false);
    for (const auto & voice : parsed.voices) {
        for (const auto & segment : voice.segments) {
            if (segment.text.find('"') == std::string::npos) continue;
            // Chord symbols take no time, so cutting them out of the text
            // leaves every note and rest where it was.
            std::string text;
            for (std::size_t offset = 0; offset < segment.text.size();) {
                if (segment.text[offset] == '"') {
                    offset = segment.text.find('"', offset + 1) + 1;
                    continue;
                }
                text += segment.text[offset++];
            }
            line_segments[segment.line][segment.index] = text;
            line_changed[segment.line] = true;
        }
    }
    bool any = false;
    for (const bool changed : line_changed) any = any || changed;
    if (!any) return abc_text;
    return abc::assemble(abc_text, line_segments, line_changed, {}, "removing the chords");
}

} // namespace yue2
