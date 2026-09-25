#include "yue2/transcription.h"
#include "yue2/audio.h"
#include "yue2/mert2_encoder.h"
#include "yue2/mert2_frontend.h"
#include "yue2/sheetsage2_tokens.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace yue2 {
namespace {

constexpr std::array<std::int32_t, 24> duration_steps = {
    1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64,
    96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096,
};

constexpr std::array<const char *, 12> chromatic_sharps = {
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
};

constexpr std::array<const char *, 23> structure_labels = {
    "silence", "intro", "outro", "verse", "chorus", "bridge", "pre-chorus",
    "post-chorus", "interlude", "fade-out", "loop", "rap", "preshot", "irregular",
    "instrumental", "intro and verse", "pre-chorus and chorus", "verse and pre-chorus",
    "solo", "theme", "development", "variation", "pre-outro",
};

const std::vector<std::string> & full_chord_labels() {
    static const std::vector<std::string> labels = [] {
        struct Quality { const char * name; std::vector<const char *> inversions; };
        const std::array<Quality, 15> qualities = {{
            {"maj", {"/2", "/3", "/5"}}, {"min", {"/2", "/b3", "/5"}},
            {"dim", {}}, {"aug", {}}, {"maj7", {"/3", "/5", "/7"}},
            {"min7", {"/b3", "/5", "/b7"}}, {"7", {"/3", "/5", "/b7"}},
            {"hdim7", {}}, {"dim7", {}}, {"minmaj7", {}}, {"sus2", {}},
            {"sus4", {}}, {"sus4(b7)", {}}, {"maj6", {}}, {"min6", {}},
        }};
        std::vector<std::string> result = {"N"};
        for (const auto & quality : qualities) {
            for (const auto * root : chromatic_sharps) {
                for (const auto * inversion : quality.inversions) {
                    result.push_back(std::string(root) + ":" + quality.name + inversion);
                }
                result.push_back(std::string(root) + ":" + quality.name);
            }
        }
        return result;
    }();
    return labels;
}

int payload_field_index(std::int32_t token) {
    using Type = sheetsage2::TokenLayout::Type;
    switch (sheetsage2::TokenLayout::type(token)) {
        case Type::time: return 0;
        case Type::meter:
        case Type::eighth_position: return 1;
        case Type::structure: return 2;
        case Type::key: return 3;
        case Type::chord_majmin:
        case Type::chord_full: return 4;
        case Type::pitch:
        case Type::duration: return 5;
        default: return 6;
    }
}

bool has_field(const std::vector<std::int32_t> & payload, int field) {
    return std::any_of(payload.begin(), payload.end(),
        [field](std::int32_t token) { return payload_field_index(token) == field; });
}

struct OverlapPrefix {
    std::vector<std::int32_t> tokens;
    std::int64_t global_subbeat_base = 0;
    bool available = false;
};

OverlapPrefix build_overlap_prefix(
    const std::vector<ScoreEvent> & stitched,
    const std::vector<std::int32_t> & prompt_prefix,
    double window_start,
    double prefix_end) {
    constexpr double epsilon = 1.0e-4;
    std::vector<const ScoreEvent *> sources;
    for (const auto & event : stitched) {
        if (event.time_seconds >= window_start - epsilon &&
            event.time_seconds < prefix_end - epsilon) {
            sources.push_back(&event);
        }
    }
    std::sort(sources.begin(), sources.end(), [](const ScoreEvent * left, const ScoreEvent * right) {
        return std::tie(left->subbeat, left->time_seconds) <
            std::tie(right->subbeat, right->time_seconds);
    });
    const auto first = std::find_if(sources.begin(), sources.end(), [](const ScoreEvent * event) {
        return event->has_timestamp || event->meter_numerator > 0 || event->eighth_position >= 0;
    });
    if (first == sources.end()) return {};
    sources.erase(sources.begin(), first);
    const auto base = sources.front()->subbeat;

    std::array<std::int32_t, 5> active_context{};
    for (const auto & event : stitched) {
        if (event.time_seconds > sources.front()->time_seconds + epsilon) continue;
        for (const auto token : event.payload_tokens) {
            const int field = payload_field_index(token);
            if (field >= 2 && field <= 4) active_context[static_cast<std::size_t>(field)] = token;
            if (sheetsage2::TokenLayout::type(token) == sheetsage2::TokenLayout::Type::meter) {
                active_context[1] = token;
            }
        }
    }

    OverlapPrefix result;
    result.tokens = prompt_prefix;
    result.global_subbeat_base = base;
    result.available = true;
    std::int64_t previous = 0;
    for (std::size_t index = 0; index < sources.size(); ++index) {
        const auto & event = *sources[index];
        const auto local_subbeat = std::max<std::int64_t>(previous, event.subbeat - base);
        auto shift = local_subbeat - previous;
        while (shift > 256) {
            result.tokens.push_back(sheetsage2::TokenLayout::subbeat_shift_begin + 256);
            shift -= 256;
        }
        result.tokens.push_back(
            sheetsage2::TokenLayout::subbeat_shift_begin + static_cast<std::int32_t>(shift));
        previous = local_subbeat;

        auto payload = event.payload_tokens;
        for (auto & token : payload) {
            if (sheetsage2::TokenLayout::type(token) == sheetsage2::TokenLayout::Type::time) {
                const auto time_id = std::clamp<std::int64_t>(
                    std::llround((event.time_seconds - window_start) * 100.0), 0, 29999);
                token = sheetsage2::TokenLayout::time_begin + static_cast<std::int32_t>(time_id);
            }
        }
        if (index == 0) {
            for (int field = 2; field <= 4; ++field) {
                const auto token = active_context[static_cast<std::size_t>(field)];
                if (token && !has_field(payload, field)) payload.push_back(token);
            }
            const bool has_eighth = std::any_of(payload.begin(), payload.end(), [](std::int32_t token) {
                return sheetsage2::TokenLayout::type(token) ==
                    sheetsage2::TokenLayout::Type::eighth_position;
            });
            const bool has_meter = std::any_of(payload.begin(), payload.end(), [](std::int32_t token) {
                return sheetsage2::TokenLayout::type(token) == sheetsage2::TokenLayout::Type::meter;
            });
            if (has_eighth && !has_meter && active_context[1]) payload.push_back(active_context[1]);
        }
        std::stable_sort(payload.begin(), payload.end(), [](std::int32_t left, std::int32_t right) {
            return payload_field_index(left) < payload_field_index(right);
        });
        result.tokens.insert(result.tokens.end(), payload.begin(), payload.end());
    }
    return result;
}

std::vector<ScoreEvent> decode_events_impl(
    const std::vector<std::int32_t> & tokens,
    double audio_duration_seconds) {
    using sheetsage2::TokenLayout;
    const auto out = std::find(tokens.begin(), tokens.end(), TokenLayout::out);
    if (tokens.empty() || tokens.front() != TokenLayout::sos || out == tokens.end()) {
        throw std::runtime_error("invalid SheetSage2 output prefix");
    }
    std::vector<ScoreEvent> events;
    std::map<std::int64_t, double> time_anchors;
    std::int64_t subbeat = 0;
    auto cursor = out + 1;
    while (cursor != tokens.end() && *cursor != TokenLayout::eos) {
        if (TokenLayout::type(*cursor) != TokenLayout::Type::subbeat_shift) {
            throw std::runtime_error("SheetSage2 event is missing its subbeat shift");
        }
        while (cursor != tokens.end() && TokenLayout::type(*cursor) == TokenLayout::Type::subbeat_shift) {
            subbeat += *cursor - TokenLayout::subbeat_shift_begin;
            ++cursor;
        }
        ScoreEvent event;
        event.subbeat = subbeat;
        event.source_subbeat = subbeat;
        event.time_seconds = std::numeric_limits<double>::quiet_NaN();
        bool recognized_payload = false;
        while (cursor != tokens.end() && *cursor != TokenLayout::eos &&
               TokenLayout::type(*cursor) != TokenLayout::Type::subbeat_shift) {
            const auto type = TokenLayout::type(*cursor);
            if (type == TokenLayout::Type::time) {
                event.payload_tokens.push_back(*cursor);
                event.time_seconds = (*cursor - TokenLayout::time_begin) / 100.0;
                event.has_timestamp = true;
                recognized_payload = true;
                ++cursor;
            } else if (type == TokenLayout::Type::meter) {
                event.payload_tokens.push_back(*cursor);
                static constexpr std::array<std::int32_t, 6> denominators = {1, 2, 4, 8, 16, 32};
                const auto meter_id = *cursor - TokenLayout::meter_begin;
                event.meter_numerator = meter_id / 6 + 1;
                event.meter_denominator = denominators[static_cast<std::size_t>(meter_id % 6)];
                recognized_payload = true;
                ++cursor;
            } else if (type == TokenLayout::Type::eighth_position) {
                event.payload_tokens.push_back(*cursor);
                event.eighth_position = *cursor - TokenLayout::eighth_position_begin;
                recognized_payload = true;
                ++cursor;
            } else if (type == TokenLayout::Type::structure) {
                event.payload_tokens.push_back(*cursor);
                const auto index = static_cast<std::size_t>(*cursor - TokenLayout::structure_begin);
                event.structure = structure_labels[index];
                recognized_payload = true;
                ++cursor;
            } else if (type == TokenLayout::Type::key) {
                event.payload_tokens.push_back(*cursor);
                const auto key_id = *cursor - TokenLayout::key_begin;
                event.key = std::string(chromatic_sharps[static_cast<std::size_t>(key_id % 12)]) +
                    (key_id >= 12 ? ":minor" : ":major");
                recognized_payload = true;
                ++cursor;
            } else if (type == TokenLayout::Type::chord_majmin) {
                event.payload_tokens.push_back(*cursor);
                const auto chord_id = *cursor - TokenLayout::chord_majmin_begin;
                event.chord = chord_id == 0 ? "N" :
                    std::string(chromatic_sharps[static_cast<std::size_t>((chord_id - 1) % 12)]) +
                    (chord_id > 12 ? ":min" : ":maj");
                recognized_payload = true;
                ++cursor;
            } else if (type == TokenLayout::Type::chord_full) {
                event.payload_tokens.push_back(*cursor);
                const auto chord_id = static_cast<std::size_t>(*cursor - TokenLayout::chord_full_begin);
                event.chord = full_chord_labels()[chord_id];
                recognized_payload = true;
                ++cursor;
            } else if (type == TokenLayout::Type::pitch) {
                event.payload_tokens.push_back(*cursor);
                const auto pitch_id = *cursor - TokenLayout::pitch_begin;
                NoteEvent note;
                note.pitch = pitch_id % 128;
                note.track = pitch_id >= 128 ? 1 : 0;
                ++cursor;
                if (cursor != tokens.end() && TokenLayout::type(*cursor) == TokenLayout::Type::duration) {
                    note.duration_bin = *cursor - TokenLayout::duration_begin;
                    event.payload_tokens.push_back(*cursor);
                    ++cursor;
                }
                note.duration_steps = duration_steps[static_cast<std::size_t>(note.duration_bin)];
                event.notes.push_back(note);
                recognized_payload = true;
            } else {
                ++cursor;
            }
        }
        if (std::isfinite(event.time_seconds)) time_anchors[subbeat] = event.time_seconds;
        if (recognized_payload) events.push_back(std::move(event));
    }

    std::vector<std::pair<std::int64_t, double>> anchors(time_anchors.begin(), time_anchors.end());
    double step_seconds = 0.125;
    if (anchors.size() >= 2) {
        std::vector<double> slopes;
        for (std::size_t index = 1; index < anchors.size(); ++index) {
            const auto step_delta = anchors[index].first - anchors[index - 1].first;
            const auto time_delta = anchors[index].second - anchors[index - 1].second;
            if (step_delta > 0 && time_delta > 0.0) slopes.push_back(time_delta / step_delta);
        }
        if (!slopes.empty()) {
            std::sort(slopes.begin(), slopes.end());
            step_seconds = slopes[slopes.size() / 2];
        }
    }
    const auto event_time = [&](std::int64_t step) {
        double time = step * step_seconds;
        if (!anchors.empty()) {
            const auto upper = std::lower_bound(
                anchors.begin(), anchors.end(), step,
                [](const auto & anchor, std::int64_t value) { return anchor.first < value; });
            if (upper == anchors.begin()) {
                time = upper->second + (step - upper->first) * step_seconds;
            } else if (upper == anchors.end()) {
                const auto & last = anchors.back();
                time = last.second + (step - last.first) * step_seconds;
            } else if (upper->first == step) {
                time = upper->second;
            } else {
                const auto & left = *(upper - 1);
                const double fraction = static_cast<double>(step - left.first) /
                    static_cast<double>(upper->first - left.first);
                time = left.second + fraction * (upper->second - left.second);
            }
        }
        return std::clamp(time, 0.0, audio_duration_seconds);
    };
    for (auto & event : events) {
        event.time_seconds = event_time(event.subbeat);
        for (auto & note : event.notes) {
            note.end_time_seconds = std::max(
                event.time_seconds + 0.04,
                event_time(event.subbeat + note.duration_steps));
            note.end_time_seconds = std::min(audio_duration_seconds, note.end_time_seconds);
        }
    }
    return events;
}

std::string abc_pitch(std::int32_t midi) {
    static constexpr std::array<const char *, 12> names = {
        "=C", "^C", "=D", "^D", "=E", "=F", "^F", "=G", "^G", "=A", "^A", "=B",
    };
    midi = std::clamp(midi, 0, 127);
    const int octave = midi / 12 - 1;
    std::string result = names[static_cast<std::size_t>(midi % 12)];
    if (octave >= 5) {
        for (char & c : result) if (c >= 'A' && c <= 'G') c = static_cast<char>(c - 'A' + 'a');
        result.append(static_cast<std::size_t>(octave - 5), '\'');
    } else if (octave < 4) {
        result.append(static_cast<std::size_t>(4 - octave), ',');
    }
    return result;
}

std::string abc_chord(const std::string & label) {
    if (label.empty() || label == "N") return {};
    const auto separator = label.find(':');
    if (separator == std::string::npos) return label;
    const auto root = label.substr(0, separator);
    auto descriptor = label.substr(separator + 1);
    std::string inversion;
    if (const auto slash = descriptor.find('/'); slash != std::string::npos) {
        inversion = descriptor.substr(slash + 1);
        descriptor.resize(slash);
    }
    static const std::map<std::string, std::string> suffixes = {
        {"maj", ""}, {"min", "m"}, {"dim", "dim"}, {"aug", "aug"},
        {"7", "7"}, {"maj7", "maj7"}, {"min7", "m7"}, {"dim7", "dim7"},
        {"hdim7", "m7b5"}, {"sus4", "sus4"}, {"sus2", "sus2"},
        {"maj6", "6"}, {"min6", "m6"}, {"sus4(b7)", "7sus4"},
        {"minmaj7", "m(maj7)"},
    };
    const auto found = suffixes.find(descriptor);
    std::string result = root + (found == suffixes.end() ? descriptor : found->second);
    if (!inversion.empty()) {
        int root_id = 0;
        for (; root_id < 12 && root != chromatic_sharps[static_cast<std::size_t>(root_id)]; ++root_id) {}
        static const std::map<std::string, int> intervals = {
            {"2", 2}, {"b3", 3}, {"3", 4}, {"5", 7}, {"b7", 10}, {"7", 11},
        };
        const auto interval = intervals.find(inversion);
        if (root_id < 12 && interval != intervals.end()) {
            result += "/" + std::string(chromatic_sharps[
                static_cast<std::size_t>((root_id + interval->second) % 12)]);
        }
    }
    return result;
}

std::string abc_key(const std::string & label) {
    if (label.empty()) return {};
    static constexpr std::array<const char *, 12> major_keys = {
        "C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B",
    };
    static constexpr std::array<const char *, 12> minor_keys = {
        "Cm", "C#m", "Dm", "Ebm", "Em", "Fm", "F#m", "Gm", "G#m", "Am", "Bbm", "Bm",
    };
    const auto root = label.substr(0, label.find(':'));
    int root_id = 0;
    for (; root_id < 12 && root != chromatic_sharps[static_cast<std::size_t>(root_id)]; ++root_id) {}
    if (root_id == 12) return {};
    return label.find(":minor") != std::string::npos
        ? minor_keys[static_cast<std::size_t>(root_id)]
        : major_keys[static_cast<std::size_t>(root_id)];
}

constexpr std::array<std::int32_t, 11> abc_duration_units = {
    48, 32, 24, 16, 12, 8, 6, 4, 3, 2, 1,
};

void append_abc_duration(
    std::ostringstream & output,
    const std::string & prefix,
    const std::string & note,
    std::int64_t duration,
    bool tie_out) {
    bool first = true;
    while (duration > 0) {
        const auto found = std::find_if(
            abc_duration_units.begin(), abc_duration_units.end(),
            [duration](std::int32_t value) { return value <= duration; });
        if (found == abc_duration_units.end()) {
            throw std::runtime_error("ABC duration cannot be represented");
        }
        const auto chunk = *found;
        if (first) output << prefix;
        output << note;
        if (chunk != 1) output << chunk;
        if (note != "z" && (duration > chunk || tie_out)) output << '-';
        duration -= chunk;
        first = false;
    }
}

struct AbcCell {
    std::int32_t pitch = -1;
    std::int64_t note_id = -1;
};

struct AbcMeasure {
    std::int64_t start_step = 0;
    std::int64_t end_step = 0;
    std::int32_t numerator = 4;
    std::int32_t denominator = 4;
    std::int64_t pad_before_steps = 0;
    std::int64_t pad_after_steps = 0;
};

bool valid_meter(std::int32_t numerator, std::int32_t denominator) {
    return numerator > 0 && denominator > 0 && (denominator & (denominator - 1)) == 0;
}

std::vector<AbcMeasure> infer_abc_measures(
    const std::vector<ScoreEvent> & events,
    std::int64_t content_steps,
    std::int32_t fallback_numerator,
    std::int32_t fallback_denominator) {
    std::vector<const ScoreEvent *> ordered;
    ordered.reserve(events.size());
    for (const auto & event : events) ordered.push_back(&event);
    std::stable_sort(ordered.begin(), ordered.end(), [](const auto * left, const auto * right) {
        return left->subbeat < right->subbeat;
    });

    std::map<std::int64_t, std::pair<std::int32_t, std::int32_t>> downbeats;
    auto active_meter = std::make_pair(fallback_numerator, fallback_denominator);
    for (std::size_t begin = 0; begin < ordered.size();) {
        auto end = begin + 1;
        while (end < ordered.size() && ordered[end]->subbeat == ordered[begin]->subbeat) ++end;
        for (auto index = begin; index < end; ++index) {
            const auto & event = *ordered[index];
            if (valid_meter(event.meter_numerator, event.meter_denominator)) {
                active_meter = {event.meter_numerator, event.meter_denominator};
            }
        }
        for (auto index = begin; index < end; ++index) {
            const auto eighth = ordered[index]->eighth_position;
            if (eighth < 0) continue;
            const auto scaled = eighth * active_meter.second;
            if (scaled % 8 != 0) continue;
            const auto beat_position = scaled / 8;
            if (beat_position == 0 && beat_position < active_meter.first) {
                downbeats[std::max<std::int64_t>(0, ordered[index]->subbeat)] = active_meter;
            }
        }
        begin = end;
    }

    std::vector<AbcMeasure> result;
    const auto append_span = [&](std::int64_t start,
                                 std::int64_t end,
                                 std::pair<std::int32_t, std::int32_t> meter,
                                 bool pickup,
                                 bool final,
                                 std::vector<AbcMeasure> & measures) {
        if (end <= start) return;
        const auto expected = static_cast<std::int64_t>(meter.first) * 4;
        auto cursor = start;
        auto remaining = end - start;
        if (!pickup && !final && remaining % expected != 0) {
            const auto inferred_numerator = static_cast<std::int32_t>((remaining + 3) / 4);
            measures.push_back({
                start, end, inferred_numerator, meter.second, 0,
                static_cast<std::int64_t>(inferred_numerator) * 4 - remaining,
            });
            return;
        }
        while (remaining >= expected) {
            measures.push_back({cursor, cursor + expected, meter.first, meter.second, 0, 0});
            cursor += expected;
            remaining -= expected;
        }
        if (remaining == 0) return;
        if (pickup) {
            measures.push_back({cursor, end, meter.first, meter.second, expected - remaining, 0});
        } else {
            measures.push_back({cursor, end, meter.first, meter.second, 0, expected - remaining});
        }
    };

    if (downbeats.empty()) {
        append_span(0, std::max<std::int64_t>(1, content_steps),
                    {fallback_numerator, fallback_denominator},
                    false, true, result);
        return result;
    }

    const std::vector<std::pair<std::int64_t, std::pair<std::int32_t, std::int32_t>>> anchors(
        downbeats.begin(), downbeats.end());
    if (anchors.front().first > 0) {
        append_span(0, anchors.front().first, anchors.front().second, true, false, result);
    }
    for (std::size_t index = 0; index < anchors.size(); ++index) {
        const auto start = anchors[index].first;
        const auto end = index + 1 < anchors.size()
            ? anchors[index + 1].first
            : content_steps;
        append_span(start, end, anchors[index].second, false,
                    index + 1 == anchors.size(), result);
    }
    if (result.empty()) {
        const auto & [position, meter] = anchors.back();
        const auto expected = static_cast<std::int64_t>(meter.first) * 4;
        result.push_back({position, position, meter.first, meter.second, 0, expected});
    }
    return result;
}

std::vector<AbcCell> make_voice_cells(
    const std::vector<ScoreEvent> & events,
    std::int32_t track,
    std::int64_t total_units) {
    struct TimedNote {
        std::int64_t start;
        std::int64_t end;
        std::int32_t pitch;
        std::int64_t order;
    };
    std::vector<TimedNote> notes;
    std::int64_t order = 0;
    for (const auto & event : events) {
        for (const auto & note : event.notes) {
            if (note.track != track) continue;
            const auto start = std::max<std::int64_t>(0, event.subbeat);
            notes.push_back({
                start,
                start + std::max(1, note.duration_steps),
                std::clamp(note.pitch, 0, 127),
                order++,
            });
        }
    }
    std::stable_sort(notes.begin(), notes.end(), [](const auto & left, const auto & right) {
        return std::tie(left.start, left.order) < std::tie(right.start, right.order);
    });

    // SheetSage2's native score is monophonic per voice. Keep the first
    // prediction at a shared onset, and clip it at the next distinct onset.
    std::vector<TimedNote> monophonic;
    for (const auto & note : notes) {
        if (!monophonic.empty() && monophonic.back().start == note.start) continue;
        monophonic.push_back(note);
    }
    for (std::size_t index = 0; index + 1 < monophonic.size(); ++index) {
        monophonic[index].end = std::min(monophonic[index].end, monophonic[index + 1].start);
    }

    std::vector<AbcCell> cells(static_cast<std::size_t>(total_units));
    for (const auto & note : monophonic) {
        const auto begin = std::clamp<std::int64_t>(note.start, 0, total_units);
        const auto end = std::clamp<std::int64_t>(note.end, begin, total_units);
        for (auto position = begin; position < end; ++position) {
            cells[static_cast<std::size_t>(position)] = {note.pitch, note.order};
        }
    }
    return cells;
}

std::string render_voice_measure(
    const std::vector<AbcCell> & cells,
    const AbcMeasure & measure,
    std::int32_t unit_denominator,
    const std::map<std::int64_t, std::string> & keys,
    const std::map<std::int64_t, std::string> & chords,
    const std::string & header_key,
    bool show_chords) {
    const auto step_units = unit_denominator / (measure.denominator * 4);
    std::ostringstream output;
    bool has_note = false;
    bool has_annotation = false;
    if (measure.pad_before_steps > 0) {
        append_abc_duration(
            output, {}, "z", measure.pad_before_steps * step_units, false);
    }
    auto position = measure.start_step;
    std::string active_key = header_key;
    for (const auto & [change_position, candidate] : keys) {
        if (change_position >= position) break;
        active_key = candidate;
    }

    while (position < measure.end_step) {
        const auto & cell = cells[static_cast<std::size_t>(position)];
        auto next = position + 1;
        while (next < measure.end_step) {
            const auto & candidate = cells[static_cast<std::size_t>(next)];
            if (candidate.note_id != cell.note_id || candidate.pitch != cell.pitch ||
                keys.find(next) != keys.end() ||
                (show_chords && chords.find(next) != chords.end())) {
                break;
            }
            ++next;
        }

        std::string prefix;
        if (const auto changed_key = keys.find(position);
            changed_key != keys.end() && changed_key->second != active_key) {
            active_key = changed_key->second;
            prefix += "[K:" + active_key + ']';
        }
        if (show_chords) {
            if (const auto chord = chords.find(position); chord != chords.end()) {
                const auto text = abc_chord(chord->second);
                if (!text.empty()) prefix += '"' + text + '"';
            }
        }
        has_note = has_note || cell.note_id >= 0;
        has_annotation = has_annotation || !prefix.empty();

        const bool tie_out = measure.pad_after_steps == 0 && cell.note_id >= 0 &&
            next < static_cast<std::int64_t>(cells.size()) &&
            cells[static_cast<std::size_t>(next)].note_id == cell.note_id;
        append_abc_duration(
            output,
            prefix,
            cell.note_id < 0 ? "z" : abc_pitch(cell.pitch),
            (next - position) * step_units,
            tie_out);
        position = next;
    }
    if (measure.pad_after_steps > 0) {
        append_abc_duration(
            output, {}, "z", measure.pad_after_steps * step_units, false);
    }
    const auto rendered = output.str();
    if (!has_note && !has_annotation) return "Z";
    return rendered;
}

std::string sanitize_structure(const std::string & label) {
    std::string clean;
    bool space = false;
    for (const unsigned char character : label) {
        if (std::isspace(character)) {
            space = !clean.empty();
        } else {
            if (space) clean.push_back(' ');
            clean.push_back(static_cast<char>(character));
            space = false;
        }
    }
    return clean;
}

double infer_quarter_bpm(
    const std::vector<ScoreEvent> & events,
    std::int32_t denominator) {
    std::vector<double> seconds_per_step;
    const ScoreEvent * previous_anchor = nullptr;
    for (const auto & event : events) {
        if (!event.has_timestamp) continue;
        if (previous_anchor && event.subbeat > previous_anchor->subbeat &&
            event.time_seconds > previous_anchor->time_seconds) {
            seconds_per_step.push_back(
                (event.time_seconds - previous_anchor->time_seconds) /
                static_cast<double>(event.subbeat - previous_anchor->subbeat));
        }
        previous_anchor = &event;
    }
    if (seconds_per_step.empty()) return 120.0;
    std::sort(seconds_per_step.begin(), seconds_per_step.end());
    const auto seconds = seconds_per_step[seconds_per_step.size() / 2];
    return std::clamp(60.0 / (seconds * std::max(1, denominator)), 30.0, 300.0);
}

std::string make_abc(const std::vector<ScoreEvent> & events, bool melody_only) {
    int numerator = 4;
    int denominator = 4;
    for (const auto & event : events) {
        if (event.meter_numerator > 0 && event.meter_denominator > 0) {
            numerator = event.meter_numerator;
            denominator = event.meter_denominator;
            break;
        }
    }
    std::string key = "C";
    for (const auto & event : events) {
        const auto candidate = abc_key(event.key);
        if (!candidate.empty()) { key = candidate; break; }
    }
    const double tempo = infer_quarter_bpm(events, denominator);

    std::int64_t content_steps = 0;
    std::map<std::int64_t, std::string> keys;
    std::map<std::int64_t, std::string> chords;
    std::map<std::int64_t, std::string> structures;
    for (const auto & event : events) {
        const auto position = std::max<std::int64_t>(0, event.subbeat);
        content_steps = std::max(content_steps, position);
        const auto event_key = abc_key(event.key);
        if (!event_key.empty()) {
            keys[position] = event_key;
            content_steps = std::max(content_steps, position + 1);
        }
        if (!event.chord.empty()) {
            chords[position] = event.chord;
            content_steps = std::max(content_steps, position + 1);
        }
        if (!event.structure.empty()) {
            structures[position] = event.structure;
            content_steps = std::max(content_steps, position + 1);
        }
        for (const auto & note : event.notes) {
            content_steps = std::max(content_steps, position + std::max(1, note.duration_steps));
        }
    }
    const auto measures = infer_abc_measures(
        events, content_steps, numerator, denominator);
    if (measures.empty()) throw std::runtime_error("ABC measure inference produced no measures");
    std::int32_t unit_denominator = 1;
    std::int64_t cell_steps = content_steps;
    for (const auto & measure : measures) {
        unit_denominator = std::max(unit_denominator, measure.denominator * 4);
        cell_steps = std::max(cell_steps, measure.end_step);
    }
    const auto vocal = make_voice_cells(events, 0, cell_steps);
    const auto instrumental = make_voice_cells(events, 1, cell_steps);

    const auto measure_count = measures.size();
    std::map<std::int64_t, std::vector<std::string>> structure_labels;
    std::string active_structure;
    for (const auto & [position, label] : structures) {
        const auto clean = sanitize_structure(label);
        if (clean.empty() || clean == active_structure) continue;
        active_structure = clean;
        std::size_t measure_index = 0;
        while (measure_index + 1 < measure_count &&
               position >= measures[measure_index].end_step) {
            ++measure_index;
        }
        structure_labels[static_cast<std::int64_t>(measure_index)].push_back(clean);
    }
    std::vector<std::string> vocal_measures;
    std::vector<std::string> instrumental_measures;
    vocal_measures.reserve(measure_count);
    instrumental_measures.reserve(measure_count);
    for (const auto & measure : measures) {
        vocal_measures.push_back(render_voice_measure(
            vocal, measure, unit_denominator, keys, chords, key, !melody_only));
        instrumental_measures.push_back(render_voice_measure(
            instrumental, measure, unit_denominator, keys, chords, key, false));
    }

    const auto & first_measure = measures.front();
    std::ostringstream abc;
    abc << "X:1\nT:\nM:" << first_measure.numerator << '/' << first_measure.denominator
        << "\nL:1/" << unit_denominator << "\nQ:1/4=" << std::llround(tempo)
        << "\nV: Vocal clef=treble name=\"Vocal Melody\" snm=\"Vocal\""
        << "\nV: Ins clef=treble name=\"Ins Melody\" snm=\"Inst.\""
        << "\nK:" << key << '\n';
    for (std::size_t group_start = 0; group_start < measure_count;) {
        auto group_end = std::min(measure_count, group_start + 4);
        for (auto probe = group_start + 1; probe < group_end; ++probe) {
            const bool meter_changed =
                measures[probe].numerator != measures[group_start].numerator ||
                measures[probe].denominator != measures[group_start].denominator;
            if (meter_changed || structure_labels.find(static_cast<std::int64_t>(probe)) != structure_labels.end()) {
                group_end = probe;
                break;
            }
        }
        if (const auto labels = structure_labels.find(static_cast<std::int64_t>(group_start));
            labels != structure_labels.end()) {
            for (const auto & label : labels->second) abc << "% " << label << '\n';
        }
        abc << "V: Vocal\n";
        const bool meter_changed = group_start > 0 &&
            (measures[group_start].numerator != measures[group_start - 1].numerator ||
             measures[group_start].denominator != measures[group_start - 1].denominator);
        if (meter_changed) {
            abc << "M:" << measures[group_start].numerator << '/'
                << measures[group_start].denominator << '\n';
        }
        for (auto measure = group_start; measure < group_end; ++measure) {
            abc << vocal_measures[measure] << '|';
        }
        abc << "\nV: Ins\n";
        if (meter_changed) {
            abc << "M:" << measures[group_start].numerator << '/'
                << measures[group_start].denominator << '\n';
        }
        for (auto measure = group_start; measure < group_end; ++measure) {
            abc << instrumental_measures[measure] << '|';
        }
        abc << '\n';
        group_start = group_end;
    }
    return abc.str();
}

void append_u32be(std::vector<std::uint8_t> & out, std::uint32_t value) {
    out.push_back(static_cast<std::uint8_t>(value >> 24));
    out.push_back(static_cast<std::uint8_t>(value >> 16));
    out.push_back(static_cast<std::uint8_t>(value >> 8));
    out.push_back(static_cast<std::uint8_t>(value));
}

void append_u16be(std::vector<std::uint8_t> & out, std::uint16_t value) {
    out.push_back(static_cast<std::uint8_t>(value >> 8));
    out.push_back(static_cast<std::uint8_t>(value));
}

void append_vlq(std::vector<std::uint8_t> & out, std::uint32_t value) {
    std::uint8_t bytes[5];
    int count = 0;
    bytes[count++] = static_cast<std::uint8_t>(value & 0x7fU);
    while ((value >>= 7U) != 0) bytes[count++] = static_cast<std::uint8_t>((value & 0x7fU) | 0x80U);
    while (count-- > 0) out.push_back(bytes[count]);
}

constexpr std::uint16_t midi_ppq = 960;

struct TimedMidiEvent {
    std::uint32_t tick = 0;
    std::int32_t order = 0;
    std::vector<std::uint8_t> bytes;
};

void add_midi_meta(
    std::vector<TimedMidiEvent> & events,
    std::uint32_t tick,
    std::int32_t order,
    std::uint8_t type,
    const std::vector<std::uint8_t> & payload) {
    std::vector<std::uint8_t> bytes = {0xff, type};
    append_vlq(bytes, static_cast<std::uint32_t>(payload.size()));
    bytes.insert(bytes.end(), payload.begin(), payload.end());
    events.push_back({tick, order, std::move(bytes)});
}

void add_midi_text(
    std::vector<TimedMidiEvent> & events,
    std::uint32_t tick,
    std::int32_t order,
    std::uint8_t type,
    const std::string & text) {
    add_midi_meta(events, tick, order, type,
                  std::vector<std::uint8_t>(text.begin(), text.end()));
}

std::vector<std::uint8_t> make_midi_track(
    const std::string & name,
    std::vector<TimedMidiEvent> events) {
    add_midi_text(events, 0, -100, 0x03, name);
    std::stable_sort(events.begin(), events.end(), [](const auto & left, const auto & right) {
        return std::tie(left.tick, left.order) < std::tie(right.tick, right.order);
    });
    std::vector<std::uint8_t> track;
    std::uint32_t previous = 0;
    for (const auto & event : events) {
        append_vlq(track, event.tick - previous);
        previous = event.tick;
        track.insert(track.end(), event.bytes.begin(), event.bytes.end());
    }
    track.insert(track.end(), {0x00, 0xff, 0x2f, 0x00});
    std::vector<std::uint8_t> result = {'M','T','r','k'};
    append_u32be(result, static_cast<std::uint32_t>(track.size()));
    result.insert(result.end(), track.begin(), track.end());
    return result;
}

struct MidiTimeline {
    std::vector<AbcMeasure> measures;
    std::vector<std::uint32_t> measure_ticks;
    std::uint32_t end_tick = 0;

    std::uint32_t tick(std::int64_t step) const {
        if (measures.empty()) return 0;
        step = std::max<std::int64_t>(0, step);
        for (std::size_t index = 0; index < measures.size(); ++index) {
            const auto & measure = measures[index];
            if (step < measure.start_step ||
                (step >= measure.end_step && index + 1 < measures.size())) {
                continue;
            }
            const auto ticks_per_step = midi_ppq /
                static_cast<std::uint32_t>(std::max(1, measure.denominator));
            const auto offset_steps = measure.pad_before_steps +
                std::clamp<std::int64_t>(
                    step - measure.start_step, 0, measure.end_step - measure.start_step);
            const auto value = static_cast<std::uint64_t>(measure_ticks[index]) +
                static_cast<std::uint64_t>(offset_steps) * ticks_per_step;
            return static_cast<std::uint32_t>(std::min<std::uint64_t>(value, end_tick));
        }
        return end_tick;
    }
};

MidiTimeline make_midi_timeline(
    const std::vector<ScoreEvent> & events,
    double duration_seconds,
    double bpm,
    std::int32_t first_denominator) {
    std::int32_t numerator = 4;
    std::int32_t denominator = 4;
    bool found_meter = false;
    std::int64_t content_steps = 1;
    for (const auto & event : events) {
        if (!found_meter && valid_meter(event.meter_numerator, event.meter_denominator)) {
            numerator = event.meter_numerator;
            denominator = event.meter_denominator;
            found_meter = true;
        }
        content_steps = std::max(content_steps, std::max<std::int64_t>(0, event.subbeat) + 1);
        for (const auto & note : event.notes) {
            content_steps = std::max(
                content_steps,
                std::max<std::int64_t>(0, event.subbeat) + std::max(1, note.duration_steps));
        }
    }
    if (std::isfinite(duration_seconds) && duration_seconds > 0.0 && bpm > 0.0) {
        const ScoreEvent * previous_anchor = nullptr;
        const ScoreEvent * last_anchor = nullptr;
        for (const auto & event : events) {
            if (!event.has_timestamp || !std::isfinite(event.time_seconds)) continue;
            if (!last_anchor || std::tie(event.time_seconds, event.subbeat) >
                                std::tie(last_anchor->time_seconds, last_anchor->subbeat)) {
                previous_anchor = last_anchor;
                last_anchor = &event;
            }
        }
        double duration_steps_value =
            duration_seconds * bpm * std::max(1, first_denominator) / 60.0;
        if (last_anchor && previous_anchor &&
            last_anchor->subbeat > previous_anchor->subbeat &&
            last_anchor->time_seconds > previous_anchor->time_seconds &&
            duration_seconds > last_anchor->time_seconds) {
            const auto seconds_per_step =
                (last_anchor->time_seconds - previous_anchor->time_seconds) /
                static_cast<double>(last_anchor->subbeat - previous_anchor->subbeat);
            duration_steps_value = last_anchor->subbeat +
                (duration_seconds - last_anchor->time_seconds) / seconds_per_step;
        }
        const auto duration_steps = static_cast<std::int64_t>(
            std::ceil(duration_steps_value - 1.0e-6));
        content_steps = std::max(content_steps, duration_steps);
    }
    MidiTimeline result;
    result.measures = infer_abc_measures(
        events, content_steps, numerator, denominator);
    std::uint64_t tick = 0;
    for (const auto & measure : result.measures) {
        result.measure_ticks.push_back(static_cast<std::uint32_t>(tick));
        const auto ticks_per_step = midi_ppq /
            static_cast<std::uint32_t>(std::max(1, measure.denominator));
        const auto steps = measure.pad_before_steps +
            (measure.end_step - measure.start_step) + measure.pad_after_steps;
        tick += static_cast<std::uint64_t>(std::max<std::int64_t>(0, steps)) * ticks_per_step;
        if (tick > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("SheetSage2 MIDI timeline is too long");
        }
    }
    result.end_tick = static_cast<std::uint32_t>(tick);
    return result;
}

std::pair<std::int8_t, bool> midi_key_signature(const std::string & label) {
    static constexpr std::array<std::int8_t, 12> major = {
        0, -5, 2, -3, 4, -1, -6, 1, -4, 3, -2, 5,
    };
    static constexpr std::array<std::int8_t, 12> minor = {
        -3, 4, -1, -6, 1, -4, 3, -2, 5, 0, -5, 2,
    };
    const auto root = label.substr(0, label.find(':'));
    std::size_t root_id = 0;
    while (root_id < chromatic_sharps.size() && root != chromatic_sharps[root_id]) ++root_id;
    if (root_id == chromatic_sharps.size()) return {0, false};
    const bool is_minor = label.find(":minor") != std::string::npos;
    return {is_minor ? minor[root_id] : major[root_id], is_minor};
}

std::vector<std::int32_t> chord_pitches(const std::string & label) {
    if (label.empty() || label == "N" || label == "X" || label == "?") return {};
    const auto colon = label.find(':');
    if (colon == std::string::npos) return {};
    const auto root_name = label.substr(0, colon);
    std::int32_t root = 0;
    while (root < static_cast<std::int32_t>(chromatic_sharps.size()) &&
           root_name != chromatic_sharps[static_cast<std::size_t>(root)]) ++root;
    if (root == static_cast<std::int32_t>(chromatic_sharps.size())) return {};
    auto quality = label.substr(colon + 1);
    std::string inversion;
    if (const auto slash = quality.find('/'); slash != std::string::npos) {
        inversion = quality.substr(slash + 1);
        quality.resize(slash);
    }
    std::vector<std::int32_t> intervals;
    if (quality == "maj") intervals = {0, 4, 7};
    else if (quality == "min") intervals = {0, 3, 7};
    else if (quality == "dim") intervals = {0, 3, 6};
    else if (quality == "aug") intervals = {0, 4, 8};
    else if (quality == "maj7") intervals = {0, 4, 7, 11};
    else if (quality == "min7") intervals = {0, 3, 7, 10};
    else if (quality == "7") intervals = {0, 4, 7, 10};
    else if (quality == "hdim7") intervals = {0, 3, 6, 10};
    else if (quality == "dim7") intervals = {0, 3, 6, 9};
    else if (quality == "minmaj7") intervals = {0, 3, 7, 11};
    else if (quality == "sus2") intervals = {0, 2, 7};
    else if (quality == "sus4") intervals = {0, 5, 7};
    else if (quality == "sus4(b7)") intervals = {0, 5, 7, 10};
    else if (quality == "maj6") intervals = {0, 4, 7, 9};
    else if (quality == "min6") intervals = {0, 3, 7, 9};
    else return {};
    static const std::map<std::string, std::int32_t> bass_intervals = {
        {"2", 2}, {"3", 4}, {"b3", 3}, {"5", 7}, {"7", 11}, {"b7", 10},
    };
    auto bass = 0;
    if (const auto found = bass_intervals.find(inversion); found != bass_intervals.end()) {
        bass = found->second;
    }
    std::vector<std::int32_t> pitches = {36 + (root + bass) % 12};
    for (const auto interval : intervals) pitches.push_back(48 + root + interval);
    std::sort(pitches.begin(), pitches.end());
    pitches.erase(std::unique(pitches.begin(), pitches.end()), pitches.end());
    return pitches;
}

std::vector<std::uint8_t> assemble_midi(
    const std::vector<const std::vector<std::uint8_t> *> & tracks) {
    std::vector<std::uint8_t> result = {'M','T','h','d'};
    append_u32be(result, 6);
    append_u16be(result, 1);
    append_u16be(result, static_cast<std::uint16_t>(tracks.size()));
    append_u16be(result, midi_ppq);
    for (const auto * track : tracks) result.insert(result.end(), track->begin(), track->end());
    return result;
}

TranscriptionMidiExports make_midis(
    const std::vector<ScoreEvent> & events,
    bool melody_only,
    double duration_seconds) {
    std::int32_t first_denominator = 4;
    for (const auto & event : events) {
        if (valid_meter(event.meter_numerator, event.meter_denominator)) {
            first_denominator = event.meter_denominator;
            break;
        }
    }
    const auto bpm = std::max<std::int64_t>(1, std::llround(
        infer_quarter_bpm(events, first_denominator)));
    const auto timeline = make_midi_timeline(
        events, duration_seconds, static_cast<double>(bpm), first_denominator);
    const auto tempo = static_cast<std::uint32_t>(std::clamp<std::int64_t>(
        std::llround(60000000.0 / static_cast<double>(bpm)), 1, 0xffffff));

    std::vector<TimedMidiEvent> conductor;
    add_midi_meta(conductor, 0, -90, 0x51, {
        static_cast<std::uint8_t>(tempo >> 16),
        static_cast<std::uint8_t>(tempo >> 8),
        static_cast<std::uint8_t>(tempo),
    });
    std::pair<std::int32_t, std::int32_t> last_meter = {-1, -1};
    for (std::size_t index = 0; index < timeline.measures.size(); ++index) {
        const auto & measure = timeline.measures[index];
        const auto meter = std::make_pair(measure.numerator, measure.denominator);
        if (meter == last_meter) continue;
        last_meter = meter;
        std::uint8_t power = 0;
        auto denominator = static_cast<std::uint32_t>(std::max(1, measure.denominator));
        while (denominator > 1) { denominator >>= 1U; ++power; }
        add_midi_meta(conductor, timeline.measure_ticks[index], -80, 0x58, {
            static_cast<std::uint8_t>(std::clamp(measure.numerator, 1, 255)),
            power, 24, 8,
        });
    }
    std::string active_key;
    std::string active_structure;
    for (const auto & event : events) {
        const auto tick = timeline.tick(event.subbeat);
        if (!event.key.empty() && event.key != active_key) {
            active_key = event.key;
            const auto [signature, minor] = midi_key_signature(event.key);
            add_midi_meta(conductor, tick, -70, 0x59, {
                static_cast<std::uint8_t>(signature), static_cast<std::uint8_t>(minor),
            });
        }
        const auto structure = sanitize_structure(event.structure);
        if (!structure.empty() && structure != active_structure) {
            active_structure = structure;
            add_midi_text(conductor, tick, -60, 0x06, structure);
        }
    }

    std::array<std::vector<TimedMidiEvent>, 2> melody_tracks;
    for (const auto & event : events) {
        const auto start = timeline.tick(event.subbeat);
        for (const auto & note : event.notes) {
            if (note.track < 0 || note.track >= static_cast<std::int32_t>(melody_tracks.size())) continue;
            auto end = timeline.tick(event.subbeat + std::max(1, note.duration_steps));
            end = std::max(start + 1U, end);
            const auto pitch = static_cast<std::uint8_t>(std::clamp(note.pitch, 0, 127));
            const auto channel = static_cast<std::uint8_t>(note.track);
            melody_tracks[static_cast<std::size_t>(note.track)].push_back({
                start, 10, {static_cast<std::uint8_t>(0x90 | channel), pitch, 100},
            });
            melody_tracks[static_cast<std::size_t>(note.track)].push_back({
                end, 0, {static_cast<std::uint8_t>(0x80 | channel), pitch, 0},
            });
        }
    }

    std::vector<TimedMidiEvent> chord_track;
    if (!melody_only) {
        std::map<std::int64_t, std::string> chords;
        for (const auto & event : events) {
            if (!event.chord.empty()) chords[std::max<std::int64_t>(0, event.subbeat)] = event.chord;
        }
        std::vector<std::pair<std::int64_t, std::string>> ordered(chords.begin(), chords.end());
        for (std::size_t index = 0; index < ordered.size(); ++index) {
            const auto start_step = ordered[index].first;
            const auto end_step = index + 1 < ordered.size()
                ? ordered[index + 1].first
                : (timeline.measures.empty() ? start_step + 1 : timeline.measures.back().end_step);
            const auto label = ordered[index].second;
            add_midi_text(chord_track, timeline.tick(start_step), -20, 0x01, label);
            const auto pitches = chord_pitches(label);
            if (pitches.empty()) continue;
            std::vector<std::int64_t> cuts = {start_step};
            for (const auto & measure : timeline.measures) {
                if (measure.start_step > start_step && measure.start_step < end_step) {
                    cuts.push_back(measure.start_step);
                }
            }
            cuts.push_back(end_step);
            for (std::size_t cut = 0; cut + 1 < cuts.size(); ++cut) {
                const auto start = timeline.tick(cuts[cut]);
                auto end = timeline.tick(cuts[cut + 1]);
                end = std::max(start + 1U, end);
                for (const auto value : pitches) {
                    const auto pitch = static_cast<std::uint8_t>(std::clamp(value, 0, 127));
                    chord_track.push_back({start, 10, {0x92, pitch, 48}});
                    chord_track.push_back({end, 0, {0x82, pitch, 0}});
                }
            }
        }
    }

    const auto conductor_file = make_midi_track("Conductor", std::move(conductor));
    std::vector<std::uint8_t> vocal_file;
    std::vector<std::uint8_t> instrumental_file;
    std::vector<std::uint8_t> chord_file;
    if (!melody_tracks[0].empty()) {
        melody_tracks[0].push_back({0, -90, {0xc0, 0}});
        vocal_file = make_midi_track("Vocal Melody", std::move(melody_tracks[0]));
    }
    if (!melody_tracks[1].empty()) {
        melody_tracks[1].push_back({0, -90, {0xc1, 0}});
        instrumental_file = make_midi_track("Instrument Melody", std::move(melody_tracks[1]));
    }
    if (!melody_only && !chord_track.empty()) {
        chord_track.push_back({0, -90, {0xc2, 0}});
        chord_file = make_midi_track("Chords", std::move(chord_track));
    }

    std::vector<const std::vector<std::uint8_t> *> melody_files = {&conductor_file};
    if (!vocal_file.empty()) melody_files.push_back(&vocal_file);
    if (!instrumental_file.empty()) melody_files.push_back(&instrumental_file);
    auto transcription_files = melody_files;
    if (!chord_file.empty()) transcription_files.push_back(&chord_file);

    TranscriptionMidiExports result;
    result.transcription = assemble_midi(transcription_files);
    result.melody = assemble_midi(melody_files);
    result.vocal = assemble_midi(vocal_file.empty()
        ? std::vector<const std::vector<std::uint8_t> *>{&conductor_file}
        : std::vector<const std::vector<std::uint8_t> *>{&conductor_file, &vocal_file});
    result.instrumental = assemble_midi(instrumental_file.empty()
        ? std::vector<const std::vector<std::uint8_t> *>{&conductor_file}
        : std::vector<const std::vector<std::uint8_t> *>{&conductor_file, &instrumental_file});
    if (!melody_only) {
        result.chords = assemble_midi(chord_file.empty()
            ? std::vector<const std::vector<std::uint8_t> *>{&conductor_file}
            : std::vector<const std::vector<std::uint8_t> *>{&conductor_file, &chord_file});
    }
    return result;
}

} // namespace

std::string serialize_sheetsage2_abc(
    const std::vector<ScoreEvent> & events,
    bool melody_only) {
    return make_abc(events, melody_only);
}

std::vector<std::uint8_t> serialize_sheetsage2_midi(
    const std::vector<ScoreEvent> & events,
    bool melody_only,
    double duration_seconds) {
    return make_midis(events, melody_only, duration_seconds).transcription;
}

TranscriptionMidiExports serialize_sheetsage2_midis(
    const std::vector<ScoreEvent> & events,
    bool melody_only,
    double duration_seconds) {
    return make_midis(events, melody_only, duration_seconds);
}

std::vector<TranscriptionWindow> make_transcription_window_plan(
    double duration_seconds,
    double window_seconds,
    double overlap_seconds,
    double lookahead_seconds) {
    if (!std::isfinite(duration_seconds) || !std::isfinite(window_seconds) ||
        !std::isfinite(overlap_seconds) || !std::isfinite(lookahead_seconds) ||
        duration_seconds <= 0.0 || window_seconds <= 0.0) {
        throw std::invalid_argument("duration and window length must be finite and positive");
    }
    if (lookahead_seconds < 0.0 || lookahead_seconds > overlap_seconds ||
        overlap_seconds >= window_seconds) {
        throw std::invalid_argument("require 0 <= lookahead <= overlap < window length");
    }
    const double hop = window_seconds - overlap_seconds;
    double start = 0.0;
    double accepted = 0.0;
    std::vector<TranscriptionWindow> result;
    while (true) {
        const bool last = start + window_seconds >= duration_seconds - 1.0e-6;
        const double accept_end = last
            ? duration_seconds
            : start + window_seconds - lookahead_seconds;
        result.push_back({
            start,
            std::min(duration_seconds, start + window_seconds),
            accepted,
            accept_end,
            accepted,
            last ? -1.0 : window_seconds - lookahead_seconds,
        });
        if (last) return result;
        accepted = accept_end;
        // Keep a fixed hop even when the final segment is shorter than the
        // model window. Back-shifting the tail to force a full input window
        // expands its overlap prefix and can exhaust the 5120-token decoder
        // context before reaching the end of the recording.
        start += hop;
    }
}

std::vector<ScoreEvent> decode_sheetsage2_tokens(
    const std::vector<std::int32_t> & tokens,
    double audio_duration_seconds) {
    if (!std::isfinite(audio_duration_seconds) || audio_duration_seconds < 0.0) {
        throw std::invalid_argument("audio duration must be finite and nonnegative");
    }
    return decode_events_impl(tokens, audio_duration_seconds);
}

struct Transcriber::Impl {
    Impl(std::filesystem::path path, const TranscriberRuntimeOptions & runtime_options)
        : model_path(std::move(path)),
          model(model_path.string(), mert2::EncoderOptions{runtime_options.device, runtime_options.threads}) {}
    std::filesystem::path model_path;
    mert2::Encoder model;
    std::mutex inference_mutex;
};

Transcriber::Transcriber(const std::filesystem::path & model_path)
    : Transcriber(model_path, {}) {}

Transcriber::Transcriber(
    const std::filesystem::path & model_path,
    const TranscriberRuntimeOptions & runtime_options)
    : impl_(nullptr) {
    if (model_path.empty()) {
        throw std::invalid_argument("SheetSage2 GGUF model path must not be empty");
    }
    if (!std::filesystem::is_regular_file(model_path)) {
        throw std::runtime_error("SheetSage2 GGUF model does not exist: " + model_path.string());
    }
    if (runtime_options.threads < 0) {
        throw std::invalid_argument("transcriber thread count must be nonnegative");
    }
    impl_ = std::make_unique<Impl>(model_path, runtime_options);
}

Transcriber::~Transcriber() = default;
Transcriber::Transcriber(Transcriber &&) noexcept = default;
Transcriber & Transcriber::operator=(Transcriber &&) noexcept = default;

TranscriptionResult Transcriber::transcribe(
    const std::filesystem::path & audio_path,
    const TranscriptionOptions & options) {
    const auto decoded_audio = audio::read_wav_mono(audio_path);
    return transcribe_mono(
        decoded_audio.samples.data(), decoded_audio.samples.size(),
        decoded_audio.sample_rate, options);
}

TranscriptionResult Transcriber::transcribe_mono(
    const float * samples,
    std::size_t sample_count,
    std::int32_t sample_rate,
    const TranscriptionOptions & options,
    const TranscriptionControl & control) {
    if (control.should_cancel && control.should_cancel()) {
        throw std::runtime_error("YuE2 transcription cancelled");
    }
    if (!std::isfinite(options.window_seconds) ||
        options.window_seconds <= 0.0F || options.window_seconds > 300.0F) {
        throw std::invalid_argument("SheetSage2 window seconds must be in (0,300]");
    }
    if (!std::isfinite(options.overlap_seconds) || !std::isfinite(options.lookahead_seconds) ||
        options.overlap_seconds < 0.0F || options.lookahead_seconds < 0.0F) {
        throw std::invalid_argument("SheetSage2 overlap and lookahead must be finite and nonnegative");
    }
    if (options.preset == TranscriptionPreset::paper) {
        throw std::invalid_argument(
            "the SheetSage2 paper preset is not implemented in the native exporter yet");
    }
    const std::vector<std::int32_t> prefix = options.melody_only
        ? std::vector<std::int32_t>{1, 4, 11, 3}
        : std::vector<std::int32_t>{1, 4, 5, 6, 7, 9, 11, 3};
    if (options.max_tokens <= prefix.size() || options.max_tokens > 5120) {
        throw std::invalid_argument(
            "SheetSage2 max token count must exceed the prompt prefix and be at most 5120");
    }
    if (!samples || sample_count == 0 || sample_rate <= 0) {
        throw std::invalid_argument("mono PCM must be non-empty with a positive sample rate");
    }
    std::vector<float> source(samples, samples + sample_count);
    for (const auto sample : source) {
        if (!std::isfinite(sample)) throw std::invalid_argument("mono PCM contains a non-finite sample");
    }
    const double duration = static_cast<double>(sample_count) / sample_rate;
    auto waveform = sample_rate == audio::transcription_sample_rate
        ? std::move(source)
        : audio::resample_sinc(source, sample_rate, audio::transcription_sample_rate);
    std::size_t window_samples = static_cast<std::size_t>(std::llround(
        options.window_seconds * audio::transcription_sample_rate));
    window_samples = ((window_samples + 959) / 960) * 960;
    if (window_samples <= 1024) {
        throw std::invalid_argument("SheetSage2 window is too short for the MERT2 front end");
    }
    const double model_window_seconds =
        static_cast<double>(window_samples) / audio::transcription_sample_rate;
    const bool single_window = waveform.size() <= window_samples;
    const auto plan = make_transcription_window_plan(
        duration, model_window_seconds,
        single_window ? 0.0 : options.overlap_seconds,
        single_window ? 0.0 : options.lookahead_seconds);
    TranscriptionResult result;
    result.duration_seconds = duration;
    std::vector<ScoreEvent> stitched;
    // GGML schedulers retain allocation state between graphs. Calls sharing
    // one model instance are safe, but execute serially on that instance.
    std::lock_guard<std::mutex> inference_lock(impl_->inference_mutex);
    for (std::size_t window_index = 0; window_index < plan.size(); ++window_index) {
        if (control.should_cancel && control.should_cancel()) {
            throw std::runtime_error("YuE2 transcription cancelled");
        }
        const auto & window = plan[window_index];
        const auto start_sample = static_cast<std::size_t>(std::llround(
            window.start_seconds * audio::transcription_sample_rate));
        std::vector<float> segment(window_samples, 0.0F);
        if (start_sample < waveform.size()) {
            const auto count = std::min(window_samples, waveform.size() - start_sample);
            std::copy_n(waveform.begin() + static_cast<std::ptrdiff_t>(start_sample), count, segment.begin());
        }

        auto decoder_prefix = prefix;
        std::int64_t global_base = 0;
        std::size_t prefix_tokens = 0;
        if (window_index > 0) {
            const auto overlap = build_overlap_prefix(
                stitched, prefix, window.start_seconds, window.prefix_end_seconds);
            if (overlap.available) {
                decoder_prefix = overlap.tokens;
                global_base = overlap.global_subbeat_base;
                prefix_tokens = decoder_prefix.size();
                const auto reserve = std::min<std::size_t>(128, options.max_tokens / 4);
                if (decoder_prefix.size() + reserve >= options.max_tokens) {
                    throw std::runtime_error(
                        "SheetSage2 overlap prefix fills the decoder context; reduce overlap_seconds");
                }
            } else {
                result.warnings.push_back(
                    "window " + std::to_string(window_index + 1) +
                    " had no beat/timestamp overlap prefix; subbeat continuity was inferred");
            }
        }

        const auto log_mel = mert2::log_mel_spectrogram(segment);
        const auto memory = impl_->model.sheetsage_memory(log_mel);
        const double stop_time = window.generation_stop_seconds >= 0.0
            ? window.generation_stop_seconds
            : std::min(duration - window.start_seconds, model_window_seconds);
        const auto tokens = impl_->model.generate_tokens(
            memory, decoder_prefix, options.max_tokens, stop_time);
        auto decoded = decode_sheetsage2_tokens(tokens, model_window_seconds);

        std::vector<ScoreEvent> accepted;
        for (auto & event : decoded) {
            const double absolute_time = window.start_seconds + event.time_seconds;
            if (absolute_time < window.accept_start_seconds - 1.0e-4 ||
                absolute_time >= window.accept_end_seconds - 1.0e-4 ||
                absolute_time >= duration - 1.0e-4) {
                continue;
            }
            event.source_subbeat = event.subbeat;
            event.window_index = window_index;
            event.time_seconds = std::clamp(absolute_time, 0.0, duration);
            for (auto & note : event.notes) {
                note.end_time_seconds = std::clamp(
                    window.start_seconds + note.end_time_seconds,
                    event.time_seconds,
                    duration);
            }
            accepted.push_back(std::move(event));
        }
        if (window_index > 0 && prefix_tokens == 0 && !accepted.empty() && !stitched.empty()) {
            global_base = stitched.back().subbeat + 1 - accepted.front().source_subbeat;
        }
        for (auto & event : accepted) {
            event.subbeat = global_base + event.source_subbeat;
            stitched.push_back(std::move(event));
        }
        std::sort(stitched.begin(), stitched.end(), [](const ScoreEvent & left, const ScoreEvent & right) {
            return std::tie(left.time_seconds, left.subbeat) <
                std::tie(right.time_seconds, right.subbeat);
        });
        if (tokens.size() > options.max_tokens) {
            result.warnings.push_back(
                "window " + std::to_string(window_index + 1) +
                " reached the token limit; inspect its timestamp coverage");
        }
        result.windows.push_back({
            window, tokens, prefix_tokens, decoded.size(), accepted.size(),
        });
        if (control.on_progress) control.on_progress(window_index + 1, plan.size());
    }
    if (result.windows.size() == 1) result.tokens = result.windows.front().tokens;
    result.events = std::move(stitched);
    result.abc = serialize_sheetsage2_abc(result.events, options.melody_only);
    result.midi_exports = serialize_sheetsage2_midis(
        result.events, options.melody_only, result.duration_seconds);
    result.midi = result.midi_exports.transcription;
    if (options.melody_only) {
        result.warnings.push_back(
            "melody-only prompts omit meter and key; ABC uses a 4/4, C-major fallback header while retaining both melody voices");
    }
    return result;
}

std::string serialize_transcription_json(const TranscriptionResult & result) {
    auto json_string = [](const std::string & value) {
        std::ostringstream escaped;
        escaped << '"';
        for (const unsigned char character : value) {
            switch (character) {
                case '"': escaped << "\\\""; break;
                case '\\': escaped << "\\\\"; break;
                case '\n': escaped << "\\n"; break;
                case '\r': escaped << "\\r"; break;
                case '\t': escaped << "\\t"; break;
                default:
                    if (character < 0x20) {
                        escaped << "\\u" << std::hex << std::setw(4)
                                << std::setfill('0') << static_cast<int>(character)
                                << std::dec;
                    } else {
                        escaped << character;
                    }
            }
        }
        escaped << '"';
        return escaped.str();
    };

    std::ostringstream output;
    output << std::setprecision(10) << "{\n  \"duration_seconds\": "
           << result.duration_seconds << ",\n  \"tokens\": [";
    for (std::size_t index = 0; index < result.tokens.size(); ++index) {
        if (index) output << ", ";
        output << result.tokens[index];
    }
    output << "],\n  \"windows\": [";
    for (std::size_t index = 0; index < result.windows.size(); ++index) {
        const auto & record = result.windows[index];
        const auto & window = record.window;
        output << (index ? ",\n    {" : "\n    {")
               << "\"start_seconds\": " << window.start_seconds
               << ", \"end_seconds\": " << window.end_seconds
               << ", \"accept_start_seconds\": " << window.accept_start_seconds
               << ", \"accept_end_seconds\": " << window.accept_end_seconds
               << ", \"prefix_end_seconds\": " << window.prefix_end_seconds
               << ", \"generation_stop_seconds\": " << window.generation_stop_seconds
               << ", \"prefix_tokens\": " << record.prefix_tokens
               << ", \"decoded_events\": " << record.decoded_events
               << ", \"accepted_events\": " << record.accepted_events
               << ", \"tokens\": [";
        for (std::size_t token = 0; token < record.tokens.size(); ++token) {
            if (token) output << ", ";
            output << record.tokens[token];
        }
        output << "]}";
    }
    output << "\n  ],\n  \"events\": [";
    for (std::size_t index = 0; index < result.events.size(); ++index) {
        const auto & event = result.events[index];
        output << (index ? ",\n    {" : "\n    {")
               << "\"subbeat\": " << event.subbeat
               << ", \"source_subbeat\": " << event.source_subbeat
               << ", \"window_index\": " << event.window_index
               << ", \"time_seconds\": " << event.time_seconds
               << ", \"has_timestamp\": " << (event.has_timestamp ? "true" : "false")
               << ", \"meter_numerator\": " << event.meter_numerator
               << ", \"meter_denominator\": " << event.meter_denominator
               << ", \"eighth_position\": " << event.eighth_position
               << ", \"structure\": " << json_string(event.structure)
               << ", \"key\": " << json_string(event.key)
               << ", \"chord\": " << json_string(event.chord)
               << ", \"payload_tokens\": [";
        for (std::size_t token = 0; token < event.payload_tokens.size(); ++token) {
            if (token) output << ", ";
            output << event.payload_tokens[token];
        }
        output << "], \"notes\": [";
        for (std::size_t note = 0; note < event.notes.size(); ++note) {
            const auto & value = event.notes[note];
            if (note) output << ", ";
            output << "{\"pitch\": " << value.pitch
                   << ", \"track\": " << value.track
                   << ", \"duration_bin\": " << value.duration_bin
                   << ", \"duration_steps\": " << value.duration_steps
                   << ", \"end_time_seconds\": " << value.end_time_seconds << '}';
        }
        output << "]}";
    }
    output << "\n  ],\n  \"warnings\": [";
    for (std::size_t index = 0; index < result.warnings.size(); ++index) {
        if (index) output << ", ";
        output << json_string(result.warnings[index]);
    }
    output << "]\n}\n";
    return output.str();
}

const char * version() noexcept { return "0.2.0"; }
bool transcription_runtime_available() noexcept { return true; }

} // namespace yue2
