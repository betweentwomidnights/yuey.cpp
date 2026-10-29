#include "training/dataset.h"

#include "training/unicode.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>

namespace yue2::training {
namespace json = server::json;
namespace {

using Codes = std::vector<std::uint32_t>;

bool digit(std::uint32_t c) { return c >= '0' && c <= '9'; }
std::uint32_t lower(std::uint32_t c) { return c >= 'A' && c <= 'Z' ? c + 32 : c; }

bool boundary(const Codes & text, std::size_t at) {
    const bool before = at > 0 && unicode::word(text[at - 1]);
    const bool after = at < text.size() && unicode::word(text[at]);
    return before != after;
}

bool literal(const Codes & text, std::size_t at, const char * word) {
    for (std::size_t i = 0; word[i]; ++i) {
        if (at + i >= text.size() || lower(text[at + i]) != static_cast<std::uint32_t>(word[i])) return false;
    }
    return true;
}

// \d+(?:\.\d+)? from at; returns the end, or at when there are no digits.
std::size_t number(const Codes & text, std::size_t at) {
    auto end = at;
    while (end < text.size() && digit(text[end])) ++end;
    if (end == at) return at;
    if (end + 1 < text.size() && text[end] == '.' && digit(text[end + 1])) {
        end += 2;
        while (end < text.size() && digit(text[end])) ++end;
    }
    return end;
}

std::size_t spaces(const Codes & text, std::size_t at) {
    while (at < text.size() && unicode::space(text[at])) ++at;
    return at;
}

bool tonic(std::uint32_t c) { c = lower(c); return c >= 'a' && c <= 'g'; }
bool accidental(std::uint32_t c) { return c == '#' || lower(c) == 'b'; }

// \d+(?:\.\d+)?\s*bpm from at: the end of the match, if any. Every quantifier
// is followed by something it cannot consume, so no backtracking is needed.
std::optional<std::size_t> tempo_at(const Codes & text, std::size_t at, std::size_t * digits_end) {
    const auto end = number(text, at);
    if (end == at) return std::nullopt;
    const auto word = spaces(text, end);
    if (!literal(text, word, "bpm")) return std::nullopt;
    if (digits_end) *digits_end = end;
    return word + 3;
}

// [A-G](?:#|b)?\s+(major|minor), ignoring case.
std::optional<std::size_t> key_at(const Codes & text, std::size_t at, std::size_t * mode_at) {
    if (at >= text.size() || !tonic(text[at])) return std::nullopt;
    auto next = at + 1;
    if (next < text.size() && accidental(text[next])) ++next;
    const auto mode = spaces(text, next);
    if (mode == next) return std::nullopt;
    if (!literal(text, mode, "major") && !literal(text, mode, "minor")) return std::nullopt;
    if (mode_at) *mode_at = mode;
    return mode + 5;
}

Codes strip_codes(const Codes & text, std::size_t begin, std::size_t end) {
    while (begin < end && unicode::space(text[begin])) ++begin;
    while (end > begin && unicode::space(text[end - 1])) --end;
    return Codes(text.begin() + begin, text.begin() + end);
}

int pitch_class(const std::string & root) {
    static const std::map<char, int> natural{{'C', 0}, {'D', 2}, {'E', 4}, {'F', 5}, {'G', 7}, {'A', 9}, {'B', 11}};
    if (root.empty() || !natural.count(root[0])) throw std::invalid_argument("Unknown chord root: " + root);
    const auto rest = root.substr(1);
    return natural.at(root[0]) + (rest == "#" ? 1 : rest == "b" ? -1 : 0);
}

std::string duration_suffix(long long ticks) {
    // L:1/16 with 12 grid ticks per quarter: a sixteenth is 3 ticks.
    const auto divisor = std::gcd(ticks, 3LL);
    const auto numerator = ticks / divisor, denominator = 3 / divisor;
    if (numerator == 1 && denominator == 1) return "";
    return denominator == 1 ? std::to_string(numerator) : std::to_string(numerator) + "/" + std::to_string(denominator);
}

double as_double(const json::Value & value, const char * what) {
    if (value.type != json::Type::number) throw std::invalid_argument(std::string("Invalid ") + what);
    return std::strtod(value.text.c_str(), nullptr);
}

// int() of a JSON number: Python truncates a float toward zero.
long long as_int(const json::Value & value, const char * what) {
    const auto number = as_double(value, what);
    if (!std::isfinite(number)) throw std::invalid_argument(std::string("Invalid ") + what);
    return static_cast<long long>(std::trunc(number));
}

const json::Value & field(const json::Value & object, const char * key) {
    const auto * value = object.find(key);
    if (!value) throw std::invalid_argument(std::string("Event is missing ") + key);
    return *value;
}

long long optional_int(const json::Value & object, const char * key) {
    const auto * value = object.find(key);
    return value ? as_int(*value, key) : 0;
}

} // namespace

long long round_half_even(double value) {
    const double floor = std::floor(value), fraction = value - floor;
    const auto lower = static_cast<long long>(floor);
    if (fraction > 0.5) return lower + 1;
    if (fraction < 0.5) return lower;
    return lower % 2 == 0 ? lower : lower + 1;
}

std::string strip(const std::string & text) {
    const auto codes = unicode::decode(text);
    return unicode::encode(strip_codes(codes, 0, codes.size()));
}

CaptionMetadata caption_metadata(const std::string & caption) {
    Codes text;
    for (const auto c : unicode::decode(caption)) text.push_back(c == 0x266d ? 'b' : c == 0x266f ? '#' : c);
    std::optional<std::pair<std::size_t, std::size_t>> tempo;
    for (std::size_t at = 0; at < text.size() && !tempo; ++at) {
        std::size_t digits_end = 0;
        if (!boundary(text, at)) continue;
        const auto end = tempo_at(text, at, &digits_end);
        if (end && boundary(text, *end)) tempo = std::make_pair(at, digits_end);
    }
    std::optional<std::pair<std::size_t, std::size_t>> key;
    for (std::size_t at = 0; at < text.size() && !key; ++at) {
        std::size_t mode = 0;
        if (!boundary(text, at)) continue;
        const auto end = key_at(text, at, &mode);
        if (end && boundary(text, *end)) key = std::make_pair(at, mode);
    }
    if (!tempo || !key) throw std::invalid_argument("Caption needs explicit tempo and key");
    CaptionMetadata output;
    const auto digits = unicode::encode(text, tempo->first, tempo->second);
    output.bpm = std::strtod(digits.c_str(), nullptr);
    if (!(output.bpm >= 20 && output.bpm <= 400)) throw std::invalid_argument("Unsupported caption tempo");
    // group(1) is the tonic and an optional accidental exactly as written.
    output.key = static_cast<char>(text[key->first] >= 'a' ? text[key->first] - 32 : text[key->first]);
    if (accidental(text[key->first + 1])) output.key.push_back(static_cast<char>(text[key->first + 1]));
    if (literal(text, key->second, "minor")) output.key += "m";
    std::size_t begin = 0;
    for (std::size_t at = 0; at <= text.size(); ++at) {
        if (at < text.size() && text[at] != ',') continue;
        const auto part = strip_codes(text, begin, at);
        begin = at + 1;
        if (part.empty()) continue;
        const auto tempo_end = tempo_at(part, 0, nullptr);
        const auto key_end = key_at(part, 0, nullptr);
        if ((tempo_end && *tempo_end == part.size()) || (key_end && *key_end == part.size())) continue;
        if (!output.style.empty()) output.style += ", ";
        output.style += unicode::encode(part);
    }
    return output;
}

std::vector<long long> window_starts(long long total_frames, long long frames, double bpm) {
    if (frames <= 0 || total_frames < frames) throw std::invalid_argument("Recording shorter than training window");
    const double bar_frames = 240 / bpm * 25;
    std::set<long long> starts{0};
    for (long long at = frames; at < total_frames; at += frames) {
        const auto bars = static_cast<long long>(std::floor(std::min(at, total_frames - frames) / bar_frames));
        const auto candidate = round_half_even(bars * bar_frames);
        starts.insert(std::max(0LL, std::min(total_frames - frames, candidate)));
    }
    return {starts.begin(), starts.end()};
}

std::string abc_pitch(int pitch) {
    if (pitch < 0 || pitch > 127) throw std::invalid_argument("MIDI pitch out of range");
    static const char * names[] = {"=C", "^C", "=D", "^D", "=E", "=F", "^F", "=G", "^G", "=A", "^A", "=B"};
    const int octave = pitch / 12 - 1;
    std::string name = names[pitch % 12];
    if (octave >= 5) {
        for (auto & c : name) c = static_cast<char>(c >= 'A' && c <= 'Z' ? c + 32 : c);
        return name + std::string(octave - 5, '\'');
    }
    return name + std::string(std::max(0, 4 - octave), ',');
}

std::string chord_name(const std::string & text) {
    if (text.empty() || text == "N") return "";
    const auto colon = text.find(':');
    if (colon == std::string::npos) return text;
    const auto root = text.substr(0, colon);
    auto kind = text.substr(colon + 1);
    const auto slash = kind.find('/');
    const bool inverted = slash != std::string::npos;
    const auto inversion = inverted ? kind.substr(slash + 1) : "";
    if (inverted) kind = kind.substr(0, slash);
    static const std::map<std::string, std::string> suffix{
        {"maj", ""}, {"min", "m"}, {"dim", "dim"}, {"aug", "aug"}, {"7", "7"}, {"maj7", "maj7"}, {"min7", "m7"},
        {"dim7", "dim7"}, {"hdim7", "m7b5"}, {"sus4", "sus4"}, {"sus2", "sus2"}, {"maj6", "6"}, {"min6", "m6"},
        {"sus4(b7)", "7sus4"}, {"minmaj7", "m(maj7)"}};
    const auto found = suffix.find(kind);
    if (found == suffix.end()) return "";
    auto result = root + found->second;
    static const std::map<std::string, int> intervals{{"2", 2}, {"b3", 3}, {"3", 4}, {"5", 7}, {"b7", 10}, {"7", 11}};
    const auto interval = intervals.find(inversion);
    if (inverted && interval != intervals.end()) {
        static const char * names[] = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};
        result += "/" + std::string(names[((pitch_class(root) + interval->second) % 12 + 12) % 12]);
    }
    return result;
}

EngravedScore score_from_events(const json::Value & events, const json::Value & seconds_value, double bpm, const std::string & key) {
    const double seconds = as_double(seconds_value, "score timing");
    if (!std::isfinite(seconds) || seconds <= 0 || !(bpm >= 20 && bpm <= 400)) throw std::invalid_argument("Invalid score timing");
    const bool key_ok = !key.empty() && key[0] >= 'A' && key[0] <= 'G' &&
        [&]() {
            std::size_t at = 1;
            if (at < key.size() && (key[at] == '#' || key[at] == 'b')) ++at;
            if (at < key.size() && key[at] == 'm') ++at;
            return at == key.size();
        }();
    if (!key_ok) throw std::invalid_argument("Unsupported key");
    if (events.type != json::Type::array) throw std::invalid_argument("Events must be an array");
    long long numerator = 4, denominator = 4;
    for (const auto & event : events.array) {
        if (optional_int(event, "meter_numerator") > 0 && optional_int(event, "meter_denominator") > 0) {
            numerator = optional_int(event, "meter_numerator");
            denominator = optional_int(event, "meter_denominator");
            break;
        }
    }
    if ((denominator != 2 && denominator != 4 && denominator != 8 && denominator != 16) || numerator < 1 || numerator > 12) {
        throw std::invalid_argument("Unsupported meter");
    }
    if (numerator * 48 % denominator) throw std::invalid_argument("Unsupported meter grid");
    const long long bar_ticks = numerator * 48 / denominator;
    const double scale = bpm / 60 * 12;
    const auto limit = static_cast<long long>(std::floor(seconds * scale + 1e-6));
    const auto total = static_cast<long long>(std::ceil(static_cast<double>(std::max(1LL, limit)) / bar_ticks)) * bar_ticks;

    struct Note { long long start, end; int pitch; };
    std::map<long long, std::string> chords;
    std::vector<Note> voices[2];
    std::vector<double> rounding;
    long long invalid = 0;
    std::set<std::pair<long long, long long>> meter_changes;
    for (const auto & event : events.array) {
        const double start = as_double(field(event, "time_seconds"), "onset");
        if (!std::isfinite(start)) throw std::invalid_argument("Nonfinite onset");
        const auto at = std::max(0LL, round_half_even(start * scale));
        if (optional_int(event, "meter_numerator")) {
            meter_changes.emplace(optional_int(event, "meter_numerator"), as_int(field(event, "meter_denominator"), "meter"));
        }
        const auto * chord_value = event.find("chord");
        const auto chord = chord_value && chord_value->type == json::Type::string ? chord_name(chord_value->text) : "";
        if (at < limit && !chord.empty()) chords[at] = chord;
        const auto & notes = field(event, "notes");
        if (notes.type != json::Type::array) throw std::invalid_argument("Invalid note event");
        for (const auto & note : notes.array) {
            const double end = as_double(field(note, "end_time_seconds"), "note event");
            const auto pitch = as_int(field(note, "pitch"), "note event");
            const auto track = as_int(field(note, "track"), "note event");
            if (!std::isfinite(end) || end < start || pitch < 0 || pitch > 127 || (track != 0 && track != 1)) {
                throw std::invalid_argument("Invalid note event");
            }
            if (start >= seconds || end <= 0) continue;
            const auto finish = std::min(limit, std::max(at + 1, round_half_even(std::min(end, seconds) * scale)));
            if (at >= finish) { ++invalid; continue; }
            rounding.push_back(std::abs(at / scale - std::max(0.0, start)));
            rounding.push_back(std::abs(finish / scale - std::min(end, seconds)));
            voices[track].push_back({at, finish, static_cast<int>(pitch)});
        }
    }
    if (meter_changes.size() > 1) throw std::invalid_argument("Changing meter needs explicit review");

    std::vector<std::string> lines{"X:1", "T:", "M:" + std::to_string(numerator) + "/" + std::to_string(denominator), "L:1/16",
                                   "Q:1/4=" + std::to_string(round_half_even(bpm)),
                                   "V: Vocal clef=treble name=\"Vocal Melody\" snm=\"Vocal\"",
                                   "V: Ins clef=treble name=\"Ins Melody\" snm=\"Inst.\"", "K:" + key};
    long long retained = 0, duplicates = 0;
    const char * names[] = {"Vocal", "Ins"};
    for (int voice = 0; voice < 2; ++voice) {
        auto notes = voices[voice];
        std::stable_sort(notes.begin(), notes.end(), [](const Note & a, const Note & b) { return a.start < b.start; });
        std::vector<Note> mono;
        for (const auto & note : notes) {
            if (!mono.empty() && mono.back().start == note.start) { ++duplicates; continue; }
            mono.push_back(note);
        }
        for (std::size_t i = 0; i + 1 < mono.size(); ++i) mono[i].end = std::min(mono[i].end, mono[i + 1].start);
        retained += static_cast<long long>(mono.size());
        // A cell is the sounding note's pitch and index, or none (-1).
        std::vector<std::pair<int, long long>> cells(static_cast<std::size_t>(total), {-1, -1});
        for (std::size_t i = 0; i < mono.size(); ++i) {
            for (auto at = mono[i].start; at < mono[i].end; ++at) cells[at] = {mono[i].pitch, static_cast<long long>(i)};
        }
        std::vector<std::string> bars;
        for (long long begin = 0; begin < total; begin += bar_ticks) {
            const auto end = begin + bar_ticks;
            std::string body;
            for (auto at = begin; at < end;) {
                const auto cell = cells[at];
                auto finish = at + 1;
                while (finish < end && cells[finish] == cell && (voice != 0 || !chords.count(finish))) ++finish;
                if (voice == 0 && chords.count(at)) body += "\"" + chords.at(at) + "\"";
                body += cell.first < 0 ? "z" : abc_pitch(cell.first);
                body += duration_suffix(finish - at);
                if (cell.first >= 0 && finish < total && cells[finish] == cell) body += "-";
                at = finish;
            }
            bars.push_back(body + "|");
        }
        for (std::size_t index = 0; index < bars.size(); index += 4) {
            std::string row;
            for (std::size_t k = index; k < std::min(bars.size(), index + 4); ++k) row += bars[k];
            lines.push_back(std::string("V: ") + names[voice]);
            lines.push_back(row);
        }
    }
    EngravedScore output;
    for (const auto & line : lines) output.abc += line + "\n";
    auto & stats = output.stats = json::object();
    stats.object["input_notes"] = json::from_integer(static_cast<long long>(voices[0].size() + voices[1].size()));
    stats.object["retained_monophonic_notes"] = json::from_integer(retained);
    stats.object["duplicate_onsets_dropped"] = json::from_integer(duplicates);
    stats.object["clipped_notes"] = json::from_integer(invalid);
    stats.object["chord_positions"] = json::from_integer(static_cast<long long>(chords.size()));
    // max(rounding, default=0) is the integer 0 when nothing was placed.
    stats.object["max_rounding_seconds"] = rounding.empty() ? json::from_integer(0)
        : json::from_double(*std::max_element(rounding.begin(), rounding.end()));
    stats.object["max_expected_grid_error_seconds"] = json::from_double(1 / scale);
    stats.object["score_seconds"] = json::from_double(total / scale);
    stats.object["audio_seconds"] = seconds_value;
    stats.object["bpm"] = json::from_double(bpm);
    stats.object["key"] = json::from_string(key);
    stats.object["meter"] = json::from_string(std::to_string(numerator) + "/" + std::to_string(denominator));
    stats.object["grid"] = json::from_string("12 ticks/quarter; explicit pitches; real absolute timestamps");
    stats.object["manual_note_accuracy_verified"] = json::from_bool(false);
    return output;
}

} // namespace yue2::training
