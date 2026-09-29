#include "abc_score.h"

#include "yue2/transcription.h"

#include <algorithm>
#include <cctype>
#include <optional>
#include <stdexcept>
#include <utility>

namespace yue2::abc {
namespace {

std::uint32_t positive(const std::string & text) {
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos) {
        throw std::invalid_argument("invalid number in ABC field");
    }
    const auto value = std::stoul(text);
    if (value == 0) throw std::invalid_argument("ABC field must be positive");
    return static_cast<std::uint32_t>(value);
}

std::uint64_t duration_ticks(const std::string & line, std::size_t & offset, std::uint32_t unit) {
    std::uint64_t numerator = 0;
    const auto begin = offset;
    while (offset < line.size() && std::isdigit(static_cast<unsigned char>(line[offset]))) {
        numerator = numerator * 10 + static_cast<unsigned>(line[offset++] - '0');
    }
    if (offset == begin) numerator = 1;
    std::uint64_t denominator = 1;
    if (offset < line.size() && line[offset] == '/') {
        ++offset;
        const auto start = offset;
        denominator = 0;
        while (offset < line.size() && std::isdigit(static_cast<unsigned char>(line[offset]))) {
            denominator = denominator * 10 + static_cast<unsigned>(line[offset++] - '0');
        }
        if (offset == start) denominator = 2;
    }
    const auto ticks = numerator * kWholeTicks;
    if (numerator == 0 || denominator == 0 || ticks % (denominator * unit) != 0) {
        throw std::invalid_argument("ABC duration is off the tick grid");
    }
    return ticks / (denominator * unit);
}

int pitch_of(char symbol, int octave, const std::string & accidental, const std::string & key,
             std::map<std::pair<char, int>, int> & bar_accidentals) {
    static const std::string letters = "CDEFGAB";
    static const std::array<int, 7> natural = {0, 2, 4, 5, 7, 9, 11};
    const char upper = static_cast<char>(std::toupper(static_cast<unsigned char>(symbol)));
    const auto index = letters.find(upper);
    const auto identity = std::make_pair(upper, octave);
    int alteration = 0;
    if (!accidental.empty()) {
        if (accidental.front() == '^') alteration = static_cast<int>(accidental.size());
        else if (accidental.front() == '_') alteration = -static_cast<int>(accidental.size());
        bar_accidentals[identity] = alteration;
    } else if (const auto found = bar_accidentals.find(identity); found != bar_accidentals.end()) {
        alteration = found->second;
    } else {
        alteration = key_accidentals(key)[index];
    }
    return std::clamp((octave + 1) * 12 + natural[index] + alteration, 0, 127);
}

// Lengths in units, largest first, from the values the native dialect uses.
std::vector<std::uint64_t> split_units(std::uint64_t units) {
    static const std::array<std::uint64_t, 11> allowed = {48, 32, 24, 16, 12, 8, 6, 4, 3, 2, 1};
    std::vector<std::uint64_t> parts;
    for (const auto value : allowed) {
        while (units >= value) { parts.push_back(value); units -= value; }
    }
    return parts;
}

std::string units_text(std::uint64_t units) { return units == 1 ? std::string() : std::to_string(units); }

} // namespace

std::string trim(const std::string & value) {
    const auto first = value.find_first_not_of(" \t\r");
    if (first == std::string::npos) return {};
    const auto last = value.find_last_not_of(" \t\r");
    return value.substr(first, last - first + 1);
}

std::vector<std::string> split_lines(const std::string & text) {
    std::vector<std::string> out;
    std::size_t start = 0;
    while (start <= text.size()) {
        const auto end = text.find('\n', start);
        auto line = text.substr(start, end == std::string::npos ? std::string::npos : end - start);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        out.push_back(std::move(line));
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return out;
}

std::array<int, 7> key_accidentals(const std::string & key) {
    static const std::string letters = "CDEFGAB";
    static const std::map<std::string, int> fifths = {
        {"Cb",-7},{"Gb",-6},{"Db",-5},{"Ab",-4},{"Eb",-3},{"Bb",-2},{"F",-1},
        {"G",1},{"D",2},{"A",3},{"E",4},{"B",5},{"F#",6},{"C#",7},
        {"Abm",-7},{"Ebm",-6},{"Bbm",-5},{"Fm",-4},{"Cm",-3},{"Gm",-2},{"Dm",-1},
        {"Em",1},{"Bm",2},{"F#m",3},{"C#m",4},{"G#m",5},{"D#m",6},{"A#m",7},
    };
    std::array<int, 7> result{};
    const auto found = fifths.find(key);
    if (found == fifths.end()) return result;
    const std::string order = found->second > 0 ? "FCGDAEB" : "BEADGCF";
    const int count = found->second > 0 ? found->second : -found->second;
    for (int i = 0; i < count; ++i) {
        result[letters.find(order[static_cast<std::size_t>(i)])] = found->second > 0 ? 1 : -1;
    }
    return result;
}

std::uint64_t Parsed::unit_ticks() const {
    if (kWholeTicks % unit_denominator != 0) {
        throw std::invalid_argument("ABC unit length is off the tick grid");
    }
    return kWholeTicks / unit_denominator;
}

Parsed parse(const std::string & abc) {
    Parsed result;
    const auto lines = split_lines(abc);
    result.line_segments.resize(lines.size());
    std::uint32_t numerator = 4, denominator = 4;
    std::string key = "C";
    for (const auto & raw : lines) {
        const auto line = trim(raw);
        if (line == "V: Vocal") break;
        if (line.rfind("M:", 0) == 0) {
            const auto slash = line.find('/');
            if (slash == std::string::npos) throw std::invalid_argument("invalid ABC M field");
            numerator = positive(line.substr(2, slash - 2));
            denominator = positive(line.substr(slash + 1));
        } else if (line.rfind("L:1/", 0) == 0) {
            result.unit_denominator = positive(line.substr(4));
        } else if (line.rfind("K:", 0) == 0) {
            key = trim(line.substr(2));
        }
    }

    struct State {
        std::uint64_t tick = 0;
        std::uint32_t numerator = 4, denominator = 4;
        std::string key;
        std::optional<std::size_t> tied;
    };
    std::array<State, 2> state;
    for (auto & s : state) { s.numerator = numerator; s.denominator = denominator; s.key = key; }
    const auto bar_length = [](const State & s) {
        const auto ticks = static_cast<std::uint64_t>(s.numerator) * kWholeTicks;
        if (ticks % s.denominator != 0) throw std::invalid_argument("ABC meter is off the tick grid");
        return ticks / s.denominator;
    };

    int active = -1;
    for (std::size_t li = 0; li < lines.size(); ++li) {
        const auto line = trim(lines[li]);
        if (line == "V: Vocal") { active = 0; continue; }
        if (line == "V: Ins") { active = 1; continue; }
        if (active < 0 || line.empty() || line.front() == '%') continue;
        auto & s = state[static_cast<std::size_t>(active)];
        if (line.rfind("M:", 0) == 0) {
            const auto slash = line.find('/');
            if (slash == std::string::npos) throw std::invalid_argument("invalid ABC M field");
            s.numerator = positive(line.substr(2, slash - 2));
            s.denominator = positive(line.substr(slash + 1));
            continue;
        }
        if (line.rfind("K:", 0) == 0) { s.key = trim(line.substr(2)); continue; }
        if (line.size() >= 2 && std::isalpha(static_cast<unsigned char>(line[0])) && line[1] == ':') continue;
        if (line.back() != '|') throw std::invalid_argument("ABC music line must end with a barline");

        auto & voice = result.voices[static_cast<std::size_t>(active)];
        std::vector<std::string> parts;
        std::size_t start = 0;
        for (std::size_t i = 0; i < line.size(); ++i) {
            if (line[i] == '|') { parts.push_back(line.substr(start, i - start)); start = i + 1; }
        }
        result.line_segments[li] = parts;

        for (std::size_t pi = 0; pi < parts.size(); ++pi) {
            const auto & bar = parts[pi];
            Segment segment;
            segment.line = li;
            segment.index = pi;
            segment.text = bar;
            segment.begin = s.tick;
            segment.key = s.key;
            std::map<std::pair<char, int>, int> bar_accidentals;
            bool multi = false;
            for (std::size_t offset = 0; offset < bar.size();) {
                const char c = bar[offset];
                if (std::isspace(static_cast<unsigned char>(c))) { ++offset; continue; }
                if (c == '"') {
                    const auto end = bar.find('"', offset + 1);
                    if (end == std::string::npos) throw std::invalid_argument("unterminated ABC chord");
                    segment.marks.push_back({s.tick, bar.substr(offset, end - offset + 1)});
                    offset = end + 1;
                    continue;
                }
                if (c == '[') {
                    const auto end = bar.find(']', offset + 1);
                    if (end == std::string::npos) throw std::invalid_argument("unterminated ABC inline field");
                    const auto field = bar.substr(offset + 1, end - offset - 1);
                    if (field.rfind("K:", 0) == 0) {
                        s.key = trim(field.substr(2));
                        bar_accidentals.clear();
                    } else if (field.rfind("M:", 0) == 0) {
                        throw std::invalid_argument("inline meter changes are not supported");
                    }
                    segment.marks.push_back({s.tick, bar.substr(offset, end - offset + 1)});
                    offset = end + 1;
                    continue;
                }
                std::string accidental;
                while (offset < bar.size() && (bar[offset] == '^' || bar[offset] == '_' || bar[offset] == '=')) {
                    accidental.push_back(bar[offset++]);
                }
                if (offset >= bar.size()) throw std::invalid_argument("dangling ABC accidental");
                const char symbol = bar[offset++];
                if ((symbol >= 'A' && symbol <= 'G') || (symbol >= 'a' && symbol <= 'g')) {
                    int octave = std::islower(static_cast<unsigned char>(symbol)) ? 5 : 4;
                    while (offset < bar.size() && (bar[offset] == ',' || bar[offset] == '\'')) {
                        octave += bar[offset++] == '\'' ? 1 : -1;
                    }
                    const auto length = duration_ticks(bar, offset, result.unit_denominator);
                    const auto pitch = pitch_of(symbol, octave, accidental, s.key, bar_accidentals);
                    const bool tie = offset < bar.size() && bar[offset] == '-';
                    if (tie) ++offset;
                    if (s.tied) {
                        auto & held = voice.notes[*s.tied];
                        if (held.end != s.tick || held.pitch != pitch) {
                            throw std::invalid_argument("ABC tie is not followed by the same note");
                        }
                        held.end = s.tick + length;
                    } else {
                        voice.notes.push_back({s.tick, s.tick + length, pitch});
                    }
                    s.tied = tie ? std::optional<std::size_t>(s.tied ? *s.tied : voice.notes.size() - 1)
                                 : std::nullopt;
                    s.tick += length;
                    segment.has_notes = true;
                    continue;
                }
                if (symbol == 'z') {
                    if (s.tied) throw std::invalid_argument("ABC tie is not followed by its note");
                    s.tick += duration_ticks(bar, offset, result.unit_denominator);
                    continue;
                }
                if (symbol == 'Z') {
                    if (s.tied) throw std::invalid_argument("ABC tie is not followed by its note");
                    std::uint64_t count = 0;
                    while (offset < bar.size() && std::isdigit(static_cast<unsigned char>(bar[offset]))) {
                        count = count * 10 + static_cast<unsigned>(bar[offset++] - '0');
                    }
                    if (count == 0) count = 1;
                    for (std::uint64_t i = 0; i < count; ++i) {
                        s.tick += bar_length(s);
                        segment.bar_ends.push_back(s.tick);
                    }
                    multi = true;
                    continue;
                }
                throw std::invalid_argument(std::string("unsupported ABC token: ") + symbol);
            }
            if (!multi) {
                if (s.tick - segment.begin != bar_length(s)) {
                    throw std::invalid_argument("ABC bar does not fill its meter");
                }
                segment.bar_ends.push_back(s.tick);
            }
            segment.end = s.tick;
            voice.segments.push_back(std::move(segment));
        }
    }
    if (result.voices[0].segments.empty() || result.voices[1].segments.empty()) {
        throw std::invalid_argument("ABC score needs native Vocal and Ins lanes");
    }
    if (state[0].tick != state[1].tick) throw std::invalid_argument("Vocal and Ins lanes differ in length");
    if (state[0].tied || state[1].tied) throw std::invalid_argument("ABC score ends in an open tie");
    return result;
}

std::string spell(int pitch, const std::string & key) {
    const auto accidentals = key_accidentals(key);
    const bool flats = std::find(accidentals.begin(), accidentals.end(), -1) != accidentals.end();
    static const std::array<const char *, 12> sharp = {"=C","^C","=D","^D","=E","=F","^F","=G","^G","=A","^A","=B"};
    static const std::array<const char *, 12> flat = {"=C","_D","=D","_E","=E","=F","_G","=G","_A","=A","_B","=B"};
    const int pc = ((pitch % 12) + 12) % 12;
    std::string name = (flats ? flat : sharp)[static_cast<std::size_t>(pc)];
    int octave = pitch / 12 - 1;
    std::string out = name.substr(0, 1);
    char letter = name[1];
    if (octave >= 5) {
        out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(letter))));
        for (int o = 5; o < octave; ++o) out.push_back('\'');
    } else {
        out.push_back(letter);
        for (int o = 4; o > octave; --o) out.push_back(',');
    }
    return out;
}

std::string write_bars(const Segment & segment, const std::vector<Span> & notes,
                       std::uint64_t unit_ticks, const std::string & key) {
    std::string out;
    std::uint64_t bar_begin = segment.begin;
    for (std::size_t b = 0; b < segment.bar_ends.size(); ++b) {
        const auto bar_end = segment.bar_ends[b];
        if (b > 0) out += '|';
        std::vector<std::uint64_t> cuts = {bar_begin, bar_end};
        std::vector<const Mark *> marks;
        for (const auto & mark : segment.marks) {
            if (mark.tick >= bar_begin && mark.tick < bar_end) {
                marks.push_back(&mark);
                cuts.push_back(mark.tick);
            }
        }
        for (const auto & n : notes) {
            if (n.end <= bar_begin || n.begin >= bar_end) continue;
            cuts.push_back(std::max(n.begin, bar_begin));
            cuts.push_back(std::min(n.end, bar_end));
        }
        std::sort(cuts.begin(), cuts.end());
        cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());

        for (std::size_t i = 0; i + 1 < cuts.size(); ++i) {
            const auto from = cuts[i], to = cuts[i + 1];
            for (const auto * mark : marks) if (mark->tick == from) out += mark->text;
            const Span * note = nullptr;
            for (const auto & n : notes) {
                if (n.begin <= from && n.end >= to) { note = &n; break; }
            }
            if ((to - from) % unit_ticks != 0) throw std::invalid_argument("rewritten note is off the unit grid");
            const auto parts = split_units((to - from) / unit_ticks);
            if (!note) {
                for (const auto part : parts) out += "z" + units_text(part);
                continue;
            }
            const auto spelled = spell(note->pitch, key);
            for (std::size_t p = 0; p < parts.size(); ++p) {
                out += spelled + units_text(parts[p]);
                const bool last_part = p + 1 == parts.size();
                // A note carries on past this run: into the next run of the same
                // note, or across the barline.
                if (!last_part || note->end > to) out += '-';
            }
        }
        bar_begin = bar_end;
    }
    return out;
}

std::string assemble(const std::string & abc,
                     const std::vector<std::vector<std::string>> & line_segments,
                     const std::vector<bool> & line_changed,
                     const std::map<std::size_t, std::string> & replaced,
                     const char * what) {
    const auto lines = split_lines(abc);
    std::string out;
    for (std::size_t li = 0; li < lines.size(); ++li) {
        if (li > 0) out += '\n';
        if (const auto found = replaced.find(li); found != replaced.end()) {
            out += found->second;
        } else if (li < line_changed.size() && line_changed[li]) {
            for (const auto & bar : line_segments[li]) out += bar + '|';
        } else {
            out += lines[li];
        }
    }
    std::string reason;
    if (!yue2_abc_score_is_renderable(out, &reason)) {
        throw std::runtime_error(std::string(what) + " produced an unrenderable score: " + reason);
    }
    return out;
}

} // namespace yue2::abc
