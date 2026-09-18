#include "yue2/transcription.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace yue2 {
namespace {

constexpr std::uint16_t midi_ppq = 960;

std::string trim(const std::string & value) {
    auto begin = value.begin();
    while (begin != value.end() && std::isspace(static_cast<unsigned char>(*begin))) ++begin;
    auto end = value.end();
    while (end != begin && std::isspace(static_cast<unsigned char>(*(end - 1)))) --end;
    return std::string(begin, end);
}

std::vector<std::string> lines(const std::string & text) {
    std::vector<std::string> result;
    std::istringstream input(text);
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        result.push_back(std::move(line));
    }
    return result;
}

std::uint32_t positive(const std::string & text, const char * field) {
    const auto value_text = trim(text);
    std::size_t used = 0;
    const auto value = std::stoul(value_text, &used);
    if (used != value_text.size() || value == 0 ||
        value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(std::string("invalid ABC ") + field);
    }
    return static_cast<std::uint32_t>(value);
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
    while ((value >>= 7U) != 0) {
        bytes[count++] = static_cast<std::uint8_t>((value & 0x7fU) | 0x80U);
    }
    while (count-- > 0) out.push_back(bytes[count]);
}

struct TimedEvent {
    std::uint32_t tick = 0;
    std::int32_t order = 0;
    std::vector<std::uint8_t> bytes;
};

void add_meta(std::vector<TimedEvent> & events, std::uint32_t tick,
              std::int32_t order, std::uint8_t type,
              const std::vector<std::uint8_t> & payload) {
    std::vector<std::uint8_t> bytes = {0xff, type};
    append_vlq(bytes, static_cast<std::uint32_t>(payload.size()));
    bytes.insert(bytes.end(), payload.begin(), payload.end());
    events.push_back({tick, order, std::move(bytes)});
}

void add_text(std::vector<TimedEvent> & events, std::uint32_t tick,
              std::int32_t order, std::uint8_t type, const std::string & text) {
    add_meta(events, tick, order, type,
             std::vector<std::uint8_t>(text.begin(), text.end()));
}

std::vector<std::uint8_t> make_track(
    const std::string & name, std::vector<TimedEvent> events) {
    add_text(events, 0, -100, 0x03, name);
    std::stable_sort(events.begin(), events.end(), [](const auto & left, const auto & right) {
        return std::tie(left.tick, left.order) < std::tie(right.tick, right.order);
    });
    std::vector<std::uint8_t> body;
    std::uint32_t previous = 0;
    for (const auto & event : events) {
        append_vlq(body, event.tick - previous);
        previous = event.tick;
        body.insert(body.end(), event.bytes.begin(), event.bytes.end());
    }
    body.insert(body.end(), {0x00, 0xff, 0x2f, 0x00});
    std::vector<std::uint8_t> result = {'M','T','r','k'};
    append_u32be(result, static_cast<std::uint32_t>(body.size()));
    result.insert(result.end(), body.begin(), body.end());
    return result;
}

std::vector<std::uint8_t> assemble(
    const std::vector<const std::vector<std::uint8_t> *> & tracks) {
    std::vector<std::uint8_t> result = {'M','T','h','d'};
    append_u32be(result, 6);
    append_u16be(result, 1);
    append_u16be(result, static_cast<std::uint16_t>(tracks.size()));
    append_u16be(result, midi_ppq);
    for (const auto * track : tracks) result.insert(result.end(), track->begin(), track->end());
    return result;
}

struct Duration {
    std::uint64_t numerator = 1;
    std::uint64_t denominator = 1;
};

Duration duration_at(const std::string & line, std::size_t & offset) {
    const auto begin = offset;
    std::uint64_t numerator = 0;
    while (offset < line.size() && std::isdigit(static_cast<unsigned char>(line[offset]))) {
        numerator = numerator * 10 + static_cast<unsigned>(line[offset++] - '0');
    }
    const bool has_numerator = offset != begin;
    if (offset >= line.size() || line[offset] != '/') {
        return {has_numerator ? numerator : 1, 1};
    }
    ++offset;
    const auto denominator_begin = offset;
    std::uint64_t denominator = 0;
    while (offset < line.size() && std::isdigit(static_cast<unsigned char>(line[offset]))) {
        denominator = denominator * 10 + static_cast<unsigned>(line[offset++] - '0');
    }
    if (!has_numerator) numerator = 1;
    if (offset == denominator_begin) denominator = 2;
    if (numerator == 0 || denominator == 0) {
        throw std::invalid_argument("ABC durations must be positive");
    }
    return {numerator, denominator};
}

std::uint32_t duration_ticks(const Duration & duration, std::uint32_t unit_denominator) {
    const auto numerator = duration.numerator * 4ULL * midi_ppq;
    const auto denominator = duration.denominator * unit_denominator;
    if (numerator % denominator != 0 || numerator / denominator == 0 ||
        numerator / denominator > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("ABC duration is not representable on the MIDI timeline");
    }
    return static_cast<std::uint32_t>(numerator / denominator);
}

std::uint32_t bar_ticks(std::uint32_t numerator, std::uint32_t denominator) {
    const auto value = static_cast<std::uint64_t>(numerator) * 4ULL * midi_ppq;
    if (value % denominator != 0 || value / denominator > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("ABC meter is not representable on the MIDI timeline");
    }
    return static_cast<std::uint32_t>(value / denominator);
}

int root_pitch_class(const std::string & root) {
    static const std::map<std::string, int> roots = {
        {"C",0},{"B#",0},{"C#",1},{"Db",1},{"D",2},{"D#",3},{"Eb",3},
        {"E",4},{"Fb",4},{"E#",5},{"F",5},{"F#",6},{"Gb",6},{"G",7},
        {"G#",8},{"Ab",8},{"A",9},{"A#",10},{"Bb",10},{"B",11},{"Cb",11},
    };
    const auto found = roots.find(root);
    return found == roots.end() ? -1 : found->second;
}

std::array<int, 7> key_accidentals(const std::string & key) {
    static const std::array<char, 7> letters = {'C','D','E','F','G','A','B'};
    static const std::string sharps = "FCGDAEB";
    static const std::string flats = "BEADGCF";
    static const std::map<std::string, int> sharp_keys = {
        {"G",1},{"D",2},{"A",3},{"E",4},{"B",5},{"F#",6},{"C#",7},
        {"Em",1},{"Bm",2},{"F#m",3},{"C#m",4},{"G#m",5},{"D#m",6},{"A#m",7},
    };
    static const std::map<std::string, int> flat_keys = {
        {"F",1},{"Bb",2},{"Eb",3},{"Ab",4},{"Db",5},{"Gb",6},{"Cb",7},
        {"Dm",1},{"Gm",2},{"Cm",3},{"Fm",4},{"Bbm",5},{"Ebm",6},{"Abm",7},
    };
    std::array<int, 7> result{};
    const auto apply = [&](const std::string & order, int count, int value) {
        for (int index = 0; index < count; ++index) {
            const auto found = std::find(letters.begin(), letters.end(), order[index]);
            if (found != letters.end()) result[static_cast<std::size_t>(found - letters.begin())] = value;
        }
    };
    if (const auto found = sharp_keys.find(key); found != sharp_keys.end()) apply(sharps, found->second, 1);
    if (const auto found = flat_keys.find(key); found != flat_keys.end()) apply(flats, found->second, -1);
    return result;
}

std::pair<std::int8_t, bool> midi_key_signature(const std::string & key) {
    static const std::map<std::string, std::int8_t> major = {
        {"Cb",-7},{"Gb",-6},{"Db",-5},{"Ab",-4},{"Eb",-3},{"Bb",-2},{"F",-1},
        {"C",0},{"G",1},{"D",2},{"A",3},{"E",4},{"B",5},{"F#",6},{"C#",7},
    };
    static const std::map<std::string, std::int8_t> minor = {
        {"Abm",-7},{"Ebm",-6},{"Bbm",-5},{"Fm",-4},{"Cm",-3},{"Gm",-2},{"Dm",-1},
        {"Am",0},{"Em",1},{"Bm",2},{"F#m",3},{"C#m",4},{"G#m",5},{"D#m",6},{"A#m",7},
    };
    if (const auto found = minor.find(key); found != minor.end()) return {found->second, true};
    if (const auto found = major.find(key); found != major.end()) return {found->second, false};
    return {0, false};
}

int pitch_for(char letter, int octave, const std::string & accidental,
              const std::string & key, std::map<std::pair<char,int>, int> & measure) {
    static const std::map<char, int> natural = {
        {'C',0},{'D',2},{'E',4},{'F',5},{'G',7},{'A',9},{'B',11},
    };
    const char upper = static_cast<char>(std::toupper(static_cast<unsigned char>(letter)));
    auto alteration = 0;
    const auto identity = std::make_pair(upper, octave);
    if (!accidental.empty()) {
        if (accidental.front() == '^') alteration = static_cast<int>(accidental.size());
        else if (accidental.front() == '_') alteration = -static_cast<int>(accidental.size());
        measure[identity] = alteration;
    } else if (const auto found = measure.find(identity); found != measure.end()) {
        alteration = found->second;
    } else {
        static const std::string letters = "CDEFGAB";
        const auto index = letters.find(upper);
        alteration = key_accidentals(key)[index];
    }
    const auto found = natural.find(upper);
    if (found == natural.end()) throw std::invalid_argument("invalid ABC pitch");
    return std::clamp((octave + 1) * 12 + found->second + alteration, 0, 127);
}

std::vector<int> chord_pitches(const std::string & label) {
    const auto slash = label.find('/');
    const auto body = label.substr(0, slash);
    std::size_t root_end = 1;
    if (body.size() > 1 && (body[1] == '#' || body[1] == 'b')) root_end = 2;
    const auto root = root_pitch_class(body.substr(0, root_end));
    if (root < 0) return {};
    auto quality = body.substr(root_end);
    std::vector<int> intervals;
    if (quality.empty() || quality == "maj") intervals = {0,4,7};
    else if (quality == "m" || quality == "min") intervals = {0,3,7};
    else if (quality == "dim") intervals = {0,3,6};
    else if (quality == "aug") intervals = {0,4,8};
    else if (quality == "7") intervals = {0,4,7,10};
    else if (quality == "maj7") intervals = {0,4,7,11};
    else if (quality == "m7") intervals = {0,3,7,10};
    else if (quality == "dim7") intervals = {0,3,6,9};
    else if (quality == "m7b5") intervals = {0,3,6,10};
    else if (quality == "sus2") intervals = {0,2,7};
    else if (quality == "sus4") intervals = {0,5,7};
    else if (quality == "7sus4") intervals = {0,5,7,10};
    else if (quality == "6") intervals = {0,4,7,9};
    else if (quality == "m6") intervals = {0,3,7,9};
    else if (quality == "m(maj7)") intervals = {0,3,7,11};
    else return {};
    std::vector<int> result = {36 + root};
    for (const auto interval : intervals) result.push_back(48 + root + interval);
    if (slash != std::string::npos) {
        const auto bass = root_pitch_class(label.substr(slash + 1));
        if (bass >= 0) result.front() = 36 + bass;
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

struct NoteSpan { std::uint32_t begin = 0, end = 0; int pitch = 0; bool tied = false; };
struct ChordMark { std::uint32_t tick = 0; std::string label; };
struct VoiceState {
    std::uint64_t tick = 0;
    std::uint64_t bar_start = 0;
    std::uint32_t numerator = 4;
    std::uint32_t denominator = 4;
    std::string key = "C";
    std::map<std::pair<char,int>, int> accidentals;
    std::optional<std::size_t> tied_note;
    std::uint64_t expected_span = 0;
};

struct ParsedScore {
    std::uint32_t bpm = 120;
    std::array<std::vector<NoteSpan>, 2> notes;
    std::vector<ChordMark> chords;
    std::vector<TimedEvent> conductor;
    std::uint32_t end_tick = 0;
};

std::uint32_t checked_tick(std::uint64_t value) {
    if (value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("ABC score is too long for Standard MIDI");
    }
    return static_cast<std::uint32_t>(value);
}

void add_meter(std::vector<TimedEvent> & conductor, std::uint32_t tick,
               std::uint32_t numerator, std::uint32_t denominator) {
    if ((denominator & (denominator - 1U)) != 0 || denominator > 128 || numerator > 255) {
        throw std::invalid_argument("ABC meter denominator must be a supported power of two");
    }
    std::uint8_t power = 0;
    for (auto value = denominator; value > 1; value >>= 1U) ++power;
    add_meta(conductor, tick, -80, 0x58,
             {static_cast<std::uint8_t>(numerator), power, 24, 8});
}

void add_key(std::vector<TimedEvent> & conductor, std::uint32_t tick, const std::string & key) {
    const auto [signature, minor] = midi_key_signature(key);
    add_meta(conductor, tick, -70, 0x59,
             {static_cast<std::uint8_t>(signature), static_cast<std::uint8_t>(minor)});
}

ParsedScore parse_score(const std::string & abc) {
    if (abc.empty()) throw std::invalid_argument("ABC score is empty");
    std::uint32_t numerator = 4, denominator = 4, unit_denominator = 32, bpm = 120;
    std::string key = "C";
    for (const auto & raw : lines(abc)) {
        const auto line = trim(raw);
        if (line.rfind("M:", 0) == 0) {
            const auto slash = line.find('/', 2);
            if (slash == std::string::npos) throw std::invalid_argument("invalid ABC M field");
            numerator = positive(line.substr(2, slash - 2), "meter numerator");
            denominator = positive(line.substr(slash + 1), "meter denominator");
        } else if (line.rfind("L:1/", 0) == 0) {
            unit_denominator = positive(line.substr(4), "unit length");
        } else if (line.rfind("Q:", 0) == 0) {
            const auto equals = line.rfind('=');
            if (equals != std::string::npos) bpm = positive(line.substr(equals + 1), "tempo");
        } else if (line.rfind("K:", 0) == 0) {
            key = trim(line.substr(2));
        }
        if (line == "V: Vocal") break;
    }
    if (bpm < 4 || bpm > 1000) throw std::invalid_argument("ABC tempo is out of range");

    ParsedScore result;
    result.bpm = bpm;
    const auto micros = static_cast<std::uint32_t>(
        std::llround(60000000.0 / static_cast<double>(bpm)));
    add_meta(result.conductor, 0, -90, 0x51,
             {static_cast<std::uint8_t>(micros >> 16),
              static_cast<std::uint8_t>(micros >> 8),
              static_cast<std::uint8_t>(micros)});
    add_meter(result.conductor, 0, numerator, denominator);
    add_key(result.conductor, 0, key);

    std::array<VoiceState, 2> voices;
    for (auto & voice : voices) {
        voice.numerator = numerator;
        voice.denominator = denominator;
        voice.key = key;
    }
    int active_voice = -1;
    std::string pending_section;
    bool found_vocal = false, found_instrumental = false;

    const auto set_meter = [&](VoiceState & voice, const std::string & value, int lane) {
        const auto slash = value.find('/');
        if (slash == std::string::npos) throw std::invalid_argument("invalid ABC M field");
        voice.numerator = positive(value.substr(0, slash), "meter numerator");
        voice.denominator = positive(value.substr(slash + 1), "meter denominator");
        (void) bar_ticks(voice.numerator, voice.denominator);
        if (lane == 0) add_meter(result.conductor, checked_tick(voice.tick),
                                 voice.numerator, voice.denominator);
    };
    const auto set_key = [&](VoiceState & voice, const std::string & value, int lane) {
        voice.key = trim(value);
        voice.accidentals.clear();
        if (lane == 0) add_key(result.conductor, checked_tick(voice.tick), voice.key);
    };

    for (const auto & raw : lines(abc)) {
        const auto line = trim(raw);
        if (line == "V: Vocal") {
            active_voice = 0;
            found_vocal = true;
            if (!pending_section.empty()) {
                add_text(result.conductor, checked_tick(voices[0].tick), -60, 0x06, pending_section);
                pending_section.clear();
            }
            continue;
        }
        if (line == "V: Ins") {
            active_voice = 1;
            found_instrumental = true;
            continue;
        }
        if (!line.empty() && line.front() == '%') {
            pending_section = trim(line.substr(1));
            continue;
        }
        if (active_voice < 0 || line.empty()) continue;
        auto & voice = voices[static_cast<std::size_t>(active_voice)];
        if (line.rfind("M:", 0) == 0) {
            set_meter(voice, line.substr(2), active_voice);
            continue;
        }
        if (line.rfind("K:", 0) == 0) {
            set_key(voice, line.substr(2), active_voice);
            continue;
        }
        if (line.size() >= 2 && std::isalpha(static_cast<unsigned char>(line[0])) && line[1] == ':') {
            continue;
        }

        for (std::size_t offset = 0; offset < line.size();) {
            if (std::isspace(static_cast<unsigned char>(line[offset]))) { ++offset; continue; }
            if (line[offset] == '|') {
                const auto span = voice.tick - voice.bar_start;
                const auto expected = voice.expected_span != 0
                    ? voice.expected_span
                    : bar_ticks(voice.numerator, voice.denominator);
                if (span != expected) {
                    throw std::invalid_argument("ABC voice has music outside a complete bar");
                }
                voice.bar_start = voice.tick;
                voice.expected_span = 0;
                voice.accidentals.clear();
                ++offset;
                continue;
            }
            if (line[offset] == '"') {
                const auto end = line.find('"', offset + 1);
                if (end == std::string::npos) throw std::invalid_argument("unterminated ABC chord annotation");
                if (active_voice == 0) {
                    result.chords.push_back({checked_tick(voice.tick), line.substr(offset + 1, end - offset - 1)});
                }
                offset = end + 1;
                continue;
            }
            if (line[offset] == '[') {
                const auto end = line.find(']', offset + 1);
                if (end == std::string::npos) throw std::invalid_argument("unterminated ABC inline field");
                const auto field = line.substr(offset + 1, end - offset - 1);
                if (field.rfind("K:", 0) == 0) set_key(voice, field.substr(2), active_voice);
                else if (field.rfind("M:", 0) == 0) set_meter(voice, field.substr(2), active_voice);
                offset = end + 1;
                continue;
            }

            std::string accidental;
            while (offset < line.size() && (line[offset] == '^' || line[offset] == '_' || line[offset] == '=')) {
                accidental.push_back(line[offset++]);
            }
            if (offset >= line.size()) throw std::invalid_argument("dangling ABC accidental");
            const auto symbol = line[offset++];
            if ((symbol >= 'A' && symbol <= 'G') || (symbol >= 'a' && symbol <= 'g')) {
                int octave = std::islower(static_cast<unsigned char>(symbol)) ? 5 : 4;
                while (offset < line.size() && (line[offset] == ',' || line[offset] == '\'')) {
                    octave += line[offset++] == '\'' ? 1 : -1;
                }
                const auto duration = duration_ticks(duration_at(line, offset), unit_denominator);
                const auto begin = checked_tick(voice.tick);
                voice.tick += duration;
                const auto pitch = pitch_for(symbol, octave, accidental, voice.key, voice.accidentals);
                bool tied = false;
                if (offset < line.size() && line[offset] == '-') { tied = true; ++offset; }
                auto & notes = result.notes[static_cast<std::size_t>(active_voice)];
                if (voice.expected_span != 0) {
                    throw std::invalid_argument("ABC multi-bar rest must end at a barline");
                }
                if (voice.tied_note) {
                    if (*voice.tied_note >= notes.size() || !notes[*voice.tied_note].tied ||
                        notes[*voice.tied_note].end != begin ||
                        notes[*voice.tied_note].pitch != pitch) {
                        throw std::invalid_argument("ABC tie is not followed by the same note");
                    }
                    notes[*voice.tied_note].end = checked_tick(voice.tick);
                    notes[*voice.tied_note].tied = tied;
                } else {
                    notes.push_back({begin, checked_tick(voice.tick), pitch, tied});
                    voice.tied_note = notes.size() - 1;
                }
                if (!tied) voice.tied_note.reset();
                continue;
            }
            if (symbol == 'z' || symbol == 'Z') {
                if (voice.tied_note) throw std::invalid_argument("ABC tie is not followed by its note");
                const auto duration = duration_at(line, offset);
                if (symbol == 'Z') {
                    if (duration.denominator != 1) throw std::invalid_argument("fractional ABC multi-bar rest");
                    if (voice.tick != voice.bar_start || voice.expected_span != 0) {
                        throw std::invalid_argument("ABC multi-bar rest must occupy a complete bar span");
                    }
                    voice.expected_span = static_cast<std::uint64_t>(bar_ticks(
                        voice.numerator, voice.denominator)) * duration.numerator;
                    voice.tick += voice.expected_span;
                } else {
                    if (voice.expected_span != 0) {
                        throw std::invalid_argument("ABC multi-bar rest must end at a barline");
                    }
                    voice.tick += duration_ticks(duration, unit_denominator);
                }
                continue;
            }
            throw std::invalid_argument(std::string("unsupported ABC music token: ") + symbol);
        }
    }
    if (!found_vocal || !found_instrumental) {
        throw std::invalid_argument("ABC MIDI export requires native Vocal and Ins lanes");
    }
    for (const auto & voice : voices) {
        if (voice.tied_note) throw std::invalid_argument("ABC score ends with an unresolved tie");
        if (voice.tick != voice.bar_start) {
            throw std::invalid_argument("ABC voice has music outside a complete bar");
        }
    }
    if (voices[0].tick == 0 || voices[0].tick != voices[1].tick) {
        throw std::invalid_argument("ABC MIDI export requires equal nonempty Vocal and Ins timelines");
    }
    result.end_tick = checked_tick(voices[0].tick);
    return result;
}

} // namespace

TranscriptionMidiExports serialize_yue2_abc_midis(const std::string & abc) {
    const auto score = parse_score(abc);
    std::array<std::vector<TimedEvent>, 2> melody_events;
    for (std::size_t lane = 0; lane < score.notes.size(); ++lane) {
        const auto channel = static_cast<std::uint8_t>(lane);
        melody_events[lane].push_back({0, -90, {static_cast<std::uint8_t>(0xc0 | channel), 0}});
        for (const auto & note : score.notes[lane]) {
            const auto pitch = static_cast<std::uint8_t>(note.pitch);
            melody_events[lane].push_back({note.begin, 10,
                {static_cast<std::uint8_t>(0x90 | channel), pitch, 100}});
            melody_events[lane].push_back({note.end, 0,
                {static_cast<std::uint8_t>(0x80 | channel), pitch, 0}});
        }
    }

    std::map<std::uint32_t, std::string> chord_marks;
    for (const auto & chord : score.chords) chord_marks[chord.tick] = chord.label;
    std::vector<TimedEvent> chord_events;
    std::vector<std::pair<std::uint32_t, std::string>> chords(chord_marks.begin(), chord_marks.end());
    for (std::size_t index = 0; index < chords.size(); ++index) {
        const auto begin = chords[index].first;
        const auto end = index + 1 < chords.size() ? chords[index + 1].first : score.end_tick;
        add_text(chord_events, begin, -20, 0x01, chords[index].second);
        for (const auto pitch_value : chord_pitches(chords[index].second)) {
            const auto pitch = static_cast<std::uint8_t>(pitch_value);
            chord_events.push_back({begin, 10, {0x92, pitch, 48}});
            chord_events.push_back({std::max(begin + 1, end), 0, {0x82, pitch, 0}});
        }
    }
    if (!chord_events.empty()) chord_events.push_back({0, -90, {0xc2, 0}});

    const auto conductor = make_track("Conductor", score.conductor);
    const auto vocal = make_track("Vocal Melody", std::move(melody_events[0]));
    const auto instrumental = make_track("Instrument Melody", std::move(melody_events[1]));
    std::vector<std::uint8_t> chords_track;
    if (!chord_events.empty()) chords_track = make_track("Chords", std::move(chord_events));

    std::vector<const std::vector<std::uint8_t> *> melody_tracks = {
        &conductor, &vocal, &instrumental,
    };
    auto combined_tracks = melody_tracks;
    if (!chords_track.empty()) combined_tracks.push_back(&chords_track);

    TranscriptionMidiExports result;
    result.transcription = assemble(combined_tracks);
    result.melody = assemble(melody_tracks);
    result.vocal = assemble({&conductor, &vocal});
    result.instrumental = assemble({&conductor, &instrumental});
    if (!chords_track.empty()) result.chords = assemble({&conductor, &chords_track});
    return result;
}

} // namespace yue2
