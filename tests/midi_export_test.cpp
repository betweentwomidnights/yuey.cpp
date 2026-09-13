#include "yue2/transcription.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct MidiNote {
    std::uint32_t tick = 0;
    std::uint8_t channel = 0;
    std::uint8_t pitch = 0;
    std::uint8_t velocity = 0;
};

struct MidiMeta {
    std::uint32_t tick = 0;
    std::uint8_t type = 0;
    std::vector<std::uint8_t> payload;
};

struct MidiTrack {
    std::string name;
    std::vector<MidiNote> notes;
    std::vector<MidiMeta> meta;
    std::uint32_t end_tick = 0;
};

struct MidiFile {
    std::uint16_t format = 0;
    std::uint16_t ppq = 0;
    std::vector<MidiTrack> tracks;
};

std::uint16_t u16(const std::vector<std::uint8_t> & data, std::size_t offset) {
    if (offset + 2 > data.size()) throw std::runtime_error("short MIDI u16");
    return static_cast<std::uint16_t>((data[offset] << 8U) | data[offset + 1]);
}

std::uint32_t u32(const std::vector<std::uint8_t> & data, std::size_t offset) {
    if (offset + 4 > data.size()) throw std::runtime_error("short MIDI u32");
    return (static_cast<std::uint32_t>(data[offset]) << 24U) |
        (static_cast<std::uint32_t>(data[offset + 1]) << 16U) |
        (static_cast<std::uint32_t>(data[offset + 2]) << 8U) |
        data[offset + 3];
}

std::uint32_t vlq(
    const std::vector<std::uint8_t> & data,
    std::size_t & offset,
    std::size_t end) {
    std::uint32_t value = 0;
    for (int count = 0; count < 4; ++count) {
        if (offset >= end) throw std::runtime_error("short MIDI VLQ");
        const auto byte = data[offset++];
        value = (value << 7U) | (byte & 0x7fU);
        if ((byte & 0x80U) == 0) return value;
    }
    throw std::runtime_error("oversized MIDI VLQ");
}

MidiFile parse_midi(const std::vector<std::uint8_t> & data) {
    if (data.size() < 14 || std::string(data.begin(), data.begin() + 4) != "MThd" ||
        u32(data, 4) != 6) {
        throw std::runtime_error("invalid MIDI header");
    }
    MidiFile result;
    result.format = u16(data, 8);
    const auto track_count = u16(data, 10);
    result.ppq = u16(data, 12);
    std::size_t offset = 14;
    for (std::uint16_t track_index = 0; track_index < track_count; ++track_index) {
        if (offset + 8 > data.size() ||
            std::string(data.begin() + offset, data.begin() + offset + 4) != "MTrk") {
            throw std::runtime_error("invalid MIDI track header");
        }
        const auto length = u32(data, offset + 4);
        offset += 8;
        const auto end = offset + length;
        if (end > data.size()) throw std::runtime_error("short MIDI track");
        MidiTrack track;
        std::uint32_t tick = 0;
        std::uint8_t running = 0;
        while (offset < end) {
            tick += vlq(data, offset, end);
            if (offset >= end) throw std::runtime_error("missing MIDI event");
            auto status = data[offset];
            if (status < 0x80) {
                if (running == 0) throw std::runtime_error("invalid MIDI running status");
                status = running;
            } else {
                ++offset;
                if (status < 0xf0) running = status;
            }
            if (status == 0xff) {
                if (offset >= end) throw std::runtime_error("short MIDI meta event");
                const auto type = data[offset++];
                const auto size = vlq(data, offset, end);
                if (offset + size > end) throw std::runtime_error("short MIDI meta payload");
                MidiMeta meta{tick, type, {data.begin() + offset, data.begin() + offset + size}};
                if (type == 0x03) track.name.assign(meta.payload.begin(), meta.payload.end());
                track.meta.push_back(std::move(meta));
                offset += size;
                continue;
            }
            if (status == 0xf0 || status == 0xf7) {
                const auto size = vlq(data, offset, end);
                if (offset + size > end) throw std::runtime_error("short MIDI sysex payload");
                offset += size;
                continue;
            }
            const auto kind = status & 0xf0U;
            const auto size = kind == 0xc0 || kind == 0xd0 ? 1U : 2U;
            if (offset + size > end) throw std::runtime_error("short MIDI channel event");
            if (kind == 0x90 && data[offset + 1] != 0) {
                track.notes.push_back({
                    tick,
                    static_cast<std::uint8_t>(status & 0x0fU),
                    data[offset],
                    data[offset + 1],
                });
            }
            offset += size;
        }
        track.end_tick = tick;
        result.tracks.push_back(std::move(track));
    }
    if (offset != data.size()) throw std::runtime_error("trailing MIDI bytes");
    return result;
}

const MidiTrack & track_named(const MidiFile & midi, const std::string & name) {
    const auto found = std::find_if(midi.tracks.begin(), midi.tracks.end(), [&](const auto & track) {
        return track.name == name;
    });
    if (found == midi.tracks.end()) throw std::runtime_error("missing MIDI track " + name);
    return *found;
}

const MidiMeta & meta_of_type(const MidiTrack & track, std::uint8_t type) {
    const auto found = std::find_if(track.meta.begin(), track.meta.end(), [&](const auto & meta) {
        return meta.type == type;
    });
    if (found == track.meta.end()) throw std::runtime_error("missing MIDI meta event");
    return *found;
}

} // namespace

int main() {
    std::vector<yue2::ScoreEvent> events(4);
    events[0].subbeat = 0;
    events[0].time_seconds = 0.0;
    events[0].has_timestamp = true;
    events[0].meter_numerator = 4;
    events[0].meter_denominator = 4;
    events[0].eighth_position = 0;
    events[0].structure = "intro";
    events[0].key = "C#:minor";
    events[0].chord = "C#:min";
    events[0].notes.push_back({72, 0, 3, 4, 0.0});
    events[0].notes.push_back({76, 1, 3, 4, 0.0});

    events[1].subbeat = 8;
    events[1].time_seconds = 1.263157894736842;
    events[1].has_timestamp = true;
    events[1].chord = "A:maj";

    events[2].subbeat = 16;
    events[2].time_seconds = 2.526315789473684;
    events[2].has_timestamp = true;
    events[2].meter_numerator = 4;
    events[2].meter_denominator = 4;
    events[2].eighth_position = 0;
    events[2].structure = "verse";
    events[2].chord = "B:maj";
    events[2].notes.push_back({67, 1, 3, 4, 0.0});

    events[3].subbeat = 24;
    events[3].time_seconds = 3.789473684210526;
    events[3].has_timestamp = true;

    constexpr double duration = 5.052631578947368;
    const auto exports = yue2::serialize_sheetsage2_midis(events, false, duration);
    assert(exports.transcription == yue2::serialize_sheetsage2_midi(events, false, duration));
    const auto midi = parse_midi(exports.transcription);
    assert(midi.format == 1);
    assert(midi.ppq == 960);
    assert(midi.tracks.size() == 4);

    const auto & conductor = track_named(midi, "Conductor");
    const auto & tempo = meta_of_type(conductor, 0x51);
    assert(tempo.tick == 0 && tempo.payload == std::vector<std::uint8_t>({0x09, 0xa3, 0x1b}));
    const auto & meter = meta_of_type(conductor, 0x58);
    assert(meter.tick == 0 && meter.payload == std::vector<std::uint8_t>({4, 2, 24, 8}));
    const auto & key = meta_of_type(conductor, 0x59);
    assert(key.tick == 0 && key.payload == std::vector<std::uint8_t>({4, 1}));
    const auto marker_count = std::count_if(conductor.meta.begin(), conductor.meta.end(), [](const auto & meta) {
        return meta.type == 0x06;
    });
    assert(marker_count == 2);

    const auto & vocal = track_named(midi, "Vocal Melody");
    assert(vocal.notes.size() == 1 && vocal.notes[0].channel == 0 && vocal.notes[0].pitch == 72);
    const auto & instrumental = track_named(midi, "Instrument Melody");
    assert(instrumental.notes.size() == 2);
    assert(instrumental.notes[0].channel == 1 && instrumental.notes[0].pitch == 76);

    const auto & chords = track_named(midi, "Chords");
    assert(chords.notes.size() == 12);
    assert(chords.notes.front().tick == 0 && chords.notes.front().channel == 2);
    assert(std::any_of(chords.notes.begin(), chords.notes.end(), [](const auto & note) {
        return note.tick == 0 && note.pitch == 37;
    }));
    assert(std::any_of(chords.notes.begin(), chords.notes.end(), [](const auto & note) {
        return note.tick == 1920 && note.pitch == 45;
    }));
    assert(std::any_of(chords.notes.begin(), chords.notes.end(), [](const auto & note) {
        return note.tick == 3840 && note.pitch == 47;
    }));
    assert(chords.end_tick == 7680);

    assert(parse_midi(exports.melody).tracks.size() == 3);
    assert(parse_midi(exports.vocal).tracks.size() == 2);
    assert(parse_midi(exports.instrumental).tracks.size() == 2);
    assert(parse_midi(exports.chords).tracks.size() == 2);

    const auto melody_exports = yue2::serialize_sheetsage2_midis(events, true, duration);
    assert(melody_exports.chords.empty());
    const auto melody = parse_midi(melody_exports.transcription);
    assert(melody.format == 1 && melody.tracks.size() == 3);
    assert(std::none_of(melody.tracks.begin(), melody.tracks.end(), [](const auto & track) {
        return track.name == "Chords";
    }));

    std::cout << "format-1 MIDI conductor, split melodies, chords, and grid timing: ok\n";
    return 0;
}
