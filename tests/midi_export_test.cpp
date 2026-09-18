#include "yue2/transcription.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <iterator>
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

    std::vector<yue2::ScoreEvent> changing_meter(4);
    changing_meter[0].subbeat = 0;
    changing_meter[0].time_seconds = 0.0;
    changing_meter[0].has_timestamp = true;
    changing_meter[0].meter_numerator = 4;
    changing_meter[0].meter_denominator = 4;
    changing_meter[0].eighth_position = 0;
    changing_meter[0].notes.push_back({60, 0, 10, 48, 0.0});
    changing_meter[1].subbeat = 16;
    changing_meter[1].time_seconds = 2.0;
    changing_meter[1].has_timestamp = true;
    changing_meter[1].meter_numerator = 3;
    changing_meter[1].meter_denominator = 4;
    changing_meter[1].eighth_position = 0;
    changing_meter[2].subbeat = 28;
    changing_meter[2].time_seconds = 3.5;
    changing_meter[2].has_timestamp = true;
    changing_meter[2].meter_numerator = 5;
    changing_meter[2].meter_denominator = 8;
    changing_meter[2].eighth_position = 0;
    changing_meter[3].subbeat = 48;
    changing_meter[3].time_seconds = 4.75;
    changing_meter[3].has_timestamp = true;
    changing_meter[3].meter_numerator = 5;
    changing_meter[3].meter_denominator = 8;
    changing_meter[3].eighth_position = 0;
    const auto changing = parse_midi(
        yue2::serialize_sheetsage2_midi(changing_meter, true));
    const auto & changing_conductor = track_named(changing, "Conductor");
    std::vector<MidiMeta> signatures;
    std::copy_if(
        changing_conductor.meta.begin(), changing_conductor.meta.end(),
        std::back_inserter(signatures), [](const auto & meta) { return meta.type == 0x58; });
    assert(signatures.size() == 3);
    assert(signatures[0].tick == 0 && signatures[0].payload[0] == 4 && signatures[0].payload[1] == 2);
    assert(signatures[1].tick == 3840 && signatures[1].payload[0] == 3 && signatures[1].payload[1] == 2);
    assert(signatures[2].tick == 6720 && signatures[2].payload[0] == 5 && signatures[2].payload[1] == 3);
    assert(track_named(changing, "Vocal Melody").end_tick == 9120);

    std::vector<yue2::ScoreEvent> pickup(2);
    pickup[0].subbeat = 0;
    pickup[0].time_seconds = 0.0;
    pickup[0].has_timestamp = true;
    pickup[0].notes.push_back({64, 1, 3, 4, 0.0});
    pickup[1].subbeat = 4;
    pickup[1].time_seconds = 0.631578947368421;
    pickup[1].has_timestamp = true;
    pickup[1].meter_numerator = 4;
    pickup[1].meter_denominator = 4;
    pickup[1].eighth_position = 0;
    pickup[1].notes.push_back({67, 1, 7, 16, 0.0});
    const auto pickup_midi = parse_midi(
        yue2::serialize_sheetsage2_midi(pickup, true, 3.157894736842105));
    const auto & pickup_notes = track_named(pickup_midi, "Instrument Melody").notes;
    assert(pickup_notes.size() == 2);
    assert(pickup_notes[0].tick == 2880);
    assert(pickup_notes[1].tick == 3840);

    std::vector<yue2::ScoreEvent> inversion(2);
    inversion[0].subbeat = 0;
    inversion[0].time_seconds = 0.0;
    inversion[0].has_timestamp = true;
    inversion[0].meter_numerator = 4;
    inversion[0].meter_denominator = 4;
    inversion[0].eighth_position = 0;
    inversion[0].chord = "C:maj/3";
    inversion[1].subbeat = 8;
    inversion[1].time_seconds = 1.0;
    inversion[1].has_timestamp = true;
    const auto inversion_midi = parse_midi(
        yue2::serialize_sheetsage2_midi(inversion, false, 2.0));
    const auto & inversion_notes = track_named(inversion_midi, "Chords").notes;
    assert(inversion_notes.size() == 4);
    assert(std::any_of(inversion_notes.begin(), inversion_notes.end(), [](const auto & note) {
        return note.pitch == 40;
    }));

    const std::string native_abc =
        "X:1\nT:\nM:4/4\nL:1/32\nQ:1/4=95\n"
        "V: Vocal clef=treble\nV: Ins clef=treble\nK:C#m\n% verse\n"
        "V: Vocal\n\"C#m\"^F8-^F8G16|Z|\n"
        "V: Ins\nZ|C,32|\n% bridge\n"
        "V: Vocal\nZ2|\n"
        "V: Ins\nD,32|E,32|\n";
    const auto native_exports = yue2::serialize_yue2_abc_midis(native_abc);
    const auto native_midi = parse_midi(native_exports.transcription);
    assert(native_midi.format == 1 && native_midi.ppq == 960);
    assert(native_midi.tracks.size() == 4);
    const auto & native_conductor = track_named(native_midi, "Conductor");
    assert(meta_of_type(native_conductor, 0x51).payload ==
        std::vector<std::uint8_t>({0x09, 0xa3, 0x1b}));
    assert(meta_of_type(native_conductor, 0x59).payload ==
        std::vector<std::uint8_t>({4, 1}));
    const auto & native_vocal = track_named(native_midi, "Vocal Melody");
    assert(native_vocal.notes.size() == 2);
    assert(native_vocal.notes[0].pitch == 66 && native_vocal.notes[0].tick == 0);
    assert(native_vocal.notes[1].pitch == 68 && native_vocal.notes[1].tick == 1920);
    assert(track_named(native_midi, "Instrument Melody").notes.size() == 3);
    assert(!native_exports.melody.empty() && !native_exports.vocal.empty() &&
           !native_exports.instrumental.empty() && !native_exports.chords.empty());

    bool rejected_partial_bar = false;
    try {
        (void)yue2::serialize_yue2_abc_midis(
            "X:1\nM:4/4\nL:1/32\nQ:1/4=95\nV: Vocal\nC8|\nV: Ins\nZ|\n");
    } catch (const std::invalid_argument &) {
        rejected_partial_bar = true;
    }
    assert(rejected_partial_bar);

    bool rejected_missing_barline = false;
    try {
        (void)yue2::serialize_yue2_abc_midis(
            "X:1\nM:4/4\nL:1/32\nQ:1/4=95\nV: Vocal\nC32D32|\nV: Ins\nZ2|\n");
    } catch (const std::invalid_argument &) {
        rejected_missing_barline = true;
    }
    assert(rejected_missing_barline);

    bool rejected_broken_tie = false;
    try {
        (void)yue2::serialize_yue2_abc_midis(
            "X:1\nM:4/4\nL:1/32\nQ:1/4=95\nV: Vocal\nC16-D16|\nV: Ins\nZ|\n");
    } catch (const std::invalid_argument &) {
        rejected_broken_tie = true;
    }
    assert(rejected_broken_tie);

    std::cout << "format-1 MIDI tracks, grid timing, native ABC, meter changes, pickups, and inversions: ok\n";
    return 0;
}
