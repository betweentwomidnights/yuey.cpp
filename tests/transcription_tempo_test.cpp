// A transcription that opens with a pickup in 4/8 must still write the tempo
// of the song's 4/4 body. The tempo used to be converted with the first
// meter's denominator, so an 8 there halved Q:, and every render of the score
// played the melody at half speed for twice the source's length.
#include "yue2/transcription.h"

#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

namespace {

// 120 bpm: a quarter is 0.5 s, and a step is a sixteenth in 4/4.
constexpr double kSecondsPerStep = 0.125;

yue2::ScoreEvent downbeat(std::int64_t subbeat, int numerator, int denominator, int pitch) {
    yue2::ScoreEvent event;
    event.subbeat = subbeat;
    event.source_subbeat = subbeat;
    event.time_seconds = static_cast<double>(subbeat) * kSecondsPerStep;
    event.has_timestamp = true;
    event.meter_numerator = numerator;
    event.meter_denominator = denominator;
    event.eighth_position = 0;
    event.chord = "C";
    yue2::NoteEvent note;
    note.pitch = pitch;
    note.track = 1;
    note.duration_steps = 4;
    note.end_time_seconds = event.time_seconds + 4 * kSecondsPerStep;
    event.notes.push_back(note);
    return event;
}

std::string tempo_line(const std::string & abc) {
    const auto at = abc.find("\nQ:");
    assert(at != std::string::npos);
    return abc.substr(at + 1, abc.find('\n', at + 1) - at - 1);
}

} // namespace

int main() {
    std::vector<yue2::ScoreEvent> events;
    // A 4/8 pickup of 16 steps, then eight bars of 4/4.
    events.push_back(downbeat(0, 4, 8, 60));
    for (int bar = 0; bar < 8; ++bar) {
        events.push_back(downbeat(16 + 16 * bar, 4, 4, 62 + bar % 3));
    }

    const auto abc = yue2::serialize_sheetsage2_abc(events, false);
    std::printf("%s\n", tempo_line(abc).c_str());
    assert(tempo_line(abc) == "Q:1/4=120");

    // The same body without the pickup has always read correctly.
    events.erase(events.begin());
    assert(tempo_line(yue2::serialize_sheetsage2_abc(events, false)) == "Q:1/4=120");
    return 0;
}
