// A transcription that opens with a pickup in 4/8 must still write the tempo
// of the song's 4/4 body, and a cover can drop that pickup and start on the
// first detected downbeat. The tempo used to be converted with the first
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

    // A lead-in the tracker did not lock onto: a 4/8 label and an anacrusis
    // note before the first downbeat at step 14.
    std::vector<yue2::ScoreEvent> late;
    {
        auto lead = downbeat(0, 4, 8, 55);
        lead.eighth_position = -1;  // not a downbeat
        lead.structure = "intro";
        lead.notes.front().duration_steps = 2;
        late.push_back(lead);
        for (int bar = 0; bar < 8; ++bar) late.push_back(downbeat(14 + 16 * bar, 4, 4, 62 + bar % 3));
    }
    const auto kept = yue2::serialize_sheetsage2_abc(late, false, yue2::Pickup::keep);
    const auto dropped = yue2::serialize_sheetsage2_abc(late, false, yue2::Pickup::drop);
    // Kept, the lead-in is a bar of its own ahead of the first downbeat.
    assert(kept.find("\nM:4/4") != std::string::npos);
    const auto count_bars = [](const std::string & abc) {
        std::size_t bars = 0;
        bool ins = false;
        std::size_t start = 0;
        while (start < abc.size()) {
            const auto end = abc.find('\n', start);
            const auto line = abc.substr(start, end == std::string::npos ? std::string::npos : end - start);
            if (line.rfind("V: Ins", 0) == 0) ins = true;
            else if (line.rfind("V: Vocal", 0) == 0) ins = false;
            else if (ins && !line.empty() && line[0] != '%' && line.find(':') != 1) {
                for (const char c : line) if (c == '|') ++bars;
            }
            if (end == std::string::npos) break;
            start = end + 1;
        }
        return bars;
    };
    // Dropped: bar one is the first downbeat, so exactly the eight bars remain,
    // one fewer than kept, in the body's meter throughout.
    assert(count_bars(dropped) == 8);
    assert(count_bars(kept) == 9);
    assert(dropped.find("M:4/8") == std::string::npos);
    assert(dropped.find("\nM:4/4\n") != std::string::npos);
    // The anacrusis note (G3, written G,) is gone; the section label is not.
    assert(kept.find("G,") != std::string::npos);
    assert(dropped.find("G,") == std::string::npos);
    assert(dropped.find("% intro") != std::string::npos);
    assert(tempo_line(dropped) == "Q:1/4=120");

    // The other shape, from a real clip: the tracker opens on a downbeat but
    // calls the lead-in a 7/8 bar, then switches to the song's 4/4 where the
    // melody enters. The 7/8 bar has a chord label and the section, no notes.
    std::vector<yue2::ScoreEvent> odd;
    {
        auto intro = downbeat(0, 7, 8, 55);
        intro.notes.clear();
        intro.chord = "C:min";
        intro.structure = "intro";
        odd.push_back(intro);
        for (int bar = 0; bar < 8; ++bar) odd.push_back(downbeat(14 + 16 * bar, 4, 4, 62 + bar % 3));
    }
    const auto odd_kept = yue2::serialize_sheetsage2_abc(odd, false, yue2::Pickup::keep);
    const auto odd_dropped = yue2::serialize_sheetsage2_abc(odd, false, yue2::Pickup::drop);
    assert(count_bars(odd_kept) == 9);
    assert(count_bars(odd_dropped) == 8);
    assert(odd_dropped.find("M:4/8") == std::string::npos);
    assert(odd_dropped.find("M:7/8") == std::string::npos);
    assert(odd_dropped.find("% intro") != std::string::npos);
    assert(tempo_line(odd_dropped) == "Q:1/4=120");
    return 0;
}
