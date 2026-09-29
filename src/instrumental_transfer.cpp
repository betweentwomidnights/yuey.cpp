// Instrumental by melody transfer: the Vocal lane's notes move into the Ins
// lane, and the Vocal lane keeps only rests and its chord symbols. This is
// upstream YuE2's instrumental method (its yue2-music skill), which keeps the
// sung melody playing on an instrument where resting the Vocal lane drops it.
//
// Overlaps follow upstream's vocal priority: every Vocal note is kept, and an
// Ins note it overlaps is trimmed around it. Only bars that change are
// rewritten; every other bar keeps its exact text, so a score with nothing on
// the Vocal lane comes back byte for byte. Rewritten notes carry explicit
// accidentals, as SheetSage2's own scores do, so no bar-local accidental state
// can leak between the two voices' spellings.
#include "yue2/generation.h"

#include "abc_score.h"

#include <algorithm>
#include <map>
#include <string>
#include <vector>

namespace yue2 {

using abc::Segment;
using abc::Span;

namespace {

// The segment holding `tick`; a lane's segments are contiguous from tick 0.
std::size_t segment_at(const std::vector<Segment> & segments, std::uint64_t tick) {
    const auto after = std::upper_bound(
        segments.begin(), segments.end(), tick,
        [](std::uint64_t value, const Segment & segment) { return value < segment.begin; });
    return static_cast<std::size_t>(after - segments.begin()) - 1;
}

} // namespace

std::string make_instrumental_transfer_abc(const std::string & abc_text) {
    const auto parsed = abc::parse(abc_text);
    const auto & vocal = parsed.voices[0];
    const auto & ins = parsed.voices[1];
    if (vocal.notes.empty()) return abc_text;  // nothing to move

    // Vocal priority: keep every Vocal note, and trim Ins notes around them.
    std::vector<Span> merged = vocal.notes;
    for (const auto & note : ins.notes) {
        std::vector<Span> pieces = {note};
        for (auto index = abc::first_reaching(vocal.notes, note.begin);
             index < vocal.notes.size() && vocal.notes[index].begin < note.end; ++index) {
            const auto & v = vocal.notes[index];
            std::vector<Span> next;
            for (const auto & p : pieces) {
                if (v.end <= p.begin || v.begin >= p.end) { next.push_back(p); continue; }
                if (p.begin < v.begin) next.push_back({p.begin, v.begin, p.pitch});
                if (v.end < p.end) next.push_back({v.end, p.end, p.pitch});
            }
            pieces.swap(next);
        }
        merged.insert(merged.end(), pieces.begin(), pieces.end());
    }
    std::sort(merged.begin(), merged.end(), [](const Span & a, const Span & b) { return a.begin < b.begin; });

    // Bars to rewrite: every Ins bar a Vocal note reaches, then any bar a note
    // is tied across into one of those, old or new, until nothing changes. The
    // old notes count too: a tie out of a kept bar must not land on a bar that
    // no longer continues it.
    std::vector<bool> ins_touched(ins.segments.size(), false);
    for (std::size_t i = 0; i < ins.segments.size(); ++i) {
        const auto & segment = ins.segments[i];
        const auto first = abc::first_reaching(vocal.notes, segment.begin);
        ins_touched[i] = first < vocal.notes.size() && vocal.notes[first].begin < segment.end;
    }
    // joined[i]: some note, old or new, runs across the barline into segment
    // i. Bars joined that way are rewritten together, so a run of them with
    // one touched bar is touched throughout.
    std::vector<bool> joined(ins.segments.size(), false);
    const auto join = [&](const Span & note) {
        const auto last = segment_at(ins.segments, note.end - 1);
        for (auto i = segment_at(ins.segments, note.begin) + 1; i <= last; ++i) joined[i] = true;
    };
    for (const auto & note : merged) join(note);
    for (const auto & note : ins.notes) join(note);
    for (std::size_t i = 1; i < ins.segments.size(); ++i) {
        if (joined[i] && ins_touched[i - 1]) ins_touched[i] = true;
    }
    for (std::size_t i = ins.segments.size(); i-- > 1;) {
        if (joined[i] && ins_touched[i]) ins_touched[i - 1] = true;
    }

    auto line_segments = parsed.line_segments;
    std::vector<bool> line_changed(line_segments.size(), false);
    const auto unit_ticks = parsed.unit_ticks();
    for (const auto & segment : vocal.segments) {
        if (!segment.has_notes) continue;
        line_segments[segment.line][segment.index] =
            abc::write_bars(segment, {}, unit_ticks, segment.key);
        line_changed[segment.line] = true;
    }
    for (std::size_t i = 0; i < ins.segments.size(); ++i) {
        if (!ins_touched[i]) continue;
        const auto & segment = ins.segments[i];
        line_segments[segment.line][segment.index] =
            abc::write_bars(segment, merged, unit_ticks, segment.key);
        line_changed[segment.line] = true;
    }
    return abc::assemble(abc_text, line_segments, line_changed, {}, "melody transfer");
}

} // namespace yue2
