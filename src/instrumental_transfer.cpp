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

std::string make_instrumental_transfer_abc(const std::string & abc_text) {
    const auto parsed = abc::parse(abc_text);
    const auto & vocal = parsed.voices[0];
    const auto & ins = parsed.voices[1];
    if (vocal.notes.empty()) return abc_text;  // nothing to move

    // Vocal priority: keep every Vocal note, and trim Ins notes around them.
    std::vector<Span> merged = vocal.notes;
    for (const auto & note : ins.notes) {
        std::vector<Span> pieces = {note};
        for (const auto & v : vocal.notes) {
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
    const auto overlaps = [](const Segment & s, const Span & n) { return n.begin < s.end && n.end > s.begin; };
    for (std::size_t i = 0; i < ins.segments.size(); ++i) {
        for (const auto & v : vocal.notes) if (overlaps(ins.segments[i], v)) { ins_touched[i] = true; break; }
    }
    std::vector<Span> tied = merged;
    tied.insert(tied.end(), ins.notes.begin(), ins.notes.end());
    for (bool changed = true; changed;) {
        changed = false;
        for (const auto & n : tied) {
            bool any = false;
            for (std::size_t i = 0; i < ins.segments.size(); ++i) if (ins_touched[i] && overlaps(ins.segments[i], n)) any = true;
            if (!any) continue;
            for (std::size_t i = 0; i < ins.segments.size(); ++i) {
                if (!ins_touched[i] && overlaps(ins.segments[i], n)) { ins_touched[i] = true; changed = true; }
            }
        }
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
