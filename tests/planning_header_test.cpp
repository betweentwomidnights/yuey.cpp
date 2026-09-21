#include "yue2/generation.h"
#include "yue2/transcription.h"

#include <cassert>
#include <cmath>
#include <stdexcept>
#include <string>

namespace {

bool rejected(const yue2::PlanningHeader & header) {
    try {
        (void)yue2::make_planning_abc_prefix(header);
    } catch (const std::invalid_argument &) {
        return true;
    }
    return false;
}

} // namespace

int main() {
    assert(yue2::normalize_abc_key("C# minor") == "C#m");
    assert(yue2::normalize_abc_key(" c#:MINOR ") == "C#m");
    // Flats are accepted but never returned: the plugin, the web UI and this
    // API all name pitches with sharps, and Db and C# are the same key.
    assert(yue2::normalize_abc_key("Db major") == "C#");
    assert(yue2::normalize_abc_key("Eb minor") == "D#m");
    assert(yue2::normalize_abc_key("Bb major") == "A#");
    assert(yue2::normalize_abc_key("Cb major") == "B");
    assert(yue2::normalize_abc_key("Fb minor") == "Em");
    assert(yue2::normalize_abc_key("Am") == "Am");
    assert(yue2::normalize_abc_key("F#maj") == "F#");

    const auto prefix = yue2::make_planning_abc_prefix({95, 4, 4, "C# minor"});
    assert(prefix ==
        "X:1\nT:\nM:4/4\nL:1/32\nQ:1/4=95\n"
        "V: Vocal clef=treble name=\"Vocal Melody\" snm=\"Vocal\"\n"
        "V: Ins clef=treble name=\"Ins Melody\" snm=\"Inst.\"\n"
        "K:C#m\n% intro\n");
    assert(prefix.back() == '\n');

    // A Create request sends typed planning controls, which become this
    // header-only prefix. Planning inspects its prefix before sampling, so an
    // empty score has to be a legitimate answer or every planned song fails
    // before the model runs. A completed plan still requires both lanes.
    const auto prefix_info = yue2::inspect_abc_score(prefix, true);
    assert(prefix_info.bars == 0);
    assert(prefix_info.bpm == 95);
    assert(prefix_info.meter_numerator == 4 && prefix_info.meter_denominator == 4);
    assert(prefix_info.duration_seconds == 0.0);

    bool strict_prefix_rejected = false;
    try {
        (void)yue2::inspect_abc_score(prefix);
    } catch (const std::invalid_argument &) {
        strict_prefix_rejected = true;
    }
    assert(strict_prefix_rejected);

    const std::string score =
        "X:1\nM:4/4\nL:1/32\nQ:1/4=95\n"
        "V: Vocal clef=treble name=\"Vocal Melody\" snm=\"Vocal\"\n"
        "V: Ins clef=treble name=\"Ins Melody\" snm=\"Inst.\"\n"
        "K:C#m\n% verse\n"
        "V: Vocal\nM:4/4\n\"C#m\"^F8-^F8G16|\n"
        "V: Ins\nM:4/4\nZ|\n"
        "% bridge\n"
        "V: Vocal\nZ2|\n"
        "V: Ins\nc8B8A16|G32|\n";
    const std::string instrumental =
        "X:1\nM:4/4\nL:1/32\nQ:1/4=95\n"
        "V: Vocal clef=treble name=\"Vocal Melody\" snm=\"Vocal\"\n"
        "V: Ins clef=treble name=\"Ins Melody\" snm=\"Inst.\"\n"
        "K:C#m\n% verse\n"
        "V: Vocal\nM:4/4\n\"C#m\"z8z8z16|\n"
        "V: Ins\nM:4/4\nZ|\n"
        "% bridge\n"
        "V: Vocal\nZ2|\n"
        "V: Ins\nc8B8A16|G32|\n";
    assert(yue2::make_vocal_rest_abc(score) == instrumental);
    assert(yue2::make_vocal_rest_abc(instrumental) == instrumental);
    assert(yue2::inspect_abc_score(score).bars == 3);
    assert(yue2::make_instrumental_lyrics(score) == "[Verse]\n\n[Bridge]");
    assert(yue2::make_instrumental_lyrics("X:1\nM:4/4\nK:C\nC4|\n") ==
        "[Intro]\n\n[Verse]\n\n[Chorus]\n\n[Verse]\n\n[Chorus]\n\n[Outro]");
    assert(yue2::make_instrumental_lyrics(
        "X:1\n% pre_chorus\n% VERSE 2\n% ignored!\n") ==
        "[Pre Chorus]\n\n[Verse 2]");

    const std::string long_score =
        "X:1\nM:4/4\nL:1/32\nQ:1/4=120\n"
        "V: Vocal clef=treble\nV: Ins clef=treble\nK:C\n% intro\n"
        "V: Vocal\n\"C\"C32|\"F\"D32|\"G\"E32|\"C\"F32|\n"
        "V: Ins\nC,32|F,32|G,32|C,32|\n% outro\n"
        "V: Vocal\n\"Am\"G32|\"F\"A32|\"G\"B32|\"C\"c32|\n"
        "V: Ins\nA,32|F,32|G,32|C32|\n";
    const auto long_info = yue2::inspect_abc_score(long_score);
    assert(long_info.bars == 8 && long_info.bpm == 120);
    assert(long_info.meter_numerator == 4 && long_info.meter_denominator == 4);
    assert(std::abs(long_info.duration_seconds - 16.0) < 1.0e-9);
    assert(yue2::score_aligned_semantic_budget(long_info) == 1150);

    const auto fitted = yue2::fit_abc_score_to_bars(long_score, 4, 2);
    const auto fitted_info = yue2::inspect_abc_score(fitted);
    assert(fitted_info.bars == 4);
    assert(fitted.find("\"C\"C32|\"F\"D32|") != std::string::npos);
    assert(fitted.find("\"G\"B32|\"C\"c32|") != std::string::npos);
    assert(fitted.find("\"G\"E32|") == std::string::npos);
    assert(fitted.find("% outro") != std::string::npos);

    // long_score is 8 bars of 4/4 at 120bpm, so 2 seconds a bar, 16 total.
    // A ceiling above it leaves the plan alone; zero and a negative lift it.
    assert(yue2::fit_natural_plan_to_ceiling(long_score, 16.0) == long_score);
    assert(yue2::fit_natural_plan_to_ceiling(long_score, 100.0) == long_score);
    assert(yue2::fit_natural_plan_to_ceiling(long_score, 0.0) == long_score);

    // A ten second ceiling allows five bars, and the score still ends: the
    // planner's real tail is lifted in as the outro rather than cut off.
    const auto held = yue2::fit_natural_plan_to_ceiling(long_score, 10.0);
    const auto held_info = yue2::inspect_abc_score(held);
    assert(held_info.bars == 5);
    assert(held_info.duration_seconds <= 10.0);
    assert(held.find("% outro") != std::string::npos);
    assert(held.find("\"C\"C32|") != std::string::npos);

    // A ceiling tighter than one bar still yields a renderable bar rather than
    // an empty score, and a tiny allowance shrinks the outro to fit.
    const auto floor_bar = yue2::fit_natural_plan_to_ceiling(long_score, 0.5);
    assert(yue2::inspect_abc_score(floor_bar).bars == 1);

    // A plan that runs out of token budget stops part way through a bar and
    // leaves an unpaired Vocal block. This is the shape a dense full-mode
    // instrumental plan actually produced against the model; everything before
    // the cut is still real music.
    const std::string truncated = long_score +
        "% verse\n"
        "V: Vocal\n\"Cm\"z32|\"Cm\"z32|\n"
        "V: Ins\nc12c12e8|d12B12c8|\n"
        "V: Vocal\n\"Cm\"z32|\"Cm\"z32|\"Cm";
    bool truncated_rejected = false;
    try {
        (void)yue2::inspect_abc_score(truncated);
    } catch (const std::invalid_argument &) {
        truncated_rejected = true;
    }
    assert(truncated_rejected);

    // The complete Vocal/Ins pair that did finish is kept, the unfinished one
    // is dropped, and the result parses and renders.
    const auto salvaged = yue2::trim_to_complete_score(truncated);
    const auto salvaged_info = yue2::inspect_abc_score(salvaged);
    assert(salvaged_info.bars == 10);
    assert(salvaged.find("% verse") != std::string::npos);
    assert(salvaged.find("c12c12e8|d12B12c8|") != std::string::npos);

    // A score that already parses is handed back untouched.
    assert(yue2::trim_to_complete_score(long_score) == long_score);

    // Nothing salvageable is returned as-is, so the caller still sees the
    // original failure rather than a silent empty score.
    const std::string headerless = "X:1\nM:4/4\nL:1/32\nQ:1/4=120\nK:C\n% intro\nV: Vocal\nc4";
    assert(yue2::trim_to_complete_score(headerless) == headerless);

    // The margin past the final bar is absolute, not proportional. It used to
    // be max(1.75x, +30s), which meant the delivered audio grew without bound
    // because generation spends the whole budget rather than stopping at
    // MUSIC_END: a 180s score rendered 315s and a 460s one rendered 805s.
    {
        yue2::AbcScoreInfo timed;
        timed.bars = 8;
        timed.duration_seconds = 16.0;
        // Short scores are untouched: +30s already won there.
        assert(yue2::score_aligned_semantic_budget(timed) == 1150);

        timed.bars = 90;
        timed.duration_seconds = 180.0;
        assert(yue2::score_aligned_semantic_budget(timed) == 5250); // was 7875

        timed.bars = 230;
        timed.duration_seconds = 460.0;
        assert(yue2::score_aligned_semantic_budget(timed) == 12250); // was 20125
    }

    // A continuation is measured by what it adds, so its prefix is retained
    // whatever the ceiling says and the result is never shorter than its source.
    {
        // long_score is 8 bars of 4/4 at 120bpm: 2 seconds a bar.
        // Six bars of prefix plus a four second allowance keeps eight bars, so
        // the score is already within its budget and comes back untouched.
        assert(yue2::fit_natural_plan_to_ceiling(long_score, 4.0, 4, 6) == long_score);

        // The same six-bar prefix with a two second allowance keeps seven.
        const auto extended = yue2::fit_natural_plan_to_ceiling(long_score, 2.0, 2, 6);
        assert(yue2::inspect_abc_score(extended).bars == 7);

        // A prefix at or beyond the score is handed back whole rather than cut.
        assert(yue2::fit_natural_plan_to_ceiling(long_score, 2.0, 2, 8) == long_score);
        assert(yue2::fit_natural_plan_to_ceiling(long_score, 2.0, 2, 99) == long_score);

        // With no prefix the ceiling still governs the whole score.
        assert(yue2::inspect_abc_score(
            yue2::fit_natural_plan_to_ceiling(long_score, 10.0, 2, 0)).bars == 5);
    }

    // A bar ending on a tie promises the next note continues it. Fitting
    // splices the head onto the tail, so the bar before the splice no longer
    // leads anywhere and its tie cannot resolve. The MIDI exporter rejects a
    // tie it cannot resolve and fails the whole job, which is how a natural
    // length instrumental create died on a guitar prompt.
    const std::string tied_score =
        "X:1\nM:4/4\nL:1/32\nQ:1/4=120\n"
        "V: Vocal clef=treble\nV: Ins clef=treble\nK:C\n% intro\n"
        "V: Vocal\n\"C\"C32|\"F\"D32|\"G\"E32|\"C\"F32|\n"
        "V: Ins\nC,32|F,16F,16-|F,32|C,32|\n% outro\n"
        "V: Vocal\n\"Am\"G32|\"F\"A32|\"G\"B32|\"C\"c32|\n"
        "V: Ins\nA,32|F,32|G,32|C,32|\n";

    // Untouched it is valid: the tie in bar 2 is followed by its own note.
    assert(tied_score.find("-|") != std::string::npos);
    (void)yue2::serialize_yue2_abc_midis(tied_score);

    // Keeping two head bars and two tail bars cuts between bar 2 and bar 3,
    // stranding that tie. It has to go, or the export throws.
    const auto spliced = yue2::fit_abc_score_to_bars(tied_score, 4, 2);
    assert(yue2::inspect_abc_score(spliced).bars == 4);
    assert(spliced.find("-|") == std::string::npos);
    (void)yue2::serialize_yue2_abc_midis(spliced);

    // A fit that keeps every bar splices nothing, so the tie is still resolved
    // and must survive: the repair has to be confined to the actual cut.
    const auto whole = yue2::fit_abc_score_to_bars(tied_score, 8, 2);
    assert(yue2::inspect_abc_score(whole).bars == 8);
    assert(whole.find("-|") != std::string::npos);
    (void)yue2::serialize_yue2_abc_midis(whole);

    bool bad_instrumental = false;
    try {
        (void)yue2::make_vocal_rest_abc("X:1\nM:4/4\nK:C\nC4|\n");
    } catch (const std::invalid_argument &) {
        bad_instrumental = true;
    }
    assert(bad_instrumental);

    assert(rejected({0, 4, 4, "C"}));
    assert(rejected({401, 4, 4, "C"}));
    assert(rejected({95, 0, 4, "C"}));
    assert(rejected({95, 4, 3, "C"}));
    assert(rejected({95, 4, 4, "H minor"}));
    assert(rejected({95, 4, 4, "C#m\nV: injected"}));
    return 0;
}
