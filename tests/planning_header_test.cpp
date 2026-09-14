#include "yue2/generation.h"

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
    assert(yue2::normalize_abc_key("Db major") == "Db");
    assert(yue2::normalize_abc_key("Am") == "Am");
    assert(yue2::normalize_abc_key("F#maj") == "F#");

    const auto prefix = yue2::make_planning_abc_prefix({95, 4, 4, "C# minor"});
    assert(prefix ==
        "X:1\nT:\nM:4/4\nL:1/32\nQ:1/4=95\n"
        "V: Vocal clef=treble name=\"Vocal Melody\" snm=\"Vocal\"\n"
        "V: Ins clef=treble name=\"Ins Melody\" snm=\"Inst.\"\n"
        "K:C#m\n% intro\n");
    assert(prefix.back() == '\n');

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
