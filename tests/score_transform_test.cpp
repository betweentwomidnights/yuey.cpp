// The score editor's transforms, on one small score: E flat major, a Vocal note
// tied across the barline, chords on Vocal, and a multi-bar rest on Ins.
#include "yue2/score_transform.h"

#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace {

std::string score(const std::string & key, const std::string & tempo,
                  const std::string & vocal, const std::string & ins) {
    return "X:1\nL:1/32\nM:4/4\nK:" + key + "\nQ:1/4=" + tempo + "\n"
        "V: Vocal clef=treble name=\"Vocal Melody\" snm=\"Vocal\"\n"
        "V: Ins clef=treble name=\"Ins Melody\" snm=\"Inst.\"\n"
        "% verse\n"
        "V: Vocal\n" + vocal + "\n"
        "V: Ins\n" + ins + "\n";
}

template <typename Call>
bool rejects(Call call) {
    try {
        (void)call();
    } catch (const std::invalid_argument &) {
        return true;
    }
    return false;
}

void check(const std::string & got, const std::string & expected) {
    if (got != expected) {
        std::printf("got:\n%s\nexpected:\n%s\n", got.c_str(), expected.c_str());
        std::fflush(stdout);
    }
    assert(got == expected);
}

} // namespace

int main() {
    // In E flat: e, f and g are Eb5, F5 and G5; E, is Eb3.
    const auto source = score("Eb", "100",
        "\"Eb\"e8 f8 g16-|\"Cm\"g16 z16|",
        "E,32|Z1|");

    // Tempo: only the Q: line moves.
    check(yue2::scale_abc_tempo(source, 0.5), score("Eb", "50",
        "\"Eb\"e8 f8 g16-|\"Cm\"g16 z16|", "E,32|Z1|"));
    check(yue2::scale_abc_tempo(source, 2.0), score("Eb", "200",
        "\"Eb\"e8 f8 g16-|\"Cm\"g16 z16|", "E,32|Z1|"));
    const auto quarter = yue2::scale_abc_tempo(yue2::scale_abc_tempo(source, 0.5), 0.5);
    assert(rejects([&] { return yue2::scale_abc_tempo(quarter, 0.5); }));  // 13 bpm
    assert(rejects([&] { return yue2::scale_abc_tempo(score("C", "300", "z32|", "z32|"), 2.0); }));

    // Up a whole tone: E flat major becomes F major, and so do its chords.
    check(yue2::transpose_abc(source, 2), score("F", "100",
        "\"F\"=f8=g8=a16-|\"Dm\"=a16z16|",
        "=F,32|Z1|"));
    // Up a semitone: the key is spelled E, the chord C#m, the notes in sharps.
    check(yue2::transpose_abc(source, 1), score("E", "100",
        "\"E\"=e8^f8^g16-|\"C#m\"^g16z16|",
        "=E,32|Z1|"));
    // A whole octave leaves the key and chords alone and moves only notes.
    check(yue2::transpose_abc(source, 12), score("Eb", "100",
        "\"Eb\"_e'8=f'8=g'16-|\"Cm\"=g'16z16|",
        "_E32|Z1|"));
    // Down again lands on the same notes, spelled explicitly, and on chord
    // symbols in sharps, as SheetSage2 writes them: Eb comes back as D#.
    check(yue2::transpose_abc(yue2::transpose_abc(source, 2), -2), score("Eb", "100",
        "\"D#\"_e8=f8=g16-|\"Cm\"=g16z16|",
        "_E,32|Z1|"));
    assert(yue2::transpose_abc(source, 0) == source);
    assert(rejects([&] { return yue2::transpose_abc(source, 25); }));
    assert(rejects([&] {
        return yue2::transpose_abc(score("C", "100", "c''''32|", "z32|"), 12);
    }));

    // Swap: the bass is sung and the melody played; chords stay on Vocal, and
    // the melody's tie into the second bar opens the Ins multi-bar rest.
    check(yue2::swap_abc_lanes(source), score("Eb", "100",
        "\"Eb\"_E,32|\"Cm\"z32|",
        "_e8=f8=g16-|=g16z16|"));
    const auto silent = score("C", "100", "\"C\"z32|", "Z1|");
    assert(yue2::swap_abc_lanes(silent) == silent);

    // Drop chords: nothing else in the bar moves, spacing included.
    check(yue2::drop_abc_chords(source), score("Eb", "100",
        "e8 f8 g16-|g16 z16|",
        "E,32|Z1|"));
    const auto chordless = yue2::drop_abc_chords(source);
    assert(yue2::drop_abc_chords(chordless) == chordless);

    assert(rejects([&] { return yue2::drop_abc_chords("X:1\nM:4/4\nK:C\nC4|\n"); }));
    return 0;
}
