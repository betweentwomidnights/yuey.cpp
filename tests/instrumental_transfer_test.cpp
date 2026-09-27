// Upstream YuE2's instrumental score: the Vocal lane's notes move into Ins,
// Vocal keeps its chords over rests, and an Ins note a moved note overlaps is
// trimmed around it. Bars nothing reaches keep their exact text.
#include "yue2/generation.h"

#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace {

const std::string kHeader =
    "X:1\n"
    "L:1/32\n"
    "M:4/4\n"
    "K:Bb\n"
    "Q:1/4=120\n"
    "V: Vocal clef=treble name=\"Vocal Melody\" snm=\"Vocal\"\n"
    "V: Ins clef=treble name=\"Ins Melody\" snm=\"Inst.\"\n";

} // namespace

int main() {
    // In B flat: B is Bb4, c and d are C5 and D5, f is F5, G, is G3. The chorus
    // ties f across a barline over an Ins G, that is itself tied across the
    // same barline, and its last Vocal note lands inside a multi-bar rest.
    const auto source = kHeader +
        "% verse\n"
        "V: Vocal\n"
        "\"Bb\"B8 c8 d16|\"F\"z32|\n"
        "V: Ins\n"
        "F,32|A,16 B,16|\n"
        "% chorus\n"
        "V: Vocal\n"
        "\"Eb\"z16 f16-|f16 z16|z32|c32|\n"
        "V: Ins\n"
        "G,32-|G,32|Z2|\n";
    const auto expected = kHeader +
        "% verse\n"
        "V: Vocal\n"
        // The chord stays; the bar with no notes is untouched.
        "\"Bb\"z32|\"F\"z32|\n"
        "V: Ins\n"
        // The whole-bar F, is covered by the melody and goes; bar two keeps
        // its original spacing and key-signature spelling.
        "_B8=c8=d16|A,16 B,16|\n"
        "% chorus\n"
        "V: Vocal\n"
        "\"Eb\"z32|z32|z32|z32|\n"
        "V: Ins\n"
        // G, survives on both sides of the moved f; Z2 becomes its two bars.
        "=G,16=f16-|=f16=G,16|z32|=c32|\n";
    const auto converted = yue2::make_instrumental_transfer_abc(source);
    std::printf("%s\n", converted.c_str());
    assert(converted == expected);

    // Nothing on the Vocal lane: nothing to move, byte for byte.
    const auto instrumental = kHeader +
        "V: Vocal\n"
        "\"Bb\"z32|Z1|\n"
        "V: Ins\n"
        "F,32|A,16 B,16|\n";
    assert(yue2::make_instrumental_transfer_abc(instrumental) == instrumental);

    // A bar accidental carries to the repeated F, and each moved note is
    // written with its own, so it cannot leak into the Ins spelling; the Ins
    // C held under them keeps the part after them.
    const auto accidentals =
        "X:1\nL:1/32\nM:4/4\nK:C\nQ:1/4=100\n"
        "V: Vocal\n"
        "\"D\"^F8 F8 z16|\n"
        "V: Ins\n"
        "C32|\n";
    const auto accidentals_converted = yue2::make_instrumental_transfer_abc(accidentals);
    assert(accidentals_converted ==
        "X:1\nL:1/32\nM:4/4\nK:C\nQ:1/4=100\n"
        "V: Vocal\n"
        "\"D\"z32|\n"
        "V: Ins\n"
        "^F8^F8=C16|\n");

    // Chords split the rests they sit in, and a Vocal note tied through a chord
    // change moves as one note.
    const auto chords =
        "X:1\nL:1/16\nM:3/4\nK:C\nQ:1/4=90\n"
        "V: Vocal\n"
        "\"C\"e4- \"G\"e4 z4|\n"
        "V: Ins\n"
        "z12|\n";
    const auto chords_converted = yue2::make_instrumental_transfer_abc(chords);
    std::printf("%s\n", chords_converted.c_str());
    assert(chords_converted ==
        "X:1\nL:1/16\nM:3/4\nK:C\nQ:1/4=90\n"
        "V: Vocal\n"
        "\"C\"z4\"G\"z8|\n"
        "V: Ins\n"
        "=e8z4|\n");

    bool threw = false;
    try {
        (void)yue2::make_instrumental_transfer_abc("X:1\nM:4/4\nK:C\nC4|\n");
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    assert(threw);
    return 0;
}
