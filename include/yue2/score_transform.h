// Quick, deterministic edits to a native two-voice score, for a score editor
// to offer as buttons before a render. Each one returns a score that renders,
// or throws std::invalid_argument saying why it cannot apply; a score that the
// edit does not change comes back exactly as it went in. The melody transfer
// that instrumental requests use is make_instrumental_transfer_abc in
// generation.h.
#pragma once

#include <string>

namespace yue2 {

// Multiplies the Q: tempo by `factor` (0.5 is half-time, 2 double-time) and
// leaves every note alone, so the same score plays at the new speed. The
// result must stay within 20-400 bpm.
std::string scale_abc_tempo(const std::string & abc, double factor);

// Moves every note on both lanes by `semitones` (-24 to 24). Unless the shift
// is whole octaves, the key (header, voice and inline K:) and every chord
// symbol move too, spelled the way SheetSage2's scores spell them. Throws if a
// note would leave the MIDI range.
std::string transpose_abc(const std::string & abc, int semitones);

// Exchanges the notes of the Vocal and Ins lanes: the instrument part is sung
// and the melody is played. Chord symbols stay where they are.
std::string swap_abc_lanes(const std::string & abc);

// Removes every chord symbol, so the model harmonises the melody itself. On
// its own a chordless scaffold gives little harmony; it is meant for a score
// that still has a melody.
std::string drop_abc_chords(const std::string & abc);

} // namespace yue2
