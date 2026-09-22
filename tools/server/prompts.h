// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace yue2::server {

// Curated style prompts for a client's dice button.
//
// Shape follows what YuE2 upstream documents and what the ComfyUI prompt
// compiler formalises: genre and subgenre, then mood and energy, then the
// instruments as an actual band rather than a wish list, then a melody or
// groove descriptor. Comma separated, since that is what the style field is
// concatenated into.
//
// Two departures from that guidance, both deliberate.
//
// No tempo. A caller sets tempo separately, through a planning header or the
// Q: field of a score it supplies, and in a DAW that tempo is the project's.
// A prompt reading "128 bpm" against a score reading Q:1/4=100 is exactly the
// contradictory-token case the upstream guidance warns about, and the score
// wins, so the words would only mislead.
//
// No length hint. Length comes from a bar count or the natural ceiling, and
// "about two minutes" in the style field cannot move either.
//
// Instrumental prompts never mention voices: the pipeline already prefixes
// "Instrumental, no vocals, no singing, no humming." to the style, so a prompt
// that reaches for a vocal here would be arguing with its own preamble.
struct DicePrompts {
    std::vector<std::string> instrumental;
    std::vector<std::string> vocal;
};

const DicePrompts & dice_prompts();

// The pool as a JSON object, shaped to match the dice payload other services
// in this family already serve, so a client parses one thing:
//   {"dice":{"instrumental":[...],"vocal":[...]}}
std::string dice_prompts_json();

} // namespace yue2::server
