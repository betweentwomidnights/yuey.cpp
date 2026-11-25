// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <string>

namespace yue2::server {

// How the server bounds a generation, as advertised to clients. /health and
// /props both have to report these, and they used to be two hand-maintained
// string concatenations a few hundred lines apart: planning_overrun and
// planning_loop_bars went into one and not the other, so a UI reading /props
// could not discover them. One builder, so they cannot drift again.
struct GenerationPolicy {
    double natural_max_seconds = 180.0;
    double planning_overrun = 2.0;
    std::uint32_t planning_loop_bars = 16;
    // What an instrumental request gets when it does not say: its
    // instrumental_method, and whether the instrumental adapter is stacked.
    std::string instrumental_method = "transfer";
    bool use_instrumental_adapter = true;
};

// Emits the fields without enclosing braces or a leading comma, so a caller
// can splice it into a larger object. Besides the configured fields it reports
// continue_max_source_seconds, derived from natural_max_seconds, so a client
// can refuse an over-long continuation before uploading it.
std::string generation_policy_json(const GenerationPolicy & policy);

// A continuation is held to the natural-length ceiling as a whole, source plus
// what it adds, and the source has to leave room for some new music. That room
// is ten seconds, or half the ceiling when the ceiling is under twenty.
double continuation_room_seconds(double natural_max_seconds);

// The longest source a continuation accepts under a ceiling; 0 when the
// ceiling is 0, which is unbounded.
double continuation_max_source_seconds(double natural_max_seconds);

// What a source-bearing request is, for the length check below.
enum class SourceUse {
    continuation,  // /continue, or /plan and /generate with a scored abc_prefix
    cover,         // /cover, whose render is as long as its source
};

// Empty when source_seconds fits under natural_max_seconds for this use;
// otherwise the message for the 400 that refuses the request. Checked at
// submit, before transcription or any model loads, so a refusal is instant.
// route names the endpoint in the message, as "/continue".
std::string source_length_refusal(
    SourceUse use, const char * route, double source_seconds, double natural_max_seconds);

} // namespace yue2::server
