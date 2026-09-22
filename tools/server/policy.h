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
};

// Emits the fields without enclosing braces or a leading comma, so a caller
// can splice it into a larger object.
std::string generation_policy_json(const GenerationPolicy & policy);

} // namespace yue2::server
