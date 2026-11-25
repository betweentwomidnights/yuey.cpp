// SPDX-License-Identifier: MIT
#include "server/policy.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace yue2::server {

namespace {

std::string seconds_text(double seconds) {
    char text[32];
    std::snprintf(text, sizeof text, "%.1f", seconds);
    std::string value = text;
    if (value.size() > 2 && value.compare(value.size() - 2, 2, ".0") == 0) {
        value.resize(value.size() - 2);
    }
    return value;
}

} // namespace

double continuation_room_seconds(double natural_max_seconds) {
    if (!(natural_max_seconds > 0.0) || !std::isfinite(natural_max_seconds)) return 0.0;
    return std::min(10.0, natural_max_seconds / 2.0);
}

double continuation_max_source_seconds(double natural_max_seconds) {
    if (!(natural_max_seconds > 0.0) || !std::isfinite(natural_max_seconds)) return 0.0;
    return natural_max_seconds - continuation_room_seconds(natural_max_seconds);
}

std::string source_length_refusal(
    SourceUse use, const char * route, double source_seconds, double natural_max_seconds) {
    if (!(natural_max_seconds > 0.0) || !std::isfinite(natural_max_seconds)) return {};
    if (!std::isfinite(source_seconds) || source_seconds < 0.0) return {};
    const auto ceiling = seconds_text(natural_max_seconds);
    if (use == SourceUse::cover) {
        if (source_seconds <= natural_max_seconds) return {};
        return std::string(route) + " source is " + seconds_text(source_seconds) +
            " s, but a cover is as long as its source and the natural-length ceiling "
            "(natural_max_seconds) is " + ceiling + " s. Trim the source to " +
            ceiling + " s or less, or raise the ceiling.";
    }
    const double limit = continuation_max_source_seconds(natural_max_seconds);
    if (source_seconds <= limit) return {};
    return std::string(route) + " source is " + seconds_text(source_seconds) +
        " s, but a continuation is held to " + ceiling +
        " s in all (natural_max_seconds), source plus new music, so the source can be "
        "at most " + seconds_text(limit) + " s. Keep the last " + seconds_text(limit) +
        " s of the source or less, or raise the ceiling.";
}

std::string generation_policy_json(const GenerationPolicy & policy) {
    return "\"natural_max_seconds\":" + std::to_string(policy.natural_max_seconds) +
        ",\"planning_overrun\":" + std::to_string(policy.planning_overrun) +
        ",\"planning_loop_bars\":" + std::to_string(policy.planning_loop_bars) +
        ",\"instrumental_method\":\"" + policy.instrumental_method + "\"" +
        ",\"use_instrumental_adapter\":" + (policy.use_instrumental_adapter ? "true" : "false") +
        ",\"continue_max_source_seconds\":" +
            std::to_string(continuation_max_source_seconds(policy.natural_max_seconds));
}

} // namespace yue2::server
