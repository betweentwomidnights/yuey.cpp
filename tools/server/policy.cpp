// SPDX-License-Identifier: MIT
#include "server/policy.h"

namespace yue2::server {

std::string generation_policy_json(const GenerationPolicy & policy) {
    return "\"natural_max_seconds\":" + std::to_string(policy.natural_max_seconds) +
        ",\"planning_overrun\":" + std::to_string(policy.planning_overrun) +
        ",\"planning_loop_bars\":" + std::to_string(policy.planning_loop_bars);
}

} // namespace yue2::server
