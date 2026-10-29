// Running native tools from the workflow.
#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace yue2::training {

// Standard output of a program run to completion, or nothing when it cannot
// start or outlives the timeout (it is then killed). Standard error is
// discarded.
std::optional<std::string> capture_output(const std::filesystem::path & program, const std::vector<std::string> & arguments,
                                          int timeout_seconds);

} // namespace yue2::training
