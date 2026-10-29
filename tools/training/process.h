// Running native tools from the workflow.
#pragma once

#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace yue2::training {

// Standard output of a program run to completion, or nothing when it cannot
// start or outlives the timeout (it is then killed). Standard error is
// discarded.
std::optional<std::string> capture_output(const std::filesystem::path & program, const std::vector<std::string> & arguments,
                                          int timeout_seconds);

// Run a program from cwd with standard output and error written to log
// (replaced, or appended to); returns its exit code. Throws when it cannot start.
int run_to_log(const std::filesystem::path & program, const std::vector<std::string> & arguments,
               const std::filesystem::path & cwd, const std::filesystem::path & log, bool append = false);

// Like run_to_log, appending each output line to log and handing it, without
// its line ending, to on_line. If on_line throws, the rest of the output is
// still logged and the program awaited before the exception propagates.
int run_streaming(const std::filesystem::path & program, const std::vector<std::string> & arguments,
                  const std::filesystem::path & cwd, const std::filesystem::path & log,
                  const std::function<void(const std::string &)> & on_line);

// Set a variable for this process and the programs it starts.
void set_environment(const char * name, const char * value);

} // namespace yue2::training
