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
                                          int timeout_seconds, int * exit_code = nullptr);

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

// A program on PATH (with .exe on Windows), or nothing.
std::optional<std::filesystem::path> find_program(const std::string & name);

// A native tool run in the background with standard output and error written
// to a log, so its owner can report progress while it works.
class Child {
public:
    Child(const std::filesystem::path & program, const std::vector<std::string> & arguments,
          const std::filesystem::path & cwd, const std::filesystem::path & log);
    ~Child();
    Child(const Child &) = delete;
    Child & operator=(const Child &) = delete;

    bool running();
    int wait();
    void terminate();
    long pid() const { return pid_; }
    // Peak working set and peak private bytes, in GiB, on Windows only.
    std::optional<std::pair<double, double>> peak_memory_gib() const;

private:
#ifdef _WIN32
    void * handle_ = nullptr;
#endif
    long pid_ = 0;
    std::optional<int> code_;
};

} // namespace yue2::training
