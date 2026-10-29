// The durable two-expert training workflow yue2-server launches for a named
// run: prepare and review the dataset, fit the AR expert, release its teacher
// cache, prepare frozen-AR conditioning, fit the NAR expert, and publish the
// pair. Ported from training_workflow.py's Workflow and main(). Each stage runs
// as a child yue2-train-run command with its own log, and the run's
// state.json carries status and progress as the Python wrote it.
#pragma once

#include "server/json.h"

#include <filesystem>
#include <optional>
#include <string>
#include <utility>

namespace yue2::training {

// action is prepare, train, resume or publish. Waits for the server to commit
// this process as the run's owner first. Returns the process exit code.
int run_workflow(const std::filesystem::path & run, const std::string & action);

// Never touch the device before the server commits this process as the run's
// owner in owner.json; throws once timeout_seconds pass without it.
void wait_for_owner(const std::filesystem::path & run, double timeout_seconds = 5);

// A stage's progress file, as it changes after the stage starts. A resumed
// stage may hash models before replacing its old status, so the previous
// segment's stopped or completed snapshot is never reported.
class ProgressWatch {
public:
    explicit ProgressWatch(std::optional<std::filesystem::path> path);
    // The file's contents when it changed since construction, else nothing.
    std::optional<server::json::Value> read() const;

private:
    std::optional<std::pair<long long, long long>> signature() const;
    std::optional<std::filesystem::path> path_;
    std::optional<std::pair<long long, long long>> initial_;
};

// setup_training.py without Python: the training runtime a server is started
// with (--training-config), naming the native tools' folder, the run root, the
// models folder and each required model, found by file name under it.
void write_training_runtime(const std::filesystem::path & bin, const std::filesystem::path & models_dir,
                            const std::filesystem::path & run_root, const std::filesystem::path & out);

} // namespace yue2::training
