// The durable two-expert training workflow yue2-server launches for a named
// run: prepare and review the dataset, fit the AR expert, release its teacher
// cache, prepare frozen-AR conditioning, fit the NAR expert, and publish the
// pair. Ported from training_workflow.py's Workflow and main(). Each stage runs
// as a child yue2-train-run command with its own log, and the run's
// state.json carries status and progress as the Python wrote it.
#pragma once

#include <filesystem>
#include <string>

namespace yue2::training {

// action is prepare, train, resume or publish. Waits for the server to commit
// this process as the run's owner first. Returns the process exit code.
int run_workflow(const std::filesystem::path & run, const std::string & action);

// setup_training.py without Python: the training runtime a server is started
// with (--training-config), naming the native tools' folder, the run root, the
// models folder and each required model, found by file name under it.
void write_training_runtime(const std::filesystem::path & bin, const std::filesystem::path & models_dir,
                            const std::filesystem::path & run_root, const std::filesystem::path & out);

} // namespace yue2::training
