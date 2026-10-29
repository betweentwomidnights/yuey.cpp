// Named native AR and NAR fits: probe, fit (in resumable segments), post-probe
// and heldout checkpoint selection. Ported from tools/train_ar_run.py and
// train_nar_run.py, which stay the reference until the native workflow
// replaces them. Selection, prompts, identities, the teacher index and every
// checkpoint equal the Python's; progress and run records differ only in
// timings and process IDs.
#pragma once

#include <filesystem>
#include <optional>
#include <string>

namespace yue2::training {

struct FitOptions {
    std::filesystem::path config;
    std::string family;  // AR only: lora or dora
    std::string phase;   // probe, fit, post-probe or select
    std::optional<std::filesystem::path> resume;
    std::string segment = "native";
    bool prune_checkpoints = false;
};

void fit_ar(const FitOptions & options);
void fit_nar(const FitOptions & options);

// The optimizer step a native training state records (yue2.training.step).
long long state_step(const std::filesystem::path & state);

} // namespace yue2::training
