#pragma once
#include "yue2/autoregressive.h"
#include <memory>
#include <string>
#include <vector>

namespace yue2 {
struct ArTrainingOptions {
    std::string device;
    int threads = 4;
    int rank = 8;
    float learning_rate = 0.0001F;
    float weight_decay = 0.01F;
    float gradient_clip = 1.0F;
    float kl_weight = 0.0F;
    bool dora = false; // Full norm gradients around base + frozen companions.
    // Build the common-model graph (no learned adapter) for every block, so
    // teacher() can compute frozen distributions after updates.
    bool online_teacher = false;
    std::uint32_t seed = 831001;
    std::vector<LoraAdapterSpec> fixed_adapters;
    // Hash model, runtime, fixed adapters, exact prefix and target streams.
    std::string run_identity;
};
struct ArTrainingSample {
    std::vector<std::int32_t> prefix;
    std::vector<std::int32_t> codes;          // Raw codec IDs [0,32767].
    std::vector<float> teacher_probabilities; // Optional frozen common-model distribution.
    float teacher_entropy = 0;
    // Set by ArTrainer::prepare_teacher/validate_teacher once the complete
    // distribution has passed its finite/nonnegative/normalized checks.
    // Clear it if teacher_probabilities change; unset rows are checked per use.
    bool teacher_validated = false;
};
struct ArTrainingResult {
    float loss = 0;
    double gradient_norm = 0;
    std::uint64_t optimizer_step = 0;
    float cross_entropy = 0;
    float teacher_kl = 0;
    // Host wall time, as for NarTrainingResult.
    double inputs_seconds = 0;
    double optimizer_seconds = 0;
};

// Experimental fixed-frame semantic teacher forcing with a maximum prefix
// capacity. Shorter prefixes are left padded and masked. Cross-entropy is over
// MUSIC_END + codec vocabulary only, averaged over target tokens only.
// Cropped windows must use predict_end=false. No planning-token loss is used;
// AR weights are nevertheless shared with planning at inference.
class ArTrainer {
  public:
    ArTrainer(const std::string &model, std::uint32_t prefix_tokens, std::uint32_t frames,
              bool predict_end, const ArTrainingOptions &options);
    ~ArTrainer();
    ArTrainer(const ArTrainer &) = delete;
    ArTrainer &operator=(const ArTrainer &) = delete;
    ArTrainingResult step(const ArTrainingSample &, bool update = true);
    float evaluate(const ArTrainingSample &);
    // Call before any update/resume, while learned B factors are zero.
    void prepare_teacher(ArTrainingSample &);
    // The same frozen distribution at any step; requires online_teacher.
    void teacher(ArTrainingSample &);
    // Check a supplied frozen distribution once, then mark it validated.
    void validate_teacher(ArTrainingSample &);
    void save_adapter(const std::string &path) const;
    void save_state(const std::string &path, const std::string &rng_state) const;
    std::string load_state(const std::string &path);
    // Snapshot after step(update=false): factors, gradients, inputs and logits.
    void save_diagnostics(const std::string &path) const;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace yue2
