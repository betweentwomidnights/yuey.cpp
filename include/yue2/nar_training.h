#pragma once
#include "yue2/autoregressive.h"
#include <memory>
#include <string>
#include <vector>

namespace yue2 {
struct NarTrainingOptions {
    std::string device;
    int threads = 4;
    int rank = 8;
    float learning_rate = 0.0001F;
    float weight_decay = 0.01F;
    float gradient_clip = 1.0F;
    std::uint32_t seed = 831001;
    std::vector<LoraAdapterSpec> fixed_adapters;
    // Frozen AR adapters used to prepare conditioning; never loaded into NAR.
    std::vector<LoraAdapterSpec> conditioning_adapters;
    bool f32_projections = false;
    // Experimental output-row DoRA, full norm gradients. Requires F32
    // projections and an unquantized base, with no fixed NAR adapters.
    bool dora = false;
    // Caller hashes dataset, preprocessing and model/adapter content identities.
    std::string run_identity;
};
struct NarTrainingSample {
    FrozenNarConditioning conditioning;
    std::vector<float> clean_latents; // [frames,64], posterior means
};
struct NarTrainingResult {
    float loss = 0;
    double gradient_norm = 0;
    std::uint64_t optimizer_step = 0;
    // Host wall time: input validation/upload, and gradient download, scans,
    // AdamW and parameter upload. Diagnostics only; never part of numerics.
    double inputs_seconds = 0;
    double optimizer_seconds = 0;
};

// Experimental batch-one acoustic training. All seven NAR projections in all
// 28 layers get functional F32 LoRAs. Frozen weights, AR, boundaries and VAE
// remain unchanged. Per-layer recomputation shares one backward workspace.
class NarTrainer {
  public:
    NarTrainer(const std::string &model, std::uint32_t frames, std::uint32_t max_ar_tokens,
               const NarTrainingOptions &options);
    ~NarTrainer();
    NarTrainer(const NarTrainer &) = delete;
    NarTrainer &operator=(const NarTrainer &) = delete;
    NarTrainingResult step(const NarTrainingSample &, const std::vector<float> &noise, float sigma,
                           bool update = true);
    float evaluate(const NarTrainingSample &, const std::vector<float> &noise, float sigma);
    std::vector<float> velocity() const;
    void save_adapter(const std::string &path) const;
    void save_state(const std::string &path, const std::string &rng_state) const;
    std::string load_state(const std::string &path);
    // Call after step(update=false) for a numerical oracle: includes current
    // factors, gradients, frozen AR K/V, mixed state, target and prediction.
    void save_diagnostics(const std::string &path) const;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace yue2
