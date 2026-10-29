#pragma once

#include "gguf_model.h"
#include "yue2/autoregressive.h"

#include <map>
#include <string>
#include <vector>

namespace yue2::detail {

// Resident, functional LoRA adapters for YuE2 generation projections. Base
// weights stay immutable (including quantized weights); adapter factors live
// on the same backend and are evaluated as W*x + scale*B*(A*x).
class LoraStack {
public:
    LoraStack() = default;
    LoraStack(GgufModel & base, const std::vector<LoraAdapterSpec> & specs);

    ggml_tensor * linear(
        ggml_context * context,
        ggml_tensor * weight,
        ggml_tensor * input,
        bool f32_weights = false) const;

    // Apply full adapter replacements (used by Mothersuperior's jointly
    // trained NAR vae2llm/llm2vae projections) as strength-scaled deltas.
    ggml_tensor * bias(ggml_context * context, ggml_tensor * base) const;

    // Whether any adapter changes this base tensor. A caller that multiplies
    // by a slice of a weight bypasses linear(), so it must check first.
    bool adapts(const ggml_tensor * base) const {
        auto * key = const_cast<ggml_tensor *>(base);
        return bindings_.count(key) != 0 || replacements_.count(key) != 0;
    }

    std::size_t adapter_count() const noexcept { return adapters_.size(); }
    // Exact-parameter compatibility ID (non-cryptographic), excluding paths.
    // Used in addition to external SHA-256 run manifests for frozen DoRA bases.
    std::string reference_signature() const;
    bool has_non_additive_targets() const;
    // Transient per-target effective W + frozen LoRAs; never resident merged W.
    std::vector<float> row_norm_squared(ggml_tensor *weight) const;
    std::size_t target_count() const noexcept {
        return bindings_.size() + replacements_.size();
    }

private:
    struct Adapter {
        GgufModel model;
        float scale = 0.0F;
    };
    struct Factor {
        ggml_tensor * a = nullptr;
        ggml_tensor * b = nullptr;
        float scale = 0.0F;
        ggml_tensor * row_gain = nullptr;
        float strength = 1.0F;
    };
    struct Replacement {
        ggml_tensor * tensor = nullptr;
        float strength = 0.0F;
    };

    std::vector<Adapter> adapters_;
    ggml_backend_t backend_ = nullptr;
    std::map<ggml_tensor *, std::vector<Factor>> bindings_;
    std::map<ggml_tensor *, std::vector<Replacement>> replacements_;
};

} // namespace yue2::detail
