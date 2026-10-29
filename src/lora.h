#pragma once

#include "gguf_model.h"
#include "yue2/autoregressive.h"

#include <map>
#include <memory>
#include <string>
#include <utility>
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

    // A frozen base weight (or rows [first, first + count) of it) as F32 in a
    // training graph: F32 as is, F16/BF16 cast, a k-quant read through
    // get_rows, which dequantizes exactly on every backend. The row indices
    // live in small buffers this stack owns.
    ggml_tensor * f32_weight(ggml_context * context, ggml_tensor * weight, int64_t first = 0, int64_t count = -1) const;

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
    struct RowIds {
        std::unique_ptr<ggml_context, void (*)(ggml_context *)> context{nullptr, ggml_free};
        std::unique_ptr<ggml_backend_buffer, void (*)(ggml_backend_buffer_t)> buffer{nullptr, ggml_backend_buffer_free};
        ggml_tensor * ids = nullptr;
    };
    mutable std::map<std::pair<int64_t, int64_t>, std::unique_ptr<RowIds>> row_ids_;

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
