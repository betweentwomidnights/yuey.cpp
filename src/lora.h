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
        ggml_tensor * input) const;

    std::size_t adapter_count() const noexcept { return adapters_.size(); }
    std::size_t target_count() const noexcept { return bindings_.size(); }

private:
    struct Adapter {
        GgufModel model;
        float scale = 0.0F;
    };
    struct Factor {
        ggml_tensor * a = nullptr;
        ggml_tensor * b = nullptr;
        float scale = 0.0F;
    };

    std::vector<Adapter> adapters_;
    std::map<ggml_tensor *, std::vector<Factor>> bindings_;
};

} // namespace yue2::detail
