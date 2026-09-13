#include "lora.h"

#include <cmath>
#include <cstdio>
#include <map>
#include <stdexcept>

namespace yue2::detail {
namespace {

std::runtime_error lora_error(const std::string & message) {
    return std::runtime_error("[yue2:lora] " + message);
}

bool ends_with(const std::string & value, const char * suffix) {
    const std::string ending(suffix);
    return value.size() >= ending.size() &&
        value.compare(value.size() - ending.size(), ending.size(), ending) == 0;
}

std::string remove_suffix(const std::string & value, const char * suffix) {
    return value.substr(0, value.size() - std::char_traits<char>::length(suffix));
}

void validate_factor_type(const ggml_tensor * tensor, const std::string & name) {
    if (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16) {
        throw lora_error("unsupported factor type for " + name + ": " +
            ggml_type_name(tensor->type));
    }
}

} // namespace

LoraStack::LoraStack(GgufModel & base, const std::vector<LoraAdapterSpec> & specs) {
    adapters_.reserve(specs.size());
    const auto base_sha = base.optional_string("yue2.checkpoint.sha256");
    for (const auto & spec : specs) {
        if (spec.path.empty()) throw lora_error("adapter path is empty");
        if (!std::isfinite(spec.strength)) {
            throw lora_error("adapter strength must be finite: " + spec.path);
        }
        Adapter adapter;
        adapter.model = load_gguf_raw_on_backend(spec.path.c_str(), base.backend());
        if (adapter.model.string("general.architecture") != "yue2_lora" ||
            adapter.model.string("yue2.component") != "generation-adapter" ||
            adapter.model.string("yue2.adapter.type") != "lora") {
            throw lora_error("not a YuE2 generation LoRA: " + spec.path);
        }
        const auto rank = adapter.model.u32("yue2.adapter.rank");
        const auto alpha = adapter.model.f32("yue2.adapter.alpha");
        if (rank == 0 || !std::isfinite(alpha) || alpha <= 0.0F) {
            throw lora_error("invalid rank or alpha: " + spec.path);
        }
        if (const auto intended = adapter.model.optional_string(
                "yue2.adapter.base_sha256")) {
            if (!base_sha) {
                throw lora_error(
                    "adapter declares a base fingerprint but the base GGUF does not: " +
                    spec.path);
            }
            if (*intended != *base_sha) {
                throw lora_error("adapter base fingerprint mismatch: " + spec.path);
            }
        }
        adapter.scale = alpha / static_cast<float>(rank) * spec.strength;

        std::map<std::string, std::pair<ggml_tensor *, ggml_tensor *>> pairs;
        for (const auto & name : adapter.model.tensor_names()) {
            if (ends_with(name, ".lora_A")) {
                auto & pair = pairs[remove_suffix(name, ".lora_A")];
                if (pair.first) throw lora_error("duplicate factor: " + name);
                pair.first = adapter.model.get(name);
            } else if (ends_with(name, ".lora_B")) {
                auto & pair = pairs[remove_suffix(name, ".lora_B")];
                if (pair.second) throw lora_error("duplicate factor: " + name);
                pair.second = adapter.model.get(name);
            } else {
                throw lora_error("unsupported adapter tensor: " + name);
            }
        }
        if (pairs.size() != adapter.model.u32("yue2.adapter.target_count")) {
            throw lora_error("adapter target count does not match its tensors: " + spec.path);
        }
        if (pairs.empty()) throw lora_error("adapter has no targets: " + spec.path);

        adapters_.push_back(std::move(adapter));
        auto & loaded = adapters_.back();
        for (const auto & entry : pairs) {
            const auto weight_name = entry.first + ".weight";
            if (!base.has(weight_name)) {
                throw lora_error("base model has no target: " + weight_name);
            }
            auto * weight = base.get(weight_name);
            auto * a = entry.second.first;
            auto * b = entry.second.second;
            if (!a || !b) throw lora_error("incomplete factor pair: " + entry.first);
            validate_factor_type(a, entry.first + ".lora_A");
            validate_factor_type(b, entry.first + ".lora_B");
            if (a->ne[0] != weight->ne[0] || a->ne[1] != rank ||
                b->ne[0] != rank || b->ne[1] != weight->ne[1]) {
                throw lora_error("factor shape does not match base weight: " + weight_name);
            }
            bindings_[weight].push_back({a, b, loaded.scale});
        }
        std::fprintf(
            stderr, "[yue2] LoRA: %s (rank %u, alpha %.6g, strength %.6g, %zu targets)\n",
            spec.path.c_str(), rank, alpha, spec.strength, pairs.size());
    }
}

ggml_tensor * LoraStack::linear(
    ggml_context * context,
    ggml_tensor * weight,
    ggml_tensor * input) const {
    auto * output = ggml_mul_mat(context, weight, input);
    const auto found = bindings_.find(weight);
    if (found == bindings_.end()) return output;
    for (const auto & factor : found->second) {
        if (factor.scale == 0.0F) continue;
        auto * ax = ggml_mul_mat(context, factor.a, input);
        auto * bax = ggml_mul_mat(context, factor.b, ax);
        output = ggml_add(context, output, ggml_scale(context, bax, factor.scale));
    }
    return output;
}

} // namespace yue2::detail
