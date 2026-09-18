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
        std::map<std::string, ggml_tensor *> replacements;
        for (const auto & name : adapter.model.tensor_names()) {
            if (ends_with(name, ".lora_A")) {
                auto & pair = pairs[remove_suffix(name, ".lora_A")];
                if (pair.first) throw lora_error("duplicate factor: " + name);
                pair.first = adapter.model.get(name);
            } else if (ends_with(name, ".lora_B")) {
                auto & pair = pairs[remove_suffix(name, ".lora_B")];
                if (pair.second) throw lora_error("duplicate factor: " + name);
                pair.second = adapter.model.get(name);
            } else if (ends_with(name, ".replacement")) {
                const auto target = remove_suffix(name, ".replacement");
                if (!replacements.emplace(target, adapter.model.get(name)).second) {
                    throw lora_error("duplicate replacement: " + name);
                }
            } else {
                throw lora_error("unsupported adapter tensor: " + name);
            }
        }
        if (pairs.size() + replacements.size() !=
            adapter.model.u32("yue2.adapter.target_count")) {
            throw lora_error("adapter target count does not match its tensors: " + spec.path);
        }
        if (pairs.empty() && replacements.empty()) {
            throw lora_error("adapter has no targets: " + spec.path);
        }

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
        for (const auto & entry : replacements) {
            if (!base.has(entry.first)) {
                throw lora_error("base model has no replacement target: " + entry.first);
            }
            auto * base_tensor = base.get(entry.first);
            auto * replacement = entry.second;
            validate_factor_type(replacement, entry.first + ".replacement");
            if (replacement->type != base_tensor->type &&
                replacement->type != GGML_TYPE_F32 && replacement->type != GGML_TYPE_F16) {
                throw lora_error("unsupported replacement type: " + entry.first);
            }
            if (replacement->ne[0] != base_tensor->ne[0] ||
                replacement->ne[1] != base_tensor->ne[1] ||
                replacement->ne[2] != base_tensor->ne[2] ||
                replacement->ne[3] != base_tensor->ne[3]) {
                throw lora_error("replacement shape does not match base tensor: " + entry.first);
            }
            replacements_[base_tensor].push_back({replacement, spec.strength});
        }
        std::fprintf(
            stderr, "[yue2] LoRA: %s (rank %u, alpha %.6g, strength %.6g, "
            "%zu low-rank + %zu replacement targets)\n",
            spec.path.c_str(), rank, alpha, spec.strength,
            pairs.size(), replacements.size());
    }
}

ggml_tensor * LoraStack::linear(
    ggml_context * context,
    ggml_tensor * weight,
    ggml_tensor * input) const {
    auto * output = ggml_mul_mat(context, weight, input);
    const auto replacement = replacements_.find(weight);
    if (replacement != replacements_.end()) {
        for (const auto & item : replacement->second) {
            if (item.strength == 0.0F) continue;
            auto * replaced = ggml_mul_mat(context, item.tensor, input);
            output = ggml_add(
                context, output,
                ggml_scale(context, ggml_sub(context, replaced,
                                               ggml_mul_mat(context, weight, input)),
                           item.strength));
        }
    }
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

ggml_tensor * LoraStack::bias(ggml_context * context, ggml_tensor * base) const {
    const auto found = replacements_.find(base);
    if (found == replacements_.end()) return base;
    auto * base_f32 = base->type == GGML_TYPE_F32
        ? base : ggml_cast(context, base, GGML_TYPE_F32);
    ggml_tensor * output = base_f32;
    for (const auto & item : found->second) {
        if (item.strength == 0.0F) continue;
        auto * replacement = item.tensor->type == GGML_TYPE_F32
            ? item.tensor : ggml_cast(context, item.tensor, GGML_TYPE_F32);
        output = ggml_add(
            context, output,
            ggml_scale(context, ggml_sub(context, replacement, base_f32), item.strength));
    }
    return output;
}

} // namespace yue2::detail
