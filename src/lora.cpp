#include "lora.h"

#include <cmath>
#include <algorithm>
#include "ggml-alloc.h"
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <type_traits>
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
    backend_ = base.backend();
    adapters_.reserve(specs.size());
    const auto base_sha = base.optional_string("yue2.checkpoint.sha256");
    for (const auto & spec : specs) {
        if (spec.path.empty()) throw lora_error("adapter path is empty");
        if (!std::isfinite(spec.strength)) {
            throw lora_error("adapter strength must be finite: " + spec.path);
        }
        Adapter adapter;
        adapter.model = load_gguf_raw_on_backend(spec.path.c_str(), base.backend());
        const auto kind = adapter.model.string("yue2.adapter.type");
        if (adapter.model.string("general.architecture") != "yue2_lora" ||
            adapter.model.string("yue2.component") != "generation-adapter" ||
            (kind != "lora" && kind != "dora" && kind != "mixed_lora_dora")) {
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
        std::map<std::string, ggml_tensor *> gains, magnitudes;
        for (const auto & name : adapter.model.tensor_names()) {
            if (ends_with(name, ".lora_A")) {
                auto & pair = pairs[remove_suffix(name, ".lora_A")];
                if (pair.first) throw lora_error("duplicate factor: " + name);
                pair.first = adapter.model.get(name);
            } else if (ends_with(name, ".lora_B")) {
                auto & pair = pairs[remove_suffix(name, ".lora_B")];
                if (pair.second) throw lora_error("duplicate factor: " + name);
                pair.second = adapter.model.get(name);
            } else if (ends_with(name, ".dora_scale")) {
                gains.emplace(remove_suffix(name, ".dora_scale"), adapter.model.get(name));
            } else if (ends_with(name, ".dora_magnitude")) {
                magnitudes.emplace(remove_suffix(name, ".dora_magnitude"), adapter.model.get(name));
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
        if (gains.size() != magnitudes.size() ||
            (kind == "lora" && !gains.empty()) ||
            (kind == "dora" && (gains.size() != pairs.size() || !replacements.empty())) ||
            (kind == "mixed_lora_dora" && gains.empty()))
            throw lora_error("invalid DoRA target layout: " + spec.path);
        if (!gains.empty() && adapter.model.u32("yue2.adapter.dora.target_count") != gains.size())
            throw lora_error("DoRA target count mismatch: " + spec.path);
        for (const auto &entry : gains)
            if (!pairs.count(entry.first) || !magnitudes.count(entry.first))
                throw lora_error("orphan DoRA normalization: " + entry.first);
        for (const auto &entry : magnitudes)
            if (!gains.count(entry.first)) throw lora_error("orphan DoRA magnitude: " + entry.first);
        const auto reference = adapter.model.optional_string("yue2.adapter.dora.reference_prefix");
        const bool references_fixed = reference.has_value();
        if (references_fixed && (gains.empty() || has_non_additive_targets() ||
                                  *reference != reference_signature()))
            throw lora_error("DoRA frozen companion prefix does not match its training base");

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
                b->ne[0] != rank || b->ne[1] != weight->ne[1] ||
                a->ne[2] != 1 || a->ne[3] != 1 || b->ne[2] != 1 || b->ne[3] != 1) {
                throw lora_error("factor shape does not match base weight: " + weight_name);
            }
            auto *gain = gains.count(entry.first) ? gains.at(entry.first) : nullptr;
            if (gain) {
                auto *magnitude = magnitudes.at(entry.first);
                for (auto *value : {gain, magnitude}) {
                    validate_factor_type(value, entry.first + ".dora");
                    if (value->ne[0] != weight->ne[1] || value->ne[1] != 1 ||
                        value->ne[2] != 1 || value->ne[3] != 1)
                        throw lora_error("DoRA row shape mismatch: " + entry.first);
                    std::vector<float> values(value->ne[0]);
                    if (value->type == GGML_TYPE_F32)
                        ggml_backend_tensor_get(value, values.data(), 0, values.size() * sizeof(float));
                    else {
                        std::vector<ggml_fp16_t> half(values.size());
                        ggml_backend_tensor_get(value, half.data(), 0, half.size() * sizeof(ggml_fp16_t));
                        ggml_fp16_to_fp32_row(half.data(), values.data(), values.size());
                    }
                    for (auto x : values)
                        if (!std::isfinite(x)) throw lora_error("nonfinite DoRA normalization");
                }
            }
            const auto previous = bindings_.find(weight);
            if ((gain && !references_fixed && (previous != bindings_.end() || replacements_.count(weight))) ||
                (previous != bindings_.end() && std::any_of(previous->second.begin(), previous->second.end(),
                    [](const Factor &factor) { return factor.row_gain != nullptr; })))
                throw lora_error("DoRA targets cannot overlap another adapter: " + weight_name);
            bindings_[weight].push_back({a, b,
                gain ? alpha / static_cast<float>(rank) : loaded.scale, gain, spec.strength});
        }
        for (const auto & entry : replacements) {
            if (!base.has(entry.first)) {
                throw lora_error("base model has no replacement target: " + entry.first);
            }
            auto * base_tensor = base.get(entry.first);
            const auto existing = bindings_.find(base_tensor);
            if (existing != bindings_.end() && std::any_of(existing->second.begin(), existing->second.end(),
                    [](const Factor &factor) { return factor.row_gain != nullptr; }))
                throw lora_error("DoRA target cannot have a replacement: " + entry.first);
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
    ggml_tensor * input,
    bool f32_weights) const {
    // Training may need F32 projection arithmetic while frozen weights stay
    // in F16 storage. Cast within the current checkpointed block's workspace;
    // keep original pointers for adapter binding lookup. Inference is unchanged.
    auto compute_weight = [&](ggml_tensor * tensor) {
        return f32_weights && tensor->type != GGML_TYPE_F32
            ? ggml_cast(context, tensor, GGML_TYPE_F32) : tensor;
    };
    auto * output = ggml_mul_mat(context, compute_weight(weight), input);
    const auto replacement = replacements_.find(weight);
    if (replacement != replacements_.end()) {
        for (const auto & item : replacement->second) {
            if (item.strength == 0.0F) continue;
            auto * replaced = ggml_mul_mat(context, compute_weight(item.tensor), input);
            output = ggml_add(
                context, output,
                ggml_scale(context, ggml_sub(context, replaced,
                                               ggml_mul_mat(context, compute_weight(weight), input)),
                           item.strength));
        }
    }
    const auto found = bindings_.find(weight);
    if (found == bindings_.end()) return output;
    for (const auto & factor : found->second) {
        if (factor.row_gain && factor.strength == 0.0F) continue;
        if (factor.scale == 0.0F) continue;
        auto * ax = ggml_mul_mat(context, compute_weight(factor.a), input);
        auto * bax = ggml_mul_mat(context, compute_weight(factor.b), ax);
        // Every shipped adapter has alpha == rank at strength 1, so this scale
        // is 1.0, and as its own op it was ~6% of decode and of each flow
        // step for nothing. Multiplying by one is exact, so leaving it out
        // changes no output.
        if (factor.scale != 1.0F) bax = ggml_scale(context, bax, factor.scale);
        if (factor.row_gain) {
            auto *adapted = ggml_mul(context, ggml_add(context, output, bax),
                compute_weight(factor.row_gain));
            output = factor.strength == 1.0F ? adapted : ggml_add(context, output,
                ggml_scale(context, ggml_sub(context, adapted, output), factor.strength));
        } else output = ggml_add(context, output, bax);
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

bool LoraStack::has_non_additive_targets() const {
    if (!replacements_.empty()) return true;
    for (const auto &entry : bindings_)
        for (const auto &factor : entry.second) if (factor.row_gain) return true;
    return false;
}

std::string LoraStack::reference_signature() const {
    uint64_t hash = 14695981039346656037ULL;
    auto bytes = [&](const void *data, size_t n) {
        auto *p=static_cast<const uint8_t *>(data);
        for(size_t i=0;i<n;++i) { hash^=p[i];hash*=1099511628211ULL; }
    };
    auto integer = [&](uint64_t v) { for(int i=0;i<8;++i) { uint8_t b=(v>>(8*i))&255;bytes(&b,1); } };
    integer(adapters_.size());
    for (const auto &adapter : adapters_) {
        uint32_t scale;std::memcpy(&scale,&adapter.scale,sizeof(scale));integer(scale);
        auto names=adapter.model.tensor_names();std::sort(names.begin(),names.end());
        for(const auto &name:names) {
            bytes(name.c_str(),name.size()+1);auto *t=adapter.model.get(name);
            integer(t->type);for(auto n:t->ne) integer(n);
            std::vector<uint8_t> data(ggml_nbytes(t));
            ggml_backend_tensor_get(t,data.data(),0,data.size());bytes(data.data(),data.size());
        }
    }
    std::ostringstream out;out<<"fnv1a64-v1:"<<std::hex<<std::setw(16)<<std::setfill('0')<<hash;
    return out.str();
}

std::vector<float> LoraStack::row_norm_squared(ggml_tensor *weight) const {
    if(has_non_additive_targets()) throw lora_error("DoRA reference supports ordinary additive frozen LoRAs only");
    if(weight->type!=GGML_TYPE_F16 && weight->type!=GGML_TYPE_F32)
        throw lora_error("DoRA training base must be unquantized");
    auto *ctx=ggml_init({1024*ggml_tensor_overhead()+ggml_graph_overhead_custom(1024,false),nullptr,true});
    if(!ctx) throw lora_error("DoRA norm context allocation failed");
    std::unique_ptr<ggml_context,decltype(&ggml_free)> guard(ctx,ggml_free);
    auto cast=[&](ggml_tensor *t) { return t->type==GGML_TYPE_F32?t:ggml_cast(ctx,t,GGML_TYPE_F32); };
    auto *effective=cast(weight);
    const auto found=bindings_.find(weight);
    if(found!=bindings_.end()) for(const auto &factor:found->second) {
        if(factor.scale==0) continue;
        auto *a=ggml_cont(ctx,ggml_transpose(ctx,cast(factor.a)));
        auto *delta=ggml_mul_mat(ctx,a,cast(factor.b));
        effective=ggml_add(ctx,effective,ggml_scale(ctx,delta,factor.scale));
    }
    auto *norm=ggml_reshape_1d(ctx,ggml_sum_rows(ctx,ggml_sqr(ctx,effective)),weight->ne[1]);
    ggml_set_output(norm);auto *graph=ggml_new_graph_custom(ctx,1024,false);ggml_build_forward_expand(graph,norm);
    for(int i=0;i<ggml_graph_n_nodes(graph);++i) {
        auto *node=ggml_graph_node(graph,i);if(node->op==GGML_OP_MUL_MAT)ggml_mul_mat_set_prec(node,GGML_PREC_F32);
    }
    if(!backend_) throw lora_error("reference norm needs a backend-bound stack");
    const auto device=backend_;
    auto *allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(device));
    std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>,decltype(&ggml_gallocr_free)> allocation(allocator,ggml_gallocr_free);
    if(!allocator || !ggml_gallocr_alloc_graph(allocator,graph) ||
        ggml_backend_graph_compute(device,graph)!=GGML_STATUS_SUCCESS)
        throw lora_error("DoRA reference row-norm computation failed");
    std::vector<float> values(weight->ne[1]);ggml_backend_tensor_get(norm,values.data(),0,values.size()*sizeof(float));
    for(auto x:values) if(!std::isfinite(x)||x<=1e-12F) throw lora_error("DoRA frozen base has a zero/nonfinite row");
    return values;
}

} // namespace yue2::detail
