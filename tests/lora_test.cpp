#include "gguf_model.h"
#include "lora.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <random>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr const char * kStem = "model.layers.0.self_attn.q_proj";
constexpr const char * kHash =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

struct TemporaryDirectory {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("yue2-lora-test-" + std::to_string(std::random_device{}()));
    TemporaryDirectory() { std::filesystem::create_directories(path); }
    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

ggml_tensor * tensor_2d(
    ggml_context * context,
    const char * name,
    std::int64_t ne0,
    std::int64_t ne1,
    const std::vector<float> & values) {
    auto * tensor = ggml_new_tensor_2d(context, GGML_TYPE_F32, ne0, ne1);
    ggml_set_name(tensor, name);
    assert(values.size() == static_cast<std::size_t>(ne0 * ne1));
    std::memcpy(tensor->data, values.data(), values.size() * sizeof(float));
    return tensor;
}

void write_base(const std::filesystem::path & path) {
    auto * context = ggml_init({4096, nullptr, false});
    assert(context);
    auto * weight = tensor_2d(
        context, "model.layers.0.self_attn.q_proj.weight", 3, 2,
        {1, 2, 3, 4, 5, 6});
    auto * file = gguf_init_empty();
    gguf_set_val_str(file, "general.architecture", "test");
    gguf_set_val_str(file, "yue2.checkpoint.sha256", kHash);
    gguf_add_tensor(file, weight);
    auto *nar_weight = tensor_2d(context, "model.layers.0.nar_self_attn.q_proj.weight", 3, 2,
        {1, 2, 3, 4, 5, 6});
    gguf_add_tensor(file, nar_weight);
    assert(gguf_write_to_file(file, path.string().c_str(), false));
    gguf_free(file);
    ggml_free(context);
}

void write_adapter(
    const std::filesystem::path & path,
    const char * base_hash = kHash) {
    auto * context = ggml_init({8192, nullptr, false});
    assert(context);
    auto * a = tensor_2d(
        context, "model.layers.0.self_attn.q_proj.lora_A", 3, 1,
        {1, 2, 0});
    auto * b = tensor_2d(
        context, "model.layers.0.self_attn.q_proj.lora_B", 1, 2,
        {3, -1});
    auto * file = gguf_init_empty();
    gguf_set_val_str(file, "general.architecture", "yue2_lora");
    gguf_set_val_str(file, "yue2.component", "generation-adapter");
    gguf_set_val_str(file, "yue2.adapter.type", "lora");
    gguf_set_val_u32(file, "yue2.adapter.rank", 1);
    gguf_set_val_f32(file, "yue2.adapter.alpha", 2.0F);
    gguf_set_val_u32(file, "yue2.adapter.target_count", 1);
    gguf_set_val_str(file, "yue2.adapter.base_sha256", base_hash);
    gguf_add_tensor(file, a);
    gguf_add_tensor(file, b);
    assert(gguf_write_to_file(file, path.string().c_str(), false));
    gguf_free(file);
    ggml_free(context);
}

void write_replacement_adapter(const std::filesystem::path & path) {
    auto * context = ggml_init({8192, nullptr, false});
    assert(context);
    auto * replacement = tensor_2d(
        context, "model.layers.0.self_attn.q_proj.weight.replacement", 3, 2,
        {0, 0, 0, 0, 0, 0});
    auto * file = gguf_init_empty();
    gguf_set_val_str(file, "general.architecture", "yue2_lora");
    gguf_set_val_str(file, "yue2.component", "generation-adapter");
    gguf_set_val_str(file, "yue2.adapter.type", "lora");
    gguf_set_val_u32(file, "yue2.adapter.rank", 1);
    gguf_set_val_f32(file, "yue2.adapter.alpha", 1.0F);
    gguf_set_val_u32(file, "yue2.adapter.target_count", 1);
    gguf_set_val_u32(file, "yue2.adapter.replacement_count", 1);
    gguf_set_val_str(file, "yue2.adapter.base_sha256", kHash);
    gguf_add_tensor(file, replacement);
    assert(gguf_write_to_file(file, path.string().c_str(), false));
    gguf_free(file);
    ggml_free(context);
}

void write_dora(const std::filesystem::path &path, bool omit_magnitude = false,
                bool nonfinite = false, const std::string &reference = "") {
    auto *ctx = ggml_init({16384, nullptr, false});
    auto *a = tensor_2d(ctx, "model.layers.0.self_attn.q_proj.lora_A", 3, 1, {1,2,0});
    auto *b = tensor_2d(ctx, "model.layers.0.self_attn.q_proj.lora_B", 1, 2, {3,-1});
    auto *m = tensor_2d(ctx, "model.layers.0.self_attn.q_proj.dora_magnitude", 2, 1, {1,1});
    auto *g = tensor_2d(ctx, "model.layers.0.self_attn.q_proj.dora_scale", 2, 1,
        {nonfinite ? std::numeric_limits<float>::infinity() : .5F,1.5F});
    auto *file = gguf_init_empty();
    gguf_set_val_str(file,"general.architecture","yue2_lora");
    gguf_set_val_str(file,"yue2.component","generation-adapter");
    gguf_set_val_str(file,"yue2.adapter.type","dora");
    gguf_set_val_u32(file,"yue2.adapter.rank",1);
    gguf_set_val_f32(file,"yue2.adapter.alpha",2.F);
    gguf_set_val_u32(file,"yue2.adapter.target_count",1);
    gguf_set_val_u32(file,"yue2.adapter.dora.target_count",1);
    gguf_set_val_str(file,"yue2.adapter.base_sha256",kHash);
    if(!reference.empty())gguf_set_val_str(file,"yue2.adapter.dora.reference_prefix",reference.c_str());
    for (auto *t : {a,b,g}) gguf_add_tensor(file,t);
    if (!omit_magnitude) gguf_add_tensor(file,m);
    assert(gguf_write_to_file(file,path.string().c_str(),false));
    gguf_free(file);ggml_free(ctx);
}

void write_mixed(const std::filesystem::path &path) {
    auto *ctx=ggml_init({32768,nullptr,false});auto *file=gguf_init_empty();
    gguf_set_val_str(file,"general.architecture","yue2_lora");
    gguf_set_val_str(file,"yue2.component","generation-adapter");
    gguf_set_val_str(file,"yue2.adapter.type","mixed_lora_dora");
    gguf_set_val_u32(file,"yue2.adapter.rank",1);gguf_set_val_f32(file,"yue2.adapter.alpha",2.F);
    gguf_set_val_u32(file,"yue2.adapter.target_count",2);gguf_set_val_u32(file,"yue2.adapter.dora.target_count",1);
    gguf_set_val_str(file,"yue2.adapter.base_sha256",kHash);
    for (const char *stem : {kStem,"model.layers.0.nar_self_attn.q_proj"}) {
        gguf_add_tensor(file,tensor_2d(ctx,(std::string(stem)+".lora_A").c_str(),3,1,{1,2,0}));
        gguf_add_tensor(file,tensor_2d(ctx,(std::string(stem)+".lora_B").c_str(),1,2,{3,-1}));
    }
    gguf_add_tensor(file,tensor_2d(ctx,"model.layers.0.nar_self_attn.q_proj.dora_magnitude",2,1,{1,1}));
    gguf_add_tensor(file,tensor_2d(ctx,"model.layers.0.nar_self_attn.q_proj.dora_scale",2,1,{.5F,1.5F}));
    assert(gguf_write_to_file(file,path.string().c_str(),false));gguf_free(file);ggml_free(ctx);
}

std::vector<float> run(
    yue2::detail::GgufModel & base,
    const yue2::detail::LoraStack & loras,
    const std::string &stem = kStem) {
    auto * context = ggml_init({1024 * 1024, nullptr, true});
    assert(context);
    auto * input = ggml_new_tensor_2d(context, GGML_TYPE_F32, 3, 2);
    ggml_set_input(input);
    auto * output = loras.linear(
        context, base.get(stem + ".weight"), input);
    auto * graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, output);
    auto * allocator = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(base.backend()));
    assert(ggml_gallocr_alloc_graph(allocator, graph));
    const std::vector<float> values = {1, 0, -1, 2, 1, 0};
    ggml_backend_tensor_set(
        input, values.data(), 0, values.size() * sizeof(values.front()));
    assert(ggml_backend_graph_compute(base.backend(), graph) == GGML_STATUS_SUCCESS);
    std::vector<float> result(4);
    ggml_backend_tensor_get(
        output, result.data(), 0, result.size() * sizeof(result.front()));
    ggml_gallocr_free(allocator);
    ggml_free(context);
    return result;
}

void require_close(const std::vector<float> & actual, const std::vector<float> & expected) {
    assert(actual.size() == expected.size());
    for (std::size_t index = 0; index < actual.size(); ++index) {
        assert(std::abs(actual[index] - expected[index]) < 1.0e-5F);
    }
}

} // namespace

int main(int argc, char ** argv) {
    TemporaryDirectory temporary;
    const auto base_path = temporary.path / "base.gguf";
    const auto adapter_path = temporary.path / "adapter.gguf";
    const auto wrong_path = temporary.path / "wrong.gguf";
    const auto replacement_path = temporary.path / "replacement.gguf";
    write_base(base_path);
    write_adapter(adapter_path);
    write_adapter(
        wrong_path,
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff");
    write_replacement_adapter(replacement_path);

    const char * device = argc > 1 ? argv[1] : "cpu";
    auto base = yue2::detail::load_gguf_raw(base_path.string().c_str(), device);
    std::vector<float> original(6);
    ggml_backend_tensor_get(
        base.get(std::string(kStem) + ".weight"), original.data(), 0,
        original.size() * sizeof(original.front()));

    yue2::detail::LoraStack enabled(
        base, {{adapter_path.string(), 0.5F}});
    assert(enabled.adapter_count() == 1 && enabled.target_count() == 1);
    require_close(run(base, enabled), {1, -3, 16, 9});
    require_close(enabled.row_norm_squared(base.get(std::string(kStem)+".weight")),{89,54});
    const auto referenced_path=temporary.path/"referenced-dora.gguf";
    write_dora(referenced_path,false,false,enabled.reference_signature());
    yue2::detail::LoraStack referenced(base,{{adapter_path.string(),.5F},{referenced_path.string(),1.F}});
    require_close(run(base,referenced),{3.5F,-7.5F,20,1.5F});
    for(float strength:{0.F,1.F}) {
        bool bad=false;
        try { yue2::detail::LoraStack wrong_reference(base,{{adapter_path.string(),strength},{referenced_path.string(),1.F}}); }
        catch(const std::runtime_error &){bad=true;}
        assert(bad);
    }

    yue2::detail::LoraStack disabled(
        base, {{adapter_path.string(), 0.0F}});
    require_close(run(base, disabled), {-2, -2, 4, 13});

    const auto dora_path = temporary.path / "dora.gguf";
    write_dora(dora_path);
    yue2::detail::LoraStack dora(base, {{dora_path.string(),1.F}});
    require_close(run(base,dora),{2,-6,14,7.5F});
    yue2::detail::LoraStack half_dora(base, {{dora_path.string(),.5F}});
    require_close(run(base,half_dora),{0,-4,9,10.25F});
    yue2::detail::LoraStack zero_dora(base, {{dora_path.string(),0.F}});
    assert(run(base,zero_dora)==run(base,disabled));
    const auto mixed_path=temporary.path/"mixed.gguf";write_mixed(mixed_path);
    yue2::detail::LoraStack mixed(base,{{mixed_path.string(),1.F}});
    require_close(run(base,mixed),{4,-4,28,5});
    require_close(run(base,mixed,"model.layers.0.nar_self_attn.q_proj"),{2,-6,14,7.5F});
    for (int test=0;test<3;++test) {
        bool invalid=false;
        try {
            if(test==0) { yue2::detail::LoraStack bad(base,{{dora_path.string(),1.F},{adapter_path.string(),1.F}}); }
            else {
                const auto bad_path=temporary.path/(test==1?"missing-dora.gguf":"nonfinite-dora.gguf");
                write_dora(bad_path,test==1,test==2);
                yue2::detail::LoraStack bad(base,{{bad_path.string(),1.F}});
            }
        } catch(const std::runtime_error &) { invalid=true; }
        assert(invalid);
    }

    yue2::detail::LoraStack replaced(
        base, {{replacement_path.string(), 0.5F}});
    assert(replaced.target_count() == 1);
    require_close(run(base, replaced), {-1, -1, 2, 6.5F});

    std::vector<float> after(6);
    ggml_backend_tensor_get(
        base.get(std::string(kStem) + ".weight"), after.data(), 0,
        after.size() * sizeof(after.front()));
    assert(after == original);

    bool rejected = false;
    try {
        yue2::detail::LoraStack wrong(
            base, {{wrong_path.string(), 1.0F}});
    } catch (const std::runtime_error & error) {
        rejected = std::string(error.what()).find("fingerprint mismatch") != std::string::npos;
    }
    assert(rejected);
    return 0;
}
