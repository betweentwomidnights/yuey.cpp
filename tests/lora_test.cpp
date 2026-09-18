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

std::vector<float> run(
    yue2::detail::GgufModel & base,
    const yue2::detail::LoraStack & loras) {
    auto * context = ggml_init({1024 * 1024, nullptr, true});
    assert(context);
    auto * input = ggml_new_tensor_2d(context, GGML_TYPE_F32, 3, 2);
    ggml_set_input(input);
    auto * output = loras.linear(
        context, base.get("model.layers.0.self_attn.q_proj.weight"), input);
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

    yue2::detail::LoraStack disabled(
        base, {{adapter_path.string(), 0.0F}});
    require_close(run(base, disabled), {-2, -2, 4, 13});

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
