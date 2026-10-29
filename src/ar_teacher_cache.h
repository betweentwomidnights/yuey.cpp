#pragma once
#include "ggml.h"
#include "gguf.h"
#include "yue2/ar_training.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace yue2::detail {
// One frozen common-model teacher distribution per AR training window. The
// fitter streams these instead of holding every [targets, 32769] F32 row set
// in host RAM. Files are immutable and written via .partial -> rename. The
// coordinator owns SHA-256 provenance of the encoded files; native loading
// rejects any file whose recorded configuration or exact prefix/codes differ.
constexpr const char *kArTeacherFormat = "yue2-ar-teacher-v1";

struct ArTeacherProvenance {
    std::string identity;         // Coordinator SHA-256 over model/runtime/device/shape.
    std::string base_sha256;      // Frozen base GGUF.
    std::string companion_sha256; // Frozen fixed adapter stack (empty when none).
    std::string fixed_adapters;   // Native absolute-path identity of fixed adapters.
    std::string device;
    uint32_t frames = 0, ar_slots = 0;
    bool predict_end = false;
    uint32_t targets() const { return frames + (predict_end ? 1 : 0); }
};

struct ArTeacherWindow {
    uint32_t index = 0;
    std::string prefix_sha256, codes_sha256; // Manifest SHA-256s; empty without one.
};

namespace ar_teacher {
using File = std::unique_ptr<gguf_context, decltype(&gguf_free)>;
using Tensors = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
struct Opened {
    File file{nullptr, gguf_free};
    Tensors tensors{nullptr, ggml_free};
    uint64_t bytes = 0;
};
inline Opened open(const std::string &path) {
    Opened result;
    ggml_context *source = nullptr;
    result.file.reset(gguf_init_from_file(path.c_str(), {true, &source}));
    result.tensors.reset(source);
    if (!result.file || !result.tensors)
        throw std::runtime_error("cannot read AR teacher cache: " + path);
    result.bytes = std::filesystem::file_size(path);
    const auto data = gguf_get_data_offset(result.file.get());
    if (data > result.bytes)
        throw std::runtime_error("truncated AR teacher cache: " + path);
    for (int64_t i = 0; i < gguf_get_n_tensors(result.file.get()); ++i) {
        const auto offset = gguf_get_tensor_offset(result.file.get(), i);
        const auto size = gguf_get_tensor_size(result.file.get(), i);
        if (offset > result.bytes - data || size > result.bytes - data - offset)
            throw std::runtime_error("truncated AR teacher cache tensor: " + path);
    }
    return result;
}
inline std::string text(const Opened &o, const char *name) {
    const auto key = gguf_find_key(o.file.get(), name);
    if (key < 0 || gguf_get_kv_type(o.file.get(), key) != GGUF_TYPE_STRING)
        throw std::runtime_error(std::string("missing/invalid AR teacher string: ") + name);
    return gguf_get_val_str(o.file.get(), key);
}
inline uint32_t u32(const Opened &o, const char *name) {
    const auto key = gguf_find_key(o.file.get(), name);
    if (key < 0 || gguf_get_kv_type(o.file.get(), key) != GGUF_TYPE_UINT32)
        throw std::runtime_error(std::string("missing/invalid AR teacher value: ") + name);
    return gguf_get_val_u32(o.file.get(), key);
}
inline float f32(const Opened &o, const char *name) {
    const auto key = gguf_find_key(o.file.get(), name);
    if (key < 0 || gguf_get_kv_type(o.file.get(), key) != GGUF_TYPE_FLOAT32)
        throw std::runtime_error(std::string("missing/invalid AR teacher value: ") + name);
    return gguf_get_val_f32(o.file.get(), key);
}
template <class T>
std::vector<T> read(const Opened &o, const std::string &path, const char *name, ggml_type type,
                    int64_t a, int64_t b = 1) {
    const auto id = gguf_find_tensor(o.file.get(), name);
    auto *t = ggml_get_tensor(o.tensors.get(), name);
    if (id < 0 || !t || t->type != type || t->ne[0] != a || t->ne[1] != b || t->ne[2] != 1 ||
        t->ne[3] != 1)
        throw std::runtime_error(std::string("invalid AR teacher tensor: ") + name);
    std::vector<T> values(size_t(a) * size_t(b));
    if (gguf_get_tensor_size(o.file.get(), id) != values.size() * sizeof(T))
        throw std::runtime_error(std::string("invalid AR teacher tensor size: ") + name);
    std::ifstream input(path, std::ios::binary);
    input.seekg(std::streamoff(gguf_get_data_offset(o.file.get()) +
                               gguf_get_tensor_offset(o.file.get(), id)));
    if (!input.read(reinterpret_cast<char *>(values.data()), std::streamsize(values.size() * sizeof(T))))
        throw std::runtime_error("short AR teacher cache read: " + path);
    return values;
}
// Header, provenance and exact window identity. Reads only prefix/codes data.
inline Opened check(const std::string &path, const ArTeacherProvenance &expected,
                    const ArTeacherWindow &window, const ArTrainingSample &sample) {
    auto o = open(path);
    if (text(o, "general.architecture") != "yue2_ar_teacher" ||
        text(o, "yue2.teacher.format") != kArTeacherFormat ||
        text(o, "yue2.teacher.precision") != "f32")
        throw std::runtime_error("not a supported AR teacher cache: " + path);
    if (text(o, "yue2.teacher.identity") != expected.identity ||
        text(o, "yue2.teacher.base_sha256") != expected.base_sha256 ||
        text(o, "yue2.teacher.companion_sha256") != expected.companion_sha256 ||
        text(o, "yue2.training.fixed_adapters") != expected.fixed_adapters ||
        text(o, "yue2.teacher.device") != expected.device ||
        u32(o, "yue2.training.frames") != expected.frames ||
        u32(o, "yue2.training.ar_slots") != expected.ar_slots ||
        u32(o, "yue2.training.predict_end") != uint32_t(expected.predict_end))
        throw std::runtime_error("AR teacher cache belongs to a different model/configuration: " + path);
    if (u32(o, "yue2.teacher.window") != window.index ||
        text(o, "yue2.teacher.prefix_sha256") != window.prefix_sha256 ||
        text(o, "yue2.teacher.codes_sha256") != window.codes_sha256)
        throw std::runtime_error("AR teacher cache belongs to a different window: " + path);
    if (read<int32_t>(o, path, "prefix", GGML_TYPE_I32, int64_t(sample.prefix.size())) != sample.prefix ||
        read<int32_t>(o, path, "codes", GGML_TYPE_I32, int64_t(sample.codes.size())) != sample.codes)
        throw std::runtime_error("AR teacher cache prefix/codes differ from the window: " + path);
    if (!std::isfinite(f32(o, "yue2.teacher.entropy")))
        throw std::runtime_error("invalid AR teacher entropy: " + path);
    return o;
}
} // namespace ar_teacher

inline void inspect_ar_teacher_cache(const std::string &path, const ArTeacherProvenance &expected,
                                     const ArTeacherWindow &window, const ArTrainingSample &sample) {
    auto o = ar_teacher::check(path, expected, window, sample);
    auto *t = ggml_get_tensor(o.tensors.get(), "probabilities");
    if (!t || t->type != GGML_TYPE_F32 || t->ne[0] != kCodecSize + 1 ||
        t->ne[1] != expected.targets() || t->ne[2] != 1 || t->ne[3] != 1)
        throw std::runtime_error("invalid AR teacher distribution shape/precision: " + path);
}

// Decodes one window: prefix/codes plus its frozen distribution and entropy.
// The trainer revalidates finiteness, nonnegativity and row normalization.
inline ArTrainingSample load_ar_teacher_cache(const std::string &path,
                                              const ArTeacherProvenance &expected,
                                              const ArTeacherWindow &window,
                                              const ArTrainingSample &sample) {
    auto o = ar_teacher::check(path, expected, window, sample);
    ArTrainingSample result{sample.prefix, sample.codes, {}, 0};
    result.teacher_probabilities = ar_teacher::read<float>(o, path, "probabilities", GGML_TYPE_F32,
                                                           kCodecSize + 1, expected.targets());
    result.teacher_entropy = ar_teacher::f32(o, "yue2.teacher.entropy");
    return result;
}

inline void write_ar_teacher_cache(const std::string &path, const ArTeacherProvenance &provenance,
                                   const ArTeacherWindow &window, const ArTrainingSample &sample) {
    if (provenance.identity.empty())
        throw std::runtime_error("AR teacher cache requires an identity");
    if (sample.teacher_probabilities.size() != size_t(kCodecSize + 1) * provenance.targets() ||
        !std::isfinite(sample.teacher_entropy))
        throw std::runtime_error("AR teacher distribution does not match the cache shape");
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
        ggml_init({ggml_tensor_overhead() * 4, nullptr, true}), ggml_free);
    ar_teacher::File file(gguf_init_empty(), gguf_free);
    if (!ctx || !file)
        throw std::runtime_error("AR teacher cache allocation failed");
    auto *f = file.get();
    gguf_set_val_str(f, "general.architecture", "yue2_ar_teacher");
    gguf_set_val_str(f, "yue2.teacher.format", kArTeacherFormat);
    gguf_set_val_str(f, "yue2.teacher.precision", "f32");
    gguf_set_val_str(f, "yue2.teacher.identity", provenance.identity.c_str());
    gguf_set_val_str(f, "yue2.teacher.base_sha256", provenance.base_sha256.c_str());
    gguf_set_val_str(f, "yue2.teacher.companion_sha256", provenance.companion_sha256.c_str());
    gguf_set_val_str(f, "yue2.training.fixed_adapters", provenance.fixed_adapters.c_str());
    gguf_set_val_str(f, "yue2.teacher.device", provenance.device.c_str());
    gguf_set_val_u32(f, "yue2.training.frames", provenance.frames);
    gguf_set_val_u32(f, "yue2.training.ar_slots", provenance.ar_slots);
    gguf_set_val_u32(f, "yue2.training.predict_end", provenance.predict_end ? 1 : 0);
    gguf_set_val_u32(f, "yue2.teacher.window", window.index);
    gguf_set_val_str(f, "yue2.teacher.prefix_sha256", window.prefix_sha256.c_str());
    gguf_set_val_str(f, "yue2.teacher.codes_sha256", window.codes_sha256.c_str());
    gguf_set_val_f32(f, "yue2.teacher.entropy", sample.teacher_entropy);
    auto add = [&](const char *name, ggml_type type, const void *data, int64_t a, int64_t b = 1) {
        auto *t = ggml_new_tensor_2d(ctx.get(), type, a, b);
        ggml_set_name(t, name);
        t->data = const_cast<void *>(data);
        gguf_add_tensor(f, t);
    };
    add("probabilities", GGML_TYPE_F32, sample.teacher_probabilities.data(), kCodecSize + 1,
        provenance.targets());
    add("prefix", GGML_TYPE_I32, sample.prefix.data(), int64_t(sample.prefix.size()));
    add("codes", GGML_TYPE_I32, sample.codes.data(), int64_t(sample.codes.size()));
    const std::filesystem::path target(path);
    if (std::filesystem::exists(target))
        throw std::runtime_error("AR teacher cache exists: " + path);
    if (!target.parent_path().empty())
        std::filesystem::create_directories(target.parent_path());
    const auto temporary = path + ".partial";
    if (!gguf_write_to_file(f, temporary.c_str(), false))
        throw std::runtime_error("cannot write " + temporary);
    std::filesystem::rename(temporary, target);
}
} // namespace yue2::detail
