// Minimal GGUF/backend support for yue2.cpp.
//
// The overall loader pattern is adapted from the MIT-licensed sa3.cpp model
// loader so all three repositories can keep using the same GGML fork and
// tensor conventions without depending on one another's source trees.
#pragma once

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"

#include <cerrno>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <optional>
#include <utility>
#include <vector>

namespace yue2::detail {

inline std::runtime_error gguf_error(const std::string & message) {
    return std::runtime_error("[yue2:gguf] " + message);
}

inline bool file_seek(FILE * file, std::uint64_t offset) {
#ifdef _WIN32
    return _fseeki64(file, static_cast<long long>(offset), SEEK_SET) == 0;
#else
    return fseeko(file, static_cast<off_t>(offset), SEEK_SET) == 0;
#endif
}

inline std::uint64_t file_size(FILE * file) {
#ifdef _WIN32
    const auto position = _ftelli64(file);
    if (position < 0 || _fseeki64(file, 0, SEEK_END) != 0) throw gguf_error("failed to determine file size");
    const auto size = _ftelli64(file);
    if (size < 0 || _fseeki64(file, position, SEEK_SET) != 0) throw gguf_error("failed to determine file size");
#else
    const auto position = ftello(file);
    if (position < 0 || fseeko(file, 0, SEEK_END) != 0) throw gguf_error("failed to determine file size");
    const auto size = ftello(file);
    if (size < 0 || fseeko(file, position, SEEK_SET) != 0) throw gguf_error("failed to determine file size");
#endif
    return static_cast<std::uint64_t>(size);
}

inline int threads_from_environment() {
    const char * value = std::getenv("YUE2_THREADS");
    if (!value || !*value) return 0;
    char * end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (!end || *end != '\0' || parsed <= 0 || parsed > 1024) {
        std::fprintf(stderr, "[yue2] ignoring invalid YUE2_THREADS='%s'\n", value);
        return 0;
    }
    return static_cast<int>(parsed);
}

inline void load_dynamic_backends_once() {
    static const bool loaded = []() {
        ggml_backend_load_all();
        return true;
    }();
    (void) loaded;
}

inline ggml_backend_t make_backend(
    const char * requested_device = nullptr,
    int cpu_threads = 0,
    bool announce = true) {
    load_dynamic_backends_once();
    std::string request = requested_device ? requested_device : "";
    if (request.empty()) {
        if (const char * value = std::getenv("YUE2_DEVICE")) request = value;
    }

    std::string normalized_request = request;
    for (char & c : normalized_request) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    const bool request_cpu = normalized_request == "cpu";
    const bool request_accelerator = !request.empty() && normalized_request != "auto" && !request_cpu;

    ggml_backend_t backend = nullptr;
    if (!request_cpu) {
        std::vector<ggml_backend_dev_t> devices;
        for (std::size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto * device = ggml_backend_dev_get(i);
            const auto type = ggml_backend_dev_type(device);
            if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                devices.push_back(device);
            }
        }

        ggml_backend_dev_t selected = nullptr;
        const char * selector = std::getenv("YUE2_GPU");
        if ((!selector || !*selector) && request_accelerator) selector = request.c_str();
        const bool explicit_accelerator = selector && *selector;
        if (selector && *selector) {
            char * end = nullptr;
            const long index = std::strtol(selector, &end, 10);
            if (end && *end == '\0' && index >= 0 && static_cast<std::size_t>(index) < devices.size()) {
                selected = devices[static_cast<std::size_t>(index)];
            } else {
                std::string needle = selector;
                for (char & c : needle) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                for (auto * device : devices) {
                    std::string haystack = std::string(ggml_backend_dev_name(device)) + " " +
                        ggml_backend_dev_description(device);
                    for (char & c : haystack) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                    if (haystack.find(needle) != std::string::npos) {
                        selected = device;
                        break;
                    }
                }
            }
            if (!selected) {
                throw gguf_error("requested accelerator was not found: " + std::string(selector));
            }
        }
        if (!selected) {
            std::size_t best_memory = 0;
            for (auto * device : devices) {
                if (ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
                std::size_t free = 0, total = 0;
                ggml_backend_dev_memory(device, &free, &total);
                if (!selected || total > best_memory) {
                    selected = device;
                    best_memory = total;
                }
            }
            if (!selected && !devices.empty()) selected = devices.front();
        }
        if (selected) backend = ggml_backend_dev_init(selected, nullptr);
        if (explicit_accelerator && !backend) {
            throw gguf_error("could not initialize requested accelerator: " + std::string(selector));
        }
    }

    if (!backend) backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) throw gguf_error("no usable GGML backend");

    auto * device = ggml_backend_get_device(backend);
    if (device && ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        const int threads = cpu_threads > 0 ? cpu_threads : threads_from_environment();
        if (threads > 0) {
            auto * registration = ggml_backend_dev_backend_reg(device);
            auto set_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(
                ggml_backend_reg_get_proc_address(registration, "ggml_backend_set_n_threads"));
            if (set_threads) set_threads(backend, threads);
        }
    }
    if (announce) {
        std::fprintf(stderr, "[yue2] backend: %s (%s)\n", ggml_backend_name(backend),
            device ? ggml_backend_dev_description(device) : "unknown device");
    }
    return backend;
}

class GgufModel {
public:
    GgufModel() = default;
    ~GgufModel() { reset(); }
    GgufModel(const GgufModel &) = delete;
    GgufModel & operator=(const GgufModel &) = delete;
    GgufModel(GgufModel && other) noexcept { move_from(other); }
    GgufModel & operator=(GgufModel && other) noexcept {
        if (this != &other) {
            reset();
            move_from(other);
        }
        return *this;
    }

    ggml_tensor * get(const std::string & name) const {
        const auto found = tensors_.find(name);
        if (found == tensors_.end()) throw gguf_error("missing tensor: " + name);
        return found->second;
    }

    bool has(const std::string & name) const { return tensors_.count(name) != 0; }
    ggml_backend_t backend() const { return backend_; }

    std::vector<std::string> tensor_names() const {
        std::vector<std::string> result;
        result.reserve(tensors_.size());
        for (const auto & entry : tensors_) result.push_back(entry.first);
        return result;
    }

    std::string string(const char * key) const {
        const int index = gguf_find_key(gguf_, key);
        if (index < 0 || gguf_get_kv_type(gguf_, index) != GGUF_TYPE_STRING) {
            throw gguf_error("missing string key: " + std::string(key));
        }
        return gguf_get_val_str(gguf_, index);
    }

    std::uint32_t u32(const char * key) const {
        const int index = gguf_find_key(gguf_, key);
        if (index < 0 || gguf_get_kv_type(gguf_, index) != GGUF_TYPE_UINT32) {
            throw gguf_error("missing uint32 key: " + std::string(key));
        }
        return gguf_get_val_u32(gguf_, index);
    }

    float f32(const char * key) const {
        const int index = gguf_find_key(gguf_, key);
        if (index < 0 || gguf_get_kv_type(gguf_, index) != GGUF_TYPE_FLOAT32) {
            throw gguf_error("missing float32 key: " + std::string(key));
        }
        return gguf_get_val_f32(gguf_, index);
    }

    std::optional<std::string> optional_string(const char * key) const {
        const int index = gguf_find_key(gguf_, key);
        if (index < 0) return std::nullopt;
        if (gguf_get_kv_type(gguf_, index) != GGUF_TYPE_STRING) {
            throw gguf_error("key is not a string: " + std::string(key));
        }
        return std::string(gguf_get_val_str(gguf_, index));
    }

private:
    friend GgufModel load_gguf(const char *, const char *, int);
    friend GgufModel load_gguf_raw(const char *, const char *, int);
    friend GgufModel load_gguf_raw_impl(const char *, ggml_backend_t, bool);
    friend GgufModel load_gguf_raw_on_backend(const char *, ggml_backend_t);
    friend GgufModel load_gguf_raw_on_backend(const char *, ggml_backend_t);

    void reset() {
        if (buffer_) ggml_backend_buffer_free(buffer_);
        if (gguf_) gguf_free(gguf_);
        if (context_) ggml_free(context_);
        if (backend_ && owns_backend_) ggml_backend_free(backend_);
        buffer_ = nullptr;
        gguf_ = nullptr;
        context_ = nullptr;
        backend_ = nullptr;
        owns_backend_ = false;
        tensors_.clear();
    }

    void move_from(GgufModel & other) noexcept {
        context_ = std::exchange(other.context_, nullptr);
        gguf_ = std::exchange(other.gguf_, nullptr);
        backend_ = std::exchange(other.backend_, nullptr);
        owns_backend_ = std::exchange(other.owns_backend_, false);
        buffer_ = std::exchange(other.buffer_, nullptr);
        tensors_ = std::move(other.tensors_);
    }

    ggml_context * context_ = nullptr;
    gguf_context * gguf_ = nullptr;
    ggml_backend_t backend_ = nullptr;
    bool owns_backend_ = false;
    ggml_backend_buffer_t buffer_ = nullptr;
    std::map<std::string, ggml_tensor *> tensors_;
};

inline GgufModel load_gguf_raw_impl(
    const char * path,
    ggml_backend_t backend,
    bool owns_backend) {
    GgufModel model;
    if (!backend) throw gguf_error("cannot load weights onto a null backend");
    model.backend_ = backend;
    model.owns_backend_ = owns_backend;
    const gguf_init_params params = {true, &model.context_};
    model.gguf_ = gguf_init_from_file(path, params);
    if (!model.gguf_) throw gguf_error("failed to open " + std::string(path));

    std::unique_ptr<FILE, int (*)(FILE *)> file(std::fopen(path, "rb"), std::fclose);
    if (!file) throw gguf_error("cannot read " + std::string(path));
    const auto bytes = file_size(file.get());
    const auto data_offset = static_cast<std::uint64_t>(gguf_get_data_offset(model.gguf_));

    struct Read {
        ggml_tensor * tensor;
        std::string name;
        std::uint64_t offset;
        std::size_t size;
    };
    std::vector<Read> reads;
    for (auto * tensor = ggml_get_first_tensor(model.context_); tensor;
         tensor = ggml_get_next_tensor(model.context_, tensor)) {
        const std::string name = ggml_get_name(tensor);
        const int index = gguf_find_tensor(model.gguf_, name.c_str());
        if (index < 0) throw gguf_error("missing tensor offset: " + name);
        const auto offset = data_offset + gguf_get_tensor_offset(model.gguf_, index);
        const auto size = ggml_nbytes(tensor);
        if (offset > bytes || size > bytes - offset) throw gguf_error("truncated tensor: " + name);
        reads.push_back({tensor, name, offset, size});
    }

    model.buffer_ = ggml_backend_alloc_ctx_tensors(model.context_, model.backend_);
    if (!model.buffer_) throw gguf_error("could not allocate model weights");
    std::vector<std::uint8_t> scratch;
    for (const auto & read : reads) {
        scratch.resize(read.size);
        if (!file_seek(file.get(), read.offset)) throw gguf_error("seek failed for tensor: " + read.name);
        if (std::fread(scratch.data(), 1, read.size, file.get()) != read.size) {
            throw gguf_error("short read for tensor: " + read.name);
        }
        ggml_backend_tensor_set(read.tensor, scratch.data(), 0, read.size);
        model.tensors_[read.name] = read.tensor;
    }
    return model;
}

inline GgufModel load_gguf_raw(const char * path, const char * device = nullptr, int threads = 0) {
    return load_gguf_raw_impl(path, make_backend(device, threads), true);
}

// Load another immutable GGUF buffer on an already selected backend. The
// returned model borrows the backend and owns only its context and buffer.
inline GgufModel load_gguf_raw_on_backend(const char * path, ggml_backend_t backend) {
    return load_gguf_raw_impl(path, backend, false);
}

inline GgufModel load_gguf(const char * path, const char * device = nullptr, int threads = 0) {
    auto model = load_gguf_raw(path, device, threads);
    if (model.string("yue2.transcription.architecture") != "sheetsage2-mert2-fs") {
        throw gguf_error("not a yue2 SheetSage2/MERT2 transcription model");
    }
    if (model.u32("yue2.transcription.sample_rate") != 24000) {
        throw gguf_error("unsupported MERT2 sample rate");
    }
    return model;
}

} // namespace yue2::detail
