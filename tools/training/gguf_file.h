// Reading GGUF metadata and tensor payloads for provenance checks.
#pragma once

#include "server/json.h"
#include "training/files.h"

#include "ggml.h"
#include "gguf.h"

#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace yue2::training {

// A GGUF's metadata, with tensor payloads read from the file on demand.
class GgufFile {
public:
    explicit GgufFile(const std::filesystem::path & path) : path_(path) {
        gguf_init_params params{true, &meta_};
        file_ = gguf_init_from_file(path.u8string().c_str(), params);
        if (!file_) throw std::invalid_argument("Cannot read " + python_path(path));
    }
    ~GgufFile() {
        gguf_free(file_);
        if (meta_) ggml_free(meta_);
    }
    GgufFile(const GgufFile &) = delete;
    GgufFile & operator=(const GgufFile &) = delete;

    std::optional<server::json::Value> field(const char * name) const {
        const auto id = gguf_find_key(file_, name);
        if (id < 0) return std::nullopt;
        switch (gguf_get_kv_type(file_, id)) {
            case GGUF_TYPE_STRING: return server::json::from_string(gguf_get_val_str(file_, id));
            case GGUF_TYPE_UINT32: return server::json::from_integer(gguf_get_val_u32(file_, id));
            case GGUF_TYPE_INT32: return server::json::from_integer(gguf_get_val_i32(file_, id));
            case GGUF_TYPE_UINT64: return server::json::from_integer(static_cast<long long>(gguf_get_val_u64(file_, id)));
            case GGUF_TYPE_INT64: return server::json::from_integer(gguf_get_val_i64(file_, id));
            default: return server::json::Value{};
        }
    }
    // Metadata keys in file order, and typed access to their values.
    std::vector<std::string> keys() const {
        std::vector<std::string> names;
        for (int64_t i = 0; i < gguf_get_n_kv(file_); ++i) names.emplace_back(gguf_get_key(file_, i));
        return names;
    }
    gguf_type type(const char * name) const { return gguf_get_kv_type(file_, id(name)); }
    std::string string(const char * name) const { return gguf_get_val_str(file_, id(name)); }
    std::uint32_t u32(const char * name) const { return gguf_get_val_u32(file_, id(name)); }
    float f32(const char * name) const { return gguf_get_val_f32(file_, id(name)); }
    std::vector<std::string> tensors() const {
        std::vector<std::string> names;
        for (int64_t i = 0; i < gguf_get_n_tensors(file_); ++i) names.emplace_back(gguf_get_tensor_name(file_, i));
        return names;
    }
    const ggml_tensor & tensor(const std::string & name) const {
        const auto * found = ggml_get_tensor(meta_, name.c_str());
        if (!found) throw std::invalid_argument("No tensor " + name + " in " + python_path(path_));
        return *found;
    }
    std::size_t elements(const std::string & name) const { return static_cast<std::size_t>(ggml_nelements(&tensor(name))); }
    std::string payload(const std::string & name) const {
        const auto id = gguf_find_tensor(file_, name.c_str());
        std::ifstream stream(path_, std::ios::binary);
        stream.seekg(static_cast<std::streamoff>(gguf_get_data_offset(file_) + gguf_get_tensor_offset(file_, id)));
        std::string bytes(gguf_get_tensor_size(file_, id), '\0');
        if (!stream.read(bytes.data(), static_cast<std::streamsize>(bytes.size()))) throw std::invalid_argument("Truncated " + python_path(path_));
        return bytes;
    }

private:
    int64_t id(const char * name) const {
        const auto found = gguf_find_key(file_, name);
        if (found < 0) throw std::invalid_argument(std::string("Missing GGUF field ") + name + " in " + python_path(path_));
        return found;
    }
    std::filesystem::path path_;
    ggml_context * meta_ = nullptr;
    gguf_context * file_ = nullptr;
};

} // namespace yue2::training
