#pragma once
#include "ggml.h"
#include "gguf.h"
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace yue2::detail {
// Versioned encoded-file fingerprint, independent of GGML allocation or F16
// expansion. Coordinator SHA-256 checks remain the cryptographic provenance
// check; this native 64-bit identity replaces the legacy 64-bit decoded hash.
inline uint64_t training_file_fingerprint(const std::string &path) {
    std::ifstream input(path,std::ios::binary);
    if (!input) throw std::runtime_error("cannot open training cache: "+path);
    std::vector<unsigned char> buffer(1024*1024);
    uint64_t hash=14695981039346656037ULL;
    while (input) {
        input.read(reinterpret_cast<char *>(buffer.data()),buffer.size());
        const auto count=input.gcount();
        for (std::streamsize i=0;i<count;++i) {
            hash ^= buffer[static_cast<size_t>(i)];
            hash *= 1099511628211ULL;
        }
    }
    if (!input.eof()) throw std::runtime_error("cannot read training cache: "+path);
    return hash;
}

struct NarCacheHeader {
    uint32_t tokens=0;
    size_t frames=0;
    uint64_t bytes=0;
};

inline NarCacheHeader inspect_nar_cache_header(const std::string &path,
    const std::string &model, const std::string &fixed, const std::string &conditioning) {
    ggml_context *source=nullptr;
    std::unique_ptr<gguf_context,decltype(&gguf_free)> file(
        gguf_init_from_file(path.c_str(),{true,&source}),gguf_free);
    std::unique_ptr<ggml_context,decltype(&ggml_free)> tensors(source,ggml_free);
    if (!file || !tensors) throw std::runtime_error("cannot read training cache header: "+path);
    auto text = [&](const char *name,bool optional=false) {
        const auto key=gguf_find_key(file.get(),name);
        if (key<0 && optional) return std::string();
        if (key<0 || gguf_get_kv_type(file.get(),key)!=GGUF_TYPE_STRING)
            throw std::runtime_error(std::string("missing/invalid cache string: ")+name);
        return std::string(gguf_get_val_str(file.get(),key));
    };
    if (text("general.architecture")!="yue2_nar_sample")
        throw std::runtime_error("not a NAR sample cache");
    if (!std::filesystem::equivalent(text("yue2.training.model"),model) ||
        text("yue2.training.fixed_adapters")!=fixed ||
        text("yue2.training.conditioning_adapters",true)!=conditioning)
        throw std::runtime_error("sample was prepared with different base weights or fixed adapters");
    const auto key=gguf_find_key(file.get(),"yue2.training.ar_tokens");
    if (key<0 || gguf_get_kv_type(file.get(),key)!=GGUF_TYPE_UINT32)
        throw std::runtime_error("invalid cached AR token count");
    NarCacheHeader result;
    result.tokens=gguf_get_val_u32(file.get(),key);
    result.bytes=std::filesystem::file_size(path);
    const auto data_offset=gguf_get_data_offset(file.get());
    if (data_offset>result.bytes) throw std::runtime_error("truncated training cache");
    for (int64_t i=0;i<gguf_get_n_tensors(file.get());++i) {
        const auto offset=gguf_get_tensor_offset(file.get(),i);
        const auto bytes=gguf_get_tensor_size(file.get(),i);
        if (offset>result.bytes-data_offset || bytes>result.bytes-data_offset-offset)
            throw std::runtime_error("truncated training cache tensor");
    }
    auto tensor = [&](const char *name) {
        auto *value=ggml_get_tensor(tensors.get(),name);
        if (!value) throw std::runtime_error(std::string("missing cache tensor: ")+name);
        return value;
    };
    auto *clean=tensor("clean");auto *keys=tensor("keys");auto *values=tensor("values");
    const auto count=ggml_nelements(clean);
    if (clean->type!=GGML_TYPE_F32 || count<64 || count%64 || !result.tokens)
        throw std::runtime_error("invalid NAR cache target shape/precision");
    const auto kv_count=uint64_t(result.tokens)*128*8*28;
    for (auto *value : {keys,values}) {
        if ((value->type!=GGML_TYPE_F32 && value->type!=GGML_TYPE_F16) ||
            uint64_t(ggml_nelements(value))!=kv_count)
            throw std::runtime_error("invalid NAR cache conditioning shape/precision");
    }
    result.frames=static_cast<size_t>(count/64);
    return result;
}
} // namespace yue2::detail
