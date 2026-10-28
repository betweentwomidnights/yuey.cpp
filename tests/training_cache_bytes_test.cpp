#include "training_cache_bytes.h"
#include <cassert>
#include <chrono>

int main() {
    namespace fs=std::filesystem;
    const auto root=fs::temp_directory_path()/
        ("yue2-byte-cache-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directory(root);
    const auto model=root/"model.gguf",cache=root/"sample.gguf";
    std::ofstream(model)<<"model identity sentinel";
    auto write=[&](uint32_t tokens=3) {
        auto *ctx=ggml_init({ggml_tensor_overhead()*8,nullptr,true});assert(ctx);
        std::vector<float> clean(128,0);
        std::vector<ggml_fp16_t> kv(128*3*8*28,0);
        auto *target=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,clean.size());
        auto *keys=ggml_new_tensor_1d(ctx,GGML_TYPE_F16,kv.size());
        auto *values=ggml_new_tensor_1d(ctx,GGML_TYPE_F16,kv.size());
        target->data=clean.data();keys->data=kv.data();values->data=kv.data();
        ggml_set_name(target,"clean");ggml_set_name(keys,"keys");ggml_set_name(values,"values");
        auto *file=gguf_init_empty();assert(file);
        gguf_set_val_str(file,"general.architecture","yue2_nar_sample");
        gguf_set_val_str(file,"yue2.training.model",model.string().c_str());
        gguf_set_val_str(file,"yue2.training.fixed_adapters","");
        gguf_set_val_str(file,"yue2.training.conditioning_adapters","");
        gguf_set_val_u32(file,"yue2.training.ar_tokens",tokens);
        for (auto *tensor : {target,keys,values}) gguf_add_tensor(file,tensor);
        assert(gguf_write_to_file(file,cache.string().c_str(),false));
        gguf_free(file);ggml_free(ctx);
    };
    auto inspect=[&]() {return yue2::detail::inspect_nar_cache_header(cache.string(),model.string(),"","");};
    write();const auto header=inspect();
    assert(header.frames==2 && header.tokens==3 && header.bytes==fs::file_size(cache));
    const auto before=yue2::detail::training_file_fingerprint(cache.string());
    const auto stamp=fs::last_write_time(cache);
    auto *metadata=gguf_init_from_file(cache.string().c_str(),{true,nullptr});assert(metadata);
    const auto offset=gguf_get_data_offset(metadata)+gguf_get_tensor_offset(metadata,0);
    gguf_free(metadata);
    {
        std::fstream file(cache,std::ios::binary|std::ios::in|std::ios::out);
        file.seekg(offset);char value=0;file.read(&value,1);value^=1;
        file.seekp(offset);file.write(&value,1);assert(file.good());
    }
    fs::last_write_time(cache,stamp);
    assert(inspect().bytes==header.bytes && fs::last_write_time(cache)==stamp);
    assert(yue2::detail::training_file_fingerprint(cache.string())!=before);
    bool rejected=false;
    try {yue2::detail::inspect_nar_cache_header(cache.string(),model.string(),"wrong adapter","");}
    catch(const std::exception &) {rejected=true;}
    assert(rejected);
    write(4);rejected=false;
    try {inspect();} catch(const std::exception &) {rejected=true;}
    assert(rejected);
    write();fs::resize_file(cache,fs::file_size(cache)-1);rejected=false;
    try {inspect();} catch(const std::exception &) {rejected=true;}
    assert(rejected);
    fs::remove(cache);fs::remove(model);fs::remove(root);
}
