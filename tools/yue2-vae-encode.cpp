#include "yue2/vae.h"
#include "yue2/audio.h"
#include "server/json.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <cmath>
#include <limits>

namespace fs=std::filesystem;
namespace json=yue2::server::json;
template<class T> std::vector<T> read(const fs::path &path) {
    const auto size=fs::file_size(path);
    if(!size || size%sizeof(T))throw std::runtime_error("Invalid input: "+path.string());
    std::vector<T> result(size/sizeof(T));std::ifstream in(path,std::ios::binary);
    if(!in.read(reinterpret_cast<char *>(result.data()),size))throw std::runtime_error("Short input");
    return result;
}
void write(const fs::path &target,const std::vector<float> &values) {
    if(fs::exists(target))throw std::runtime_error("Output exists: "+target.string());
    if(!target.parent_path().empty())fs::create_directories(target.parent_path());
    auto temporary=target;temporary+=".partial";
    {std::ofstream out(temporary,std::ios::binary);if(!out.write(reinterpret_cast<const char *>(values.data()),values.size()*sizeof(float)))throw std::runtime_error("Cannot write posterior means");}
    fs::rename(temporary,target);
}
std::vector<float> clip_48k(const yue2::audio::PcmAudio &source,int64_t begin,int64_t end,bool trimmed) {
    if(begin<0 || end<=begin)throw std::runtime_error("Invalid source context");
    const int64_t total=source.interleaved_samples.size()/source.channels;
    const long double first_sample=static_cast<long double>(begin)*source.sample_rate/25;
    const long double last_sample=static_cast<long double>(end)*source.sample_rate/25;
    if(first_sample>total || last_sample>static_cast<long double>(total)+(trimmed?64:0)+.5L)
        throw std::runtime_error("Source context exceeds recording");
    const int64_t first=std::llround(first_sample);
    int64_t last=std::llround(last_sample);
    if(trimmed)last=std::min(last,total);
    if(first>=last || last>total)throw std::runtime_error("Source context exceeds recording");
    std::vector<std::vector<float>> channels(source.channels);
    for(int c=0;c<source.channels;++c) {
        auto &pcm=channels[c];pcm.resize(last-first);
        for(int64_t i=first;i<last;++i)pcm[i-first]=source.interleaved_samples[i*source.channels+c];
        if(source.sample_rate!=48000)pcm=yue2::audio::resample_vae(pcm,source.sample_rate,48000);
    }
    std::vector<float> result(channels[0].size()*source.channels);
    for(size_t i=0;i<channels[0].size();++i)for(int c=0;c<source.channels;++c)result[i*source.channels+c]=channels[c][i];
    return result;
}
int main(int argc,char **argv) {
    try {
        std::map<std::string,std::string> args;
        for(int i=1;i<argc;++i) {
            const std::string key=argv[i];
            if(key=="--help") {std::cout<<"Native VAE posterior means: --model VAE.gguf (--pcm-f32 interleaved.f32 --channels 1|2 --out means.f32 | --plan windows.json) [--device cuda|vulkan|metal|cpu] [--core-frames 128] [--full] [--stop-file PATH]\nPlan entries: audio,begin_frame,end_frame,core_offset,frames,out,identity,decoder_trim_samples(optional).\n";return 0;}
            if(key=="--full"){args[key]="1";continue;}
            if(i+1==argc)throw std::runtime_error("Missing argument value");
            if(key!="--model" && key!="--pcm-f32" && key!="--channels" && key!="--out" && key!="--device" && key!="--threads" && key!="--core-frames" && key!="--plan" && key!="--stop-file")throw std::runtime_error("Unknown option: "+key);
            if(!args.emplace(key,argv[++i]).second)throw std::runtime_error("Duplicate option");
        }
        auto required=[&](const char *key){if(!args.count(key))throw std::runtime_error(std::string("Missing ")+key);return args.at(key);};
        yue2::VaeRuntimeOptions options;options.device=args.count("--device")?args.at("--device"):"cpu";
        options.threads=args.count("--threads")?std::stoi(args.at("--threads")):4;
        if(args.count("--core-frames"))options.encode_core_frames=std::stoll(args.at("--core-frames"));
        options.tiled=!args.count("--full");
        const auto start=std::chrono::steady_clock::now();
        yue2::VaeEncoder encoder(required("--model"),options);
        yue2::VaeEncodeControl control;
        control.should_cancel=[&](){return args.count("--stop-file") && fs::exists(args.at("--stop-file"));};
        if(args.count("--plan")) {
            if(args.count("--pcm-f32") || args.count("--out") || args.count("--channels"))throw std::runtime_error("Plan and raw inputs are exclusive");
            const auto bytes=read<char>(args.at("--plan"));const auto plan=json::parse(std::string(bytes.begin(),bytes.end()));
            if(plan.type!=json::Type::array || plan.array.empty())throw std::runtime_error("Plan must be a nonempty array");
            std::string previous;yue2::audio::PcmAudio source;
            for(size_t i=0;i<plan.array.size();++i) {
                if(control.should_cancel())break;
                const auto &item=plan.array[i];const auto audio=json::string(item,"audio"),target=json::string(item,"out"),identity=json::string(item,"identity");
                if(audio.empty() || target.empty() || identity.empty())throw std::runtime_error("Incomplete VAE plan entry");
                const auto begin=json::u64(item,"begin_frame",0),end=json::u64(item,"end_frame",0),offset=json::u64(item,"core_offset",0),frames=json::u64(item,"frames",0);
                const auto trim=json::u32(item,"decoder_trim_samples",0);
                if(!frames || frames>24576 || offset>24576 || end<=begin ||
                   end>static_cast<uint64_t>(std::numeric_limits<int64_t>::max()/48000) ||
                   offset+frames>end-begin || (trim!=0 && trim!=64))throw std::runtime_error("Invalid VAE plan core");
                const auto window_start=std::chrono::steady_clock::now();
                if(audio!=previous) {source=yue2::audio::read_audio_pcm(audio);previous=audio;}
                if(trim && source.sample_rate!=48000)throw std::runtime_error("Native decoder edge requires 48 kHz");
                auto pcm=clip_48k(source,begin,end,trim!=0);
                std::vector<float> values;
                try { values=encoder.encode(pcm.data(),pcm.size()/source.channels,source.channels,control); }
                catch(const std::exception &) {if(control.should_cancel())break;throw;}
                if((offset+frames)*64>values.size())throw std::runtime_error("Encoded context does not cover target core");
                std::vector<float> core(values.begin()+offset*64,values.begin()+(offset+frames)*64);write(target,core);
                double squared=0;for(float value:core)squared+=static_cast<double>(value)*value;
                std::cout<<"VAE_EVENT {\"index\":"<<i<<",\"identity\":"<<json::quote(identity)<<",\"frames\":"<<frames
                         <<",\"latent_rms\":"<<std::sqrt(squared/core.size())<<",\"seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-window_start).count()<<"}"<<std::endl;
            }
            return 0;
        }
        const auto pcm=read<float>(required("--pcm-f32"));
        const auto channels=args.count("--channels")?std::stoi(args.at("--channels")):2;
        if(channels<1 || channels>2 || pcm.size()%channels)throw std::runtime_error("Invalid channel count/alignment");
        const auto values=encoder.encode(pcm.data(),pcm.size()/channels,channels);
        write(required("--out"),values);
        std::cout<<"{\"frames\":"<<values.size()/64<<",\"seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<"}\n";
        return 0;
    } catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}
}
