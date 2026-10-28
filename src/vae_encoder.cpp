#include "yue2/vae.h"
#include "gguf_model.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>

namespace yue2 {
namespace {
constexpr size_t kGraphSize=4096;
constexpr int64_t kRatio=1920;
constexpr std::array<int,6> kStrides={2,2,4,4,5,6};

ggml_tensor *f32(ggml_context *ctx,ggml_tensor *x) {
    return x->type==GGML_TYPE_F32?x:ggml_cast(ctx,x,GGML_TYPE_F32);
}
ggml_tensor *channel(ggml_context *ctx,ggml_tensor *x) {
    return ggml_reshape_2d(ctx,f32(ctx,x),1,ggml_nelements(x));
}
ggml_tensor *snake(ggml_context *ctx,detail::GgufModel &model,ggml_tensor *x,const std::string &name) {
    auto *alpha=ggml_exp(ctx,channel(ctx,model.get(name+".alpha")));
    auto *beta=ggml_scale_bias(ctx,ggml_exp(ctx,channel(ctx,model.get(name+".beta"))),1.0f,1e-9f);
    auto *sine=ggml_sin(ctx,ggml_mul(ctx,x,alpha));
    return ggml_add(ctx,x,ggml_div(ctx,ggml_sqr(ctx,sine),beta));
}
ggml_tensor *conv(ggml_context *ctx,detail::GgufModel &model,ggml_tensor *x,
                  const std::string &name,int stride=1,int dilation=1,int padding=-1) {
    auto *weight=f32(ctx,model.get(name+".weight"));
    if(padding<0)padding=static_cast<int>((weight->ne[0]-1)*dilation/2);
    // Keep activations F32. ggml_conv_1d's default F16 im2col would round the
    // reference encoder's inputs before every convolution.
    auto *columns=ggml_im2col(ctx,weight,x,stride,0,padding,0,dilation,0,false,GGML_TYPE_F32);
    const auto length=columns->ne[1];
    // Match ggml_conv_1d's [time,channels] output layout. Besides avoiding a
    // transpose, this keeps short final bins out of CUDA's reduced-precision
    // small-batch MMF path (which does not honor an explicit F32 request).
    auto *output=ggml_mul_mat(ctx,ggml_reshape_2d(ctx,columns,columns->ne[0],length),
                             ggml_reshape_2d(ctx,weight,weight->ne[0]*weight->ne[1],weight->ne[2]));
    ggml_mul_mat_set_prec(output,GGML_PREC_F32);
    return ggml_add(ctx,output,channel(ctx,model.get(name+".bias")));
}
ggml_tensor *encode_graph(ggml_context *ctx,detail::GgufModel &model,ggml_tensor *x) {
    x=conv(ctx,model,x,"encoder.layers.0");
    for(size_t layer=0;layer<kStrides.size();++layer) {
        const auto prefix="encoder.layers."+std::to_string(layer+1)+".layers";
        for(int unit=0;unit<3;++unit) {
            const auto name=prefix+"."+std::to_string(unit)+".layers";
            const int dilation=unit==0?1:unit==1?3:9;
            auto *y=conv(ctx,model,snake(ctx,model,x,name+".0"),name+".1",1,dilation);
            y=conv(ctx,model,snake(ctx,model,y,name+".2"),name+".3");
            x=ggml_add(ctx,x,y);
        }
        x=conv(ctx,model,snake(ctx,model,x,prefix+".3"),prefix+".4",kStrides[layer],1,(kStrides[layer]+1)/2);
    }
    return conv(ctx,model,snake(ctx,model,x,"encoder.layers.7"),"encoder.layers.8");
}
}

class VaeEncoder::Impl {
public:
    Impl(const std::string &path,const VaeRuntimeOptions &options)
        : model(detail::load_gguf_raw_selected(path.c_str(),options.device.empty()?nullptr:options.device.c_str(),
                    options.threads,[](const std::string &name){return name.rfind("encoder.",0)==0;})),
          tiled(options.tiled),core(options.encode_core_frames),halo(options.encode_halo_frames) {
        if(options.threads<0 || core<1 || core>24576 || halo<16 || halo>24576)
            throw std::invalid_argument("VAE encoder requires positive core frames and at least 16 halo frames");
        if(model.string("general.architecture")!="yue2_vae" || model.string("yue2.component")!="vae" ||
           model.u32("yue2.vae.sample_rate")!=48000 || model.u32("yue2.vae.audio_channels")!=2 ||
           model.u32("yue2.vae.latent_dim")!=64 || model.u32("yue2.vae.downsampling_ratio")!=1920)
            throw std::runtime_error("[yue2:vae] unsupported encoder model");
        for(const auto &name:model.tensor_names()) {
            const auto type=model.get(name)->type;
            if(type!=GGML_TYPE_F16 && type!=GGML_TYPE_F32)throw std::runtime_error("VAE encoder requires folded F16/F32 weights");
        }
        auto vector_shape=[&](const std::string &name,int64_t count) {
            if(ggml_nelements(model.get(name))!=count)throw std::runtime_error("Encoder vector shape differs: "+name);
        };
        auto convolution_shape=[&](const std::string &name,int64_t kernel,int64_t input,int64_t output) {
            const auto *w=model.get(name+".weight");
            if(w->ne[0]!=kernel || w->ne[1]!=input || w->ne[2]!=output || w->ne[3]!=1)
                throw std::runtime_error("Encoder convolution shape differs: "+name);
            vector_shape(name+".bias",output);
        };
        auto snake_shape=[&](const std::string &name,int64_t channels) {
            vector_shape(name+".alpha",channels);vector_shape(name+".beta",channels);
        };
        convolution_shape("encoder.layers.0",7,2,64);
        const std::array<int,6> outputs={64,128,256,512,1024,2048};int channels=64;
        for(size_t layer=0;layer<kStrides.size();++layer) {
            const auto prefix="encoder.layers."+std::to_string(layer+1)+".layers";
            for(int unit=0;unit<3;++unit) {
                const auto name=prefix+"."+std::to_string(unit)+".layers";
                snake_shape(name+".0",channels);snake_shape(name+".2",channels);
                convolution_shape(name+".1",7,channels,channels);convolution_shape(name+".3",1,channels,channels);
            }
            snake_shape(prefix+".3",channels);convolution_shape(prefix+".4",kStrides[layer]*2,channels,outputs[layer]);channels=outputs[layer];
        }
        snake_shape("encoder.layers.7",2048);convolution_shape("encoder.layers.8",3,2048,128);
        ggml_backend_t backends[]={model.backend(),nullptr};size_t count=1;
        auto *device=ggml_backend_get_device(model.backend());
        if(device && ggml_backend_dev_type(device)!=GGML_BACKEND_DEVICE_TYPE_CPU) {
            cpu=detail::make_backend("cpu",options.threads,false);backends[count++]=cpu;
        }
        scheduler=ggml_backend_sched_new(backends,nullptr,count,kGraphSize,false,true);
        if(!scheduler) {if(cpu){ggml_backend_free(cpu);cpu=nullptr;}throw std::runtime_error("[yue2:vae] cannot create encoder scheduler");}
    }
    ~Impl(){if(scheduler)ggml_backend_sched_free(scheduler);if(cpu)ggml_backend_free(cpu);}
    std::vector<float> encode(const float *pcm,int64_t samples,int channels,const VaeEncodeControl &control) {
        if(!pcm || samples<1920 || samples>24576*kRatio || (channels!=1 && channels!=2))
            throw std::invalid_argument("VAE encoder expects nonempty 48 kHz mono/stereo PCM within the released context");
        for(int64_t i=0;i<samples*channels;++i)if(!std::isfinite(pcm[i]))throw std::invalid_argument("Nonfinite VAE encoder PCM");
        std::lock_guard<std::mutex> lock(mutex);
        const int64_t frames=(samples+64)/kRatio;
        const int64_t step=tiled?core:frames;
        std::vector<float> result(static_cast<size_t>(frames*64));
        uint32_t index=0,total=static_cast<uint32_t>((frames+step-1)/step);
        for(int64_t start=0;start<frames;start+=step) {
            if(control.should_cancel && control.should_cancel())throw std::runtime_error("VAE encoding cancelled");
            const int64_t end=std::min(frames,start+step);
            const int64_t left=std::max<int64_t>(0,start-halo),right=std::min(frames,end+halo);
            // Include actual trailing samples at the recording edge, including
            // the native decoder's -64-sample edge and partial final bins.
            const auto stop=right==frames?samples:right*kRatio;
            auto values=window(pcm+left*kRatio*channels,stop-left*kRatio,channels);
            const auto offset=static_cast<size_t>((start-left)*64),count=static_cast<size_t>((end-start)*64);
            if(offset+count>values.size())throw std::runtime_error("Encoder tile does not cover its core");
            std::copy_n(values.begin()+offset,count,result.begin()+start*64);
            if(control.on_progress)control.on_progress(++index,total);
        }
        return result;
    }
private:
    std::vector<float> window(const float *pcm,int64_t samples,int channels) {
        const size_t bytes=ggml_tensor_overhead()*kGraphSize+ggml_graph_overhead_custom(kGraphSize,false);
        std::vector<uint8_t> storage(bytes);
        std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({bytes,storage.data(),true}),ggml_free);
        if(!ctx)throw std::runtime_error("Cannot allocate encoder graph context");
        auto *input=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,2,samples);ggml_set_input(input);
        auto *x=ggml_cont(ctx.get(),ggml_transpose(ctx.get(),input));
        auto *posterior=encode_graph(ctx.get(),model,x);
        if(posterior->ne[1]!=128)throw std::runtime_error("Unexpected VAE posterior shape");
        auto *mean=ggml_view_2d(ctx.get(),posterior,posterior->ne[0],64,posterior->nb[1],0);
        auto *output=ggml_cont(ctx.get(),ggml_transpose(ctx.get(),mean));
        auto *graph=ggml_new_graph_custom(ctx.get(),kGraphSize,false);ggml_build_forward_expand(graph,output);
        ggml_backend_sched_reset(scheduler);
        if(!ggml_backend_sched_alloc_graph(scheduler,graph))throw std::runtime_error("Cannot allocate VAE encoder graph");
        if(std::getenv("YUE2_DEBUG_TIMING"))std::fprintf(stderr,"[yue2] encoder: %lld samples, %.1f MiB scratch\n",
            static_cast<long long>(samples),ggml_backend_sched_get_buffer_size(scheduler,model.backend())/1048576.0);
        std::vector<float> stereo;
        if(channels==1) {stereo.resize(samples*2);for(int64_t i=0;i<samples;++i)stereo[2*i]=stereo[2*i+1]=pcm[i];pcm=stereo.data();}
        ggml_backend_tensor_set(input,pcm,0,static_cast<size_t>(samples*2)*sizeof(float));
        const auto status=ggml_backend_sched_graph_compute(scheduler,graph);
        if(status!=GGML_STATUS_SUCCESS)throw std::runtime_error("VAE encoder graph failed: "+std::string(ggml_status_to_string(status)));
        if(output->ne[0]!=64 || output->ne[1]!=(samples+64)/kRatio)throw std::runtime_error("Unexpected VAE encoded length");
        std::vector<float> result(static_cast<size_t>(ggml_nelements(output)));
        ggml_backend_tensor_get(output,result.data(),0,result.size()*sizeof(float));
        for(float x:result)if(!std::isfinite(x))throw std::runtime_error("Nonfinite VAE posterior mean");
        return result;
    }
    detail::GgufModel model;
    ggml_backend_t cpu=nullptr;
    ggml_backend_sched_t scheduler=nullptr;
    std::mutex mutex;
    bool tiled;
    int64_t core,halo;
};
VaeEncoder::VaeEncoder(const std::string &path,const VaeRuntimeOptions &options):impl_(std::make_unique<Impl>(path,options)){}
VaeEncoder::~VaeEncoder()=default;
VaeEncoder::VaeEncoder(VaeEncoder &&) noexcept=default;
VaeEncoder &VaeEncoder::operator=(VaeEncoder &&) noexcept=default;
std::vector<float> VaeEncoder::encode(const float *pcm,int64_t samples,int channels,const VaeEncodeControl &control) {
    return impl_->encode(pcm,samples,channels,control);
}
} // namespace yue2
