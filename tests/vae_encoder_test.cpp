// Manual real-weight oracle: MODEL PCM.f32 EXPECTED.f32 CHANNELS DEVICE.
#include "yue2/vae.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <stdexcept>

std::vector<float> read(const char *path) {
    std::ifstream in(path,std::ios::binary|std::ios::ate);
    const auto size=in.tellg();if(!in || size<=0 || size%4)throw std::runtime_error("Invalid F32 fixture");
    std::vector<float> result(static_cast<size_t>(size)/4);in.seekg(0);
    if(!in.read(reinterpret_cast<char *>(result.data()),size))throw std::runtime_error("Short fixture");return result;
}
double compare(const std::vector<float> &actual,const std::vector<float> &expected) {
    if(actual.size()!=expected.size())throw std::runtime_error("Encoded shape differs");
    double error=0,scale=0,maximum=0;
    for(size_t i=0;i<actual.size();++i) {
        if(!std::isfinite(actual[i]))throw std::runtime_error("Nonfinite output");
        const double delta=actual[i]-expected[i];error+=delta*delta;scale+=static_cast<double>(expected[i])*expected[i];maximum=std::max(maximum,std::abs(delta));
    }
    const double relative=std::sqrt(error/std::max(scale,1e-30));
    std::cout<<"frames="<<actual.size()/64<<" relative_l2="<<relative<<" max="<<maximum<<'\n';
    if(relative>1e-4 || maximum>.002)throw std::runtime_error("Encoder parity tolerance exceeded");return relative;
}
int main(int argc,char **argv) {
    if(argc!=6){std::cerr<<"usage: yue2-vae-encoder-test MODEL PCM.f32 EXPECTED.f32 CHANNELS DEVICE\n";return 2;}
    try {
        auto pcm=read(argv[2]);const auto expected=read(argv[3]);const int channels=std::stoi(argv[4]);
        if(channels<1 || channels>2 || pcm.size()%channels)throw std::runtime_error("Invalid PCM channels");
        yue2::VaeRuntimeOptions options;options.device=argv[5];options.threads=4;
        yue2::VaeEncoder encoder(argv[1],options);const auto samples=pcm.size()/channels;
        const auto actual=encoder.encode(pcm.data(),samples,channels);compare(actual,expected);
        if(expected.size()/64<=150) {
            options.tiled=false;yue2::VaeEncoder full(argv[1],options);compare(actual,full.encode(pcm.data(),samples,channels));
        }
        bool rejected=false;
        try {auto invalid=pcm;invalid[0]=std::numeric_limits<float>::quiet_NaN();encoder.encode(invalid.data(),samples,channels);}
        catch(const std::invalid_argument &){rejected=true;}
        if(!rejected)throw std::runtime_error("Nonfinite PCM was accepted");
        yue2::VaeEncodeControl cancelled;cancelled.should_cancel=[](){return true;};rejected=false;
        try {encoder.encode(pcm.data(),samples,channels,cancelled);}catch(const std::runtime_error &){rejected=true;}
        if(!rejected)throw std::runtime_error("Cancellation was ignored");
        if(expected.size()/64<=4) {
            auto a=std::async(std::launch::async,[&](){return encoder.encode(pcm.data(),samples,channels);});
            auto b=std::async(std::launch::async,[&](){return encoder.encode(pcm.data(),samples,channels);});
            if(a.get()!=actual || b.get()!=actual)throw std::runtime_error("Encoder same-instance concurrency differs");
        }
        return 0;
    }catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}
}
