#include "yue2/generation.h"
#include "yue2/transcription.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

int main(int argc,char **argv) {
    try {
        std::map<std::string,std::string> args;
        bool instrumental=false;
        for(int i=1;i<argc;++i) {
            std::string key=argv[i];
            if(key=="--instrumental")instrumental=true;
            else if(key=="--input"||key=="--output"||key=="--midi-dir") {
                if(++i==argc)throw std::runtime_error("missing value");
                args[key]=argv[i];
            } else throw std::runtime_error("Usage: yue2-score --input SCORE.abc --output NEW.abc [--instrumental --midi-dir DIR]");
        }
        std::ifstream input(args.at("--input"),std::ios::binary);
        if(!input)throw std::runtime_error("cannot read score");
        std::string score{std::istreambuf_iterator<char>(input),{}};
        if(instrumental)score=yue2::make_instrumental_transfer_abc(score);
        const auto midi=yue2::serialize_yue2_abc_midis(score);
        auto write=[](const std::filesystem::path &path,const auto &bytes) {
            if(std::filesystem::exists(path))throw std::runtime_error("output exists: "+path.string());
            if(!path.parent_path().empty())std::filesystem::create_directories(path.parent_path());
            std::ofstream file(path,std::ios::binary);
            if(!file.write(reinterpret_cast<const char *>(bytes.data()),bytes.size()))throw std::runtime_error("cannot write output");
        };
        write(args.at("--output"),score);
        if(args.count("--midi-dir")) {
            std::filesystem::path dir=args.at("--midi-dir");
            write(dir/"score.mid",midi.transcription);
            write(dir/"vocal.mid",midi.vocal);
            write(dir/"instrumental.mid",midi.instrumental);
            if(!midi.chords.empty())write(dir/"chords.mid",midi.chords);
        }
        std::cout<<"validated score ("<<score.size()<<" bytes)\n";
        return 0;
    }catch(const std::exception &error){std::cerr<<"yue2-score: "<<error.what()<<'\n';return 2;}
}
