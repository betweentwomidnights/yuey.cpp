#include "yue2/generation.h"

#include <exception>
#include <iostream>

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::cerr << "usage: yue2-inspect-generation MODEL.gguf VAE.gguf\n";
        return 2;
    }
    try {
        const auto info = yue2::inspect_generation_package(argv[1], argv[2]);
        std::cout << "YuE2 generation package: valid\n"
                  << "model tensors: " << info.model_tensor_count << " ("
                  << info.model_storage_type << ", " << info.model_file_bytes << " bytes)\n"
                  << "VAE tensors: " << info.vae_tensor_count << " ("
                  << info.vae_storage_type << ", " << info.vae_file_bytes << " bytes)\n"
                  << "context: " << info.model.context_length << " tokens\n"
                  << "audio: " << info.vae.sample_rate << " Hz, "
                  << info.vae.audio_channels << " channels, 1 latent frame / "
                  << info.vae.downsampling_ratio << " samples\n";
        if (!info.model_checkpoint_sha256.empty()) {
            std::cout << "model source SHA-256: " << info.model_checkpoint_sha256 << '\n';
        }
        if (!info.vae_checkpoint_sha256.empty()) {
            std::cout << "VAE source SHA-256: " << info.vae_checkpoint_sha256 << '\n';
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
