// yue2-quantize: re-encode a YuE2 generation GGUF to a K-quant, Q8_0 or float
// mix. The tensor plan follows sa3-quantize; see include/yue2/quantize.h.
#include "yue2/quantize.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>

namespace {

void usage() {
    std::fprintf(stderr,
        "usage: yue2-quantize --in MODEL.gguf [--out OUTPUT.gguf] [--mix MIX] [--threads N] [--overwrite]\n"
        "  --mix q4_k_m  V/down/embed/lm_head -> Q6_K, other matrices -> Q4_K (default)\n"
        "        q5_k_m  the same promotion over Q5_K\n"
        "        q8_0    every eligible matrix -> Q8_0\n"
        "        f16     re-encode to F16 (dequantizes a quantized input)\n"
        "        f32     re-encode to F32\n"
        "  --out defaults to the input name with its -<Encoding>.gguf suffix replaced,\n"
        "        e.g. yue2-3.6B-v1.0-BF16.gguf -> yue2-3.6B-v1.0-Q4_K_M.gguf\n"
        "The YuE2 VAE and transcription GGUFs are refused; they stay unquantized.\n");
}

// Swap the Encoding field of a conventionally named GGUF.
std::string derived_output(const std::string & input, const char * encoding) {
    static const char * const kEncodings[] = {"BF16", "F16", "F32", "Q8_0", "Q5_K_M", "Q4_K_M"};
    const std::string extension = ".gguf";
    if (input.size() <= extension.size() ||
        input.compare(input.size() - extension.size(), extension.size(), extension) != 0) {
        return {};
    }
    const std::string stem = input.substr(0, input.size() - extension.size());
    for (const char * candidate : kEncodings) {
        const std::string tail = std::string("-") + candidate;
        if (stem.size() > tail.size() &&
            stem.compare(stem.size() - tail.size(), tail.size(), tail) == 0) {
            return stem.substr(0, stem.size() - tail.size()) + "-" + encoding + extension;
        }
    }
    return {};
}

} // namespace

int main(int argc, char ** argv) {
    std::string input;
    std::string output;
    std::string mix = "q4_k_m";
    yue2::QuantizeOptions options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        const bool has_value = index + 1 < argc;
        if (argument == "--in" && has_value) {
            input = argv[++index];
        } else if (argument == "--out" && has_value) {
            output = argv[++index];
        } else if (argument == "--mix" && has_value) {
            mix = argv[++index];
        } else if (argument == "--threads" && has_value) {
            options.threads = static_cast<unsigned>(std::strtoul(argv[++index], nullptr, 10));
        } else if (argument == "--overwrite") {
            options.overwrite = true;
        } else {
            usage();
            return argument == "-h" || argument == "--help" ? 0 : 2;
        }
    }
    if (input.empty()) {
        usage();
        return 2;
    }

    try {
        options.mix = yue2::parse_quantization_mix(mix);
        const char * encoding = yue2::quantization_mix_name(options.mix);
        if (output.empty()) {
            output = derived_output(input, encoding);
            if (output.empty()) {
                throw std::invalid_argument(
                    "cannot derive an output name from " + input + "; pass --out");
            }
        }
        options.on_tensor = [](const yue2::QuantizedTensor & tensor, std::size_t index, std::size_t count) {
            const bool changed = tensor.source_type != tensor.output_type;
            std::fprintf(stderr, "  [%3zu/%zu] %-52s %s%s%s\n", index, count, tensor.name.c_str(),
                tensor.source_type.c_str(), changed ? " -> " : "",
                changed ? tensor.output_type.c_str() : " (kept)");
        };
        std::fprintf(stderr, "[yue2-quantize] %s -> %s (%s)\n", input.c_str(), output.c_str(), encoding);
        const auto report = yue2::quantize_generation_gguf(input, output, options);
        std::fprintf(stderr,
            "[yue2-quantize] done: %zu converted, %zu kept, %.1f MiB -> %.1f MiB\n",
            report.converted_count, report.kept_count,
            static_cast<double>(report.input_bytes) / (1024.0 * 1024.0),
            static_cast<double>(report.output_bytes) / (1024.0 * 1024.0));
        std::printf("%s\n", output.c_str());
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "yue2-quantize: %s\n", error.what());
        return 1;
    }
}
