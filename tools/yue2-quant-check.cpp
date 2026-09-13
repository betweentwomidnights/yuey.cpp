// yue2-quant-check: per-tensor dequantization check of a quantized YuE2 GGUF
// against its source, after sa3-quant-check. Every tensor whose storage type
// changed is dequantized on the host and compared by cosine similarity.
#include "yue2/quantize.h"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>

namespace {

void usage() {
    std::fprintf(stderr,
        "usage: yue2-quant-check --ref REFERENCE.gguf --quant CANDIDATE.gguf\n"
        "                        [--threshold 0.99] [--threads N] [--all]\n"
        "Exits 2 when any converted tensor falls below the threshold.\n");
}

} // namespace

int main(int argc, char ** argv) {
    std::string reference;
    std::string candidate;
    double threshold = 0.99;
    unsigned threads = 0;
    bool all = false;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        const bool has_value = index + 1 < argc;
        if (argument == "--ref" && has_value) {
            reference = argv[++index];
        } else if (argument == "--quant" && has_value) {
            candidate = argv[++index];
        } else if (argument == "--threshold" && has_value) {
            threshold = std::strtod(argv[++index], nullptr);
        } else if (argument == "--threads" && has_value) {
            threads = static_cast<unsigned>(std::strtoul(argv[++index], nullptr, 10));
        } else if (argument == "--all") {
            all = true;
        } else {
            usage();
            return argument == "-h" || argument == "--help" ? 0 : 2;
        }
    }
    if (reference.empty() || candidate.empty()) {
        usage();
        return 2;
    }

    try {
        const auto report = yue2::check_quantized_gguf(reference, candidate, threshold, threads);
        for (const auto & result : report.compared) {
            const bool bad = !(result.cosine >= threshold);
            if (!bad && !all) continue;
            std::printf("  %s cos=%.6f  %-52s %s -> %s\n", bad ? "BAD " : "ok  ", result.cosine,
                result.name.c_str(), result.reference_type.c_str(), result.candidate_type.c_str());
        }
        std::printf("compared=%zu below-threshold=%zu min-cosine=%.6f threshold=%.4f\n",
            report.compared.size(), report.below_threshold, report.minimum_cosine, threshold);
        return report.below_threshold ? 2 : 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "yue2-quant-check: %s\n", error.what());
        return 1;
    }
}
