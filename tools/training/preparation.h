// Dataset preparation for a named run: semantic caches and scored AR
// examples, VAE targets, and frozen-AR conditioning for NAR fitting. Ported
// from tools/prepare_ar_dataset.py, prepare_nar_targets.py and
// prepare_nar_dataset.py, which stay the reference until the native workflow
// replaces them. Every file written equals the Python's byte for byte, except
// logs, timings and the source.wav header (its samples are equal).
//
// Only the product path is ported: inference_urls, reused semantic or
// transcription manifests, generated targets and auxiliary replay datasets
// are research options that stay in the Python and are rejected here.
#pragma once

#include "server/json.h"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace yue2::training {

// The engraving and conditioning rules this port equals: SHA-256 of the
// reference scripts as committed (LF line endings). Preparation identities
// record them where the Python recorded its own source digest.
extern const char * const normalizer_sha256;      // tools/ar_dataset.py
extern const char * const conditioning_sha256;    // tools/prepare_nar_dataset.py
// A run the Python started on a Windows checkout recorded the CRLF digests;
// resuming it keeps whichever of the two equal-content digests it recorded.
extern const char * const normalizer_sha256_crlf;
extern const char * const conditioning_sha256_crlf;
std::string recorded_digest(const std::optional<std::string> & recorded, const char * committed, const char * crlf);

// prepare_ar_dataset.py --phase semantic / scores, from a yue2-ar-training-v1 job.
void prepare_semantic(const std::filesystem::path & config);
void prepare_scores(const std::filesystem::path & config);

struct TargetOptions {
    std::filesystem::path dataset, vae, binary, out;
    std::string device = "cuda";
    long long threads = 4, core_frames = 128, halo_frames = 50;
    std::optional<std::filesystem::path> stop_file;
};
// prepare_nar_targets.py: posterior-mean VAE targets for every eligible example.
void prepare_targets(const TargetOptions & options);

struct ConditioningOptions {
    std::filesystem::path config, targets, ar_adapter, binary, out;
    std::string device = "cuda", kv_type = "f16", storage = "cache";
    long long threads = 6, batch = 0;
    std::optional<std::filesystem::path> stop_file;
};
// prepare_nar_dataset.py: bind targets to one frozen AR adapter.
void prepare_conditioning(const ConditioningOptions & options);

// prepare_nar_dataset.materialize(): regenerate absent rolling caches, each
// byte-identical to the one preparation recorded. Returns how many it made.
long long materialize(const std::filesystem::path & manifest, const std::vector<server::json::Value> & rows,
                      const std::filesystem::path & binary, const std::filesystem::path & base,
                      const std::filesystem::path & common, const std::filesystem::path & adapter,
                      const std::optional<std::filesystem::path> & stop_file);

// Python's == on JSON values: numbers by value (1 == 1.0), objects by members.
bool python_equal(const server::json::Value & a, const server::json::Value & b);

} // namespace yue2::training
