// The durable two-expert workflow's run settings, dataset inventory and
// resource review. Ported from tools/training_workflow.py, which stays the
// reference until the native workflow replaces it; results must equal it
// (tests/test_native_training_workflow.py).
#pragma once

#include "server/json.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace yue2::training {

namespace json = server::json;

// Validated settings for a run request, with the request's recipe pinned:
// the object training_workflow.py's settings() returns. Throws
// std::invalid_argument with the same messages.
json::Value settings(const json::Value & request);

// Rows of a prepared training dataset (data/dataset.json) that are eligible.
std::vector<json::Value> eligible_rows(const json::Value & dataset);

// Peak AR host RAM and teacher-cache disk, in GiB.
json::Value ar_resource_estimate(const std::vector<json::Value> & eligible, const json::Value & options,
                                 const std::string & device, std::uint64_t model_bytes = 0);
double checkpoint_allowance(const json::Value & options, int experts = 2);
long long conditioning_bytes(const std::vector<json::Value> & eligible, const std::string & storage,
                             const std::string & evaluation_scope);

// Base weights the NAR fitter and the conditioning-only AR model load.
struct ExpertBytes { std::uint64_t nar = 0, ar = 0; };
ExpertBytes expert_bytes(const std::filesystem::path & base);
json::Value storage_options(const std::vector<json::Value> & eligible, const json::Value & options, const ExpertBytes & experts);

// Recordings with TXT captions in a dataset folder, hashed and split into
// train and recording-level validation holdouts.
json::Value inventory(const std::filesystem::path & folder, long long window_seconds, long long seed);

// What preparation records for dataset review, from a prepared run folder
// (request.json, runtime.json, data/dataset.json). Free disk is left out.
json::Value review(const std::filesystem::path & run);

// Supported recipes and storage modes, for downstream trainers. With a
// training runtime, also whether its native tools support each mode.
json::Value capabilities(const std::optional<std::filesystem::path> & runtime);

// SHA-256 of an executable and every shared library beside it, plus the
// precision environment: an unchanged executable can load a new GGML.
json::Value runtime_identity(const std::filesystem::path & executable);

// Files of a folder in Python's sorted(Path.iterdir()) order: case-folded on
// Windows, by code point elsewhere.
std::vector<std::filesystem::path> sorted_entries(const std::filesystem::path & folder);

} // namespace yue2::training
