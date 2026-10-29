// Publishing a reviewed run as one style: the selected AR and NAR adapters in
// one GGUF beside a sidecar with its caption pool. Ported from
// tools/publish_training_adapter.py and Workflow.release_caches() in
// training_workflow.py, which stay the reference until the native workflow
// replaces them; the published GGUF and sidecar equal the Python's byte for byte.
#pragma once

#include "server/json.h"

#include <filesystem>

namespace yue2::training {

// publish_training_adapter.publish() for a run folder (its runtime.json,
// request.json and state.json). Returns the sidecar, or the existing one when
// this run already published the same bytes.
server::json::Value publish(const std::filesystem::path & run);

// Workflow.release_caches(): delete what only further training needs, keeping
// the selected adapters, manifests, metrics and logs. Returns bytes released.
long long release_caches(const std::filesystem::path & run);

} // namespace yue2::training
