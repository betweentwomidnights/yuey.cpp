// Publishing a reviewed run as one style: the selected AR and NAR adapters in
// one GGUF beside a sidecar with its caption pool. Ported from
// the Python publish_training_adapter.py and Workflow.release_caches(); the
// published GGUF and sidecar were checked byte-equal to theirs before they
// were removed (docs/training-history.md).
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
