// Internals of NAR conditioning preparation, shared with their unit tests.
#pragma once

#include "server/json.h"

#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace yue2::training::conditioning_detail {

// prepare_nar_dataset.verify_cache(): a native cache's provenance, its exact
// input payloads, its conditioning shape and that every K/V value is finite.
void verify_cache(const std::filesystem::path & cache, const std::string & key, const server::json::Value & row,
                  const server::json::Value & target, const std::filesystem::path & base, const std::filesystem::path & common,
                  const std::filesystem::path & adapter, const std::string & kv_type);

// Each prepared example's VAE target: the same recording, start and length,
// in the same split, with no target left over. Pairs keep the examples' order.
std::vector<std::pair<server::json::Value, server::json::Value>> match_targets(const std::vector<server::json::Value> & rows,
                                                                               const std::vector<server::json::Value> & targets);

} // namespace yue2::training::conditioning_detail
