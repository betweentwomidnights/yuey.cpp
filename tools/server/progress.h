#pragma once

#include "yue2/generation_pipeline.h"

#include <algorithm>
#include <cstdint>

namespace yue2::server {

// Stage-weighted work estimate, not elapsed-time prediction. Flow is the
// dominant stage on local CPU/Metal; don't report 80% before it even starts.
// Reserve 100% for successful result finalization in the server.
inline int generation_progress(GenerationStage stage, std::uint32_t current,
                               std::uint32_t total, int base = 0) {
    int from = 0, to = 0;
    switch (stage) {
        case GenerationStage::abc:      from = 0;  to = 2;  break;
        case GenerationStage::semantic: from = 2;  to = 20; break;
        case GenerationStage::flow:     from = 20; to = 97; break;
        case GenerationStage::decode:   from = 97; to = 99; break;
        case GenerationStage::complete: from = 99; to = 99; break;
    }
    base = std::clamp(base, 0, 99);
    // A missing total means unknown work, not a completed stage. Use 64-bit
    // arithmetic to handle maximum token counts without overflow.
    const auto denominator = std::max<std::uint64_t>(1, total);
    const auto done = total ? std::min<std::uint64_t>(current, total) : 0;
    const auto local = std::uint64_t(from) * denominator + (to - from) * done;
    return std::min(99, base + int((100 - base) * local / (100 * denominator)));
}

} // namespace yue2::server
