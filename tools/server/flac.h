#pragma once

#include <cstdint>
#include <vector>

namespace yue2::server {

// Encodes interleaved 16-bit PCM as a FLAC stream: a STREAMINFO block, then
// 4096-sample frames with the best of four stereo decorrelations, fixed
// predictors of order 0 to 4, and partitioned Rice residuals. It is the part of
// FLAC that pays on music; LPC would add a few percent for a lot more code.
//
// Lossless: any FLAC decoder returns exactly `interleaved`. The STREAMINFO MD5
// is left zero, which the format defines as "not computed".
std::vector<std::uint8_t> encode_flac_pcm16(
    const std::vector<std::int16_t> & interleaved,
    std::uint32_t sample_rate,
    std::uint32_t channels);

} // namespace yue2::server
