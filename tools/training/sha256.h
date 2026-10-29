// SHA-256 (FIPS 180-4) for training provenance: every run identity, cache and
// published adapter is keyed by the hex digest of the bytes it was built from.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace yue2::training {

class Sha256 {
public:
    Sha256();
    void update(const void * data, std::size_t size);
    void update(std::string_view text) { update(text.data(), text.size()); }
    // Lowercase hex, like hashlib's hexdigest(). The object is spent after.
    std::string hex();

private:
    void block(const std::uint8_t * data);
    std::array<std::uint32_t, 8> state_;
    std::array<std::uint8_t, 64> buffer_{};
    std::size_t buffered_ = 0;
    std::uint64_t length_ = 0;
};

std::string sha256_hex(std::string_view bytes);
std::string sha256_file(const std::filesystem::path & path);
// Digests of several files, in input order, hashed on up to eight threads.
std::vector<std::string> sha256_files(const std::vector<std::filesystem::path> & paths);

} // namespace yue2::training
