#include "training/sha256.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <exception>
#include <fstream>
#include <stdexcept>
#include <thread>

namespace yue2::training {
namespace {

constexpr std::uint32_t k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

constexpr std::uint32_t rotate(std::uint32_t value, int bits) { return (value >> bits) | (value << (32 - bits)); }

} // namespace

Sha256::Sha256()
    : state_{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19} {}

void Sha256::block(const std::uint8_t * data) {
    std::uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
        w[i] = (std::uint32_t(data[4 * i]) << 24) | (std::uint32_t(data[4 * i + 1]) << 16) |
               (std::uint32_t(data[4 * i + 2]) << 8) | std::uint32_t(data[4 * i + 3]);
    }
    for (int i = 16; i < 64; ++i) {
        const auto s0 = rotate(w[i - 15], 7) ^ rotate(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const auto s1 = rotate(w[i - 2], 17) ^ rotate(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    auto [a, b, c, d, e, f, g, h] = state_;
    for (int i = 0; i < 64; ++i) {
        const auto t1 = h + (rotate(e, 6) ^ rotate(e, 11) ^ rotate(e, 25)) + ((e & f) ^ (~e & g)) + k[i] + w[i];
        const auto t2 = (rotate(a, 2) ^ rotate(a, 13) ^ rotate(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    const std::uint32_t mixed[8] = {a, b, c, d, e, f, g, h};
    for (int i = 0; i < 8; ++i) state_[i] += mixed[i];
}

void Sha256::update(const void * data, std::size_t size) {
    auto * bytes = static_cast<const std::uint8_t *>(data);
    length_ += size;
    if (buffered_) {
        const auto take = std::min(size, buffer_.size() - buffered_);
        std::memcpy(buffer_.data() + buffered_, bytes, take);
        buffered_ += take; bytes += take; size -= take;
        if (buffered_ < buffer_.size()) return;
        block(buffer_.data());
        buffered_ = 0;
    }
    for (; size >= 64; bytes += 64, size -= 64) block(bytes);
    std::memcpy(buffer_.data(), bytes, size);
    buffered_ = size;
}

std::string Sha256::hex() {
    const auto bits = length_ * 8;
    const std::uint8_t one = 0x80, zero = 0;
    update(&one, 1);
    while (buffered_ != 56) update(&zero, 1);
    std::uint8_t tail[8];
    for (int i = 0; i < 8; ++i) tail[i] = static_cast<std::uint8_t>(bits >> (56 - 8 * i));
    update(tail, 8);
    static constexpr char digits[] = "0123456789abcdef";
    std::string output;
    for (const auto word : state_) {
        for (int shift = 28; shift >= 0; shift -= 4) output.push_back(digits[(word >> shift) & 15]);
    }
    return output;
}

std::string sha256_hex(std::string_view bytes) {
    Sha256 hash;
    hash.update(bytes);
    return hash.hex();
}

std::string sha256_file(const std::filesystem::path & path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("Cannot read " + path.string());
    Sha256 hash;
    std::vector<char> block(1 << 20);
    while (stream) {
        stream.read(block.data(), static_cast<std::streamsize>(block.size()));
        hash.update(block.data(), static_cast<std::size_t>(stream.gcount()));
    }
    if (!stream.eof()) throw std::runtime_error("Cannot read " + path.string());
    return hash.hex();
}

std::vector<std::string> sha256_files(const std::vector<std::filesystem::path> & paths) {
    std::vector<std::string> output(paths.size());
    std::vector<std::exception_ptr> errors(paths.size());
    std::atomic<std::size_t> next{0};
    const auto workers = std::max<std::size_t>(1, std::min<std::size_t>({8, std::thread::hardware_concurrency(), paths.size()}));
    std::vector<std::thread> threads;
    for (std::size_t t = 0; t < workers; ++t) {
        threads.emplace_back([&]() {
            for (std::size_t i; (i = next++) < paths.size();) {
                try { output[i] = sha256_file(paths[i]); } catch (...) { errors[i] = std::current_exception(); }
            }
        });
    }
    for (auto & thread : threads) thread.join();
    for (const auto & error : errors) if (error) std::rethrow_exception(error);
    return output;
}

} // namespace yue2::training
