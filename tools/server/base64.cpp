#include "base64.h"

#include <stdexcept>

namespace yue2::server {
namespace {

constexpr char kAlphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

int decode_character(char character) {
    if (character >= 'A' && character <= 'Z') return character - 'A';
    if (character >= 'a' && character <= 'z') return character - 'a' + 26;
    if (character >= '0' && character <= '9') return character - '0' + 52;
    if (character == '+') return 62;
    if (character == '/') return 63;
    return -1;
}

} // namespace

std::string base64_encode(const std::uint8_t * data, std::size_t size) {
    std::string output;
    output.reserve((size + 2) / 3 * 4);
    for (std::size_t offset = 0; offset < size; offset += 3) {
        const std::size_t remaining = size - offset;
        const std::uint32_t value = static_cast<std::uint32_t>(data[offset]) << 16 |
            (remaining > 1 ? static_cast<std::uint32_t>(data[offset + 1]) << 8 : 0U) |
            (remaining > 2 ? static_cast<std::uint32_t>(data[offset + 2]) : 0U);
        output.push_back(kAlphabet[(value >> 18) & 63]);
        output.push_back(kAlphabet[(value >> 12) & 63]);
        output.push_back(remaining > 1 ? kAlphabet[(value >> 6) & 63] : '=');
        output.push_back(remaining > 2 ? kAlphabet[value & 63] : '=');
    }
    return output;
}

std::vector<std::uint8_t> base64_decode(std::string_view text) {
    if (text.rfind("data:", 0) == 0) {
        const auto comma = text.find(',');
        if (comma == std::string_view::npos ||
            text.substr(0, comma).find(";base64") == std::string_view::npos) {
            throw std::invalid_argument("data URI is not base64 encoded");
        }
        text.remove_prefix(comma + 1);
    }
    std::vector<std::uint8_t> output;
    output.reserve(text.size() / 4 * 3);
    std::uint32_t quartet[4] = {};
    int count = 0;
    int padding = 0;
    for (const char character : text) {
        if (character == ' ' || character == '\n' || character == '\r' || character == '\t') continue;
        if (padding > 0 && character != '=') {
            throw std::invalid_argument("base64 data continues after padding");
        }
        if (character == '=') {
            if (count < 2) throw std::invalid_argument("misplaced base64 padding");
            ++padding;
            quartet[count++] = 0;
        } else {
            const int value = decode_character(character);
            if (value < 0) throw std::invalid_argument("invalid base64 character");
            quartet[count++] = static_cast<std::uint32_t>(value);
        }
        if (count == 4) {
            const std::uint32_t value =
                quartet[0] << 18 | quartet[1] << 12 | quartet[2] << 6 | quartet[3];
            output.push_back(static_cast<std::uint8_t>((value >> 16) & 255));
            if (padding < 2) output.push_back(static_cast<std::uint8_t>((value >> 8) & 255));
            if (padding < 1) output.push_back(static_cast<std::uint8_t>(value & 255));
            count = 0;
        }
    }
    if (count != 0) throw std::invalid_argument("truncated base64 data");
    return output;
}

} // namespace yue2::server
