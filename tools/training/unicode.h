// The few Unicode rules the training tools share with Python's str and re.
#pragma once

#include "yue2_unicode_tables.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace yue2::training::unicode {

inline std::vector<std::uint32_t> decode(const std::string & text) {
    std::vector<std::uint32_t> output;
    for (std::size_t i = 0; i < text.size();) {
        const auto byte = static_cast<unsigned char>(text[i]);
        const std::size_t length = byte < 0x80 ? 1 : byte >= 0xf0 ? 4 : byte >= 0xe0 ? 3 : byte >= 0xc2 ? 2 : 0;
        if (!length || i + length > text.size()) throw std::invalid_argument("Text is not UTF-8");
        std::uint32_t code = length == 1 ? byte : byte & (0x7f >> length);
        for (std::size_t k = 1; k < length; ++k) {
            const auto next = static_cast<unsigned char>(text[i + k]);
            if ((next & 0xc0) != 0x80) throw std::invalid_argument("Text is not UTF-8");
            code = (code << 6) | (next & 0x3f);
        }
        const std::uint32_t smallest[] = {0, 0, 0x80, 0x800, 0x10000};
        if (code < smallest[length] || code > 0x10ffff || (code >= 0xd800 && code <= 0xdfff)) {
            throw std::invalid_argument("Text is not UTF-8");
        }
        output.push_back(code);
        i += length;
    }
    return output;
}

inline std::string encode(const std::vector<std::uint32_t> & codes, std::size_t begin = 0, std::size_t end = std::string::npos) {
    std::string output;
    end = std::min(end, codes.size());
    for (std::size_t i = begin; i < end; ++i) {
        const auto c = codes[i];
        if (c < 0x80) output.push_back(static_cast<char>(c));
        else if (c < 0x800) {
            output.push_back(static_cast<char>(0xc0 | (c >> 6)));
            output.push_back(static_cast<char>(0x80 | (c & 0x3f)));
        } else if (c < 0x10000) {
            output.push_back(static_cast<char>(0xe0 | (c >> 12)));
            output.push_back(static_cast<char>(0x80 | ((c >> 6) & 0x3f)));
            output.push_back(static_cast<char>(0x80 | (c & 0x3f)));
        } else {
            output.push_back(static_cast<char>(0xf0 | (c >> 18)));
            output.push_back(static_cast<char>(0x80 | ((c >> 12) & 0x3f)));
            output.push_back(static_cast<char>(0x80 | ((c >> 6) & 0x3f)));
            output.push_back(static_cast<char>(0x80 | (c & 0x3f)));
        }
    }
    return output;
}

// str.isspace() and re's \s for str patterns.
inline bool space(std::uint32_t c) {
    return (c >= 0x09 && c <= 0x0d) || (c >= 0x1c && c <= 0x20) || c == 0x85 || c == 0xa0 || c == 0x1680 ||
           (c >= 0x2000 && c <= 0x200a) || c == 0x2028 || c == 0x2029 || c == 0x202f || c == 0x205f || c == 0x3000;
}

inline bool in(const tokenizer_detail::CodepointRange * begin, const tokenizer_detail::CodepointRange * end, std::uint32_t c) {
    const auto found = std::upper_bound(begin, end, c, [](std::uint32_t value, const auto & range) { return value < range.first; });
    return found != begin && c <= std::prev(found)->last;
}

// re's \w for str patterns: letters, numbers and underscore.
inline bool word(std::uint32_t c) {
    if (c < 0x80) return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_';
    return in(std::begin(tokenizer_detail::letter_ranges), std::end(tokenizer_detail::letter_ranges), c) ||
           in(std::begin(tokenizer_detail::number_ranges), std::end(tokenizer_detail::number_ranges), c);
}

} // namespace yue2::training::unicode
