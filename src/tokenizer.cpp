#include "yue2/tokenizer.h"

#include "yue2/generation.h"
#include "yue2_unicode_tables.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace yue2 {
namespace {

namespace unicode = tokenizer_detail;

constexpr std::int32_t kTextVocabularySize = 151851;
constexpr std::uint32_t replacement_character = 0xFFFD;

bool in_ranges(
    std::uint32_t codepoint,
    const unicode::CodepointRange * begin,
    const unicode::CodepointRange * end) {
    const auto found = std::upper_bound(
        begin,
        end,
        codepoint,
        [](std::uint32_t value, const unicode::CodepointRange & range) {
            return value < range.first;
        });
    if (found == begin) return false;
    const auto & range = *(found - 1);
    return codepoint <= range.last;
}

bool is_letter(std::uint32_t codepoint) {
    return in_ranges(
        codepoint,
        std::begin(unicode::letter_ranges),
        std::end(unicode::letter_ranges));
}

bool is_number(std::uint32_t codepoint) {
    return in_ranges(
        codepoint,
        std::begin(unicode::number_ranges),
        std::end(unicode::number_ranges));
}

bool is_whitespace(std::uint32_t codepoint) {
    return (codepoint >= 0x09 && codepoint <= 0x0D) ||
        codepoint == 0x20 || codepoint == 0x85 || codepoint == 0xA0 ||
        codepoint == 0x1680 || (codepoint >= 0x2000 && codepoint <= 0x200A) ||
        codepoint == 0x2028 || codepoint == 0x2029 || codepoint == 0x202F ||
        codepoint == 0x205F || codepoint == 0x3000;
}

bool is_newline(std::uint32_t codepoint) {
    return codepoint == '\r' || codepoint == '\n';
}

std::uint8_t combining_class(std::uint32_t codepoint) {
    const auto begin = std::begin(unicode::combining_ranges);
    const auto end = std::end(unicode::combining_ranges);
    const auto found = std::upper_bound(
        begin,
        end,
        codepoint,
        [](std::uint32_t value, const unicode::CombiningRange & range) {
            return value < range.first;
        });
    if (found == begin) return 0;
    const auto & range = *(found - 1);
    return codepoint <= range.last ? range.value : 0;
}

void append_utf8(std::string & output, std::uint32_t codepoint) {
    if (codepoint <= 0x7F) {
        output.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7FF) {
        output.push_back(static_cast<char>(0xC0 | (codepoint >> 6)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else if (codepoint <= 0xFFFF) {
        output.push_back(static_cast<char>(0xE0 | (codepoint >> 12)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else {
        output.push_back(static_cast<char>(0xF0 | (codepoint >> 18)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    }
}

std::vector<std::uint32_t> decode_utf8(const std::string & input) {
    std::vector<std::uint32_t> output;
    output.reserve(input.size());
    for (std::size_t offset = 0; offset < input.size();) {
        const auto first = static_cast<unsigned char>(input[offset]);
        std::uint32_t codepoint = 0;
        std::size_t length = 0;
        std::uint32_t minimum = 0;
        if (first <= 0x7F) {
            codepoint = first;
            length = 1;
        } else if (first >= 0xC2 && first <= 0xDF) {
            codepoint = first & 0x1F;
            length = 2;
            minimum = 0x80;
        } else if (first >= 0xE0 && first <= 0xEF) {
            codepoint = first & 0x0F;
            length = 3;
            minimum = 0x800;
        } else if (first >= 0xF0 && first <= 0xF4) {
            codepoint = first & 0x07;
            length = 4;
            minimum = 0x10000;
        } else {
            throw std::invalid_argument("YuE2 tokenizer input is not valid UTF-8");
        }
        if (offset + length > input.size()) {
            throw std::invalid_argument("YuE2 tokenizer input has truncated UTF-8");
        }
        for (std::size_t index = 1; index < length; ++index) {
            const auto byte = static_cast<unsigned char>(input[offset + index]);
            if ((byte & 0xC0) != 0x80) {
                throw std::invalid_argument("YuE2 tokenizer input is not valid UTF-8");
            }
            codepoint = (codepoint << 6) | (byte & 0x3F);
        }
        if ((length > 1 && codepoint < minimum) || codepoint > 0x10FFFF ||
            (codepoint >= 0xD800 && codepoint <= 0xDFFF)) {
            throw std::invalid_argument("YuE2 tokenizer input contains an invalid codepoint");
        }
        output.push_back(codepoint);
        offset += length;
    }
    return output;
}

const unicode::Decomposition * find_decomposition(std::uint32_t codepoint) {
    const auto begin = std::begin(unicode::decompositions);
    const auto end = std::end(unicode::decompositions);
    const auto found = std::lower_bound(
        begin,
        end,
        codepoint,
        [](const unicode::Decomposition & entry, std::uint32_t value) {
            return entry.codepoint < value;
        });
    return found != end && found->codepoint == codepoint ? found : nullptr;
}

void canonical_decompose(std::uint32_t codepoint, std::vector<std::uint32_t> & output) {
    constexpr std::uint32_t s_base = 0xAC00;
    constexpr std::uint32_t l_base = 0x1100;
    constexpr std::uint32_t v_base = 0x1161;
    constexpr std::uint32_t t_base = 0x11A7;
    constexpr std::uint32_t l_count = 19;
    constexpr std::uint32_t v_count = 21;
    constexpr std::uint32_t t_count = 28;
    constexpr std::uint32_t n_count = v_count * t_count;
    constexpr std::uint32_t s_count = l_count * n_count;
    if (codepoint >= s_base && codepoint < s_base + s_count) {
        const auto index = codepoint - s_base;
        output.push_back(l_base + index / n_count);
        output.push_back(v_base + (index % n_count) / t_count);
        if (index % t_count != 0) output.push_back(t_base + index % t_count);
        return;
    }
    const auto * decomposition = find_decomposition(codepoint);
    if (!decomposition) {
        output.push_back(codepoint);
        return;
    }
    for (std::size_t index = 0; index < decomposition->length; ++index) {
        canonical_decompose(
            unicode::decomposition_data[decomposition->offset + index], output);
    }
}

std::uint32_t compose_pair(std::uint32_t first, std::uint32_t second) {
    constexpr std::uint32_t s_base = 0xAC00;
    constexpr std::uint32_t l_base = 0x1100;
    constexpr std::uint32_t v_base = 0x1161;
    constexpr std::uint32_t t_base = 0x11A7;
    constexpr std::uint32_t l_count = 19;
    constexpr std::uint32_t v_count = 21;
    constexpr std::uint32_t t_count = 28;
    constexpr std::uint32_t n_count = v_count * t_count;
    constexpr std::uint32_t s_count = l_count * n_count;
    if (first >= l_base && first < l_base + l_count &&
        second >= v_base && second < v_base + v_count) {
        return s_base + ((first - l_base) * v_count + second - v_base) * t_count;
    }
    if (first >= s_base && first < s_base + s_count &&
        (first - s_base) % t_count == 0 &&
        second > t_base && second < t_base + t_count) {
        return first + second - t_base;
    }
    const auto begin = std::begin(unicode::compositions);
    const auto end = std::end(unicode::compositions);
    const auto found = std::lower_bound(
        begin,
        end,
        std::pair<std::uint32_t, std::uint32_t>{first, second},
        [](const unicode::Composition & entry, const auto & value) {
            return entry.first < value.first ||
                (entry.first == value.first && entry.second < value.second);
        });
    return found != end && found->first == first && found->second == second
        ? found->result
        : 0;
}

std::vector<std::uint32_t> normalize_nfc(const std::vector<std::uint32_t> & input) {
    std::vector<std::uint32_t> decomposed;
    decomposed.reserve(input.size());
    for (const auto codepoint : input) canonical_decompose(codepoint, decomposed);
    for (std::size_t index = 1; index < decomposed.size(); ++index) {
        const auto current_class = combining_class(decomposed[index]);
        if (current_class == 0) continue;
        std::size_t position = index;
        while (position > 0) {
            const auto previous_class = combining_class(decomposed[position - 1]);
            if (previous_class == 0 || previous_class <= current_class) break;
            std::swap(decomposed[position], decomposed[position - 1]);
            --position;
        }
    }
    if (decomposed.empty()) return {};

    std::vector<std::uint32_t> composed;
    composed.reserve(decomposed.size());
    composed.push_back(decomposed.front());
    std::size_t starter_position = 0;
    std::uint32_t starter = decomposed.front();
    std::uint8_t last_class = 0;
    for (std::size_t index = 1; index < decomposed.size(); ++index) {
        const auto codepoint = decomposed[index];
        const auto current_class = combining_class(codepoint);
        const auto replacement = compose_pair(starter, codepoint);
        if (replacement != 0 && (last_class == 0 || last_class < current_class)) {
            composed[starter_position] = replacement;
            starter = replacement;
            continue;
        }
        if (current_class == 0) {
            starter_position = composed.size();
            starter = codepoint;
        }
        composed.push_back(codepoint);
        last_class = current_class;
    }
    return composed;
}

std::string encode_codepoints(
    const std::vector<std::uint32_t> & values,
    std::size_t begin,
    std::size_t end) {
    std::string output;
    for (std::size_t index = begin; index < end; ++index) append_utf8(output, values[index]);
    return output;
}

std::size_t contraction_length(
    const std::vector<std::uint32_t> & text,
    std::size_t offset) {
    if (text[offset] != '\'' || offset + 1 >= text.size()) return 0;
    static const std::array<const char *, 7> endings = {"s", "t", "re", "ve", "m", "ll", "d"};
    for (const char * ending : endings) {
        std::size_t length = 0;
        while (ending[length] != '\0') ++length;
        if (offset + 1 + length > text.size()) continue;
        bool matches = true;
        for (std::size_t index = 0; index < length; ++index) {
            auto codepoint = text[offset + 1 + index];
            if (codepoint >= 'A' && codepoint <= 'Z') codepoint += 'a' - 'A';
            if (codepoint != static_cast<unsigned char>(ending[index])) {
                matches = false;
                break;
            }
        }
        if (matches) return length + 1;
    }
    return 0;
}

std::vector<std::string> qwen2_pieces(const std::vector<std::uint32_t> & text) {
    std::vector<std::string> pieces;
    for (std::size_t offset = 0; offset < text.size();) {
        const auto contraction = contraction_length(text, offset);
        if (contraction != 0) {
            pieces.push_back(encode_codepoints(text, offset, offset + contraction));
            offset += contraction;
            continue;
        }

        std::size_t end = offset;
        if (is_letter(text[offset])) {
            end = offset + 1;
        } else if (!is_newline(text[offset]) && !is_letter(text[offset]) &&
                   !is_number(text[offset]) && offset + 1 < text.size() &&
                   is_letter(text[offset + 1])) {
            end = offset + 2;
        }
        if (end != offset) {
            while (end < text.size() && is_letter(text[end])) ++end;
            pieces.push_back(encode_codepoints(text, offset, end));
            offset = end;
            continue;
        }

        if (is_number(text[offset])) {
            pieces.push_back(encode_codepoints(text, offset, offset + 1));
            ++offset;
            continue;
        }

        end = offset;
        if (text[end] == 0x20 && end + 1 < text.size() &&
            !is_whitespace(text[end + 1]) && !is_letter(text[end + 1]) &&
            !is_number(text[end + 1])) {
            ++end;
        }
        const auto symbols_begin = end;
        while (end < text.size() && !is_whitespace(text[end]) &&
               !is_letter(text[end]) && !is_number(text[end])) {
            ++end;
        }
        if (end > symbols_begin) {
            while (end < text.size() && is_newline(text[end])) ++end;
            pieces.push_back(encode_codepoints(text, offset, end));
            offset = end;
            continue;
        }

        if (is_whitespace(text[offset])) {
            end = offset;
            std::size_t last_newline_end = offset;
            while (end < text.size() && is_whitespace(text[end])) {
                ++end;
                if (is_newline(text[end - 1])) last_newline_end = end;
            }
            if (last_newline_end > offset) {
                pieces.push_back(encode_codepoints(text, offset, last_newline_end));
                offset = last_newline_end;
                continue;
            }
            if (end == text.size()) {
                pieces.push_back(encode_codepoints(text, offset, end));
                offset = end;
                continue;
            }
            if (end - offset > 1) {
                pieces.push_back(encode_codepoints(text, offset, end - 1));
                offset = end - 1;
                continue;
            }
            pieces.push_back(encode_codepoints(text, offset, end));
            offset = end;
            continue;
        }
        throw std::logic_error("YuE2 Qwen2 pre-tokenizer did not consume input");
    }
    return pieces;
}

std::string decode_base64(const std::string & input) {
    static const std::array<std::int8_t, 256> table = [] {
        std::array<std::int8_t, 256> values{};
        values.fill(-1);
        for (int index = 0; index < 26; ++index) {
            values[static_cast<std::size_t>('A' + index)] = static_cast<std::int8_t>(index);
            values[static_cast<std::size_t>('a' + index)] = static_cast<std::int8_t>(26 + index);
        }
        for (int index = 0; index < 10; ++index) {
            values[static_cast<std::size_t>('0' + index)] = static_cast<std::int8_t>(52 + index);
        }
        values[static_cast<std::size_t>('+')] = 62;
        values[static_cast<std::size_t>('/')] = 63;
        return values;
    }();
    if (input.empty() || input.size() % 4 != 0) {
        throw std::runtime_error("YuE2 tiktoken vocabulary contains invalid base64");
    }
    std::string output;
    int accumulator = 0;
    int bits = 0;
    bool padding = false;
    for (const auto raw : input) {
        const auto character = static_cast<unsigned char>(raw);
        if (character == '=') {
            padding = true;
            continue;
        }
        if (padding || table[character] < 0) {
            throw std::runtime_error("YuE2 tiktoken vocabulary contains invalid base64");
        }
        accumulator = (accumulator << 6) | table[character];
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            output.push_back(static_cast<char>((accumulator >> bits) & 0xFF));
        }
    }
    return output;
}

std::string valid_utf8_or_replacement(const std::string & bytes) {
    std::string output;
    for (std::size_t offset = 0; offset < bytes.size();) {
        const auto first = static_cast<unsigned char>(bytes[offset]);
        std::size_t length = 0;
        std::uint32_t minimum = 0;
        std::uint32_t codepoint = 0;
        if (first <= 0x7F) {
            length = 1;
            codepoint = first;
        } else if (first >= 0xC2 && first <= 0xDF) {
            length = 2; minimum = 0x80; codepoint = first & 0x1F;
        } else if (first >= 0xE0 && first <= 0xEF) {
            length = 3; minimum = 0x800; codepoint = first & 0x0F;
        } else if (first >= 0xF0 && first <= 0xF4) {
            length = 4; minimum = 0x10000; codepoint = first & 0x07;
        }
        bool valid = length != 0 && offset + length <= bytes.size();
        for (std::size_t index = 1; valid && index < length; ++index) {
            const auto byte = static_cast<unsigned char>(bytes[offset + index]);
            valid = (byte & 0xC0) == 0x80;
            codepoint = (codepoint << 6) | (byte & 0x3F);
        }
        valid = valid && (length == 1 || codepoint >= minimum) && codepoint <= 0x10FFFF &&
            !(codepoint >= 0xD800 && codepoint <= 0xDFFF);
        if (!valid) {
            append_utf8(output, replacement_character);
            ++offset;
        } else {
            output.append(bytes, offset, length);
            offset += length;
        }
    }
    return output;
}

} // namespace

class TextTokenizer::Impl {
public:
    explicit Impl(const std::string & path) {
        std::ifstream input(path, std::ios::binary);
        if (!input) throw std::runtime_error("failed to open YuE2 tiktoken vocabulary: " + path);
        decoded_tokens.resize(kTextVocabularySize);
        std::vector<bool> ranks_seen(kEodToken, false);
        std::string line;
        while (std::getline(input, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            std::istringstream fields(line);
            std::string encoded;
            std::int64_t rank = -1;
            std::string extra;
            if (!(fields >> encoded >> rank) || (fields >> extra) ||
                rank < 0 || rank >= kEodToken || ranks_seen[static_cast<std::size_t>(rank)]) {
                throw std::runtime_error("YuE2 tiktoken vocabulary has an invalid line");
            }
            auto token = decode_base64(encoded);
            if (!ranks.emplace(token, static_cast<std::int32_t>(rank)).second) {
                throw std::runtime_error("YuE2 tiktoken vocabulary contains duplicate bytes");
            }
            decoded_tokens[static_cast<std::size_t>(rank)] = std::move(token);
            ranks_seen[static_cast<std::size_t>(rank)] = true;
        }
        if (ranks.size() != static_cast<std::size_t>(kEodToken) ||
            std::find(ranks_seen.begin(), ranks_seen.end(), false) != ranks_seen.end()) {
            throw std::runtime_error("YuE2 tiktoken vocabulary must contain 151643 ordinary ranks");
        }
        const std::array<const char *, 8> base = {
            "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<R>",
            "<S>", "<X>", "<mask>", "<sep>"};
        for (std::size_t index = 0; index < base.size(); ++index) {
            decoded_tokens[static_cast<std::size_t>(kEodToken) + index] = base[index];
        }
        for (int index = 0; index < 196; ++index) {
            decoded_tokens[151651 + static_cast<std::size_t>(index)] =
                "<extra_" + std::to_string(index) + ">";
        }
        decoded_tokens[kAbcStartToken] = "<abc>";
        decoded_tokens[kAbcEndToken] = "</abc>";
        decoded_tokens[151849] = "<extra_198>";
        decoded_tokens[151850] = "<extra_199>";
    }

    std::vector<std::int32_t> encode(const std::string & text) const {
        const auto normalized = normalize_nfc(decode_utf8(text));
        std::vector<std::int32_t> output;
        for (const auto & piece : qwen2_pieces(normalized)) encode_piece(piece, output);
        return output;
    }

    std::string decode(const std::vector<std::int32_t> & ids) const {
        std::string bytes;
        for (const auto id : ids) {
            if (id >= 0 && id < kTextVocabularySize) {
                bytes += decoded_tokens[static_cast<std::size_t>(id)];
            }
        }
        return valid_utf8_or_replacement(bytes);
    }

private:
    struct Symbol {
        std::size_t start = 0;
        std::size_t length = 0;
        int previous = -1;
        int next = -1;
    };

    struct Candidate {
        std::int32_t rank = 0;
        int left = -1;
        int right = -1;
        std::string bytes;
        bool operator<(const Candidate & other) const {
            return rank > other.rank || (rank == other.rank && left > other.left);
        }
    };

    void encode_piece(const std::string & piece, std::vector<std::int32_t> & output) const {
        const auto whole = ranks.find(piece);
        if (whole != ranks.end()) {
            output.push_back(whole->second);
            return;
        }
        std::vector<Symbol> symbols(piece.size());
        for (std::size_t index = 0; index < symbols.size(); ++index) {
            symbols[index] = {
                index,
                1,
                index == 0 ? -1 : static_cast<int>(index - 1),
                index + 1 == symbols.size() ? -1 : static_cast<int>(index + 1)};
        }
        std::priority_queue<Candidate> queue;
        auto add_candidate = [&](int left, int right) {
            if (left < 0 || right < 0 || symbols[static_cast<std::size_t>(left)].length == 0 ||
                symbols[static_cast<std::size_t>(right)].length == 0 ||
                symbols[static_cast<std::size_t>(left)].next != right) return;
            const auto & lhs = symbols[static_cast<std::size_t>(left)];
            const auto & rhs = symbols[static_cast<std::size_t>(right)];
            auto bytes = piece.substr(lhs.start, lhs.length + rhs.length);
            const auto found = ranks.find(bytes);
            if (found != ranks.end()) queue.push({found->second, left, right, std::move(bytes)});
        };
        for (std::size_t index = 1; index < symbols.size(); ++index) {
            add_candidate(static_cast<int>(index - 1), static_cast<int>(index));
        }
        while (!queue.empty()) {
            auto candidate = queue.top();
            queue.pop();
            auto & left = symbols[static_cast<std::size_t>(candidate.left)];
            auto & right = symbols[static_cast<std::size_t>(candidate.right)];
            if (left.length == 0 || right.length == 0 || left.next != candidate.right ||
                piece.compare(left.start, left.length + right.length, candidate.bytes) != 0) {
                continue;
            }
            left.length += right.length;
            right.length = 0;
            left.next = right.next;
            if (right.next >= 0) symbols[static_cast<std::size_t>(right.next)].previous = candidate.left;
            add_candidate(left.previous, candidate.left);
            add_candidate(candidate.left, left.next);
        }
        for (int index = symbols.empty() ? -1 : 0; index >= 0;
             index = symbols[static_cast<std::size_t>(index)].next) {
            const auto & symbol = symbols[static_cast<std::size_t>(index)];
            const auto bytes = piece.substr(symbol.start, symbol.length);
            const auto found = ranks.find(bytes);
            if (found == ranks.end()) {
                throw std::logic_error("YuE2 BPE produced bytes missing from the vocabulary");
            }
            output.push_back(found->second);
        }
    }

    std::unordered_map<std::string, std::int32_t> ranks;
    std::vector<std::string> decoded_tokens;
};

TextTokenizer::TextTokenizer(const std::string & qwen_tiktoken_path)
    : impl_(std::make_unique<Impl>(qwen_tiktoken_path)) {}

TextTokenizer::~TextTokenizer() = default;
TextTokenizer::TextTokenizer(TextTokenizer &&) noexcept = default;
TextTokenizer & TextTokenizer::operator=(TextTokenizer &&) noexcept = default;

std::vector<std::int32_t> TextTokenizer::encode(const std::string & utf8_text) const {
    return impl_->encode(utf8_text);
}

std::string TextTokenizer::decode(const std::vector<std::int32_t> & token_ids) const {
    return impl_->decode(token_ids);
}

std::size_t TextTokenizer::ordinary_vocabulary_size() const noexcept {
    return kEodToken;
}

std::size_t TextTokenizer::text_vocabulary_size() const noexcept {
    return kTextVocabularySize;
}

} // namespace yue2
