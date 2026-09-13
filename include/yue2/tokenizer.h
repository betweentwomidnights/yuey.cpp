#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace yue2 {

// Checkpoint-native Qwen/tiktoken text and ABC tokenizer used by YuE2. Encoding
// performs NFC normalization and ordinary-token BPE; strings that look like
// special tokens are intentionally not parsed as control tokens.
class TextTokenizer {
public:
    explicit TextTokenizer(const std::string & qwen_tiktoken_path);
    ~TextTokenizer();
    TextTokenizer(TextTokenizer &&) noexcept;
    TextTokenizer & operator=(TextTokenizer &&) noexcept;
    TextTokenizer(const TextTokenizer &) = delete;
    TextTokenizer & operator=(const TextTokenizer &) = delete;

    std::vector<std::int32_t> encode(const std::string & utf8_text) const;
    std::string decode(const std::vector<std::int32_t> & token_ids) const;
    std::size_t ordinary_vocabulary_size() const noexcept;
    std::size_t text_vocabulary_size() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace yue2
