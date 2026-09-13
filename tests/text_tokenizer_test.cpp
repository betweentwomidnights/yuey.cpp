#include "yue2/generation.h"
#include "yue2/tokenizer.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class FixtureReader {
public:
    explicit FixtureReader(const std::string & path) {
        std::ifstream input(path, std::ios::binary);
        if (!input) throw std::runtime_error("failed to open tokenizer fixture: " + path);
        bytes_ = std::vector<std::uint8_t>(
            std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
    }

    std::uint32_t u32() {
        require(4);
        const auto value = static_cast<std::uint32_t>(bytes_[offset_]) |
            static_cast<std::uint32_t>(bytes_[offset_ + 1]) << 8 |
            static_cast<std::uint32_t>(bytes_[offset_ + 2]) << 16 |
            static_cast<std::uint32_t>(bytes_[offset_ + 3]) << 24;
        offset_ += 4;
        return value;
    }

    std::int32_t i32() { return static_cast<std::int32_t>(u32()); }

    std::string string() {
        const auto length = u32();
        require(length);
        std::string result(
            reinterpret_cast<const char *>(bytes_.data() + offset_), length);
        offset_ += length;
        return result;
    }

    void magic() {
        static constexpr std::uint8_t expected[] = {'Y', '2', 'T', 'K', 1, 0, 0, 0};
        require(sizeof(expected));
        if (std::memcmp(bytes_.data(), expected, sizeof(expected)) != 0) {
            throw std::runtime_error("invalid tokenizer fixture header");
        }
        offset_ += sizeof(expected);
    }

    bool done() const noexcept { return offset_ == bytes_.size(); }

private:
    void require(std::size_t count) const {
        if (count > bytes_.size() - offset_) {
            throw std::runtime_error("truncated tokenizer fixture");
        }
    }

    std::vector<std::uint8_t> bytes_;
    std::size_t offset_ = 0;
};

} // namespace

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::cerr << "usage: yue2-text-tokenizer-test QWEN.TIKTOKEN FIXTURE\n";
        return 2;
    }
    yue2::TextTokenizer tokenizer(argv[1]);
    assert(tokenizer.ordinary_vocabulary_size() == yue2::kEodToken);
    assert(tokenizer.text_vocabulary_size() == 151851);

    FixtureReader fixture(argv[2]);
    fixture.magic();
    const auto cases = fixture.u32();
    for (std::uint32_t case_index = 0; case_index < cases; ++case_index) {
        const auto input = fixture.string();
        const auto count = fixture.u32();
        std::vector<std::int32_t> expected(count);
        for (auto & id : expected) id = fixture.i32();
        const auto expected_decoded = fixture.string();
        const auto actual = tokenizer.encode(input);
        if (actual != expected) {
            std::cerr << "token mismatch in case " << case_index << ": expected "
                      << expected.size() << ", got " << actual.size() << "\n";
            return 1;
        }
        const auto decoded = tokenizer.decode(actual);
        if (decoded != expected_decoded) {
            std::cerr << "decode mismatch in case " << case_index << "\n";
            return 1;
        }
    }
    if (!fixture.done()) throw std::runtime_error("trailing tokenizer fixture data");
    assert(tokenizer.decode({
        yue2::kEodToken, yue2::kAbcStartToken, yue2::kAbcEndToken,
        yue2::kMusicStartToken}) == "<|endoftext|><abc></abc>");
    std::cout << "YuE2 text tokenizer matches " << cases << " official vectors\n";
    return 0;
}
