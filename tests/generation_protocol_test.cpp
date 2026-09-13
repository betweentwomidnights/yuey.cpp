#include "yue2/generation.h"
#include "yue2/tokenizer.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class Reader {
public:
    explicit Reader(const std::string & path) {
        std::ifstream input(path, std::ios::binary);
        if (!input) throw std::runtime_error("failed to open protocol fixture: " + path);
        bytes_ = {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    }
    void magic() {
        static constexpr std::uint8_t expected[] = {'Y', '2', 'P', 'R', 1, 0, 0, 0};
        take(sizeof(expected));
        if (std::memcmp(bytes_.data(), expected, sizeof(expected)) != 0) {
            throw std::runtime_error("invalid protocol fixture header");
        }
        offset_ += sizeof(expected);
    }
    std::uint32_t u32() {
        take(4);
        const auto value = static_cast<std::uint32_t>(bytes_[offset_]) |
            static_cast<std::uint32_t>(bytes_[offset_ + 1]) << 8 |
            static_cast<std::uint32_t>(bytes_[offset_ + 2]) << 16 |
            static_cast<std::uint32_t>(bytes_[offset_ + 3]) << 24;
        offset_ += 4;
        return value;
    }
    std::int32_t i32() { return static_cast<std::int32_t>(u32()); }
    std::string string() {
        const auto size = u32();
        take(size);
        std::string output(reinterpret_cast<const char *>(bytes_.data() + offset_), size);
        offset_ += size;
        return output;
    }
    std::vector<std::int32_t> ids() {
        std::vector<std::int32_t> output(u32());
        for (auto & id : output) id = i32();
        return output;
    }
    bool done() const noexcept { return offset_ == bytes_.size(); }
private:
    void take(std::size_t size) const {
        if (size > bytes_.size() - offset_) throw std::runtime_error("truncated protocol fixture");
    }
    std::vector<std::uint8_t> bytes_;
    std::size_t offset_ = 0;
};

yue2::SymbolicMode mode(std::uint32_t value) {
    if (value == 0) return yue2::SymbolicMode::off;
    if (value == 1) return yue2::SymbolicMode::melody;
    if (value == 2) return yue2::SymbolicMode::full;
    throw std::runtime_error("invalid fixture symbolic mode");
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::cerr << "usage: yue2-generation-protocol-test QWEN.TIKTOKEN FIXTURE\n";
        return 2;
    }
    yue2::TextTokenizer tokenizer(argv[1]);
    Reader fixture(argv[2]);
    fixture.magic();
    const auto case_count = fixture.u32();
    for (std::uint32_t index = 0; index < case_count; ++index) {
        yue2::SongRequest request;
        request.symbolic_mode = mode(fixture.u32());
        request.style = fixture.string();
        request.lyrics = fixture.string();
        std::optional<std::vector<std::int32_t>> abc_ids;
        if (fixture.u32() != 0) {
            request.abc = fixture.string();
            abc_ids = fixture.ids();
        }
        const auto expected_text = fixture.string();
        const auto expected_positive = fixture.ids();
        const bool has_negative = fixture.u32() != 0;
        std::vector<std::int32_t> expected_negative;
        if (has_negative) expected_negative = fixture.ids();

        assert(yue2::generation_request_text(request) == expected_text);
        assert(yue2::make_positive_prefix(request, tokenizer) == expected_positive);
        if (has_negative) {
            assert(yue2::make_negative_prefix(request, tokenizer, abc_ids) == expected_negative);
        } else {
            bool rejected = false;
            try {
                (void)yue2::make_negative_prefix(request, tokenizer);
            } catch (const std::invalid_argument &) {
                rejected = true;
            }
            assert(rejected);
        }
    }
    assert(fixture.done());

    yue2::SongRequest request{"style", "lyrics", yue2::SymbolicMode::full, std::nullopt};
    bool rejected = false;
    try {
        (void)yue2::make_positive_prefix(request, tokenizer, {{yue2::kEodToken}});
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    assert(rejected);
    std::cout << "YuE2 generation prompt protocol matches " << case_count
              << " official vectors\n";
    return 0;
}
