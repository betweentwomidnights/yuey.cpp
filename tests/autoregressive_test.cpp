#include "yue2/autoregressive.h"
#include "yue2/generation.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class Reader {
public:
    explicit Reader(const std::string & path) {
        std::ifstream input(path, std::ios::binary);
        if (!input) throw std::runtime_error("failed to open AR fixture: " + path);
        bytes_ = {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    }
    void magic() {
        static constexpr std::uint8_t expected[] = {'Y', '2', 'A', 'R', 3, 0, 0, 0};
        require(sizeof(expected));
        if (std::memcmp(bytes_.data(), expected, sizeof(expected)) != 0) {
            throw std::runtime_error("invalid AR fixture header");
        }
        offset_ += sizeof(expected);
    }
    std::uint32_t u32() {
        require(4);
        std::uint32_t output = 0;
        for (int byte = 0; byte < 4; ++byte) {
            output |= static_cast<std::uint32_t>(bytes_[offset_ + byte]) << (8 * byte);
        }
        offset_ += 4;
        return output;
    }
    std::int32_t i32() { return static_cast<std::int32_t>(u32()); }
    float f32() {
        const auto bits = u32();
        float output = 0.0F;
        std::memcpy(&output, &bits, sizeof(output));
        return output;
    }
private:
    void require(std::size_t count) const {
        if (count > bytes_.size() - offset_) throw std::runtime_error("truncated AR fixture");
    }
    std::vector<std::uint8_t> bytes_;
    std::size_t offset_ = 0;
};

std::size_t argmax(const std::vector<float> & values) {
    return static_cast<std::size_t>(
        std::max_element(values.begin(), values.end()) - values.begin());
}

bool compare(
    const std::vector<float> & actual,
    const std::vector<float> & expected,
    const char * label) {
    double squared_error = 0.0;
    double squared_reference = 0.0;
    float maximum_error = 0.0F;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const auto difference = actual[index] - expected[index];
        maximum_error = std::max(maximum_error, std::abs(difference));
        squared_error += static_cast<double>(difference) * difference;
        squared_reference += static_cast<double>(expected[index]) * expected[index];
    }
    const auto rms = std::sqrt(squared_error / actual.size());
    const auto relative_rms = std::sqrt(squared_error / squared_reference);
    const auto actual_top = argmax(actual);
    const auto expected_top = argmax(expected);
    std::cout << label << ": max_error=" << maximum_error << " rms_error=" << rms
              << " relative_rms=" << relative_rms << " top=" << actual_top
              << " expected_top=" << expected_top << "\n";
    return actual_top == expected_top && relative_rms <= 0.003 && maximum_error <= 0.1F;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 3 || argc > 4) {
        std::cerr << "usage: yue2-autoregressive-test MODEL.gguf FIXTURE [DEVICE]\n";
        return 2;
    }
    Reader fixture(argv[2]);
    fixture.magic();
    std::vector<std::int32_t> ids(fixture.u32());
    for (auto & id : ids) id = fixture.i32();
    std::vector<float> expected(fixture.u32());
    for (auto & value : expected) value = fixture.f32();
    const auto next_token = fixture.i32();
    std::vector<float> expected_next(fixture.u32());
    for (auto & value : expected_next) value = fixture.f32();
    std::vector<std::int32_t> expected_greedy(fixture.u32());
    for (auto & id : expected_greedy) id = fixture.i32();
    if (expected.size() != yue2::kGenerationVocabSize) {
        throw std::runtime_error("unexpected AR fixture vocabulary size");
    }

    yue2::AutoregressiveOptions options;
    if (argc == 4) options.device = argv[3];
    yue2::AutoregressiveModel model(argv[1], options);
    const auto actual = model.logits(ids);
    auto session = model.create_session(ids.size() + 2);
    const auto cached = session->append(ids);
    const auto cached_next = session->append({next_token});
    if (!compare(actual, expected, "full") ||
        !compare(cached, expected, "cached-prefill") ||
        !compare(cached_next, expected_next, "cached-decode")) {
        std::cerr << "YuE2 AR logits exceed parity tolerance\n";
        return 1;
    }
    assert(session->token_count() == ids.size() + 1);
    assert(session->capacity() == ids.size() + 2);

    yue2::GenerationSampling greedy;
    greedy.temperature = 0.0F;
    greedy.min_tokens = static_cast<std::uint32_t>(expected_greedy.size());
    greedy.max_tokens = static_cast<std::uint32_t>(expected_greedy.size());
    const auto generated = model.generate(
        ids, greedy, yue2::AutoregressivePhase::abc, 1234);
    assert(generated.tokens == expected_greedy);
    assert(!generated.reached_end);
    greedy.min_tokens = 1;
    greedy.max_tokens = 1;
    const auto guided = model.generate_cfg(
        ids, ids, greedy, yue2::AutoregressivePhase::abc, 1.5F, 1234);
    assert(guided.tokens.size() == 1 && guided.tokens[0] == next_token);
    assert(!guided.reached_end);
    return 0;
}
