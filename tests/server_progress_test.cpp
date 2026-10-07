#include "server/progress.h"

#include <cassert>
#include <cstdint>
#include <limits>

int main() {
    using yue2::GenerationStage;
    using yue2::server::generation_progress;
    for (const int base : {0, 15, 25}) {
        int previous = base;
        for (const auto stage : {GenerationStage::abc, GenerationStage::semantic,
                                GenerationStage::flow, GenerationStage::decode,
                                GenerationStage::complete}) {
            for (std::uint32_t step = 0; step <= 32; ++step) {
                const auto progress = generation_progress(stage, step, 32, base);
                assert(progress >= previous && progress >= base && progress <= 99);
                previous = progress;
            }
        }
        assert(previous == 99);
    }
    assert(generation_progress(GenerationStage::abc, 1, 1) == 2);
    assert(generation_progress(GenerationStage::semantic, 1, 1) == 20);
    assert(generation_progress(GenerationStage::flow, 0, 32) == 20);
    assert(generation_progress(GenerationStage::flow, 16, 32) == 58);
    assert(generation_progress(GenerationStage::flow, 32, 32) == 97);
    assert(generation_progress(GenerationStage::decode, 1, 1) == 99);
    assert(generation_progress(GenerationStage::flow, 0, 0) == 20);
    assert(generation_progress(GenerationStage::flow, 10, 0) == 20);
    assert(generation_progress(GenerationStage::flow, 100, 32) == 97);
    const auto max = std::numeric_limits<std::uint32_t>::max();
    assert(generation_progress(GenerationStage::flow, max, max) == 97);
    assert(generation_progress(GenerationStage::complete, 1, 1, -100) == 99);
    assert(generation_progress(GenerationStage::complete, 1, 1, 200) == 99);
}
