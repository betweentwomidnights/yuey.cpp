#include "training_sampler.h"
#include "training_noise.h"
#include <cassert>
#include <set>
int main() {
    using yue2::detail::TrainingSampler;
    TrainingSampler full(36, true, 831001);
    for (int epoch = 0; epoch < 4; ++epoch) {
        std::set<size_t> seen;
        for (int i = 0; i < 36; ++i)
            seen.insert(full.next());
        assert(seen.size() == 36);
    }
    for (bool shuffle : {false, true}) {
        TrainingSampler a(7, shuffle, 123), b(7, shuffle, 99);
        for (int i = 0; i < 19; ++i)
            a.next();
        b.load(a.save());
        for (int i = 0; i < 100; ++i)
            assert(a.next() == b.next());
        bool rejected = false;
        try {
            TrainingSampler wrong(8, shuffle, 123);
            wrong.load(a.save());
        } catch (...) {
            rejected = true;
        }
        assert(rejected);
    }
    TrainingSampler one(1, false, 0);
    one.load("single-sample-v1");
    assert(one.next() == 0);
    // Each recording gets one visit per group epoch regardless of its window
    // count; each group's windows are exhausted before its own reshuffle.
    const std::vector<size_t> groups{0,0,0,1,2,2};
    TrainingSampler grouped(groups.size(),true,831001,groups),resumed(groups.size(),true,0,groups);
    std::vector<std::set<size_t>> windows(3);
    for(int epoch=0;epoch<3;++epoch) {
        std::set<size_t> tracks;
        for(int i=0;i<3;++i){const auto sample=grouped.next();tracks.insert(groups[sample]);windows[groups[sample]].insert(sample);}
        assert(tracks.size()==3);
    }
    assert(windows[0].size()==3&&windows[1].size()==1&&windows[2].size()==2);
    grouped.next();resumed.load(grouped.save());
    for(int i=0;i<200;++i)assert(grouped.next()==resumed.next());
    bool group_rejected=false;
    try {TrainingSampler changed(groups.size(),true,0,{0,0,1,1,2,2});changed.load(grouped.save());}catch(...){group_rejected=true;}
    assert(group_rejected);
    // Independent target distribution and paired-noise stream contract.
    using namespace yue2::detail;
    std::mt19937 uniform_rng(42), sigmoid_rng(42), time_a(17), time_b(17);
    double sum = 0;
    int central = 0;
    for (int i = 0; i < 100000; ++i) {
        const auto u = training_noise_time(NoiseSchedule::Uniform, uniform_rng, time_a);
        const auto s = training_noise_time(NoiseSchedule::Sigmoid, sigmoid_rng, time_b);
        assert(u >= .05F && u <= .95F && s > 0 && s < 1);
        sum += s;
        central += s >= .25F && s <= .75F;
        assert(uniform_rng() == sigmoid_rng());
    }
    assert(std::abs(sum / 100000 - .5) < .004);
    assert(std::abs(central / 100000.0 - .728063) < .006);
    bool rejected = false;
    try { noise_schedule("unknown"); } catch (const std::invalid_argument &) { rejected = true; }
    assert(rejected);
}
