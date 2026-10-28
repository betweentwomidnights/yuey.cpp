#include "training_sample_store.h"
#include <cassert>
#include <chrono>
#include <fstream>
#include <memory>

int main() {
    namespace fs = std::filesystem;
    const auto root = fs::temp_directory_path() /
        ("yue2-store-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directory(root);
    const std::vector<std::string> paths{(root / "a").string(), (root / "b").string()};
    for (size_t i = 0; i < paths.size(); ++i) std::ofstream(paths[i]) << i;
    using Sample = std::shared_ptr<int>;
    std::vector<std::weak_ptr<int>> allocated;
    auto loader = [&](const std::string &path) {
        int value = -1; std::ifstream(path) >> value;
        auto result = std::make_shared<int>(value);
        allocated.push_back(result);
        return result;
    };
    size_t inspected = 0;
    auto inspect = [&](const Sample &s) { assert(*s == int(inspected++)); };
    {
        size_t progress = 0;
        yue2::detail::TrainingSampleStore<Sample> store(paths, true, loader, inspect,
            [&](size_t current, size_t total) {
                assert(current == ++progress && total == paths.size());
                assert(inspected == current); // Report only inspected examples.
            });
        assert(store.size() == 2 && !store.empty() && inspected == 2);
        assert(progress == paths.size());
        for (const auto &weak : allocated) assert(weak.expired());
        assert(*store.get(0) == 0);
        const auto calls = allocated.size();
        assert(*store.get(0) == 0 && allocated.size() == calls);
        assert(*store.get(1) == 1);
        size_t live = 0;
        for (const auto &weak : allocated) live += !weak.expired();
        assert(live == 1);
        // A same-size rewrite is caught by its changed modification time.
        const auto original_time = fs::last_write_time(paths[1]);
        std::ofstream(paths[1]) << 9;
        fs::last_write_time(paths[1], original_time + std::chrono::seconds(2));
        bool rejected = false;
        try { store.get(1); } catch (const std::runtime_error &) { rejected = true; }
        assert(rejected);
    }
    inspected = 0; std::ofstream(paths[1]) << 1;
    {
        // Header/content validation does not decode streaming samples eagerly.
        const auto calls=allocated.size();size_t headers=0;
        yue2::detail::TrainingSampleStore<Sample> store(paths,true,loader,
            [&](const Sample &) { assert(false && "Path inspector replaces eager decoding"); },{},
            [&](const std::string &path) { assert(fs::file_size(path)==1);++headers; });
        assert(headers==2 && allocated.size()==calls);
        assert(*store.get(1)==1 && allocated.size()==calls+1);
    }
    {
        yue2::detail::TrainingSampleStore<Sample> store(paths, false, loader, inspect);
        const auto calls = allocated.size();
        assert(*store.get(0) == 0 && *store.get(1) == 1);
        assert(allocated.size() == calls);
    }
    // Replacement during initial validation fails before training starts.
    bool rejected = false;
    try {
        yue2::detail::TrainingSampleStore<Sample> store(paths, true, loader,
            [&](const Sample &) { std::ofstream(paths[0]) << "changed"; },
            [&](size_t, size_t) { assert(false && "Changed input must not report completed validation"); });
    } catch (const std::runtime_error &) { rejected = true; }
    assert(rejected);
    std::ofstream(paths[0]) << 0;
    rejected=false;
    try {
        yue2::detail::TrainingSampleStore<Sample> store(paths,true,loader,inspect,
            [&](size_t,size_t) { assert(false && "Changed header-validated input must not report success"); },
            [&](const std::string &) { std::ofstream(paths[0]) << "changed again"; });
    } catch (const std::runtime_error &) { rejected=true; }
    assert(rejected);
    for (const auto &path : paths) fs::remove(path);
    fs::remove(root);
}
