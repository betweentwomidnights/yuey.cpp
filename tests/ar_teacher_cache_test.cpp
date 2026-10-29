#include "ar_teacher_cache.h"
#include <cassert>
#include <chrono>
#include <cstring>

int main() {
    namespace fs = std::filesystem;
    using namespace yue2::detail;
    const auto root = fs::temp_directory_path() /
        ("yue2-ar-teacher-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directory(root);
    const auto path = (root / "teacher-00000.gguf").string();
    ArTeacherProvenance provenance;
    provenance.identity = "identity";
    provenance.base_sha256 = "base";
    provenance.companion_sha256 = "companion";
    provenance.fixed_adapters = "C:/companion.gguf=1.000000\n";
    provenance.device = "cpu";
    provenance.frames = 2;
    provenance.ar_slots = 3;
    const ArTeacherWindow window{0, "prefix-sha", "codes-sha"};
    yue2::ArTrainingSample sample{{7, yue2::kMusicStartToken}, {11, 12}, {}, 0};
    const size_t row = yue2::kCodecSize + 1;
    sample.teacher_probabilities.assign(row * provenance.targets(), 0);
    for (size_t i = 0; i < sample.teacher_probabilities.size(); ++i)
        sample.teacher_probabilities[i] = float(i % 97) * 1e-9F;
    sample.teacher_probabilities[0] = 0.75F;
    sample.teacher_entropy = 3.25F;
    const yue2::ArTrainingSample window_only{sample.prefix, sample.codes, {}, 0};

    write_ar_teacher_cache(path, provenance, window, sample);
    assert(fs::exists(path) && !fs::exists(path + ".partial"));
    inspect_ar_teacher_cache(path, provenance, window, window_only);
    const auto loaded = load_ar_teacher_cache(path, provenance, window, window_only);
    assert(loaded.prefix == sample.prefix && loaded.codes == sample.codes);
    assert(loaded.teacher_probabilities.size() == sample.teacher_probabilities.size());
    assert(!std::memcmp(loaded.teacher_probabilities.data(), sample.teacher_probabilities.data(),
                        sample.teacher_probabilities.size() * sizeof(float)));
    assert(loaded.teacher_entropy == sample.teacher_entropy);

    auto rejected = [&](auto &&call) {
        try { call(); } catch (const std::runtime_error &) { return true; }
        return false;
    };
    // Immutable: never replace a completed teacher file.
    assert(rejected([&] { write_ar_teacher_cache(path, provenance, window, sample); }));
    for (auto field : {&ArTeacherProvenance::identity, &ArTeacherProvenance::base_sha256,
                       &ArTeacherProvenance::companion_sha256, &ArTeacherProvenance::fixed_adapters,
                       &ArTeacherProvenance::device}) {
        auto other = provenance;
        other.*field += "-other";
        assert(rejected([&] { inspect_ar_teacher_cache(path, other, window, window_only); }));
        assert(rejected([&] { load_ar_teacher_cache(path, other, window, window_only); }));
    }
    {
        auto other = provenance;
        other.ar_slots = 4;
        assert(rejected([&] { inspect_ar_teacher_cache(path, other, window, window_only); }));
        other = provenance;
        other.predict_end = true;
        assert(rejected([&] { inspect_ar_teacher_cache(path, other, window, window_only); }));
    }
    for (auto other_window : {ArTeacherWindow{1, "prefix-sha", "codes-sha"},
                              ArTeacherWindow{0, "other", "codes-sha"},
                              ArTeacherWindow{0, "prefix-sha", "other"}})
        assert(rejected([&] { inspect_ar_teacher_cache(path, provenance, other_window, window_only); }));
    {
        auto other = window_only;
        other.codes[1] = 13;
        assert(rejected([&] { inspect_ar_teacher_cache(path, provenance, window, other); }));
        other = window_only;
        other.prefix.insert(other.prefix.begin(), 5);
        assert(rejected([&] { inspect_ar_teacher_cache(path, provenance, window, other); }));
    }
    // A wrong-shape distribution cannot be written, and truncation is detected.
    {
        auto other = sample;
        other.teacher_probabilities.pop_back();
        assert(rejected([&] { write_ar_teacher_cache((root / "short.gguf").string(), provenance, window, other); }));
        fs::copy_file(path, root / "truncated.gguf");
        fs::resize_file(root / "truncated.gguf", fs::file_size(path) / 2);
        assert(rejected([&] {
            inspect_ar_teacher_cache((root / "truncated.gguf").string(), provenance, window, window_only);
        }));
    }
    fs::remove_all(root);
    return 0;
}
