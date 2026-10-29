// The native training workflow's behaviour: progress reading, evaluation
// history, checkpoint pruning and selection, the AR teacher cache, NAR target
// matching and cache checks, rolling regeneration, cache release, disk
// checks, run ownership and dataset inventory. Ported from the Python tests
// that covered training_workflow.py and its scripts before they were removed.
//
// yue2-training-native-test [YUE2_SCORE [YUE2_TRAIN [YUE2_VAE_ENCODE]]]
//   With the tools, also checks the score MIDI roundtrip and yue2-train's
//   progress events. YUE2_VAE_ENCODER_TEST_MODEL (and _DEVICE) add the VAE
//   target integration. The test binary also stands in for yue2-ar-train and
//   yue2-train when the workflow runs them.
#include "server/json.h"
#include "training/conditioning_detail.h"
#include "training/dataset.h"
#include "training/driver.h"
#include "training/files.h"
#include "training/fitting_detail.h"
#include "training/preparation.h"
#include "training/process.h"
#include "training/publication.h"
#include "training/sha256.h"
#include "training/workflow.h"
#include "yue2/audio.h"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace {
namespace fs = std::filesystem;
namespace json = yue2::server::json;
namespace training = yue2::training;
namespace fit = yue2::training::fitting_detail;
namespace conditioning = yue2::training::conditioning_detail;

class Folder {
public:
    explicit Folder(const std::string & name) : path_(fs::temp_directory_path() / ("yue2-native-test-" + name)) {
        fs::remove_all(path_);
        fs::create_directories(path_);
    }
    ~Folder() { std::error_code error; fs::remove_all(path_, error); }
    const fs::path & path() const { return path_; }
private:
    fs::path path_;
};

void write(const fs::path & path, const std::string & bytes) {
    if (path.has_parent_path()) fs::create_directories(path.parent_path());
    std::ofstream(path, std::ios::binary | std::ios::trunc) << bytes;
}

void append(const fs::path & path, const std::string & bytes) { std::ofstream(path, std::ios::binary | std::ios::app) << bytes; }

std::string slurp(const fs::path & path) { return training::read_bytes(path); }

template <class F> std::string rejection(F && call) {
    try {
        call();
    } catch (const std::exception & error) {
        return error.what();
    }
    return {};
}

bool contains(const std::string & text, const std::string & part) { return text.find(part) != std::string::npos; }

json::Value parse(const std::string & text) { return json::parse(text); }

std::vector<json::Value> rows(const std::string & array) { return parse(array).array; }

double number(const json::Value & value) { return std::strtod(value.text.c_str(), nullptr); }

std::string text(const json::Value & value, const char * key) { return value.object.at(key).text; }

std::string metric(int step, double value) {
    return "{\"step\":" + std::to_string(step) + ",\"heldout_per_sample\":[" + json::python_repr(value) + "],\"retention_per_sample\":[]}\n";
}

void checkpoint(const fs::path & folder, int step, bool state = true) {
    write(folder / ("step-" + std::to_string(step) + ".gguf"), "adapter " + std::to_string(step));
    if (state) write(folder / ("step-" + std::to_string(step) + ".state.gguf"), "state " + std::to_string(step));
}

std::vector<std::string> checkpoint_names(const fs::path & out) {
    std::vector<std::string> names;
    for (const auto & folder : fs::directory_iterator(out)) {
        if (!folder.is_directory() || folder.path().filename().string().rfind("native", 0) != 0) continue;
        for (const auto & entry : fs::directory_iterator(folder.path())) {
            const auto name = entry.path().filename().string();
            if (std::regex_match(name, std::regex(R"(step-.*\.gguf)"))) names.push_back(folder.path().filename().string() + "/" + name);
        }
    }
    std::sort(names.begin(), names.end());
    return names;
}

// A small GGUF fixture, written through ggml's own writer.
class FixtureGguf {
public:
    explicit FixtureGguf(const std::string & architecture) : file_(gguf_init_empty()) {
        ggml_init_params params{64 * 1024 * 1024, nullptr, false};
        context_ = ggml_init(params);
        gguf_set_val_str(file_, "general.architecture", architecture.c_str());
    }
    ~FixtureGguf() { gguf_free(file_); ggml_free(context_); }
    void string(const char * key, const std::string & value) { gguf_set_val_str(file_, key, value.c_str()); }
    void u32(const char * key, std::uint32_t value) { gguf_set_val_u32(file_, key, value); }
    ggml_tensor * tensor(const char * name, ggml_type type, std::vector<int64_t> ne, const void * data = nullptr) {
        ggml_tensor * t = ne.size() == 1 ? ggml_new_tensor_1d(context_, type, ne[0])
                        : ne.size() == 2 ? ggml_new_tensor_2d(context_, type, ne[0], ne[1])
                        : ggml_new_tensor_4d(context_, type, ne[0], ne[1], ne[2], ne[3]);
        ggml_set_name(t, name);
        if (data) std::memcpy(t->data, data, ggml_nbytes(t));
        else std::memset(t->data, 0, ggml_nbytes(t));
        gguf_add_tensor(file_, t);
        return t;
    }
    void save(const fs::path & path) { assert(gguf_write_to_file(file_, path.string().c_str(), false)); }
private:
    gguf_context * file_;
    ggml_context * context_;
};

// --- stand-ins for the native tools -----------------------------------------

// yue2-ar-train --prepare-teacher-only: writes missing teacher files only, as
// the native writer does, and records how many it wrote.
int fake_teacher_writer(const std::vector<std::string> & args) {
    const auto value = [&](const char * key) { return *(std::find(args.begin(), args.end(), key) + 1); };
    assert(args.back() == "--prepare-teacher-only");
    assert(std::count(args.begin(), args.end(), "--teacher-cache") == 1);
    assert(value("--teacher-base-sha256") == "base" && value("--teacher-companion-sha256") == "companion");
    assert(value("--teacher-identity").size() == 64);
    const fs::path folder = fs::u8path(value("--teacher-cache"));
    const int count = std::stoi(value("--count"));
    int written = 0;
    for (int i = 0; i < count; ++i) {
        std::cout << "prepare_teacher=" << i + 1 << "/" << count << std::endl;
        char name[32];
        std::snprintf(name, sizeof(name), "teacher-%05d.gguf", i);
        if (!fs::exists(folder / name)) { write(folder / name, std::string(64, static_cast<char>(i))); ++written; }
    }
    write(folder.parent_path() / "written.txt", std::to_string(written));
    return 0;
}

// yue2-train --prepare-batch: YUE2_FAKE_BATCH=write regenerates each planned
// cache as the bytes "cache"; otherwise it records its command line and stops.
int fake_prepare_batch(const std::vector<std::string> & args) {
    const auto plan_path = fs::u8path(*(std::find(args.begin(), args.end(), "--prepare-batch") + 1));
    const auto plan = json::parse(slurp(plan_path));
    const char * mode = std::getenv("YUE2_FAKE_BATCH");
    if (!mode || std::string(mode) != "write") {
        std::string line;
        for (const auto & arg : args) line += arg + "\n";
        write(plan_path.parent_path() / "fake-command.txt", line);
        write(plan_path.parent_path() / "fake-cwd.txt", fs::current_path().u8string());
        return 0;
    }
    for (std::size_t i = 0; i < plan.array.size(); ++i) {
        const auto out = fs::u8path(plan.array[i].object.at("cache-out").text);
        write(out, "cache");
        std::cout << "CACHE_EVENT {\"index\":" << i << ",\"path\":" << json::quote(out.u8string())
                  << ",\"bytes\":\"5\",\"fnv\":\"0\",\"tokens\":\"1\",\"frames\":\"1\",\"half\":true}" << std::endl;
    }
    return 0;
}

// --- progress -----------------------------------------------------------------

void ar_progress() {
    Folder folder("ar-progress");
    const auto log = folder.path() / "native.log";
    write(log, "");
    fit::ArProgress progress(log, 6, 100, std::nullopt);
    auto state = progress.read();
    assert(text(state, "activity") == "Loading native trainer" && json::stringify(state.object.at("step")) == "6");
    // A split line is read once it is complete.
    write(log, "prepare_teacher=12/138\nprepare_teacher=13/");
    state = progress.read();
    assert(json::stringify(state.object.at("teacher_current")) == "12" && json::stringify(state.object.at("teacher_total")) == "138");
    assert(json::stringify(state.object.at("step")) == "6");
    append(log, "138\ninitial_loss=5.6 step=6\n");
    state = progress.read();
    assert(json::stringify(state.object.at("teacher_current")) == "13");
    assert(text(state, "activity") == "Evaluating training and validation windows");
    append(log, "step=7 loss=5.4 grad=0.4 seconds=3.6\n");
    state = progress.read();
    assert(text(state, "activity") == "Training" && json::stringify(state.object.at("step")) == "7");

    write(log, "step=100 loss=5.0\n");
    fit::ArProgress periodic(log, 0, 100, 150);
    assert(text(periodic.read(), "activity") == "Evaluating checkpoint");
    append(log, "step=101 loss=5.0\n");
    assert(text(periodic.read(), "activity") == "Training");
    append(log, "step=150 loss=5.0\n");
    state = periodic.read();
    assert(text(state, "activity") == "Final evaluation and checkpoint save" && state.object.count("activity_elapsed_seconds"));

    write(log, "teacher_cache=4/157\n");
    state = fit::ArProgress(log, 200, 100, std::nullopt).read();
    assert(text(state, "activity") == "Validating frozen teacher cache");
    assert(json::stringify(state.object.at("teacher_current")) == "4" && json::stringify(state.object.at("teacher_total")) == "157");
    assert(json::stringify(state.object.at("step")) == "200");
}

void nar_progress() {
    Folder folder("nar-progress");
    const auto log = folder.path() / "native.log";
    write(log, "");
    fit::NarProgress progress(log, 400, 0, "fit");
    const auto event = [&](const std::string & stage, int current, int total, int step = 0) {
        append(log, "TRAIN_EVENT {\"stage\":\"" + stage + "\",\"current\":" + std::to_string(current) + ",\"total\":" +
                    std::to_string(total) + ",\"step\":" + std::to_string(step) + "}\n");
        return progress.read();
    };
    // Validation and the initial evaluation are not optimizer updates.
    auto state = event("cache-validation", 127, 166);
    assert(json::stringify(state.object.at("step")) == "0" && json::stringify(state.object.at("phase_current")) == "127");
    assert(text(state, "phase_unit") == "files");
    state = event("model-loading", 0, 1);
    assert(text(state, "activity") == "Loading native NAR model" && !state.object.count("phase_current"));
    state = event("evaluation", 61, 166);
    assert(text(state, "activity") == "Initial evaluation" && json::stringify(state.object.at("phase_current")) == "61");
    assert(text(state, "phase_unit") == "windows" && json::stringify(state.object.at("step")) == "0");
    append(log, "{\"step\":1,\"loss\":0.8,\"seconds\":2.5,\"cache_load_seconds\":0.4}\n");
    state = progress.read();
    assert(json::stringify(state.object.at("step")) == "1" && text(state, "activity") == "Training");
    assert(number(state.object.at("last_update_seconds")) == 2.5 && number(state.object.at("last_cache_load_seconds")) == .4);
    assert(!state.object.count("phase_current"));

    // A split event, a resumed run and the final evaluation.
    write(log, "TRAIN_EVENT {\"stage\":\"cache-validation\",");
    fit::NarProgress resumed(log, 400, 200, "fit");
    assert(json::stringify(resumed.read().object.at("step")) == "200");
    append(log, "\"current\":1,\"total\":166,\"step\":0}\n");
    assert(json::stringify(resumed.read().object.at("step")) == "200");
    append(log, "TRAIN_EVENT {\"stage\":\"evaluation\",\"current\":28,\"total\":166,\"step\":400}\n");
    state = resumed.read();
    assert(json::stringify(state.object.at("step")) == "400" && text(state, "activity") == "Final evaluation");
    assert(json::stringify(state.object.at("phase_current")) == "28");
    append(log, "TRAIN_EVENT {\"stage\":\"checkpoint-save\",\"current\":0,\"total\":1,\"step\":400}\n");
    assert(text(resumed.read(), "activity") == "Saving checkpoint");

    // An old trainer's banner still gives a useful activity.
    write(log, "[yue2:train] 750 frames\n");
    fit::NarProgress banner(log, 400, 0, "fit");
    assert(text(banner.read(), "activity") == "Initial evaluation" && !banner.events());
    fit::NarProgress loading(log, 400, 1, "fit");
    assert(text(loading.read(), "activity") == "Loading native optimizer state");
}

// --- evaluation history, selection and pruning ----------------------------------

const auto heldout = rows(R"([{"split":"validation","group":"heldout"}])");

void evaluations() {
    const auto groups = rows(R"([{"split":"validation","group":"a"},{"split":"validation","group":"a"},{"split":"validation","group":"b"}])");
    {
        // A partial last record is ignored; groups are balanced by recording.
        Folder folder("evaluations-partial");
        const auto path = folder.path() / "native-1" / "metrics.jsonl";
        const std::string line = "{\"step\":100,\"heldout_per_sample\":[1,1,3],\"retention_per_sample\":[]}\n";
        write(path, "{\"step\":0,\"heldout_per_sample\":[1,3,7],\"retention_per_sample\":[]}\n" + line.substr(0, 20));
        auto history = fit::evaluation_history(folder.path(), groups, "ar");
        assert(history.object.at("evaluations").array.size() == 1);
        assert(number(history.object.at("best_so_far").object.at("heldout_group_mean")) == 4.5);
        assert(!fs::exists(folder.path() / "selection.json"));
        append(path, line.substr(20));
        history = fit::evaluation_history(folder.path(), groups, "ar");
        assert(history.object.at("evaluations").array.size() == 2);
        assert(number(history.object.at("best_so_far").object.at("heldout_group_mean")) == 2);
    }
    {
        // native-10 is resumed after native-2: its record for a step wins.
        Folder folder("evaluations-order");
        for (const auto & [segment, value] : std::vector<std::pair<std::string, int>>{{"native-10", 2}, {"native-2", 4}}) {
            const auto v = std::to_string(value);
            write(folder.path() / segment / "metrics.jsonl", "{\"step\":100,\"heldout_per_sample\":[" + v + "," + v + "," + v + "],\"retention_per_sample\":[]}\n");
            write(folder.path() / segment / "step-100.gguf", "identical resumed checkpoint");
        }
        assert(number(fit::evaluation_history(folder.path(), groups, "ar").object.at("best_so_far").object.at("heldout_group_mean")) == 2);
        fit::select_ar(folder.path(), groups, json::object());
        const auto selection = json::parse(slurp(folder.path() / "selection.json"));
        assert(number(selection.object.at("selected").object.at("heldout_group_mean")) == 2);
    }
    {
        // NAR histories mix optimizer records in; retention stays separate.
        Folder folder("evaluations-nar");
        write(folder.path() / "native-1" / "history.jsonl",
              "{\"step\":1,\"loss\":0.4}\n{\"step\":100,\"heldout_eval\":0.8,\"heldout_per_sample\":[1,3,7],\"retention_per_sample\":[9]}\n");
        auto all = groups;
        all.push_back(parse(R"({"split":"retention","group":"neutral"})"));
        const auto history = fit::evaluation_history(folder.path(), all, "nar");
        assert(history.object.at("evaluations").array.size() == 1);
        assert(number(history.object.at("best_so_far").object.at("heldout_group_mean")) == 4.5);
        assert(number(history.object.at("best_so_far").object.at("retention_group_mean")) == 9);
    }
    {
        Folder folder("evaluations-invalid");
        write(folder.path() / "native-1" / "metrics.jsonl", "{\"step\":100,\"heldout_per_sample\":[1]}\n");
        assert(contains(rejection([&] { fit::evaluation_history(folder.path(), groups, "ar"); }), "Metric/example count"));
    }
}

void selection_from_original_segment() {
    // A resumed fit re-evaluates its loaded checkpoint in the new segment.
    Folder folder("selection");
    const auto old = folder.path() / "native-1", next = folder.path() / "native-2";
    write(old / "step-5.gguf", "original saved adapter");
    write(old / "step-5.state.gguf", "original optimizer");
    write(next / "step-6.gguf", "next adapter");
    write(old / "metrics.jsonl", metric(0, 5));
    write(next / "metrics.jsonl", metric(5, 4) + metric(6, 4.1));
    fit::select_ar(folder.path(), heldout, json::object());
    const auto selected = json::parse(slurp(folder.path() / "selection.json"));
    assert(text(selected, "adapter") == training::python_path(old / "step-5.gguf"));
    assert(text(selected, "state") == training::python_path(old / "step-5.state.gguf"));
    write(next / "step-5.gguf", "a conflicting branch");
    assert(contains(rejection([&] { fit::select_ar(folder.path(), heldout, json::object()); }), "Conflicting adapters"));
}

void pruning() {
    {
        Folder folder("prune-segments");
        const auto old = folder.path() / "native-1", next = folder.path() / "native-2";
        checkpoint(old, 0, false);
        for (int step : {100, 200}) checkpoint(old, step);
        for (int step : {300, 400}) checkpoint(next, step);
        write(old / "metrics.jsonl", metric(0, 5) + metric(100, 4.5) + metric(200, 4));
        write(next / "metrics.jsonl", metric(200, 4) + metric(300, 4.2) + metric(400, 4.3));
        assert(fit::prune_checkpoints(folder.path(), heldout, "ar") > 0);
        assert((checkpoint_names(folder.path()) == std::vector<std::string>{"native-1/step-0.gguf", "native-1/step-200.gguf",
            "native-1/step-200.state.gguf", "native-2/step-400.gguf", "native-2/step-400.state.gguf"}));
        fit::select_ar(folder.path(), heldout, json::object());
        assert(text(json::parse(slurp(folder.path() / "selection.json")), "adapter") == training::python_path(old / "step-200.gguf"));
    }
    {
        // Unevaluated steps and a checkpoint whose state is still being written survive.
        Folder folder("prune-in-flight");
        const auto native = folder.path() / "native-1";
        for (int step : {10, 20, 30}) checkpoint(native, step);
        write(native / "step-40.gguf", "adapter 40");
        write(native / "step-40.state.gguf.partial", "partial");
        write(native / "metrics.jsonl", metric(10, 3) + metric(30, 4));
        fit::prune_checkpoints(folder.path(), heldout, "ar");
        assert((checkpoint_names(folder.path()) == std::vector<std::string>{"native-1/step-10.gguf", "native-1/step-10.state.gguf",
            "native-1/step-20.gguf", "native-1/step-20.state.gguf", "native-1/step-30.gguf", "native-1/step-30.state.gguf",
            "native-1/step-40.gguf"}));
        assert(fs::exists(native / "step-40.state.gguf.partial"));
    }
    {
        Folder folder("prune-nar");
        const auto native = folder.path() / "native-1";
        for (int step : {2, 4, 6}) checkpoint(native, step);
        write(native / "history.jsonl", "{\"step\":1,\"loss\":1.0}\n{\"step\":2,\"heldout_eval\":true,\"heldout_per_sample\":[1.2]}\n"
                                        "{\"step\":4,\"heldout_eval\":true,\"heldout_per_sample\":[1.1]}\n"
                                        "{\"step\":6,\"heldout_eval\":true,\"heldout_per_sample\":[1.3]}\n");
        fit::prune_checkpoints(folder.path(), heldout, "nar");
        assert((checkpoint_names(folder.path()) == std::vector<std::string>{"native-1/step-4.gguf", "native-1/step-4.state.gguf",
            "native-1/step-6.gguf", "native-1/step-6.state.gguf"}));
    }
#ifdef _WIN32
    {
        // Windows refuses to delete a mapped file; its adapter waits for a later pass.
        Folder folder("prune-locked");
        const auto native = folder.path() / "native-1";
        for (int step : {2, 4, 6}) checkpoint(native, step);
        write(native / "metrics.jsonl", metric(2, 3) + metric(4, 4) + metric(6, 5));
        HANDLE held = CreateFileW((native / "step-4.state.gguf").c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
        assert(held != INVALID_HANDLE_VALUE);
        fit::prune_checkpoints(folder.path(), heldout, "ar");
        const auto names = checkpoint_names(folder.path());
        assert(std::count(names.begin(), names.end(), "native-1/step-4.gguf") == 1);
        CloseHandle(held);
        fit::prune_checkpoints(folder.path(), heldout, "ar");
        assert((checkpoint_names(folder.path()) == std::vector<std::string>{"native-1/step-2.gguf", "native-1/step-2.state.gguf",
            "native-1/step-6.gguf", "native-1/step-6.state.gguf"}));
    }
#endif
    {
        Folder folder("prune-nothing");
        checkpoint(folder.path() / "native-1", 5);
        assert(fit::prune_checkpoints(folder.path(), heldout, "ar") == 0);
        write(folder.path() / "native-1" / "metrics.jsonl", metric(5, 1));
        assert(fit::prune_checkpoints(folder.path(), heldout, "ar") == 0);
        assert(checkpoint_names(folder.path()).size() == 2);
    }
}

// --- AR teacher cache -------------------------------------------------------------

void teacher_cache() {
    const auto train_rows = rows(R"([{"split":"train","prefix_sha256":"p0","codes_sha256":"c0"},
        {"split":"train","prefix_sha256":"p1","codes_sha256":"c1"},{"split":"train","prefix_sha256":"p2","codes_sha256":"c2"}])");
    const auto identity = parse(R"({"base_sha256":"base","companion_sha256":"companion","dataset_sha256":"data","runtime":{"x":"y"}})");
    const auto resolved = fit::teacher_identity(identity, "cuda", 1131, json::from_integer(750), "f32");
    const auto command = std::vector<std::string>{training::python_path(training::current_executable()), "--fake-teacher-writer", "--count", "3"};
    const auto prepare = [&](const fs::path & root, const json::Value & teacher) {
        return fit::prepare_teacher_cache(root / "teacher", command, root, root, "lora", teacher, train_rows, root / "stop.request");
    };
    {
        Folder folder("teacher-index");
        const auto & root = folder.path();
        const auto sha = prepare(root, resolved);
        fit::verify_teacher_cache(root / "teacher", sha, train_rows);
        const auto index = json::parse(slurp(root / "teacher" / "index.json"));
        assert(text(index.object.at("files").array[2], "path") == "teacher-00002.gguf");
        assert(text(index, "base_sha256") == "base" && text(index, "companion_sha256") == "companion");
        assert(json::stringify(index.object.at("ar_slots")) == "1131" && json::stringify(index.object.at("frames")) == "750");
        assert(json::stringify(index.object.at("bytes")) == "192");
        // A completed index is reused without running the writer again.
        fs::remove(root / "written.txt");
        assert(prepare(root, resolved) == sha && !fs::exists(root / "written.txt"));
        write(root / "teacher" / "teacher-00001.gguf", "changed");
        assert(contains(rejection([&] { fit::verify_teacher_cache(root / "teacher", sha, train_rows); }), "bytes changed"));
        assert(contains(rejection([&] { fit::verify_teacher_cache(root / "teacher", "other", train_rows); }), "does not match"));
        auto changed = train_rows;
        changed[0].object["codes_sha256"] = json::from_string("other");
        assert(contains(rejection([&] { fit::verify_teacher_cache(root / "teacher", sha, changed); }), "provenance changed"));
        const std::vector<json::Value> fewer(train_rows.begin(), train_rows.begin() + 2);
        assert(contains(rejection([&] { fit::verify_teacher_cache(root / "teacher", sha, fewer); }), "window count"));
    }
    {
        // An interrupted preparation resumes; the cache identity is immutable.
        Folder folder("teacher-resume");
        write(folder.path() / "teacher" / "teacher-00000.gguf", std::string(64, '\0'));
        prepare(folder.path(), resolved);
        assert(slurp(folder.path() / "written.txt") == "2");
        auto other = resolved;
        other.object["device"] = json::from_string("cpu");
        assert(contains(rejection([&] { prepare(folder.path(), other); }), "identity changed"));
        assert(contains(rejection([&] { fit::teacher_identity(identity, "cuda", 1, json::from_integer(1), "f16"); }), "Unsupported"));
    }
    {
        Folder folder("teacher-unexpected");
        write(folder.path() / "teacher" / "teacher-00009.gguf", "stale");
        assert(contains(rejection([&] { prepare(folder.path(), resolved); }), "Unexpected AR teacher"));
        assert(!fs::exists(folder.path() / "teacher" / "index.json"));
    }
}

// --- NAR targets, caches and preflight ------------------------------------------------

void nar_caches() {
    Folder folder("nar-cache");
    const auto & root = folder.path();
    const std::int32_t prefix[] = {1, 2}, reversed[] = {2, 1}, codes[] = {3, 4};
    const std::vector<float> clean(2 * 64, 0.0F);
    write(root / "prefix", std::string(reinterpret_cast<const char *>(prefix), sizeof(prefix)));
    write(root / "codes", std::string(reinterpret_cast<const char *>(codes), sizeof(codes)));
    write(root / "means", std::string(reinterpret_cast<const char *>(clean.data()), clean.size() * sizeof(float)));
    const auto row = parse("{\"prefix\":" + json::quote(root.u8string() + "/prefix") + ",\"codes\":" + json::quote(root.u8string() + "/codes") + ",\"frames\":2}");
    const auto target = parse("{\"means\":" + json::quote(root.u8string() + "/means") + "}");
    const auto base = root / "base", common = root / "common", adapter = root / "ar";
    const auto cache = [&](const std::string & name, bool bad_prefix, bool nonfinite) {
        FixtureGguf file("yue2_nar_sample");
        file.string("yue2.training.cache.identity", "key");
        file.string("yue2.training.model", base.u8string());
        file.string("yue2.training.fixed_adapters", "");
        file.string("yue2.training.conditioning_adapters", training::resolved_path(common) + "=1.000000\n" + training::resolved_path(adapter) + "=1.000000\n");
        file.u32("yue2.training.ar_tokens", 5);
        file.tensor("prefix", GGML_TYPE_I32, {2}, bad_prefix ? reversed : prefix);
        file.tensor("codes", GGML_TYPE_I32, {2}, codes);
        file.tensor("clean", GGML_TYPE_F32, {64, 2}, clean.data());
        auto * keys = file.tensor("keys", GGML_TYPE_F16, {128, 5, 8, 28});
        if (nonfinite) static_cast<std::uint16_t *>(keys->data)[0] = 0x7e00;
        file.tensor("values", GGML_TYPE_F16, {128, 5, 8, 28});
        const auto path = root / (name + ".gguf");
        file.save(path);
        return path;
    };
    conditioning::verify_cache(cache("valid", false, false), "key", row, target, base, common, adapter, "f16");
    assert(contains(rejection([&] { conditioning::verify_cache(cache("prefix", true, false), "key", row, target, base, common, adapter, "f16"); }),
                    "input payload differs"));
    assert(contains(rejection([&] { conditioning::verify_cache(cache("nan", false, true), "key", row, target, base, common, adapter, "f16"); }),
                    "Nonfinite"));

    // Each example's target: same recording, start, length and split.
    const auto a = parse(R"({"source_sha256":"a","start_frame":0,"frames":750,"split":"train"})");
    const auto b = parse(R"({"source_sha256":"a","start_frame":750,"frames":750,"split":"train"})");
    const auto pairs = conditioning::match_targets({a, b}, {b, a});
    assert(pairs.size() == 2 && training::python_equal(pairs[0].second, a) && training::python_equal(pairs[1].second, b));
    const auto held = parse(R"({"source_sha256":"a","start_frame":0,"frames":750,"split":"validation"})");
    for (const auto & [key, value] : std::vector<std::pair<const char *, json::Value>>{
             {"start_frame", json::from_integer(1)}, {"frames", json::from_integer(749)}, {"split", json::from_string("train")}}) {
        auto changed = held;
        changed.object[key] = value;
        assert(!rejection([&] { conditioning::match_targets({held}, {changed}); }).empty());
    }
    auto other = a;
    other.object["source_sha256"] = json::from_string("b");
    for (const auto & targets : std::vector<std::vector<json::Value>>{{}, {a, a}, {a, other}}) {
        assert(!rejection([&] { conditioning::match_targets({a}, targets); }).empty());
    }
}

void preflight() {
    Folder folder("preflight");
    std::vector<json::Value> windows;
    for (const auto & [split, tokens] : std::vector<std::pair<std::string, int>>{{"train", 2}, {"train", 4}, {"validation", 8}}) {
        FixtureGguf file("yue2_nar_sample");
        file.u32("yue2.training.ar_tokens", static_cast<std::uint32_t>(tokens));
        file.tensor("clean", GGML_TYPE_F32, {128});
        file.tensor("keys", GGML_TYPE_F16, {static_cast<int64_t>(tokens) * 128 * 8 * 28});
        file.tensor("values", GGML_TYPE_F16, {static_cast<int64_t>(tokens) * 128 * 8 * 28});
        const auto path = folder.path() / (std::to_string(tokens) + ".gguf");
        file.save(path);
        windows.push_back(parse("{\"split\":\"" + split + "\",\"cache\":" + json::quote(path.u8string()) + "}"));
    }
    const auto cache_path = [](const json::Value & row) { return fs::u8path(row.object.at("cache").text); };
    const auto [first, maximum] = fit::preflight_sample(windows, cache_path, 2);
    assert(first == cache_path(windows[0]) && maximum == 8);
    assert(contains(rejection([&] { fit::preflight_sample(windows, cache_path, 3); }), "shape"));
}

void decoder_trim_is_refused() {
    // Generated targets are a research option; a real row never declares a trim.
    Folder folder("vae-trim");
    const auto wav = yue2::audio::encode_wav_float(std::vector<float>(48000 * 2), 48000, 2);
    write(folder.path() / "source.wav", std::string(wav.begin(), wav.end()));
    write(folder.path() / "vae.gguf", "vae");
    write(folder.path() / "dataset.json", "{\"completed\":true,\"rows\":[{\"source\":" + json::quote((folder.path() / "source.wav").u8string()) +
        ",\"source_sha256\":\"" + training::sha256_file(folder.path() / "source.wav") +
        "\",\"track\":\"source.wav\",\"start_frame\":0,\"frames\":1,\"split\":\"train\",\"native_decoder_trim_samples\":64}]}");
    training::TargetOptions o;
    o.dataset = folder.path() / "dataset.json";
    o.vae = folder.path() / "vae.gguf";
    o.binary = training::current_executable();
    o.out = folder.path() / "targets";
    assert(contains(rejection([&] { training::prepare_targets(o); }), "decoder edge"));
}

// --- rolling regeneration -------------------------------------------------------------

void rolling_regeneration() {
    const auto self = training::current_executable();
    {
        // The sampler drew the same window twice in one segment.
        Folder folder("rolling-once");
        const auto & root = folder.path();
        write(root / "dataset.json", R"({"identity":{"device":"cuda","threads":6,"kv_type":"f16","prefill_batch":0}})");
        auto row = parse(R"({"prefix":"p","codes":"c","means":"m","cache_identity":"i","cache_bytes":"5","cache_fnv":"0",
                             "cache_tokens":"1","cache_frames":"1","cache_half":true})");
        row.object["cache"] = json::from_string((root / "a" / "sample.gguf").u8string());
        row.object["cache_sha256"] = json::from_string(training::sha256_hex("cache"));
        training::set_environment("YUE2_FAKE_BATCH", "write");
        assert(training::materialize(root / "dataset.json", {row, row}, self, "b", "c", "a", std::nullopt) == 1);
        assert(json::parse(slurp(root / "materialize.json")).array.size() == 1);
    }
    {
        // The native tool runs from its own folder, so relative paths must not reach it.
        Folder folder("rolling-absolute");
        const auto previous = fs::current_path();
        fs::current_path(folder.path());
        training::set_environment("YUE2_FAKE_BATCH", "record");
        write("run/dataset.json", R"({"identity":{"device":"cuda","threads":6,"kv_type":"f16","prefill_batch":0}})");
        const auto row = parse(R"({"prefix":"data/p","codes":"data/c","means":"data/m","cache":"run/a/sample.gguf","cache_identity":"i"})");
        const auto error = rejection([&] {
            training::materialize("run/dataset.json", {row}, self, "b.gguf", "c.gguf", "a.gguf", fs::path("run/stop.request"));
        });
        fs::current_path(previous);
        assert(contains(error, "did not regenerate"));
        std::vector<std::string> command;
        std::istringstream lines(slurp(folder.path() / "run" / "fake-command.txt"));
        for (std::string line; std::getline(lines, line);) command.push_back(line);
        const auto after = [&](const char * key) { return fs::u8path(*(std::find(command.begin(), command.end(), key) + 1)); };
        assert(after("--prepare-batch").is_absolute() && after("--model").is_absolute() && after("--stop-file").is_absolute());
        for (const auto & value : command) if (value.size() > 5 && value.substr(value.size() - 5) == ".gguf") assert(fs::u8path(value).is_absolute());
        const auto entry = json::parse(slurp(after("--prepare-batch"))).array[0];
        for (const char * key : {"prefix", "tokens", "latents", "cache-out"}) assert(fs::u8path(entry.object.at(key).text).is_absolute());
        assert(fs::equivalent(fs::u8path(slurp(folder.path() / "run" / "fake-cwd.txt")), self.parent_path()));
    }
}

// --- cache release, disk and ownership ------------------------------------------------

json::Value own_owner() {
    auto owner = json::object();
    owner.object["pid"] = json::from_integer(
#ifdef _WIN32
        static_cast<long long>(GetCurrentProcessId())
#else
        static_cast<long long>(getpid())
#endif
    );
    owner.object["identity"] = json::from_string(training::own_process_identity());
    return owner;
}

void cache_release() {
    Folder folder("release");
    const auto run = folder.path() / "run";
    write(run / "runtime.json", "{\"bin\":" + json::quote(folder.path().u8string()) + "}");
    write(run / "request.json", R"({"id":"fixture","family":"lora","rank":4})");
    const auto lora = run / "data" / "lora", nar = run / "nar-run";
    checkpoint(lora / "native-1", 0, false);
    checkpoint(lora / "native-1", 20);
    checkpoint(lora / "native-1", 40);
    write(lora / "probe" / "step-0.gguf", "probe");
    write(lora / "probe-diagnostics.gguf", "diagnostics");
    write(lora / "prompts.json", "{}");
    checkpoint(nar / "native-1", 30);
    checkpoint(nar / "native-1", 60);
    write(run / "nar-conditioning" / "window-0" / "sample.gguf", "kv");
    write(run / "nar-conditioning" / "dataset.json", "{}");
    write(run / "data" / "examples" / "window-0" / "source.wav", "audio");
    write(run / "data" / "examples" / "window-0" / "codes.i32", "codes");
    auto state = training::settings(parse(R"({"id":"fixture","family":"lora","rank":4})"));
    write(run / "state.json", "{\"settings\":" + json::stringify(state) + ",\"ar_selection\":{\"adapter\":" +
        json::quote((lora / "native-1" / "step-20.gguf").u8string()) + "},\"nar_selection\":{\"adapter\":" +
        json::quote((nar / "native-1" / "step-60.gguf").u8string()) + "},\"caches_released\":{\"gib\":0.5}}");
    assert(training::release_caches(run) > 0);
    std::vector<std::string> left;
    for (const auto & entry : fs::recursive_directory_iterator(run)) {
        if (entry.is_regular_file()) left.push_back(fs::relative(entry.path(), run).generic_string());
    }
    std::sort(left.begin(), left.end());
    assert((left == std::vector<std::string>{"data/examples/window-0/codes.i32", "data/lora/native-1/step-20.gguf", "data/lora/prompts.json",
        "nar-conditioning/dataset.json", "nar-run/native-1/step-60.gguf", "request.json", "runtime.json", "state.json"}));
    // A run that released its caches cannot train again.
    write(run / "vae-targets" / "index.json", "{}");
    training::save(run / "owner.json", own_owner());
    assert(training::run_workflow(run, "train") == 1);
    const auto after = json::parse(slurp(run / "state.json"));
    assert(text(after, "status") == "failed" && contains(text(after, "error"), "released its training caches"));
}

void disk_check() {
    // Shaped like the 38-recording laptop corpus: 138 training, 28 heldout windows.
    std::vector<json::Value> corpus;
    for (int i = 0; i < 166; ++i) corpus.push_back(parse(i < 138 ? R"({"split":"train","frames":750,"prefix_tokens":1131})"
                                                                 : R"({"split":"validation","frames":750,"prefix_tokens":1131})"));
    const double cache = training::conditioning_bytes(corpus, "cache", "heldout") / 1073741824.0;
    const double rolling = training::conditioning_bytes(corpus, "rolling", "heldout") / 1073741824.0;
    assert(std::abs(cache - 33.4) < .05);
    const std::vector<json::Value> one(corpus.begin(), corpus.begin() + 1);
    assert(std::abs(rolling - 53 * training::conditioning_bytes(one, "cache", "full") / 1073741824.0) < 1e-6 && rolling < cache / 3);
    assert(training::conditioning_bytes(corpus, "resident", "full") == 0);
    // Full-scope evaluation needs every window, so rolling saves nothing.
    assert(training::conditioning_bytes(corpus, "rolling", "full") == training::conditioning_bytes(corpus, "cache", "full"));
    const std::vector<json::Value> three(corpus.begin(), corpus.begin() + 3);
    assert(training::conditioning_bytes(three, "rolling", "heldout") == training::conditioning_bytes(three, "cache", "full"));

    // With AR selected and conditioning complete, only NAR checkpoints remain.
    std::vector<json::Value> windows;
    for (int i = 0; i < 12; ++i) windows.push_back(parse(i < 10 ? R"({"split":"train","frames":750,"prefix_tokens":1000})"
                                                                : R"({"split":"validation","frames":750,"prefix_tokens":1000})"));
    const auto options = training::settings(parse(R"({"id":"fixture","recipe_version":4,"rank":4})"));
    assert(training::remaining_disk_bytes(windows, options, false, 0, false, 0) > training::checkpoint_allowance(options, 2));
    assert(training::remaining_disk_bytes(windows, options, true, 0, true, 0) == training::checkpoint_allowance(options, 1));

    // Teacher estimates match the measured Koan corpus (recipe 4).
    std::vector<json::Value> koan;
    for (int i = 0; i < 195; ++i) koan.push_back(parse(i < 157 ? R"({"split":"train","frames":750,"prefix_tokens":1000})"
                                                               : R"({"split":"validation","frames":750,"prefix_tokens":1000})"));
    const auto gib = [](const json::Value & estimate, const char * key) { return number(estimate.object.at(key)); };
    const auto old = training::ar_resource_estimate(koan, training::settings(parse(R"({"id":"v3","recipe_version":3})")), "cuda");
    const auto streamed = training::ar_resource_estimate(koan, training::settings(parse(R"({"id":"v4","recipe_version":4})")), "cuda");
    assert(std::abs(gib(old, "estimated_ar_host_ram_gib") - 15.5) < .1 && gib(old, "estimated_teacher_disk_gib") == 0);
    assert(std::abs(gib(streamed, "estimated_ar_host_ram_gib") - 3.1) < .05);
    assert(std::abs(gib(streamed, "estimated_teacher_disk_gib") - 14.38) < .02);
    auto doubled = koan;
    doubled.insert(doubled.end(), koan.begin(), koan.end());
    const auto twice = training::ar_resource_estimate(doubled, training::settings(parse(R"({"id":"v4","recipe_version":4})")), "cuda");
    assert(gib(twice, "estimated_ar_host_ram_gib") == gib(streamed, "estimated_ar_host_ram_gib"));
    assert(std::abs(gib(twice, "estimated_teacher_disk_gib") - 2 * gib(streamed, "estimated_teacher_disk_gib")) < 1e-9);
    const auto cpu = training::ar_resource_estimate(koan, training::settings(parse(R"({"id":"v4","recipe_version":4})")), "cpu", 8ULL << 30);
    assert(std::abs(gib(cpu, "estimated_ar_host_ram_gib") - 11.1) < .05);
    const auto no_kl = training::ar_resource_estimate(koan, training::settings(parse(R"({"id":"v4","recipe_version":4,"kl_weight":0})")), "cuda");
    assert(gib(no_kl, "estimated_teacher_disk_gib") == 0);
}

void ownership_and_progress() {
    Folder folder("owner");
    training::save(folder.path() / "owner.json", own_owner());
    training::wait_for_owner(folder.path(), .01);
    auto stale = own_owner();
    stale.object["identity"] = json::from_string("stale");
    training::save(folder.path() / "owner.json", stale);
    assert(contains(rejection([&] { training::wait_for_owner(folder.path(), .01); }), "not committed"));

    // A resumed stage never shows the previous segment's terminal snapshot,
    // and a stage that finishes between ticks still reports its last progress.
    const auto progress = folder.path() / "progress.json";
    write(progress, R"({"status":"stopped","step":6})");
    assert(!training::ProgressWatch(progress).read());
    fs::remove(progress);
    const training::ProgressWatch fresh(progress);
    write(progress, R"({"current":7,"total":7})");
    const auto seen = fresh.read();
    assert(seen && json::stringify(*seen) == R"({"current":7,"total":7})");
}

#ifdef _WIN32
void status_survives_reader_lock() {
    // A reader that denies delete sharing delays the replacement; the old
    // complete JSON stays visible until it is released.
    Folder folder("status-lock");
    const auto path = folder.path() / "progress.json";
    training::save(path, parse(R"({"step":1})"));
    HANDLE reader = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, 0, nullptr);
    assert(reader != INVALID_HANDLE_VALUE);
    std::string error;
    std::thread writer([&] { error = rejection([&] { training::save(path, parse(R"({"step":2})")); }); });
    std::this_thread::sleep_for(std::chrono::milliseconds(120));
    assert(json::stringify(json::parse(slurp(path))) == R"({"step":1})");
    CloseHandle(reader);
    writer.join();
    assert(error.empty() && json::stringify(json::parse(slurp(path))) == R"({"step":2})");
}
#endif

// --- settings and inventory -------------------------------------------------------------

void settings_and_inventory() {
    const auto settings = [](const std::string & request) { return training::settings(parse(request)); };
    assert(!settings(R"({"id":"old"})").object.count("evaluation_scope"));
    assert(text(settings(R"({"id":"new","recipe_version":2})"), "evaluation_scope") == "heldout");
    assert(!settings(R"({"id":"two","recipe_version":2})").object.count("cache_fingerprint"));
    assert(text(settings(R"({"id":"three","recipe_version":3})"), "cache_fingerprint") == "bytes-v2");
    auto five = settings(R"({"id":"run","recipe_version":5})");
    assert(text(five, "conditioning_storage") == "cache");
    five.object.erase("conditioning_storage");
    assert(training::python_equal(five, settings(R"({"id":"run","recipe_version":4})")));
    const auto online = settings(R"({"id":"v5","recipe_version":5,"teacher_storage":"online"})");
    assert(!online.object.count("teacher_cache") && online.object.at("teacher_online").boolean);
    assert(!settings(R"({"id":"run"})").object.at("keep_caches").boolean && settings(R"({"id":"run","keep_caches":true})").object.at("keep_caches").boolean);
    for (const char * request : {R"({"id":"s","ar_steps":0})", R"({"id":"s","nar_steps":-1})", R"({"id":"s","rank":2.5})", R"({"id":"s","rank":1})",
                                 R"({"id":"s","seed":true})", R"({"id":"s","window_seconds":61})", R"({"id":"s","family":"other"})",
                                 R"({"id":"s","trigger":"two words"})", R"({"id":"s","keep_caches":"yes"})", R"({"id":"s","recipe_version":6})"}) {
        assert(!rejection([&] { settings(request); }).empty());
    }

    // The capabilities list exactly what the server accepts and pins.
    const auto server = slurp(fs::path(__FILE__).parent_path().parent_path() / "tools" / "server" / "training.h");
    std::set<std::string> allowed;
    const std::regex key(R"re(key!="([a-z_]+)")re");
    for (std::sregex_iterator it(server.begin(), server.end(), key), end; it != end; ++it) allowed.insert((*it)[1]);
    const auto capabilities = training::capabilities(std::nullopt);
    std::set<std::string> listed;
    for (const auto & value : capabilities.object.at("request_settings").array) listed.insert(value.text);
    assert(listed == allowed);
    std::smatch pinned;
    const std::regex recipe(R"re(recipe_version"\]=json::parse\("(\d+)"\))re");
    const bool found = std::regex_search(server, pinned, recipe);
    assert(found);
    assert(json::stringify(capabilities.object.at("default_recipe")) == pinned[1].str());
    assert(capabilities.object.at("native").type == json::Type::null);

    Folder folder("inventory");
    const auto & dataset = folder.path();
    for (int i = 0; i < 8; ++i) {
        std::vector<float> samples(16000);
        for (std::size_t k = 0; k < samples.size(); ++k) samples[k] = static_cast<float>(std::sin(k * (i + 1) * .01));
        const auto wav = yue2::audio::encode_wav_float(samples, 8000, 1);
        write(dataset / (std::to_string(i) + ".wav"), std::string(wav.begin(), wav.end()));
        write(dataset / (std::to_string(i) + ".txt"), "synthwave, 88 bpm, D minor");
    }
    std::map<std::string, std::string> before;
    for (const auto & entry : fs::directory_iterator(dataset)) before[entry.path().filename().string()] = slurp(entry.path());
    const auto a = training::inventory(dataset, 2, 123), b = training::inventory(dataset, 2, 123);
    assert(json::stringify(a) == json::stringify(b));
    assert(std::count_if(a.array.begin(), a.array.end(), [](const json::Value & row) { return row.object.at("split").text == "validation"; }) == 1);
    for (const auto & entry : fs::directory_iterator(dataset)) assert(before.at(entry.path().filename().string()) == slurp(entry.path()));
    // Two file names cannot put one recording on both sides of the split.
    write(dataset / "duplicate.wav", slurp(dataset / "0.wav"));
    write(dataset / "duplicate.txt", "synthwave, 88 bpm, D minor");
    assert(contains(rejection([&] { training::inventory(dataset, 2, 123); }), "Duplicate recording"));

    Folder short_folder("inventory-short");
    const auto wav = yue2::audio::encode_wav_float(std::vector<float>(8000), 8000, 1);
    write(short_folder.path() / "one.wav", std::string(wav.begin(), wav.end()));
    assert(contains(rejection([&] { training::inventory(short_folder.path(), 2, 123); }), "Missing caption"));
    write(short_folder.path() / "one.txt", "synthwave, 88 bpm, D minor");
    assert(contains(rejection([&] { training::inventory(short_folder.path(), 2, 123); }), "shorter"));
}

// --- the native tools -----------------------------------------------------------------

// Notes of a Standard MIDI File: (pitch, start seconds, end seconds).
std::vector<std::tuple<int, double, double>> midi_notes(const fs::path & path) {
    const auto data = slurp(path);
    const auto byte = [&](std::size_t at) { return static_cast<unsigned char>(data.at(at)); };
    const auto be = [&](std::size_t at, int count) { unsigned long v = 0; for (int i = 0; i < count; ++i) v = (v << 8) | byte(at + i); return v; };
    assert(data.substr(0, 4) == "MThd");
    const auto tracks = be(10, 2), division = be(12, 2);
    std::size_t at = 8 + be(4, 4);
    double tempo = 500000;
    std::vector<std::tuple<int, double, double>> notes;
    for (unsigned long t = 0; t < tracks; ++t) {
        assert(data.substr(at, 4) == "MTrk");
        const auto end = at + 8 + be(at + 4, 4);
        at += 8;
        unsigned long tick = 0;
        int running = 0;
        std::map<std::pair<int, int>, unsigned long> active;
        const auto variable = [&]() { unsigned long v = 0; for (;;) { const auto b = byte(at++); v = (v << 7) | (b & 127); if (b < 128) return v; } };
        while (at < end) {
            tick += variable();
            int status = byte(at);
            if (status >= 128) { ++at; running = status; } else status = running;
            if (status == 255) {
                const int kind = byte(at++);
                const auto n = variable();
                if (kind == 81) tempo = static_cast<double>(be(at, static_cast<int>(n)));
                at += n;
            } else if (status == 240 || status == 247) {
                at += variable();
            } else {
                const int kind = status & 240, n = kind == 192 || kind == 208 ? 1 : 2;
                const int pitch = byte(at), velocity = n > 1 ? byte(at + 1) : 0;
                at += n;
                const auto key = std::make_pair(status & 15, pitch);
                if (kind == 144 && velocity > 0) active[key] = tick;
                else if ((kind == 128 || (kind == 144 && velocity == 0)) && active.count(key)) {
                    notes.emplace_back(pitch, active[key] / static_cast<double>(division) * tempo / 1e6, tick / static_cast<double>(division) * tempo / 1e6);
                    active.erase(key);
                }
            }
        }
    }
    std::sort(notes.begin(), notes.end(), [](const auto & x, const auto & y) { return std::get<1>(x) < std::get<1>(y); });
    return notes;
}

void midi_roundtrip(const fs::path & score) {
    // Absolute note times, including triplets, survive engraving and yue2-score.
    const auto events = parse(R"([{"time_seconds":0,"meter_numerator":4,"meter_denominator":4,"chord":"","notes":[{"pitch":59,"track":1,"end_time_seconds":0.3333333333333333}]},
        {"time_seconds":0.3333333333333333,"chord":"","notes":[{"pitch":64,"track":1,"end_time_seconds":0.5}]},
        {"time_seconds":0.5,"chord":"","notes":[{"pitch":63,"track":1,"end_time_seconds":1}]},
        {"time_seconds":1.5,"chord":"","notes":[{"pitch":71,"track":1,"end_time_seconds":2.5}]}])");
    const std::vector<std::tuple<int, double, double>> expected{{59, 0, 1. / 3}, {64, 1. / 3, .5}, {63, .5, 1}, {71, 1.5, 2.5}};
    for (const char * key : {"Fm", "D", "G#m"}) {
        const auto score_result = training::score_from_events(events, json::from_integer(3), 120, key);
        assert(contains(score_result.abc, "Q:1/4=120") && contains(score_result.abc, std::string("K:") + key));
        assert(number(score_result.stats.object.at("max_rounding_seconds")) <= number(score_result.stats.object.at("max_expected_grid_error_seconds")));
        Folder folder(std::string("midi-") + key);
        write(folder.path() / "input.abc", score_result.abc);
        const int code = training::run_to_log(score, {"--input", (folder.path() / "input.abc").u8string(), "--output", (folder.path() / "used.abc").u8string(),
                                                      "--midi-dir", (folder.path() / "midi").u8string()}, {}, folder.path() / "score.log");
        assert(code == 0);
        const auto notes = midi_notes(folder.path() / "midi" / "instrumental.mid");
        assert(notes.size() == expected.size());
        for (std::size_t i = 0; i < notes.size(); ++i) {
            assert(std::get<0>(notes[i]) == std::get<0>(expected[i]));
            assert(std::abs(std::get<1>(notes[i]) - std::get<1>(expected[i])) < 1e-5 && std::abs(std::get<2>(notes[i]) - std::get<2>(expected[i])) < 1e-5);
        }
    }
}

void trainer_progress_events(const fs::path & train) {
    // Cache validation and its event protocol finish before model loading,
    // which a sentinel model deliberately cannot pass.
    Folder folder("trainer-events");
    const auto model = folder.path() / "sentinel.gguf", cache = folder.path() / "sample.gguf";
    {
        FixtureGguf file("progress_test_sentinel");
        file.tensor("sentinel", GGML_TYPE_F32, {1});
        file.save(model);
    }
    {
        FixtureGguf file("yue2_nar_sample");
        file.u32("yue2.training.ar_tokens", 1);
        file.string("yue2.training.model", model.u8string());
        file.string("yue2.training.fixed_adapters", "");
        file.string("yue2.training.conditioning_adapters", "");
        file.tensor("clean", GGML_TYPE_F32, {64});
        file.tensor("keys", GGML_TYPE_F32, {128 * 8 * 28});
        file.tensor("values", GGML_TYPE_F32, {128 * 8 * 28});
        file.save(cache);
    }
    write(folder.path() / "dataset.json", "{\"format\":\"yue2-nar-prepared-dataset-v1\",\"completed\":true,\"rows\":[{\"cache\":" +
                                          json::quote(cache.u8string()) + ",\"split\":\"train\"}]}");
    int code = 0;
    const auto output = training::capture_output(train, {"--model", model.u8string(), "--dataset", (folder.path() / "dataset.json").u8string(),
                                                         "--device", "cpu", "--identity", "progress-test", "--out", (folder.path() / "run").u8string(),
                                                         "--stream-samples"}, 60, &code);
    assert(output && code != 0);
    std::vector<std::string> events;
    std::istringstream lines(*output);
    for (std::string line; std::getline(lines, line);) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.rfind("TRAIN_EVENT ", 0) != 0) continue;
        const auto event = json::parse(line.substr(12));
        events.push_back(text(event, "stage") + ":" + json::stringify(event.object.at("current")) + "/" + json::stringify(event.object.at("total")));
    }
    assert((events == std::vector<std::string>{"cache-validation:0/1", "cache-validation:1/1", "model-loading:0/1"}));
    write(folder.path() / "native.log", *output);
    assert(text(fit::NarProgress(folder.path() / "native.log", 400, 0, "fit").read(), "activity") == "Loading native NAR model");
}

void vae_targets(const fs::path & encoder, const fs::path & model, const std::string & device) {
    // Preparation resumes without re-encoding, a missing provenance record
    // never blesses an old payload, and a changed profile is refused.
    Folder folder("vae-targets");
    const auto & root = folder.path();
    std::vector<float> stereo(48000 * 3 * 2);
    for (std::size_t i = 0; i < stereo.size() / 2; ++i) {
        stereo[2 * i] = static_cast<float>(.1 * std::sin(440 * 2 * 3.141592653589793 * i / 48000));
        stereo[2 * i + 1] = static_cast<float>(.07 * std::sin(311 * 2 * 3.141592653589793 * i / 48000));
    }
    const auto wav = yue2::audio::encode_wav_float(stereo, 48000, 2);
    write(root / "source.wav", std::string(wav.begin(), wav.end()));
    write(root / "dataset.json", "{\"completed\":true,\"rows\":[{\"source\":" + json::quote((root / "source.wav").u8string()) + ",\"source_sha256\":\"" +
                                 training::sha256_file(root / "source.wav") + "\",\"track\":\"source.wav\",\"start_frame\":0,\"frames\":50,\"split\":\"train\"}]}");
    training::TargetOptions o;
    o.dataset = root / "dataset.json";
    o.vae = model;
    o.binary = encoder;
    o.out = root / "targets";
    o.device = device;
    training::prepare_targets(o);
    const auto index = json::parse(slurp(o.out / "index.json"));
    assert(index.object.at("completed").boolean);
    const auto target = fs::u8path(text(index.object.at("rows").array[0], "means"));
    const auto bytes = slurp(target);
    assert(bytes.size() == 50 * 64 * 4);
    training::prepare_targets(o);
    assert(slurp(target) == bytes);
    fs::rename(target.parent_path() / "target.json", target.parent_path() / "previous.json");
    training::prepare_targets(o);
    assert(slurp(target) == bytes);
    bool quarantined = false;
    for (const auto & entry : fs::directory_iterator(target.parent_path())) quarantined = quarantined || contains(entry.path().filename().string(), "means.unregistered-");
    assert(quarantined);
    o.core_frames = 64;
    assert(contains(rejection([&] { training::prepare_targets(o); }), "profile changed"));
    assert(slurp(target) == bytes);
}

} // namespace

int main(int argc, char ** argv) {
    const std::vector<std::string> args(argv + 1, argv + argc);
    if (!args.empty() && args[0] == "--fake-teacher-writer") return fake_teacher_writer(args);
    if (!args.empty() && args[0] == "--prepare-batch") return fake_prepare_batch(args);
    ar_progress();
    nar_progress();
    evaluations();
    selection_from_original_segment();
    pruning();
    teacher_cache();
    nar_caches();
    preflight();
    decoder_trim_is_refused();
    rolling_regeneration();
    disk_check();
    ownership_and_progress();
#ifdef _WIN32
    status_survives_reader_lock();
#endif
    settings_and_inventory();
    // Last: the workflow puts this process in its job object on Windows.
    cache_release();
    if (args.size() >= 1 && fs::exists(args[0])) midi_roundtrip(fs::u8path(args[0]));
    if (args.size() >= 2 && fs::exists(args[1])) trainer_progress_events(fs::u8path(args[1]));
    const char * model = std::getenv("YUE2_VAE_ENCODER_TEST_MODEL");
    const char * device = std::getenv("YUE2_VAE_ENCODER_TEST_DEVICE");
    if (args.size() >= 3 && fs::exists(args[2]) && model) vae_targets(fs::u8path(args[2]), fs::u8path(model), device ? device : "cuda");
    std::cout << "Native training workflow behaviour checks passed\n";
    return 0;
}
