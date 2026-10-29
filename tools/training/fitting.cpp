#include "training/fitting.h"

#include "training/dataset.h"
#include "training/fitting_detail.h"
#include "training/gguf_file.h"
#include "training/preparation.h"
#include "training/preparation_detail.h"
#include "training/process.h"
#include "training/sha256.h"
#include "training/unicode.h"
#include "training/workflow.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <thread>

namespace yue2::training {
namespace fs = std::filesystem;
using namespace detail;
namespace fitting_detail {

// Peak GPU memory over a run; advisory, so a host without nvidia-smi reports 0.
class GpuPeak {
public:
    GpuPeak() : smi_(find_program("nvidia-smi")) {}
    long long sample() {
        if (!smi_) return peak_;
        int code = 1;
        const auto output = capture_output(*smi_, {"--query-gpu=memory.used", "--format=csv,noheader,nounits"}, 3, &code);
        if (output && code == 0) {
            try { peak_ = std::max(peak_, std::stoll(output->substr(0, output->find('\n')))); } catch (const std::exception &) {}
        }
        return peak_;
    }
    long long peak() const { return peak_; }

private:
    std::optional<fs::path> smi_;
    long long peak_ = 0;
};

// Recording-balanced loss: the mean of each group's mean, in first-seen order.
double grouped_loss(const json::Value & values, const std::vector<json::Value> & rows) {
    if (values.type != json::Type::array || values.array.size() != rows.size()) throw std::invalid_argument("Metric/example count differs");
    std::vector<std::pair<std::string, std::vector<double>>> groups;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const auto & row = rows[i];
        const auto * group = row.find("group");
        if (!group) group = row.find("source_sha256");
        if (!group) group = row.find("track");
        const auto key = group ? json::stringify(*group) : "null";
        auto found = std::find_if(groups.begin(), groups.end(), [&](const auto & g) { return g.first == key; });
        if (found == groups.end()) { groups.emplace_back(key, std::vector<double>{}); found = groups.end() - 1; }
        found->second.push_back(number(values.array[i]));
    }
    if (groups.empty()) return 0.0;
    double total = 0;
    for (const auto & [key, items] : groups) {
        double sum = 0;
        for (const auto value : items) sum += value;
        total += sum / static_cast<double>(items.size());
    }
    return total / static_cast<double>(groups.size());
}

// Metric files of every segment; numeric order makes the latest resumed
// segment win at a repeated step.
std::vector<fs::path> metric_paths(const fs::path & out, const std::string & filename) {
    std::vector<std::pair<std::pair<long long, std::string>, fs::path>> paths;
    if (!fs::is_directory(out)) return {};
    static const std::regex suffix(R"(-(\d+)$)");
    for (const auto & entry : fs::directory_iterator(out)) {
        const auto name = entry.path().filename().u8string();
        if (!entry.is_directory() || name.rfind("native", 0) != 0 || !fs::exists(entry.path() / filename)) continue;
        std::smatch match;
        const long long number = std::regex_search(name, match, suffix) ? std::stoll(match[1]) : 0;
        const auto path = entry.path() / filename;
        paths.push_back({{number, python_path(path)}, path});
    }
    std::sort(paths.begin(), paths.end(), [](const auto & a, const auto & b) { return a.first < b.first; });
    std::vector<fs::path> output;
    for (const auto & [key, path] : paths) output.push_back(path);
    return output;
}

// Complete lines of a metrics file: a child may be writing the last one.
std::vector<std::string> complete_lines(const fs::path & path) {
    const auto text = read_bytes(path);
    std::vector<std::string> lines;
    std::size_t start = 0;
    for (std::size_t at; (at = text.find('\n', start)) != std::string::npos; start = at + 1) lines.push_back(text.substr(start, at - start));
    return lines;
}

std::vector<std::string> all_lines(const fs::path & path) {
    auto lines = complete_lines(path);
    const auto text = read_bytes(path);
    const auto tail = text.substr(text.rfind('\n') == std::string::npos ? 0 : text.rfind('\n') + 1);
    if (!tail.empty()) lines.push_back(tail);
    std::vector<std::string> output;
    for (auto & line : lines) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        output.push_back(line);
    }
    return output;
}

json::Value evaluation_history(const fs::path & out, const std::vector<json::Value> & rows, const std::string & expert) {
    const auto validation = with_split(rows, "validation"), retention = with_split(rows, "retention");
    std::map<long long, json::Value> by_step;
    for (const auto & path : metric_paths(out, expert == "ar" ? "metrics.jsonl" : "history.jsonl")) {
        for (const auto & line : complete_lines(path)) {
            const auto metric = json::parse(line);
            if (!metric.find("heldout_per_sample")) continue;  // NAR optimizer record.
            const double heldout = grouped_loss(member(metric, "heldout_per_sample"), validation);
            const double kept = grouped_loss(get(metric, "retention_per_sample", json::array()), retention);
            if (!std::isfinite(heldout) || !std::isfinite(kept)) throw std::invalid_argument("Nonfinite evaluation metric");
            by_step[integer(member(metric, "step"))] = make({{"step", member(metric, "step")}, {"heldout_group_mean", F(heldout)},
                                                             {"retention_group_mean", F(kept)}});
        }
    }
    auto evaluations = json::array();
    const json::Value * best = nullptr;
    for (const auto & [step, entry] : by_step) evaluations.array.push_back(entry);
    for (const auto & entry : evaluations.array) {
        if (!best || number(member(entry, "heldout_group_mean")) < number(member(*best, "heldout_group_mean"))) best = &entry;
    }
    return make({{"evaluations", evaluations}, {"best_so_far", best ? *best : json::Value{}}});
}

// Keep step zero, the heldout-best and the newest resumable checkpoint. Only
// steps with a recorded evaluation are removed: their loss is known not to be
// the minimum. Returns the bytes released.
long long prune_checkpoints(const fs::path & out, const std::vector<json::Value> & rows, const std::string & expert) {
    static const std::regex checkpoint(R"(step-(\d+)(\.state)?\.gguf)");
    std::map<long long, std::vector<fs::path>> files;
    if (!fs::is_directory(out)) return 0;
    for (const auto & folder : fs::directory_iterator(out)) {
        if (!folder.is_directory() || folder.path().filename().u8string().rfind("native", 0) != 0) continue;
        for (const auto & entry : fs::directory_iterator(folder.path())) {
            const auto name = entry.path().filename().u8string();
            std::smatch match;
            if (std::regex_match(name, match, checkpoint)) files[std::stoll(match[1])].push_back(entry.path());
        }
    }
    std::vector<long long> states;
    for (const auto & [step, paths] : files) {
        for (const auto & path : paths) if (path.filename().u8string().size() > 11 && path.filename().u8string().find(".state.gguf") != std::string::npos) { states.push_back(step); break; }
    }
    const auto history = evaluation_history(out, rows, expert);
    if (states.empty() || member(history, "best_so_far").type == json::Type::null) return 0;
    const auto latest = *std::max_element(states.begin(), states.end());
    const std::set<long long> keep{0, latest, integer(member(member(history, "best_so_far"), "step"))};
    std::set<long long> evaluated;
    for (const auto & entry : member(history, "evaluations").array) evaluated.insert(integer(member(entry, "step")));
    long long released = 0;
    for (auto & [step, paths] : files) {
        if (keep.count(step) || step > latest || !evaluated.count(step)) continue;
        // State first: an adapter is never removed while its state remains.
        std::stable_sort(paths.begin(), paths.end(), [](const fs::path & a, const fs::path & b) {
            const bool sa = a.filename().u8string().find(".state.gguf") != std::string::npos;
            const bool sb = b.filename().u8string().find(".state.gguf") != std::string::npos;
            return sa && !sb;
        });
        for (const auto & path : paths) {
            std::error_code error;
            const auto size = fs::file_size(path, error);
            if (error || !fs::remove(path, error) || error) break;  // A reader may hold it; the next pass retries.
            released += static_cast<long long>(size);
        }
    }
    return released;
}

json::Value grouped_metric(json::Value metric, const std::vector<json::Value> & validation, const std::vector<json::Value> & retention,
                           const fs::path & segment, bool optional_retention) {
    const auto retained = optional_retention ? get(metric, "retention_per_sample", json::array()) : member(metric, "retention_per_sample");
    metric.object["heldout_group_mean"] = F(grouped_loss(member(metric, "heldout_per_sample"), validation));
    metric.object["retention_group_mean"] = F(grouped_loss(retained, retention));
    metric.object["segment"] = S(python_path(segment));
    return metric;
}

json::Value first_minimum(const std::vector<json::Value> & ordered) {
    const json::Value * best = &ordered.front();
    for (const auto & metric : ordered) {
        if (number(member(metric, "heldout_group_mean")) < number(member(*best, "heldout_group_mean"))) best = &metric;
    }
    return *best;
}

std::vector<json::Value> by_step_order(const std::vector<json::Value> & metrics) {
    std::map<long long, json::Value> by_step;
    for (const auto & metric : metrics) by_step[integer(member(metric, "step"))] = metric;
    std::vector<json::Value> ordered;
    for (const auto & [step, metric] : by_step) ordered.push_back(metric);
    if (ordered.empty()) throw std::invalid_argument("min() arg is an empty sequence");
    return ordered;
}

json::Value array_of(const std::vector<json::Value> & values) {
    auto output = json::array();
    output.array = values;
    return output;
}

void select_ar(const fs::path & out, const std::vector<json::Value> & rows, const json::Value & config) {
    const auto validation = with_split(rows, "validation"), retention = with_split(rows, "retention");
    std::vector<json::Value> metrics;
    for (const auto & path : metric_paths(out, "metrics.jsonl")) {
        for (const auto & line : all_lines(path)) metrics.push_back(grouped_metric(json::parse(line), validation, retention, path.parent_path(), false));
    }
    const auto ordered = by_step_order(metrics);
    const auto selected = first_minimum(ordered);
    const auto step = integer(member(selected, "step"));
    auto folder = from_utf8(text(selected, "segment"));
    auto adapter = folder / ("step-" + std::to_string(step) + ".gguf");
    // The preflight probe supplies the explicit step-zero candidate.
    if (step == 0) {
        adapter = out / "probe" / "step-0.gguf";
    } else {
        // Resume evaluates the loaded checkpoint before updating, but that
        // checkpoint belongs to its original segment, not the new metrics file.
        std::vector<std::pair<std::string, fs::path>> candidates;
        for (const auto & entry : sorted_entries(out)) {
            const auto candidate = entry / ("step-" + std::to_string(step) + ".gguf");
            if (entry.filename().u8string().rfind("native", 0) == 0 && fs::exists(candidate)) candidates.emplace_back(python_path(candidate), candidate);
        }
        if (candidates.empty()) throw StageError("No adapter for evaluated step " + std::to_string(step));
        std::set<std::string> distinct;
        for (const auto & [name, path] : candidates) distinct.insert(digest(path));
        if (distinct.size() != 1) throw std::invalid_argument("Conflicting adapters for evaluated optimizer step");
        if (!fs::exists(adapter)) { adapter = candidates.front().second; folder = adapter.parent_path(); }
    }
    const auto state = folder / ("step-" + std::to_string(step) + ".state.gguf");
    save(out / "selection.json", make({{"method", S("Minimum recording-balanced heldout semantic CE including zero; retention separate; listening required")},
        {"initial", ordered.front()}, {"selected", selected}, {"final", ordered.back()}, {"adapter", P(adapter)},
        {"adapter_sha256", S(digest(adapter))}, {"state", step ? P(state) : json::Value{}}, {"evaluations", array_of(ordered)},
        {"subjective_verdict_pending", B(true)}}));
    std::set<std::string> captions;
    for (const auto & row : rows) {
        if (text(row, "split") == "train" && get(row, "kind", S("real")).text == "real") captions.insert(text(row, "caption"));
    }
    std::vector<std::string> ordered_captions(captions.begin(), captions.end());
    save(out / "prompts.json", make({{"version", I(1)},
        {"source", make({{"name", get(config, "name", S("Style"))}, {"unique_prompts", I(static_cast<long long>(captions.size()))}})},
        {"dice", make({{"instrumental", strings(ordered_captions)}})}}));
}

void select_nar(const fs::path & out, const std::vector<json::Value> & rows) {
    const auto validation = with_split(rows, "validation"), retention = with_split(rows, "retention");
    if (validation.empty()) throw std::invalid_argument("Heldout selection requires validation");
    std::vector<json::Value> metrics;
    for (const auto & path : metric_paths(out, "history.jsonl")) {
        for (const auto & line : all_lines(path)) {
            auto metric = json::parse(line);
            if (!metric.find("heldout_eval")) continue;
            metrics.push_back(grouped_metric(metric, validation, retention, path.parent_path(), true));
        }
    }
    const auto ordered = by_step_order(metrics);
    const auto selected = first_minimum(ordered);
    const auto step = integer(member(selected, "step"));
    const auto folder = from_utf8(text(selected, "segment"));
    const auto adapter = folder / ("step-" + std::to_string(step) + ".gguf");
    save(out / "selection.json", make({{"method", S("Minimum recording-balanced heldout flow MSE including zero; neutral retention and listening separate")},
        {"initial", ordered.front()}, {"selected", selected}, {"final", ordered.back()}, {"adapter", P(adapter)},
        {"adapter_sha256", S(digest(adapter))}, {"state", step ? P(folder / ("step-" + std::to_string(step) + ".state.gguf")) : json::Value{}},
        {"evaluations", array_of(ordered)}, {"subjective_verdict_pending", B(true)}}));
}

json::Value last_record(const fs::path & history) {
    const auto lines = all_lines(history);
    if (lines.empty()) throw std::invalid_argument("Empty history: " + python_path(history));
    return json::parse(lines.back());
}

// Teacher cache (recipe 4+): frozen KL teacher distributions on disk.
const char * const teacher_format = "yue2-ar-teacher-v1";

json::Value teacher_identity(const json::Value & run_identity, const std::string & device, long long ar_slots, const json::Value & frames,
                             const std::string & precision) {
    if (precision != "f32") throw std::invalid_argument("Unsupported AR teacher cache precision");
    return make({{"format", S(teacher_format)}, {"precision", S(precision)}, {"base_sha256", member(run_identity, "base_sha256")},
        {"companion_sha256", member(run_identity, "companion_sha256")}, {"dataset_sha256", member(run_identity, "dataset_sha256")},
        {"runtime", member(run_identity, "runtime")}, {"device", S(device)}, {"ar_slots", I(ar_slots)}, {"frames", frames},
        {"predict_end", B(false)}});
}

void verify_teacher_cache(const fs::path & folder, const std::string & identity_sha256, const std::vector<json::Value> & train_rows) {
    const auto index = read_json(folder / "index.json");
    if (get(index, "format", json::Value{}).text != std::string(teacher_format) + "-index" || !truthy(get(index, "completed", B(false))) ||
        text(index, "identity_sha256") != identity_sha256) {
        throw std::invalid_argument("AR teacher cache index does not match this run");
    }
    const auto & files = member(index, "files").array;
    if (files.size() != train_rows.size()) throw std::invalid_argument("AR teacher cache window count changed");
    for (std::size_t i = 0; i < files.size(); ++i) {
        char name[32];
        std::snprintf(name, sizeof(name), "teacher-%05zu.gguf", i);
        const auto & entry = files[i];
        if (text(entry, "path") != name || integer(member(entry, "window")) != static_cast<long long>(i) ||
            text(entry, "prefix_sha256") != text(train_rows[i], "prefix_sha256") || text(entry, "codes_sha256") != text(train_rows[i], "codes_sha256")) {
            throw std::invalid_argument("AR teacher cache window provenance changed");
        }
        if (digest(folder / text(entry, "path")) != text(entry, "sha256")) throw std::invalid_argument("AR teacher cache bytes changed");
    }
}

std::vector<std::string> teacher_args(const fs::path & folder, const std::string & identity_sha256, const json::Value & identity) {
    return {"--teacher-cache", python_path(folder), "--teacher-identity", identity_sha256, "--teacher-base-sha256", text(identity, "base_sha256"),
            "--teacher-companion-sha256", text(identity, "companion_sha256")};
}

class Stopped : public std::exception {
public:
    explicit Stopped(std::string message) : message_(std::move(message)) {}
    const char * what() const noexcept override { return message_.c_str(); }
private:
    std::string message_;
};

json::Value host_peak(const Child & child, json::Value record) {
    if (const auto peak = child.peak_memory_gib()) {
        record.object["observed_host_peak_working_set_gib"] = F(peak->first);
        record.object["observed_host_peak_private_gib"] = F(peak->second);
    }
    return record;
}

std::string prepare_teacher_cache(const fs::path & folder, std::vector<std::string> command, const fs::path & cwd, const fs::path & out,
                                  const std::string & family, const json::Value & resolved, const std::vector<json::Value> & train_rows,
                                  const fs::path & stop_file) {
    const auto identity_sha256 = sha256_hex(json::python_dumps_sorted(resolved));
    if (!fs::exists(folder)) fs::create_directory(folder);
    const auto identity_path = folder / "identity.json";
    if (fs::exists(identity_path)) {
        if (!python_equal(read_json(identity_path), resolved)) throw std::invalid_argument("AR teacher cache identity changed");
    } else {
        save(identity_path, resolved);
    }
    if (fs::exists(folder / "index.json")) return identity_sha256;
    int number = 1;
    while (fs::exists(out / ("teacher-" + std::to_string(number) + ".log"))) ++number;
    const auto log = out / ("teacher-" + std::to_string(number) + ".log");
    for (const auto & argument : teacher_args(folder, identity_sha256, resolved)) command.push_back(argument);
    command.push_back("--prepare-teacher-only");
    const auto start = Clock::now();
    ArProgress progress(log, 0, 100, std::nullopt);
    std::optional<Child> child;
    child.emplace(from_utf8(command[0]), std::vector<std::string>(command.begin() + 1, command.end()), cwd, log);
    progress.values().object["activity"] = S("Loading native teacher");
    bool stopped = false;
    while (child->running()) {
        if (fs::exists(stop_file) && !stopped) {
            // Completed teacher files are atomic; a partial file is discarded
            // and recomputed by the next preparation.
            child->terminate();
            stopped = true;
        }
        try {
            auto status = make({{"stage", S("fit")}, {"status", S("running")}, {"family", S(family)}, {"elapsed_seconds", F(seconds_since(start))}});
            for (const auto & [key, value] : progress.read().object) status.object[key] = value;
            save(out / "progress.json", status);
        } catch (const std::exception & error) {
            std::cout << "Progress update deferred: " << error.what() << std::endl;
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    const int code = child->wait();
    save(with_suffix(log, ".command.json"), host_peak(*child, make({{"command", strings(command)}, {"returncode", I(code)},
                                                                    {"seconds", F(seconds_since(start))}})));
    if (stopped) throw Stopped("AR teacher preparation stopped");
    if (code) throw std::runtime_error(log_tail(log, 5000));
    std::vector<std::string> names, expected;
    for (const auto & entry : fs::directory_iterator(folder)) {
        const auto name = entry.path().filename().u8string();
        if (name.rfind("teacher-", 0) == 0 && name.size() > 5 && name.substr(name.size() - 5) == ".gguf") names.push_back(name);
    }
    std::sort(names.begin(), names.end());
    for (std::size_t i = 0; i < train_rows.size(); ++i) {
        char name[32];
        std::snprintf(name, sizeof(name), "teacher-%05zu.gguf", i);
        expected.emplace_back(name);
    }
    if (names != expected) throw std::invalid_argument("Unexpected AR teacher cache files");
    auto files = json::array();
    long long total = 0;
    for (std::size_t i = 0; i < train_rows.size(); ++i) {
        const auto path = folder / expected[i];
        const auto size = static_cast<long long>(fs::file_size(path));
        total += size;
        files.array.push_back(make({{"path", S(expected[i])}, {"window", I(static_cast<long long>(i))}, {"sha256", S(digest(path))},
            {"bytes", I(size)}, {"prefix_sha256", member(train_rows[i], "prefix_sha256")}, {"codes_sha256", member(train_rows[i], "codes_sha256")}}));
    }
    save(folder / "index.json", make({{"format", S(std::string(teacher_format) + "-index")}, {"completed", B(true)},
        {"identity_sha256", S(identity_sha256)}, {"base_sha256", member(resolved, "base_sha256")},
        {"companion_sha256", member(resolved, "companion_sha256")}, {"frames", member(resolved, "frames")},
        {"ar_slots", member(resolved, "ar_slots")}, {"bytes", I(total)}, {"files", files}}));
    return identity_sha256;
}

json::Value load_job(const fs::path & path, const char * format, const char * message) {
    auto config = read_json(path);
    if (text(config, "format") != format) throw std::invalid_argument(message);
    set_environment("GGML_CUDA_TF32", "0");
    return config;
}

void check_segment(const std::string & segment) {
    if (from_utf8(segment).filename().u8string() != segment) throw std::invalid_argument("Segment must be a directory name");
}

json::Value merge(json::Value base, const json::Value & extra) {
    for (const auto & [key, value] : extra.object) base.object[key] = value;
    return base;
}

std::vector<std::string> paths_of(const std::vector<json::Value> & rows, std::initializer_list<const char *> keys,
                                  std::vector<fs::path> & files) {
    std::vector<std::string> expected;
    for (const auto & row : rows) {
        for (const char * key : keys) {
            files.push_back(from_utf8(text(row, key)));
            expected.push_back(text(row, (std::string(key) + "_sha256").c_str()));
        }
    }
    return expected;
}

void check_examples(const std::vector<json::Value> & rows, const char * message, bool during) {
    std::vector<fs::path> files;
    const auto expected = paths_of(rows, {"prefix", "codes"}, files);
    const auto actual = sha256_files(files);
    if (actual != expected) {
        if (during) throw std::runtime_error(message);
        throw std::invalid_argument(message);
    }
}

} // namespace fitting_detail

using namespace fitting_detail;

long long state_step(const fs::path & state) {
    const GgufFile file(state);
    const auto step = file.field("yue2.training.step");
    if (!step) throw std::invalid_argument("Training state has no step: " + python_path(state));
    return integer(*step);
}

void fit_ar(const FitOptions & o) {
    const auto config = load_job(o.config, "yue2-ar-training-v1", "Unsupported training job");
    if (o.family != "lora" && o.family != "dora") throw std::invalid_argument("family must be lora or dora");
    if (o.phase != "probe" && o.phase != "fit" && o.phase != "post-probe" && o.phase != "select") throw std::invalid_argument("Unknown phase");
    check_segment(o.segment);
    const auto root = from_utf8(text(config, "workspace"));
    // The trainers run from the workspace (the native tools' folder), so every
    // path handed to them is absolute.
    const auto out = from_utf8(resolved_path(from_utf8(text(config, "output")))) / o.family;
    if (!fs::exists(out)) fs::create_directory(out);
    const auto dataset = from_utf8(resolved_path(from_utf8(text(config, "training_dataset"))));
    const auto data = read_json(dataset);
    if (!truthy(member(data, "completed"))) throw std::invalid_argument("Dataset is not ready");
    const auto rows = eligible(data);
    check_examples(rows, "Example content changed", false);
    const auto & training = member(config, "training");
    const auto binary = from_utf8(text(config, "training_binary"));
    const auto & models = member(config, "models");
    const auto model = from_utf8(text(models, "base")), companion = from_utf8(text(models, "companion"));
    const auto hashes = sha256_files({dataset, model, companion});
    const auto identity = make({{"format", S("yue2-ar-run-v1")}, {"family", S(o.family)}, {"training", training},
        {"dataset_sha256", S(hashes[0])}, {"base_sha256", S(hashes[1])}, {"companion_sha256", S(hashes[2])},
        {"runtime", runtime_identity(binary)}});
    const auto identity_path = out / "identity.json";
    if (fs::exists(identity_path)) {
        if (!python_equal(read_json(identity_path), identity)) throw std::invalid_argument("Resolved run identity changed");
    } else {
        save(identity_path, identity);
    }
    if (o.phase == "select") { select_ar(out, rows, config); return; }
    const auto run_hash = sha256_hex(json::python_dumps_sorted(identity));
    long long prefix_slots = 0;
    for (const auto & row : rows) prefix_slots = std::max(prefix_slots, static_cast<long long>(fs::file_size(from_utf8(text(row, "prefix"))) / 4));
    const long long frames = integer(member(data, "frames"));
    const auto target_steps = static_cast<long long>(std::ceil(number(member(training, "token_budget")) / static_cast<double>(frames)));
    const auto device = get(config, "device", S("cuda")).text;
    std::vector<std::string> cmd{python_path(binary), "--model", python_path(model), "--fixed-lora", python_path(companion), "--dataset",
        python_path(dataset), "--group-samples", "--shuffle-samples", "--max-prefix", py_str(prefix_slots), "--device", device,
        "--rank", py_str(member(training, "rank")), "--lr", py_str(member(training, "learning_rate")),
        "--kl-weight", py_str(member(training, "kl_weight")), "--weight-decay", py_str(member(training, "weight_decay")),
        "--clip", py_str(member(training, "clip")), "--seed", py_str(get(training, "seed", I(831001))), "--identity", run_hash};
    if (o.family == "dora") cmd.push_back("--dora");
    const bool fitting = o.phase == "fit";
    const bool kl = number(member(training, "kl_weight")) > 0;
    if (fitting && training.find("evaluation_scope")) { cmd.push_back("--eval-scope"); cmd.push_back(py_str(member(training, "evaluation_scope"))); }
    // Recipe v5 online teachers: one common-model forward per update, no cache.
    if (fitting && get(training, "teacher", json::Value{}).text == "online" && kl) cmd.push_back("--teacher-online");
    long long resumed_step = 0;
    // Recipe v4+: stream frozen teacher distributions from a SHA-256-indexed
    // disk cache instead of preparing all of them in fitter RAM each start.
    std::optional<fs::path> teacher;
    json::Value resolved_teacher;
    if (fitting && truthy(get(training, "teacher_cache", json::Value{})) && kl) {
        teacher = out / "teacher";
        resolved_teacher = teacher_identity(identity, device, prefix_slots, member(data, "frames"), text(training, "teacher_cache"));
    }
    fs::path native;
    const auto stop_file = out / "stop.request";
    const auto eval_every = get(training, "eval_every", I(100));
    std::size_t teacher_cut = 0;
    if (fitting) {
        native = out / from_utf8(o.segment);
        teacher_cut = cmd.size();
        for (const auto & value : {std::string("--out"), python_path(native), std::string("--save-every"), py_str(get(training, "checkpoint_every", I(100))),
                                   std::string("--eval-every"), py_str(eval_every), std::string("--stop-file"), python_path(stop_file)}) {
            cmd.push_back(value);
        }
        if (o.resume) {
            resumed_step = state_step(*o.resume);
            cmd.push_back("--resume");
            cmd.push_back(resolved_path(*o.resume));
        }
        if (resumed_step >= target_steps) throw std::invalid_argument("Target budget already reached");
        cmd.push_back("--steps");
        cmd.push_back(py_str(target_steps - resumed_step));
    } else {
        native = out / o.phase;
        if (o.phase == "post-probe") {
            const auto selected = read_json(out / "selection.json");
            if (integer(member(member(selected, "selected"), "step")) == 0) throw std::invalid_argument("No learned checkpoint selected");
            cmd.push_back("--resume");
            cmd.push_back(text(selected, "state"));
        }
        for (const auto & value : {std::string("--out"), python_path(native), std::string("--probe-only"), std::string("--diagnostics"),
                                   python_path(out / (o.phase + "-diagnostics.gguf"))}) {
            cmd.push_back(value);
        }
    }
    const auto log = out / (o.phase + (fitting ? "-" + o.segment : "") + ".log");
    if (fs::exists(log) || fs::exists(native)) throw StageError("Run segment already exists");
    if (truthy(get(config, "inference_urls", json::array()))) throw StageError("Waiting for inference_urls is not supported by yue2-train-run");
    const auto train_rows = with_split(rows, "train");
    std::string teacher_sha;
    if (teacher) {
        const bool indexed = fs::exists(*teacher / "index.json");
        if (indexed) {
            save(out / "progress.json", make({{"stage", S(o.phase)}, {"status", S("running")}, {"family", S(o.family)},
                {"target_steps", I(target_steps)}, {"step", I(resumed_step)}, {"activity", S("Verifying frozen teacher cache")}}));
        }
        teacher_sha = prepare_teacher_cache(*teacher, std::vector<std::string>(cmd.begin(), cmd.begin() + static_cast<long>(teacher_cut)),
                                            root, out, o.family, resolved_teacher, train_rows, stop_file);
        // A fresh index was just hashed from these bytes; an existing one is rechecked.
        if (indexed) verify_teacher_cache(*teacher, teacher_sha, train_rows);
        for (const auto & argument : teacher_args(*teacher, teacher_sha, identity)) cmd.push_back(argument);
    }
    const auto start = Clock::now();
    GpuPeak gpu;
    save(out / "run.json", make({{"format", S("yue2-ar-run-status-v1")}, {"name", S(get(config, "name", S("Style")).text + " " + o.family)},
        {"family", S(o.family)}, {"stage", S(o.phase)}, {"status", S("running")}, {"target_steps", I(target_steps)},
        {"identity_sha256", S(run_hash)}, {"output", P(native)}}));
    std::cout << "START " << o.family << " " << o.phase << std::endl;
    ArProgress progress(log, resumed_step, integer(eval_every), target_steps);
    std::optional<Child> child;
    child.emplace(from_utf8(cmd[0]), std::vector<std::string>(cmd.begin() + 1, cmd.end()), root, log);
    // Replace terminal progress from the previous segment before the native
    // teacher/evaluation warmup, which can take several minutes on resume.
    save(out / "progress.json", merge(make({{"stage", S(o.phase)}, {"status", S("running")}, {"family", S(o.family)},
                                            {"target_steps", I(target_steps)}}), progress.values()));
    long long step = 0;
    while (child->running()) {
        gpu.sample();
        try {
            auto current = progress.read();
            if (fitting) current.object["evaluation_history"] = evaluation_history(out, rows, "ar");
            if (fitting && o.prune_checkpoints) prune_checkpoints(out, rows, "ar");
            const auto history = native / "history.jsonl";
            if (fs::exists(history)) {
                const auto lines = all_lines(history);
                if (!lines.empty()) {
                    try {
                        const auto latest = json::parse(lines.back());
                        step = latest.find("step") ? integer(member(latest, "step")) : step;
                        current.object["step"] = I(step);
                        current.object["last_update_seconds"] = get(latest, "seconds", json::Value{});
                    } catch (const std::runtime_error &) {}  // Child may be appending this line.
                }
            }
            save(out / "progress.json", merge(make({{"stage", S(o.phase)}, {"status", S("running")}, {"family", S(o.family)},
                {"target_steps", I(target_steps)}, {"tokens_seen", I(integer(member(current, "step")) * frames)},
                {"elapsed_seconds", F(seconds_since(start))}, {"observed_gpu_peak_mib", I(gpu.peak())}}), current));
        } catch (const std::exception & error) {
            // Progress is advisory. A reader lock or disk error must not drop
            // ownership of the still-running native process.
            std::cout << "Progress update deferred: " << error.what() << std::endl;
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    const int code = child->wait();
    const auto record = host_peak(*child, make({{"command", strings(cmd)}, {"returncode", I(code)}, {"seconds", F(seconds_since(start))},
                                                {"observed_gpu_peak_mib", I(gpu.peak())}}));
    save(with_suffix(log, ".command.json"), record);
    if (code) {
        save(out / "run.json", make({{"status", S("failed")}, {"stage", S(o.phase)}, {"family", S(o.family)}, {"command", record}}));
        throw std::runtime_error(log_tail(log, 5000));
    }
    if (digest(dataset) != text(identity, "dataset_sha256")) throw std::runtime_error("Dataset manifest changed during the run");
    check_examples(rows, "Example changed during the run", true);
    if (teacher) verify_teacher_cache(*teacher, teacher_sha, train_rows);
    if (fitting) {
        select_ar(out, rows, config);
        if (o.prune_checkpoints) prune_checkpoints(out, rows, "ar");
    }
    save(out / "run.json", make({{"status", S(fs::exists(stop_file) ? "stopped" : "completed")}, {"stage", S(o.phase)}, {"family", S(o.family)},
        {"elapsed_seconds", F(seconds_since(start))}, {"observed_gpu_peak_mib", I(gpu.peak())}, {"identity_sha256", S(run_hash)},
        {"output", P(native)}}));
    if (fitting) {
        const auto final_step = integer(member(last_record(native / "history.jsonl"), "step"));
        save(out / "progress.json", make({{"stage", S("fit")}, {"status", S(final_step >= target_steps ? "completed" : "stopped")},
            {"family", S(o.family)}, {"step", I(final_step)}, {"target_steps", I(target_steps)}, {"tokens_seen", I(final_step * frames)},
            {"elapsed_seconds", F(seconds_since(start))}, {"observed_gpu_peak_mib", I(gpu.peak())},
            {"evaluation_history", evaluation_history(out, rows, "ar")}}));
    }
    std::cout << "END " << o.family << " " << o.phase << " " << json::stringify(record) << std::endl;
}

namespace fitting_detail {

// train_nar_run.preflight_sample(): read GGUF headers only, then probe one real
// sample at the corpus maximum.
std::pair<fs::path, long long> preflight_sample(const std::vector<json::Value> & rows, const std::function<fs::path(const json::Value &)> & cache_path,
                                                long long frames) {
    long long largest = 0;
    std::optional<fs::path> first;
    for (const auto & row : rows) {
        const GgufFile file(cache_path(row));
        const auto tokens = integer(file.field("yue2.training.ar_tokens").value_or(I(0)));
        if (tokens < 1 || static_cast<long long>(file.elements("clean")) != frames * 64 || file.tensor("clean").type != GGML_TYPE_F32) {
            throw std::invalid_argument("Invalid NAR preflight cache shape or target precision");
        }
        for (const char * key : {"keys", "values"}) {
            const auto type = file.tensor(key).type;
            if (static_cast<long long>(file.elements(key)) != tokens * 128 * 8 * 28 || (type != GGML_TYPE_F16 && type != GGML_TYPE_F32)) {
                throw std::invalid_argument("Invalid NAR preflight conditioning shape or precision");
            }
        }
        largest = std::max(largest, tokens);
        if (!first && text(row, "split") == "train") first = cache_path(row);
    }
    if (!first) throw std::invalid_argument("NAR preflight requires a training sample");
    return {*first, largest};
}

void remove_pair(std::vector<std::string> & command, const char * flag) {
    const auto found = std::find(command.begin(), command.end(), flag);
    if (found != command.end()) command.erase(found, found + 2);
}

} // namespace fitting_detail

void fit_nar(const FitOptions & o) {
    const auto config = load_job(o.config, "yue2-nar-training-v1", "Unsupported NAR job");
    if (o.phase != "probe" && o.phase != "fit" && o.phase != "post-probe" && o.phase != "select") throw std::invalid_argument("Unknown phase");
    check_segment(o.segment);
    // The trainer runs from the workspace, so every path handed to it is absolute.
    const auto out = from_utf8(resolved_path(from_utf8(text(config, "output"))));
    fs::create_directories(out);
    const auto manifest = from_utf8(resolved_path(from_utf8(text(config, "training_dataset"))));
    const auto data = read_json(manifest);
    if (text(data, "format") != "yue2-nar-prepared-dataset-v1" || !truthy(member(data, "completed"))) {
        throw std::invalid_argument("NAR data is incomplete");
    }
    const auto rows = eligible(data);
    const auto cache_path = [&](const json::Value & row) {
        const auto path = from_utf8(text(row, "cache"));
        return path.is_absolute() ? path : manifest.parent_path() / path;
    };
    const auto & models = member(config, "models");
    const auto & training = member(config, "training");
    const auto binary = from_utf8(text(config, "training_binary"));
    // Recipe 5: cached K/V files, rolling files regenerated byte-identically
    // just before use, or resident conditioning computed in the fitter.
    const auto storage = get(training, "conditioning_storage", S("cache")).text;
    const auto cache_identity = get(data, "identity", json::object());
    if (get(cache_identity, "storage", S("cache")).text != storage) {
        throw std::invalid_argument("NAR dataset was prepared for another conditioning storage");
    }
    const auto verify_caches = [&](const char * message) {
        // Rolling caches are checked as each is regenerated; resident has none.
        if (storage != "cache") return;
        std::vector<fs::path> files;
        for (const auto & row : rows) files.push_back(cache_path(row));
        const auto values = sha256_files(files);
        for (std::size_t i = 0; i < rows.size(); ++i) if (values[i] != text(rows[i], "cache_sha256")) throw std::invalid_argument(message);
    };
    verify_caches("NAR cache bytes changed");
    if (number(get(training, "weight_decay", F(.01))) != .01 || number(get(training, "clip", I(1))) != 1) {
        throw std::invalid_argument("This native CLI pins NAR weight decay .01 and clipping 1");
    }
    const auto model_hashes = sha256_files({from_utf8(text(models, "base")), from_utf8(text(models, "companion")), from_utf8(text(models, "ar"))});
    const auto identity = make({{"format", S("yue2-nar-run-v1")}, {"training", training}, {"dataset_sha256", S(digest(manifest))},
        {"models", make({{"base", S(model_hashes[0])}, {"companion", S(model_hashes[1])}, {"ar", S(model_hashes[2])}})},
        {"runtime", runtime_identity(binary)}, {"device", get(config, "device", S("cuda"))}, {"threads", get(config, "threads", I(6))}});
    for (const auto & [key, field] : std::vector<std::pair<std::string, std::string>>{{"base", "base_sha256"}, {"companion", "companion_sha256"}, {"ar", "ar_adapter_sha256"}}) {
        if (const auto * value = cache_identity.find(field); value && !python_equal(*value, member(member(identity, "models"), key.c_str()))) {
            throw std::invalid_argument("Frozen conditioning belongs to different " + key + " weights");
        }
    }
    const auto identity_path = out / "identity.json";
    if (fs::exists(identity_path) && !python_equal(read_json(identity_path), identity)) throw std::invalid_argument("Resolved NAR identity changed");
    save(identity_path, identity);
    if (o.phase == "select") { select_nar(out, rows); return; }
    const auto run_hash = sha256_hex(json::python_dumps_sorted(identity));
    std::vector<std::string> command{python_path(binary), "--model", text(models, "base"), "--conditioning-lora", text(models, "companion"),
        "--conditioning-lora", text(models, "ar"), "--dataset", python_path(manifest), "--stream-samples", "--shuffle-samples", "--group-samples",
        "--f32-projections", "--noise-schedule", py_str(get(training, "noise_schedule", S("uniform"))), "--rank", py_str(member(training, "rank")),
        "--lr", py_str(member(training, "learning_rate")), "--seed", py_str(get(training, "seed", I(831001))), "--identity", run_hash,
        "--device", py_str(get(config, "device", S("cuda"))), "--threads", py_str(get(config, "threads", I(6)))};
    if (storage == "resident") command.push_back("--resident-conditioning");
    if (storage == "rolling") command.push_back("--rolling-caches");
    const auto train_rows = with_split(rows, "train");
    const auto stop_file = out / "stop.request";
    const auto regenerate = [&](const std::vector<json::Value> & needed) {
        return materialize(manifest, needed, binary, from_utf8(text(models, "base")), from_utf8(text(models, "companion")),
                           from_utf8(text(models, "ar")), stop_file);
    };
    if (o.phase == "probe" && storage != "cache") {
        // Dataset-mode probe: the first training window at the corpus maximum
        // context. A resident probe also measures the AR conditioner's VRAM.
        if (storage == "rolling") regenerate(std::vector<json::Value>(train_rows.begin(), train_rows.begin() + std::min<std::size_t>(1, train_rows.size())));
    } else if (o.phase == "probe" && get(training, "probe_scope", json::Value{}).text == "first-at-maximum-context") {
        const auto [first, capacity] = preflight_sample(rows, cache_path, integer(member(data, "frames")));
        remove_pair(command, "--dataset");
        command.erase(std::find(command.begin(), command.end(), "--group-samples"));
        for (const auto & value : {std::string("--sample"), python_path(first), std::string("--probe-ar-slots"), py_str(capacity)}) command.push_back(value);
    }
    if (o.phase == "fit" && training.find("evaluation_scope")) { command.push_back("--eval-scope"); command.push_back(py_str(member(training, "evaluation_scope"))); }
    if ((o.phase == "fit" || o.phase == "post-probe" || storage == "rolling") && training.find("cache_fingerprint")) {
        command.push_back("--cache-fingerprint");
        command.push_back(py_str(member(training, "cache_fingerprint")));
    }
    const long long target = integer(member(training, "steps"));
    long long resumed = 0;
    fs::path native;
    const auto eval_every = integer(get(training, "eval_every", I(100)));
    if (o.phase == "fit") {
        native = out / from_utf8(o.segment);
        if (o.resume) {
            resumed = state_step(*o.resume);
            command.push_back("--resume");
            command.push_back(resolved_path(*o.resume));
        }
        if (resumed >= target) throw std::invalid_argument("Target budget already reached");
        for (const auto & value : {std::string("--out"), python_path(native), std::string("--steps"), py_str(target - resumed),
                                   std::string("--save-every"), py_str(get(training, "checkpoint_every", I(100))), std::string("--eval-every"),
                                   py_str(get(training, "eval_every", I(100))), std::string("--stop-file"), python_path(stop_file)}) {
            command.push_back(value);
        }
    } else {
        native = out / o.phase;
        if (o.phase == "post-probe") {
            const auto selected = read_json(out / "selection.json");
            if (!truthy(get(selected, "state", json::Value{}))) throw std::invalid_argument("No learned checkpoint selected");
            command.push_back("--resume");
            command.push_back(text(selected, "state"));
        }
        for (const auto & value : {std::string("--out"), python_path(native), std::string("--probe-only"), std::string("--diagnostics"),
                                   python_path(out / (o.phase + "-diagnostics.gguf"))}) {
            command.push_back(value);
        }
    }
    const auto log = out / (o.phase + (o.phase == "fit" ? "-" + o.segment : "") + ".log");
    if (fs::exists(log) || fs::exists(native)) throw StageError("NAR segment already exists");
    if (truthy(get(config, "inference_urls", json::array()))) throw StageError("Waiting for inference_urls is not supported by yue2-train-run");
    const auto start = Clock::now();
    const long long frames = integer(member(data, "frames"));
    GpuPeak gpu;
    const auto workspace = from_utf8(text(config, "workspace"));
    // One native invocation, publishing progress until it exits.
    const auto run_native = [&](const std::vector<std::string> & segment_command, const fs::path & segment_native, const fs::path & segment_log,
                                long long segment_resumed) {
        save(out / "run.json", make({{"status", S("running")}, {"stage", S(o.phase)}, {"name", get(config, "name", S("NAR"))},
                                     {"identity_sha256", S(run_hash)}, {"output", P(segment_native)}}));
        std::cout << "START NAR " << o.phase << std::endl;
        NarProgress progress(segment_log, target, segment_resumed, o.phase);
        save(out / "progress.json", merge(make({{"stage", S(o.phase)}, {"status", S("running")}, {"target_steps", I(target)}}), progress.values()));
        Child child(from_utf8(segment_command[0]), std::vector<std::string>(segment_command.begin() + 1, segment_command.end()), workspace, segment_log);
        try {
            save(out / "process.json", make({{"pid", I(child.pid())}, {"phase", S(o.phase)}, {"native", P(segment_native)}}));
        } catch (const std::exception & error) {
            std::cout << "Process status deferred: " << error.what() << std::endl;
        }
        while (child.running()) {
            gpu.sample();
            try {
                auto current = progress.read();
                const auto history = segment_native / "history.jsonl";
                if (fs::exists(history)) {
                    const auto lines = all_lines(history);
                    if (!lines.empty()) {
                        const auto step = integer(member(json::parse(lines.back()), "step"));
                        const auto evaluations = evaluation_history(out, rows, "nar");
                        bool evaluated = false;
                        for (const auto & entry : member(evaluations, "evaluations").array) evaluated = evaluated || integer(member(entry, "step")) == step;
                        if (!progress.events()) {
                            const char * activity = step == 0 ? "Initial evaluation"
                                : step == target ? (evaluated ? "Saving final checkpoint" : "Final evaluation")
                                : step % eval_every == 0 ? (evaluated ? "Saving checkpoint" : "Evaluating checkpoint") : "Training";
                            progress.activity(activity);
                            current = progress.read();
                        }
                        current.object["step"] = I(step);
                        current.object["evaluation_history"] = evaluations;
                        if (o.phase == "fit" && o.prune_checkpoints) prune_checkpoints(out, rows, "nar");
                    }
                }
                save(out / "progress.json", merge(make({{"stage", S(o.phase)}, {"status", S("running")}, {"target_steps", I(target)},
                    {"tokens_seen", I(integer(member(current, "step")) * frames)}, {"elapsed_seconds", F(seconds_since(start))},
                    {"observed_gpu_peak_mib", I(gpu.peak())}}), current));
            } catch (const std::exception &) {}
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        const int code = child.wait();
        const auto record = make({{"command", strings(segment_command)}, {"returncode", I(code)}, {"seconds", F(seconds_since(start))},
                                  {"observed_gpu_peak_mib", I(gpu.peak())}});
        save(with_suffix(segment_log, ".command.json"), record);
        if (code) {
            save(out / "run.json", make({{"status", S("failed")}, {"stage", S(o.phase)}, {"command", record}}));
            throw std::runtime_error(log_tail(segment_log, 5000));
        }
        return record;
    };
    json::Value record;
    if (o.phase == "fit" && storage == "rolling") {
        // Fit in segments whose caches exist only while needed. Before each
        // segment the native sampler reports exactly which training windows
        // it will draw; those (and the heldout set when the segment ends on an
        // evaluation) are regenerated byte-identically and everything else is
        // deleted. Segments end on evaluation boundaries and never evaluate
        // early, so updates, evaluations and checkpoints match one run.
        const auto split = std::find(command.begin(), command.end(), "--rolling-caches") + 1;
        const std::vector<std::string> base(command.begin(), split);
        std::vector<std::string> fit(split, command.end());
        remove_pair(fit, "--steps");
        remove_pair(fit, "--resume");
        remove_pair(fit, "--out");
        const auto window = integer(get(training, "rolling_window_updates", I(25)));
        std::vector<json::Value> held;
        for (const auto & row : rows) {
            if (get(training, "evaluation_scope", json::Value{}).text != "heldout" || text(row, "split") != "train") held.push_back(row);
        }
        static const std::regex name_parts(R"((.*?)-?(\d*))");
        std::smatch parts;
        const auto native_name = native.filename().u8string();
        std::regex_match(native_name, parts, name_parts);
        const std::string stem = parts[1];
        long long segment_number = parts[2].length() ? std::stoll(parts[2]) : 1;
        auto resume = o.resume;
        long long step = resumed;
        std::optional<json::Value> last;
        auto current = native;
        const auto remove_caches = [&]() {
            for (const auto & row : rows) { std::error_code error; fs::remove(from_utf8(text(row, "cache")), error); }
        };
        try {
            while (step < target && !fs::exists(stop_file)) {
                const auto end = std::min({target, step + window, (step / eval_every + 1) * eval_every});
                auto plan_command = base;
                plan_command.push_back("--sample-plan");
                plan_command.push_back(py_str(end - step));
                if (resume) { plan_command.push_back("--resume"); plan_command.push_back(python_path(*resume)); }
                int code = 1;
                const auto planned = capture_output(from_utf8(plan_command[0]), std::vector<std::string>(plan_command.begin() + 1, plan_command.end()),
                                                    24 * 3600, &code);
                std::optional<std::string> line;
                if (planned) {
                    std::size_t at = 0;
                    for (std::size_t next; at <= planned->size(); at = next + 1) {
                        next = std::min(planned->find('\n', at), planned->size());
                        auto candidate = planned->substr(at, next - at);
                        if (!candidate.empty() && candidate.back() == '\r') candidate.pop_back();
                        if (candidate.rfind("SAMPLE_PLAN ", 0) == 0) { line = candidate; break; }
                    }
                }
                if (code || !line) {
                    const auto tail = planned ? planned->substr(planned->size() > 2000 ? planned->size() - 2000 : 0) : std::string();
                    throw std::runtime_error("Sample plan failed: " + tail);
                }
                const auto plan = json::parse(line->substr(12));
                if (integer(member(plan, "step")) != step) throw std::invalid_argument("Sample plan does not start at the resumed step");
                std::vector<json::Value> needed;
                for (const auto & index : member(plan, "samples").array) needed.push_back(train_rows.at(static_cast<std::size_t>(integer(index))));
                // A fresh fit evaluates step zero; segments end on an
                // evaluation step (or the final update, which always evaluates).
                if ((step == 0 && !resume) || end % eval_every == 0 || end == target) needed.insert(needed.end(), held.begin(), held.end());
                std::set<std::string> keep;
                for (const auto & row : needed) keep.insert(resolved_path(from_utf8(text(row, "cache"))));
                for (const auto & row : rows) {
                    const auto path = from_utf8(text(row, "cache"));
                    if (fs::exists(path) && !keep.count(resolved_path(path))) fs::remove(path);
                }
                regenerate(needed);
                if (fs::exists(stop_file)) break;
                current = native.parent_path() / (stem + "-" + std::to_string(segment_number++));
                auto segment = base;
                segment.insert(segment.end(), fit.begin(), fit.end());
                for (const auto & value : {std::string("--out"), python_path(current), std::string("--steps"), py_str(end - step)}) segment.push_back(value);
                if (resume) { segment.push_back("--resume"); segment.push_back(python_path(*resume)); }
                if (end < target) segment.push_back("--no-final-eval");
                last = run_native(segment, current, out / ("fit-" + current.filename().u8string() + ".log"), step);
                std::vector<std::pair<long long, fs::path>> states;
                for (const auto & entry : fs::directory_iterator(current)) {
                    const auto name = entry.path().filename().u8string();
                    if (name.rfind("step-", 0) == 0 && name.size() > 11 && name.substr(name.size() - 11) == ".state.gguf") {
                        states.emplace_back(std::stoll(name.substr(5)), entry.path());
                    }
                }
                if (states.empty()) throw std::runtime_error("Rolling segment saved no checkpoint");
                std::sort(states.begin(), states.end());
                resume = states.back().second;
                step = state_step(*resume);
            }
        } catch (...) {
            remove_caches();
            throw;
        }
        remove_caches();
        if (!last) throw std::runtime_error("Rolling fit stopped before its first segment");
        record = *last;
        native = current;
    } else {
        record = run_native(command, native, log, resumed);
    }
    if (digest(manifest) != text(identity, "dataset_sha256")) throw std::invalid_argument("NAR manifest changed during fitting");
    verify_caches("NAR cache changed during fitting");
    if (o.phase == "fit") {
        select_nar(out, rows);
        if (o.prune_checkpoints) prune_checkpoints(out, rows, "nar");
        const auto step = integer(member(last_record(native / "history.jsonl"), "step"));
        save(out / "progress.json", make({{"stage", S("fit")}, {"status", S(step >= target ? "completed" : "stopped")}, {"step", I(step)},
            {"target_steps", I(target)}, {"tokens_seen", I(step * frames)}, {"observed_gpu_peak_mib", I(gpu.peak())},
            {"evaluation_history", evaluation_history(out, rows, "nar")}}));
    }
    save(out / "run.json", make({{"status", S(fs::exists(stop_file) ? "stopped" : "completed")}, {"stage", S(o.phase)},
        {"elapsed_seconds", F(seconds_since(start))}, {"observed_gpu_peak_mib", I(gpu.peak())}, {"identity_sha256", S(run_hash)},
        {"output", P(native)}}));
    std::cout << "END NAR " << o.phase << " " << json::stringify(record) << std::endl;
}

} // namespace yue2::training
