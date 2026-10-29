#include "training/driver.h"

#include "training/fitting.h"
#include "training/preparation.h"
#include "training/preparation_detail.h"
#include "training/process.h"
#include "training/publication.h"
#include "training/workflow.h"

#include <chrono>
#include <cmath>
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif
#include <cstdio>
#include <iostream>
#include <thread>

namespace yue2::training {
namespace fs = std::filesystem;
using namespace detail;
namespace {

constexpr double gib = 1073741824.0;

class Stopped : public std::exception {};

double now() { return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count(); }

std::string suffix(const char * name) { return executable(name); }

// Never touch the device before the server commits its ownership record.
void wait_for_owner(const fs::path & run) {
    const auto pid = static_cast<long long>(
#ifdef _WIN32
        _getpid()
#else
        getpid()
#endif
    );
    const auto identity = own_process_identity();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    for (;;) {
        try {
            const auto owner = read_json(run / "owner.json");
            if (integer(member(owner, "pid")) == pid && text(owner, "identity") == identity) return;
        } catch (const std::exception &) {}
        if (std::chrono::steady_clock::now() >= deadline) throw std::runtime_error("Training worker ownership was not committed");
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

long long folder_bytes(const fs::path & folder) {
    long long total = 0;
    if (!fs::exists(folder)) return 0;
    for (const auto & entry : fs::recursive_directory_iterator(folder)) if (entry.is_regular_file()) total += static_cast<long long>(entry.file_size());
    return total;
}

// Delete files best effort; return the bytes actually released.
long long release(const std::vector<fs::path> & paths) {
    long long released = 0;
    for (const auto & path : paths) {
        std::error_code error;
        const auto size = fs::file_size(path, error);
        if (error) continue;
        if (fs::remove(path, error) && !error) released += static_cast<long long>(size);
    }
    return released;
}

std::pair<long long, std::optional<fs::path>> latest_state(const fs::path & folder) {
    std::pair<long long, std::optional<fs::path>> best{0, std::nullopt};
    if (!fs::is_directory(folder)) return best;
    for (const auto & segment : sorted_entries(folder)) {
        if (!fs::is_directory(segment) || segment.filename().u8string().rfind("native", 0) != 0) continue;
        for (const auto & entry : sorted_entries(segment)) {
            const auto name = entry.filename().u8string();
            if (name.rfind("step-", 0) != 0 || name.size() < 11 || name.substr(name.size() - 11) != ".state.gguf") continue;
            const auto step = state_step(entry);
            if (!best.second || step > best.first) best = {step, entry};
        }
    }
    return best;
}

class Workflow {
public:
    explicit Workflow(const fs::path & run_path)
        : run_(from_utf8(resolved_path(run_path))), runtime_(read_json(run_ / "runtime.json")), request_(read_json(run_ / "request.json")),
          options_(settings(request_)) {
        // The Python ran its tools from its checkout; the native workflow from
        // the native tools' folder. A run keeps the folder its job recorded.
        workspace_ = from_utf8(resolved_path(from_utf8(get(runtime_, "workspace", member(runtime_, "bin")).text)));
        data_ = run_ / "data";
        family_ = text(options_, "family");
        if (!fs::exists(data_)) fs::create_directory(data_);
        state_ = fs::exists(run_ / "state.json") ? read_json(run_ / "state.json") : json::object();
        state_.object["name"] = get(request_, "name", member(request_, "id"));
        state_.object["settings"] = options_;
        state_.object.erase("error");
    }

    json::Value & state() { return state_; }

    void update(std::initializer_list<std::pair<const char *, json::Value>> values) {
        for (const auto & [key, value] : values) state_.object[key] = value;
        state_.object["updated_at"] = F(now());
        save(run_ / "state.json", state_);
    }

    void check_stop() const { if (fs::exists(run_ / "stop.request")) throw Stopped(); }

    // One stage as a child yue2-train-run command, publishing its progress file.
    void command(const std::vector<std::string> & arguments, const std::string & stage, const std::optional<fs::path> & progress = std::nullopt) {
        check_stop();
        update({{"status", S("running")}, {"stage", S(stage)}, {"progress", json::object()}});
        const auto signature = [&]() -> std::optional<std::pair<long long, long long>> {
            std::error_code error;
            if (!progress || !fs::exists(*progress, error)) return std::nullopt;
            const auto time = fs::last_write_time(*progress, error).time_since_epoch().count();
            const auto size = static_cast<long long>(fs::file_size(*progress, error));
            if (error) return std::nullopt;
            return std::make_pair(static_cast<long long>(time), size);
        };
        const auto previous = signature();
        // A resumed helper may hash models before replacing its old status;
        // do not show the previous segment's stopped or completed snapshot.
        const auto read_progress = [&]() {
            if (signature() == previous) return;
            try { update({{"progress", read_json(*progress)}}); } catch (const std::exception &) {}
        };
        int number = 1;
        while (fs::exists(run_ / (stage + "-" + std::to_string(number) + ".log"))) ++number;
        const auto log = run_ / (stage + "-" + std::to_string(number) + ".log");
        std::vector<std::string> cmd{python_path(current_executable())};
        cmd.insert(cmd.end(), arguments.begin(), arguments.end());
        save(with_suffix(log, ".command.json"), make({{"command", strings(cmd)}}));
        Child child(from_utf8(cmd[0]), arguments, workspace_, log);
        update({{"child_pid", I(child.pid())}, {"log", P(log)}});
        while (child.running()) {
            read_progress();
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        const int code = child.wait();
        // Fast stages can finish between ticks; read their final progress.
        read_progress();
        update({{"child_pid", json::Value{}}});
        check_stop();
        if (code) throw std::runtime_error(log_tail(log, 5000));
    }

    json::Value config() const { return read_json(run_ / "ar-job.json"); }

    void prepare() {
        const auto & o = options_;
        const auto & models = member(runtime_, "models");
        const auto binary = from_utf8(resolved_path(from_utf8(text(runtime_, "bin"))));
        for (const char * name : {"yue2-ar-train", "yue2-train", "yue2-score", "yue2-transcribe", "yue2-semantic-tokenize"}) {
            if (!fs::is_regular_file(binary / suffix(name))) throw std::invalid_argument(std::string("Training tool missing: ") + name);
        }
        const auto vae_binary = runtime_.find("vae_binary") ? from_utf8(text(runtime_, "vae_binary")) : binary / suffix("yue2-vae-encode");
        if (!fs::is_regular_file(vae_binary)) throw std::invalid_argument("Training tool missing: yue2-vae-encode");
        for (const char * name : {"base", "semantic", "transcriber", "tokenizer", "companion", "vae"}) {
            if (!fs::is_regular_file(from_utf8(text(models, name)))) throw std::invalid_argument(std::string("Training model missing: ") + name);
        }
        const auto rows = inventory(from_utf8(text(request_, "dataset")), integer(member(o, "window_seconds")), integer(member(o, "seed")));
        const auto device = get(runtime_, "device", S("cuda"));
        // Runtime is immutable. Every helper verifies source, cache and model
        // hashes on resume; resolved preparation is retained for inspection.
        auto training = make({{"rank", member(o, "rank")}, {"learning_rate", member(o, "ar_lr")}, {"kl_weight", member(o, "kl_weight")},
            {"weight_decay", F(.01)}, {"clip", I(1)},
            {"token_budget", I(integer(member(o, "ar_steps")) * integer(member(o, "window_seconds")) * 25)}, {"seed", member(o, "seed")},
            {"checkpoint_every", member(o, "eval_every")}, {"eval_every", member(o, "eval_every")}});
        if (o.find("evaluation_scope")) training.object["evaluation_scope"] = member(o, "evaluation_scope");
        if (o.find("teacher_cache")) training.object["teacher_cache"] = member(o, "teacher_cache");
        if (truthy(get(o, "teacher_online", B(false)))) training.object["teacher"] = S("online");
        const auto cfg = make({{"format", S("yue2-ar-training-v1")}, {"workspace", P(workspace_)}, {"output", P(data_)},
            {"dataset", member(request_, "dataset")}, {"bin", P(binary)}, {"device", device}, {"name", member(state_, "name")},
            {"models", models}, {"trigger", member(o, "trigger")}, {"sources", rows}, {"inference_urls", json::array()},
            {"score_binary", P(binary / suffix("yue2-score"))}, {"training_binary", P(binary / suffix("yue2-ar-train"))},
            {"training_dataset", P(data_ / "dataset.json")}, {"stop_file", P(run_ / "stop.request")},
            {"preparation", make({{"window_seconds", member(o, "window_seconds")}})}, {"training", training}});
        const auto path = run_ / "ar-job.json";
        if (fs::exists(path) && !python_equal(read_json(path), cfg)) throw std::invalid_argument("Resolved dataset/runtime changed; create a new run");
        save(path, cfg);
        update({{"dataset", rows}, {"status", S("running")}});
        command({"prepare-semantic", "--config", python_path(path)}, "semantic", data_ / "preparation-progress.json");
        command({"prepare-scores", "--config", python_path(path)}, "scores", data_ / "preparation-progress.json");
        const auto vae_device = get(runtime_, "vae_device", device);
        command({"prepare-targets", "--dataset", text(cfg, "training_dataset"), "--vae", text(models, "vae"), "--binary", python_path(vae_binary),
                 "--device", vae_device.text, "--core-frames", json::stringify(get(runtime_, "vae_core_frames", I(128))),
                 "--threads", json::stringify(get(runtime_, "threads", I(6))), "--out", python_path(run_ / "vae-targets"),
                 "--stop-file", python_path(run_ / "stop.request")},
                "vae-targets-native", run_ / "vae-targets" / "progress.json");
        const auto dataset = read_json(from_utf8(text(cfg, "training_dataset")));
        const auto eligible = eligible_rows(dataset);
        const auto weights = fs::file_size(from_utf8(text(models, "base"))) + fs::file_size(from_utf8(text(models, "companion")));
        const auto estimate = ar_resource_estimate(eligible, o, device.text, weights);
        // The teacher cache is released before NAR conditioning is written.
        const auto conditioning = conditioning_bytes(eligible, get(o, "conditioning_storage", S("cache")).text,
                                                     get(o, "evaluation_scope", S("full")).text);
        const double teacher = number(member(estimate, "estimated_teacher_disk_gib")) * gib;
        const double peak = std::max(teacher, static_cast<double>(conditioning)) + checkpoint_allowance(o);
        long long train = 0, validation = 0;
        for (const auto & row : eligible) { train += text(row, "split") == "train"; validation += text(row, "split") == "validation"; }
        const auto & all = member(dataset, "rows").array;
        update({{"status", S("ready")}, {"stage", S("dataset-review")}, {"examples", I(static_cast<long long>(eligible.size()))},
                {"excluded", I(static_cast<long long>(all.size() - eligible.size()))}, {"train_examples", I(train)},
                {"validation_examples", I(validation)}, {"estimated_conditioning_gib", F(static_cast<double>(conditioning) / gib)},
                {"estimated_peak_disk_gib", F(peak / gib)},
                {"storage_options", storage_options(eligible, o, expert_bytes(from_utf8(text(models, "base"))))},
                {"free_disk_gib", F(static_cast<double>(fs::space(run_).available) / gib)},
                {"estimated_ar_host_ram_gib", member(estimate, "estimated_ar_host_ram_gib")},
                {"estimated_teacher_disk_gib", member(estimate, "estimated_teacher_disk_gib")}});
    }

    void fit(const std::string & expert, const fs::path & config, const fs::path & folder, long long budget) {
        check_stop();
        std::vector<std::string> base{expert == "ar" ? "fit-ar" : "fit-nar", "--config", python_path(config)};
        if (expert == "ar") { base.push_back("--family"); base.push_back(family_); }
        const auto with = [&](std::initializer_list<std::string> extra) {
            auto arguments = base;
            arguments.insert(arguments.end(), extra);
            return arguments;
        };
        if (!fs::exists(folder / "probe")) command(with({"--phase", "probe"}), expert + "-probe", folder / "progress.json");
        const auto [step, resume] = latest_state(folder);
        if (step >= budget) {
            // Recover a completed native fit whose coordinator stopped before
            // writing checkpoint selection. This performs no extra updates.
            if (!fs::exists(folder / "selection.json")) command(with({"--phase", "select"}), expert + "-select");
            return;
        }
        int number = 1;
        while (fs::exists(folder / ("native-" + std::to_string(number)))) ++number;
        auto arguments = with({"--phase", "fit", "--segment", "native-" + std::to_string(number), "--prune-checkpoints"});
        if (resume) { arguments.push_back("--resume"); arguments.push_back(python_path(*resume)); }
        command(arguments, expert + "-fit", folder / "progress.json");
    }

    // Refuse to start when the remaining caches and checkpoints cannot fit.
    void check_disk(const fs::path & ar_folder, const fs::path & nar_data) {
        const auto data = read_json(from_utf8(text(config(), "training_dataset")));
        const auto eligible = eligible_rows(data);
        const bool ar_done = fs::exists(ar_folder / "selection.json");
        const double teacher = ar_done ? 0 : std::max(0.0, number(member(ar_resource_estimate(eligible, options_, "cuda"), "estimated_teacher_disk_gib")) * gib -
                                                               static_cast<double>(folder_bytes(ar_folder / "teacher")));
        const bool complete = fs::exists(nar_data / "dataset.json") && truthy(get(read_json(nar_data / "dataset.json"), "completed", B(false)));
        const auto storage = get(options_, "conditioning_storage", S("cache")).text;
        const double conditioning = complete && storage == "cache" ? 0
            : static_cast<double>(std::max(0LL, conditioning_bytes(eligible, storage, get(options_, "evaluation_scope", S("full")).text) - folder_bytes(nar_data)));
        const double needed = std::max(teacher, conditioning) + checkpoint_allowance(options_, ar_done ? 1 : 2);
        const auto free = static_cast<double>(fs::space(run_).available);
        if (free < needed) {
            char message[160];
            std::snprintf(message, sizeof(message), "Training needs about %.1f GiB free on the run drive; %.1f GiB is free", needed / gib, free / gib);
            throw std::invalid_argument(message);
        }
    }

    void train() {
        if (!fs::exists(run_ / "vae-targets" / "index.json")) throw std::invalid_argument("Prepare and review the dataset first");
        if (truthy(get(state_, "caches_released", json::Value{}))) {
            throw std::invalid_argument("This run released its training caches after publishing; start a new run to train again");
        }
        const auto & o = options_;
        const auto cfg = config();
        const auto ar_folder = data_ / family_;
        const auto nar_data = run_ / "nar-conditioning";
        check_disk(ar_folder, nar_data);
        fit("ar", run_ / "ar-job.json", ar_folder, integer(member(o, "ar_steps")));
        const auto selection = read_json(ar_folder / "selection.json");
        const auto ar = from_utf8(text(selection, "adapter"));
        if (!truthy(member(member(selection, "selected"), "step"))) {
            throw std::invalid_argument("Heldout selection chose the base AR; review the run before acoustic calibration");
        }
        update({{"ar_selection", selection}});
        // NAR never reads the KL teachers, and a completed AR fit is detected
        // from its retained latest state. Release them before conditioning.
        const auto teacher = ar_folder / "teacher";
        if (fs::exists(teacher)) {
            std::vector<fs::path> files;
            for (const auto & path : sorted_entries(teacher)) {
                const auto name = path.filename().u8string();
                if (name.rfind("teacher-", 0) == 0 && name.size() > 5 && name.substr(name.size() - 5) == ".gguf") files.push_back(path);
            }
            const auto released = release(files);
            if (released) {
                update({{"teacher_cache_released_gib", F(number(get(state_, "teacher_cache_released_gib", I(0))) + static_cast<double>(released) / gib)}});
            }
        }
        const auto binary = from_utf8(text(runtime_, "bin")) / suffix("yue2-train");
        const auto storage = get(o, "conditioning_storage", S("cache")).text;
        if (!fs::exists(nar_data / "dataset.json") || !truthy(get(read_json(nar_data / "dataset.json"), "completed", B(false)))) {
            std::vector<std::string> arguments{"prepare-conditioning", "--config", python_path(run_ / "ar-job.json"), "--targets",
                python_path(run_ / "vae-targets" / "index.json"), "--ar-adapter", python_path(ar), "--binary", python_path(binary), "--out",
                python_path(nar_data), "--conditioning-device", get(runtime_, "device", S("cuda")).text, "--conditioning-batch", "0",
                "--stop-file", python_path(run_ / "stop.request")};
            if (storage != "cache") { arguments.push_back("--storage"); arguments.push_back(storage); }
            command(arguments, "nar-conditioning", nar_data / "progress.json");
        }
        auto training = make({{"rank", member(o, "rank")}, {"learning_rate", member(o, "nar_lr")}, {"weight_decay", F(.01)}, {"clip", I(1)},
            {"seed", member(o, "seed")}, {"noise_schedule", S("uniform")}, {"steps", member(o, "nar_steps")},
            {"checkpoint_every", member(o, "eval_every")}, {"eval_every", member(o, "eval_every")}});
        if (o.find("evaluation_scope")) {
            training.object["evaluation_scope"] = member(o, "evaluation_scope");
            training.object["probe_scope"] = S("first-at-maximum-context");
        }
        if (o.find("cache_fingerprint")) training.object["cache_fingerprint"] = member(o, "cache_fingerprint");
        // Recorded only for non-default storage, so recipe 4/5-cache jobs keep their identity.
        if (storage != "cache") {
            training.object["conditioning_storage"] = S(storage);
            if (storage == "rolling") training.object["rolling_window_updates"] = I(25);
        }
        const auto & models = member(cfg, "models");
        const auto nar_cfg = make({{"format", S("yue2-nar-training-v1")}, {"workspace", P(workspace_)}, {"output", P(run_ / "nar-run")},
            {"name", member(state_, "name")}, {"training_dataset", P(nar_data / "dataset.json")}, {"training_binary", P(binary)},
            {"device", get(runtime_, "device", S("cuda"))}, {"threads", get(runtime_, "threads", I(6))}, {"inference_urls", json::array()},
            {"models", make({{"base", member(models, "base")}, {"companion", member(models, "companion")}, {"ar", P(ar)}})},
            {"training", training}});
        const auto nar_path = run_ / "nar-job.json";
        if (fs::exists(nar_path) && !python_equal(read_json(nar_path), nar_cfg)) throw std::invalid_argument("Frozen AR/NAR run changed");
        save(nar_path, nar_cfg);
        fit("nar", nar_path, run_ / "nar-run", integer(member(o, "nar_steps")));
        update({{"status", S("review")}, {"stage", S("listening-and-publication")}, {"nar_selection", read_json(run_ / "nar-run" / "selection.json")}});
    }

    void publish_run() {
        const auto status = get(state_, "status", json::Value{}).text;
        if (status != "review" && status != "published") throw std::invalid_argument("Complete both expert stages before publishing");
        // publish() reads the state this workflow last saved.
        save(run_ / "state.json", state_);
        const auto result = publish(run_);
        update({{"status", S("published")}, {"stage", S("published")}, {"export", result}});
        if (!truthy(member(options_, "keep_caches"))) release_run_caches();
    }

    // After publishing, keep the selected adapters, manifests, metrics and
    // logs. Conditioning, teachers, optimizer states, other checkpoints,
    // probe diagnostics and audio slices are only needed to keep training.
    void release_run_caches() {
        const auto released = release_caches(run_);
        const double previous = number(get(get(state_, "caches_released", json::object()), "gib", I(0)));
        update({{"caches_released", make({{"gib", F(previous + static_cast<double>(released) / gib)}, {"at", F(now())}})}});
    }

    void run(const std::string & action) {
        if (action == "prepare") prepare();
        else if (action == "train") train();
        else if (action == "publish") publish_run();
        else throw std::invalid_argument("Unknown training action: " + action);
    }

private:
    fs::path run_, workspace_, data_;
    json::Value runtime_, request_, options_, state_;
    std::string family_;
};

} // namespace

void write_training_runtime(const fs::path & bin, const fs::path & models_dir, const fs::path & run_root, const fs::path & out) {
#if !defined(_WIN32) && !defined(__linux__)
    throw std::invalid_argument("Local training ownership currently supports Windows and Linux");
#endif
    if (fs::exists(out)) throw std::invalid_argument("Output exists; choose a new config path");
    for (const char * tool : {"yue2-train-run", "yue2-ar-train", "yue2-train", "yue2-score", "yue2-transcribe", "yue2-semantic-tokenize", "yue2-vae-encode"}) {
        if (!fs::is_regular_file(bin / suffix(tool))) throw std::invalid_argument(std::string("Missing native tool: ") + tool);
    }
    const auto model = [&](const char * name) {
        std::vector<fs::path> found;
        if (fs::is_directory(models_dir)) {
            for (const auto & entry : fs::recursive_directory_iterator(models_dir)) {
                if (entry.path().filename().u8string() == name) found.push_back(entry.path());
            }
        }
        if (found.size() != 1) throw std::invalid_argument(std::string("Expected one installed ") + name + "; found " + std::to_string(found.size()));
        return S(resolved_path(found[0]));
    };
    const auto config = make({{"bin", S(resolved_path(bin))}, {"run_root", S(resolved_path(run_root))}, {"models_dir", S(resolved_path(models_dir))},
        {"device", S("cuda")}, {"threads", I(6)},
        {"models", make({{"base", model("yue2-3.6B-v1.0-BF16.gguf")}, {"vae", model("yue2-vae-v1.0-F16.gguf")},
                         {"semantic", model("yue2-semantic-tokenizer-0.7B-v1.0-F16.gguf")}, {"transcriber", model("sheetsage2-mert2-f16.gguf")},
                         {"tokenizer", model("yue2-qwen.tiktoken")}, {"companion", model("yue2-instrumental-cot-full-v1.0-F16-LoRA.gguf")}})}});
    save(out, config);
    std::cout << "Start yue2-server with --training-config " << resolved_path(out) << std::endl;
}

int run_workflow(const fs::path & run, const std::string & action) {
    set_environment("GGML_CUDA_TF32", "0");
    std::optional<Workflow> workflow;
    try {
        wait_for_owner(run);
        own_children();
        workflow.emplace(run);
        auto chosen = action;
        if (action == "resume") {
            const auto targets = run / "vae-targets" / "index.json";
            const bool complete = fs::exists(targets) && truthy(get(read_json(targets), "completed", B(false)));
            chosen = complete ? "train" : "prepare";
        }
        workflow->run(chosen);
    } catch (const Stopped &) {
        if (workflow) workflow->update({{"status", S("stopped")}, {"child_pid", json::Value{}}});
    } catch (const std::exception & error) {
        if (workflow) workflow->update({{"status", S("failed")}, {"error", S(error.what())}, {"child_pid", json::Value{}}});
        else save(run / "state.json", make({{"status", S("failed")}, {"error", S(error.what())}}));
        std::cerr << error.what() << std::endl;
        return 1;
    }
    return 0;
}

} // namespace yue2::training
