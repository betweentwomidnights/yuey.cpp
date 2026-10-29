// yue2-train-run: the native training workflow yue2-server runs for each
// training run: run settings, dataset inventory and review, capabilities,
// preparation, fitting and publication. It replaced the Python orchestration
// stage by stage (docs/training-history.md).
// Query commands print JSON on stdout; preparation prints progress lines.
// Errors go to stderr with exit code 1.
#include "training/dataset.h"
#include "training/driver.h"
#include "training/sha256.h"
#include "training/fitting.h"
#include "training/preparation.h"
#include "training/publication.h"
#include "training/workflow.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>
#endif

namespace {
namespace fs = std::filesystem;
namespace json = yue2::server::json;
namespace training = yue2::training;

constexpr const char * usage = R"(usage: yue2-train-run COMMAND [options]

  setup --bin DIR --models-dir DIR [--run-root DIR] --out FILE
                                  write the training runtime yue2-server --training-config reads
  workflow --run DIR --action prepare|train|resume|publish
                                  a named run, as yue2-server launches it
  capabilities [--runtime FILE]   supported recipes and storage modes; with a
                                  training runtime, what its native tools support
  settings --request FILE         validated settings for a run request
  inventory --dataset DIR --window-seconds N --seed N
                                  recordings, captions, hashes and holdout split
  review --run DIR                dataset review of a prepared run
  score --events FILE --seconds S --bpm B --key K
                                  re-engrave transcriber events as ABC
  identity --executable FILE      SHA-256 of a native tool and its libraries
  sha256 FILE...                  SHA-256 of files
  prepare-semantic --config AR_JOB
  prepare-scores --config AR_JOB  semantic caches, then scored AR examples
  prepare-targets --dataset DATASET --vae VAE --binary YUE2_VAE_ENCODE --out DIR
                  [--device cuda] [--threads 4] [--core-frames 128] [--halo-frames 50] [--stop-file FILE]
                                  posterior-mean VAE targets
  prepare-conditioning --config AR_JOB --targets INDEX --ar-adapter ADAPTER --binary YUE2_TRAIN --out DIR
                  [--conditioning-device cuda] [--threads 6] [--cache-kv-type f16] [--conditioning-batch 0]
                  [--storage cache|rolling|resident] [--stop-file FILE]
                                  frozen-AR conditioning for NAR fitting
  materialize --manifest DATASET --rows ROWS --binary YUE2_TRAIN --base BASE --common COMPANION
              --adapter ADAPTER [--stop-file FILE]
                                  regenerate absent rolling caches
  fit-ar --config AR_JOB --family lora|dora --phase probe|fit|post-probe|select
         [--resume STATE] [--segment native] [--prune-checkpoints]
  fit-nar --config NAR_JOB --phase probe|fit|post-probe|select
          [--resume STATE] [--segment native] [--prune-checkpoints]
                                  named native fits with heldout checkpoint selection
  publish --run DIR               the reviewed AR/NAR pair as one style GGUF and sidecar
  release-caches --run DIR        delete what only further training needs
)";

std::string read(const fs::path & path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("Cannot read " + path.u8string());
    return {std::istreambuf_iterator<char>(stream), {}};
}

struct Arguments {
    std::map<std::string, std::string> options;
    std::vector<std::string> positional;
    std::string required(const std::string & key) const {
        const auto found = options.find(key);
        if (found == options.end()) throw std::invalid_argument("missing --" + key);
        return found->second;
    }
};

// UTF-8 arguments on every platform: Windows passes argv in the ANSI code
// page, which cannot spell every dataset path.
std::vector<std::string> utf8_arguments(int argc, char ** argv) {
#ifdef _WIN32
    (void)argc;
    (void)argv;
    int count = 0;
    wchar_t ** wide = CommandLineToArgvW(GetCommandLineW(), &count);
    std::vector<std::string> output;
    for (int i = 0; i < count; ++i) output.push_back(fs::path(wide[i]).u8string());
    LocalFree(wide);
    return output;
#else
    return {argv, argv + argc};
#endif
}

Arguments arguments(const std::vector<std::string> & argv) {
    Arguments output;
    for (std::size_t i = 2; i < argv.size(); ++i) {
        const std::string value = argv[i];
        if (value == "--prune-checkpoints") {
            output.options[value.substr(2)] = "1";
        } else if (value.rfind("--", 0) == 0) {
            if (i + 1 >= argv.size()) throw std::invalid_argument(value + " needs a value");
            output.options[value.substr(2)] = argv[++i];
        } else {
            output.positional.push_back(value);
        }
    }
    return output;
}

double number(const std::string & text) {
    char * end = nullptr;
    const double value = std::strtod(text.c_str(), &end);
    if (!end || *end) throw std::invalid_argument("not a number: " + text);
    return value;
}

long long integer(const std::string & text) {
    char * end = nullptr;
    const long long value = std::strtoll(text.c_str(), &end, 10);
    if (!end || *end) throw std::invalid_argument("not an integer: " + text);
    return value;
}

json::Value engraved(const training::EngravedScore & score) {
    auto output = json::object();
    output.object["abc"] = json::from_string(score.abc);
    output.object["stats"] = score.stats;
    return output;
}

json::Value run(const std::string & command, const Arguments & args) {
    if (command == "capabilities") {
        const auto found = args.options.find("runtime");
        return training::capabilities(found == args.options.end() ? std::nullopt : std::optional<fs::path>(fs::u8path(found->second)));
    }
    if (command == "settings") return training::settings(json::parse(read(fs::u8path(args.required("request")))));
    if (command == "inventory") {
        return training::inventory(fs::u8path(args.required("dataset")), integer(args.required("window-seconds")),
                                   integer(args.required("seed")));
    }
    if (command == "review") return training::review(fs::u8path(args.required("run")));
    if (command == "score") {
        auto events = json::parse(read(fs::u8path(args.required("events"))));
        if (const auto * inner = events.find("events")) events = *inner;
        const auto seconds = args.required("seconds");
        const auto seconds_value = seconds.find_first_of(".eE") == std::string::npos ? json::from_integer(integer(seconds))
                                                                                     : json::from_double(number(seconds));
        return engraved(training::score_from_events(events, seconds_value, number(args.required("bpm")),
                                                    args.required("key")));
    }
    if (command == "identity") return training::runtime_identity(fs::u8path(args.required("executable")));
    if (command == "sha256") {
        std::vector<fs::path> paths;
        for (const auto & value : args.positional) paths.push_back(fs::u8path(value));
        const auto digests = training::sha256_files(paths);
        auto output = json::object();
        for (std::size_t i = 0; i < paths.size(); ++i) output.object[args.positional[i]] = json::from_string(digests[i]);
        return output;
    }
    const auto option = [&](const char * key, const std::string & fallback) {
        const auto found = args.options.find(key);
        return found == args.options.end() ? fallback : found->second;
    };
    const auto stop_file = [&]() {
        const auto found = args.options.find("stop-file");
        return found == args.options.end() ? std::optional<fs::path>() : std::optional<fs::path>(fs::u8path(found->second));
    };
    if (command == "prepare-semantic") { training::prepare_semantic(fs::u8path(args.required("config"))); return {}; }
    if (command == "prepare-scores") { training::prepare_scores(fs::u8path(args.required("config"))); return {}; }
    if (command == "prepare-targets") {
        training::TargetOptions o;
        o.dataset = fs::u8path(args.required("dataset"));
        o.vae = fs::u8path(args.required("vae"));
        o.binary = fs::u8path(args.required("binary"));
        o.out = fs::u8path(args.required("out"));
        o.device = option("device", "cuda");
        o.threads = integer(option("threads", "4"));
        o.core_frames = integer(option("core-frames", "128"));
        o.halo_frames = integer(option("halo-frames", "50"));
        o.stop_file = stop_file();
        training::prepare_targets(o);
        return {};
    }
    if (command == "prepare-conditioning") {
        training::ConditioningOptions o;
        o.config = fs::u8path(args.required("config"));
        o.targets = fs::u8path(args.required("targets"));
        o.ar_adapter = fs::u8path(args.required("ar-adapter"));
        o.binary = fs::u8path(args.required("binary"));
        o.out = fs::u8path(args.required("out"));
        o.device = option("conditioning-device", "cuda");
        o.threads = integer(option("threads", "6"));
        o.kv_type = option("cache-kv-type", "f16");
        o.batch = integer(option("conditioning-batch", "0"));
        o.storage = option("storage", "cache");
        o.stop_file = stop_file();
        training::prepare_conditioning(o);
        return {};
    }
    if (command == "fit-ar" || command == "fit-nar") {
        training::FitOptions o;
        o.config = fs::u8path(args.required("config"));
        o.phase = args.required("phase");
        if (command == "fit-ar") o.family = args.required("family");
        if (args.options.count("resume")) o.resume = fs::u8path(args.options.at("resume"));
        o.segment = option("segment", "native");
        o.prune_checkpoints = args.options.count("prune-checkpoints") > 0;
        if (command == "fit-ar") training::fit_ar(o);
        else training::fit_nar(o);
        return {};
    }
    if (command == "setup") {
        training::write_training_runtime(fs::u8path(args.required("bin")), fs::u8path(option("models-dir", "models")),
                                         fs::u8path(option("run-root", "outputs/training")), fs::u8path(option("out", "training-runtime.json")));
        return {};
    }
    if (command == "publish") return training::publish(fs::u8path(args.required("run")));
    if (command == "release-caches") {
        auto output = json::object();
        output.object["released_bytes"] = json::from_integer(training::release_caches(fs::u8path(args.required("run"))));
        return output;
    }
    if (command == "materialize") {
        const auto rows = json::parse(read(fs::u8path(args.required("rows"))));
        if (rows.type != json::Type::array) throw std::invalid_argument("rows must be an array");
        const auto made = training::materialize(fs::u8path(args.required("manifest")), rows.array, fs::u8path(args.required("binary")),
                                                fs::u8path(args.required("base")), fs::u8path(args.required("common")),
                                                fs::u8path(args.required("adapter")), stop_file());
        auto output = json::object();
        output.object["materialized"] = json::from_integer(made);
        return output;
    }
    throw std::invalid_argument("unknown command: " + command);
}

} // namespace

int main(int argc, char ** argv) {
    const auto args = utf8_arguments(argc, argv);
    if (args.size() < 2 || args[1] == "--help" || args[1] == "-h") {
        std::cout << usage;
        return args.size() < 2 ? 1 : 0;
    }
    try {
        // The workflow records its own failures in the run's state.json.
        if (args[1] == "workflow") {
            const auto parsed = arguments(args);
            return training::run_workflow(fs::u8path(parsed.required("run")), parsed.required("action"));
        }
        const auto output = run(args[1], arguments(args));
        if (output.type != json::Type::null) std::cout << json::pretty(output) << "\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
