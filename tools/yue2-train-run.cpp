// yue2-train-run: the native training workflow. It replaces the Python
// orchestration in tools/training_workflow.py stage by stage; this stage
// holds run settings, dataset inventory and review, and capabilities.
// Every command prints JSON on stdout; errors go to stderr with exit code 1.
#include "training/dataset.h"
#include "training/sha256.h"
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
  parity --cases FILE             evaluate a batch of cases, for the Python
                                  parity check (tests/test_native_training_workflow.py)
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
        if (value.rfind("--", 0) == 0) {
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

double as_double(const json::Value & object, const char * key) {
    const auto * value = object.find(key);
    if (!value || value->type != json::Type::number) throw std::invalid_argument(std::string("case needs ") + key);
    return std::strtod(value->text.c_str(), nullptr);
}

const json::Value & member(const json::Value & object, const char * key) {
    const auto * value = object.find(key);
    if (!value) throw std::invalid_argument(std::string("case needs ") + key);
    return *value;
}

std::vector<json::Value> rows(const json::Value & object) {
    const auto & value = member(object, "eligible");
    if (value.type != json::Type::array) throw std::invalid_argument("eligible must be an array");
    return value.array;
}

json::Value engraved(const training::EngravedScore & score) {
    auto output = json::object();
    output.object["abc"] = json::from_string(score.abc);
    output.object["stats"] = score.stats;
    return output;
}

json::Value evaluate(const json::Value & item) {
    const auto fn = json::string(item, "fn");
    if (fn == "repr") return json::from_string(json::python_repr(as_double(item, "value")));
    if (fn == "caption") {
        const auto meta = training::caption_metadata(json::string(item, "text"));
        auto output = json::object();
        output.object["bpm"] = json::from_double(meta.bpm);
        output.object["key"] = json::from_string(meta.key);
        output.object["style"] = json::from_string(meta.style);
        return output;
    }
    if (fn == "strip") return json::from_string(training::strip(json::string(item, "text")));
    if (fn == "windows") {
        auto output = json::array();
        for (const auto start : training::window_starts(static_cast<long long>(as_double(item, "total")),
                                                        static_cast<long long>(as_double(item, "frames")), as_double(item, "bpm"))) {
            output.array.push_back(json::from_integer(start));
        }
        return output;
    }
    if (fn == "chord") return json::from_string(training::chord_name(json::string(item, "text")));
    if (fn == "pitch") return json::from_string(training::abc_pitch(static_cast<int>(as_double(item, "pitch"))));
    if (fn == "settings") return training::settings(member(item, "request"));
    if (fn == "score") {
        return engraved(training::score_from_events(member(item, "events"), member(item, "seconds"), as_double(item, "bpm"),
                                                    json::string(item, "key")));
    }
    if (fn == "estimate") {
        return training::ar_resource_estimate(rows(item), member(item, "options"), json::string(item, "device"),
                                              static_cast<std::uint64_t>(as_double(item, "model_bytes")));
    }
    if (fn == "conditioning") {
        return json::from_integer(training::conditioning_bytes(rows(item), json::string(item, "storage"), json::string(item, "scope")));
    }
    if (fn == "allowance") {
        return json::from_double(training::checkpoint_allowance(member(item, "options"), static_cast<int>(as_double(item, "experts"))));
    }
    if (fn == "storage") {
        training::ExpertBytes experts{static_cast<std::uint64_t>(as_double(item, "nar")), static_cast<std::uint64_t>(as_double(item, "ar"))};
        return training::storage_options(rows(item), member(item, "options"), experts);
    }
    throw std::invalid_argument("unknown case: " + fn);
}

json::Value parity(const fs::path & path) {
    const auto cases = json::parse(read(path));
    if (cases.type != json::Type::array) throw std::invalid_argument("cases must be an array");
    auto output = json::array();
    for (const auto & item : cases.array) {
        auto result = json::object();
        try {
            result.object["ok"] = evaluate(item);
        } catch (const std::exception & error) {
            result.object["error"] = json::from_string(error.what());
        }
        output.array.push_back(std::move(result));
    }
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
    if (command == "parity") return parity(fs::u8path(args.required("cases")));
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
        std::cout << json::pretty(run(args[1], arguments(args))) << "\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
