// Internals of the native fits, shared with their unit tests.
#pragma once

#include "training/dataset.h"
#include "training/preparation_detail.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <optional>
#include <regex>
#include <string>
#include <utility>
#include <vector>

namespace yue2::training::fitting_detail {

namespace fs = std::filesystem;
using namespace detail;

using Clock = std::chrono::steady_clock;

inline double seconds_since(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }

// str() of a JSON value as Python formats a command-line argument.
inline std::string py_str(const json::Value & value) {
    switch (value.type) {
        case json::Type::number:
            return json::is_integer(value) ? value.text : json::python_repr(std::strtod(value.text.c_str(), nullptr));
        case json::Type::boolean: return value.boolean ? "True" : "False";
        case json::Type::string: return value.text;
        case json::Type::null: return "None";
        default: return json::stringify(value);
    }
}

inline std::string py_str(long long value) { return std::to_string(value); }

inline std::vector<json::Value> eligible(const json::Value & data) {
    std::vector<json::Value> rows;
    for (const auto & row : member(data, "rows").array) if (truthy(get(row, "eligible", B(true)))) rows.push_back(row);
    return rows;
}

inline std::vector<json::Value> with_split(const std::vector<json::Value> & rows, const char * split) {
    std::vector<json::Value> output;
    for (const auto & row : rows) if (text(row, "split") == split) output.push_back(row);
    return output;
}

// The lines a growing log has completed since the last read.
class LogLines {
public:
    explicit LogLines(fs::path log) : log_(std::move(log)) {}
    std::vector<std::string> read() {
        std::ifstream stream(log_, std::ios::binary);
        stream.seekg(static_cast<std::streamoff>(offset_));
        std::string text{std::istreambuf_iterator<char>(stream), {}};
        offset_ += text.size();
        // Universal newlines, like Python's text mode.
        std::string normalized = pending_;
        for (std::size_t i = 0; i < text.size(); ++i) {
            if (text[i] == '\r') {
                if (i + 1 < text.size() && text[i + 1] == '\n') continue;
                normalized.push_back('\n');
            } else {
                normalized.push_back(text[i]);
            }
        }
        std::vector<std::string> lines;
        std::size_t start = 0;
        for (std::size_t at; (at = normalized.find('\n', start)) != std::string::npos; start = at + 1) {
            lines.push_back(normalized.substr(start, at - start));
        }
        pending_ = normalized.substr(start);
        return lines;
    }

private:
    fs::path log_;
    std::size_t offset_ = 0;
    std::string pending_;
};

inline std::string trim(const std::string & text) { return strip(text); }

// train_ar_run.NativeProgress: startup separately from optimizer updates.
class ArProgress {
public:
    ArProgress(const fs::path & log, long long resumed, long long eval_every, std::optional<long long> target)
        : lines_(log), eval_every_(eval_every), target_(target) {
        values_ = make({{"activity", S("Loading native trainer")}, {"step", I(resumed)}});
    }
    json::Value & values() { return values_; }
    json::Value read() {
        static const std::regex teacher(R"(prepare_teacher=(\d+)/(\d+))"), cached(R"(teacher_cache=(\d+)/(\d+))");
        for (const auto & line : lines_.read()) {
            const auto before = values_.object["activity"].text;
            const auto stripped = trim(line);
            std::smatch match;
            if (std::regex_match(stripped, match, teacher) || std::regex_match(stripped, match, cached)) {
                const bool preparing = stripped.rfind("prepare_teacher=", 0) == 0;
                values_.object["activity"] = S(preparing ? "Preparing frozen teacher distributions" : "Validating frozen teacher cache");
                values_.object["teacher_current"] = I(std::stoll(match[1]));
                values_.object["teacher_total"] = I(std::stoll(match[2]));
            } else if (line.rfind("initial_loss=", 0) == 0) {
                values_.object["activity"] = S("Evaluating training and validation windows");
            } else if (line.rfind("step=", 0) == 0) {
                const auto first = line.substr(0, line.find_first_of(" \t"));
                const long long step = std::stoll(first.substr(first.find('=') + 1));
                const char * activity = target_ && step == *target_ ? "Final evaluation and checkpoint save"
                                        : step % eval_every_ == 0 ? "Evaluating checkpoint" : "Training";
                values_.object["step"] = I(step);
                values_.object["activity"] = S(activity);
            }
            if (values_.object["activity"].text != before) since_ = Clock::now();
        }
        auto result = values_;
        auto lowered = result.object["activity"].text;
        std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (lowered.find("evaluat") != std::string::npos) result.object["activity_elapsed_seconds"] = F(seconds_since(since_));
        return result;
    }

private:
    LogLines lines_;
    long long eval_every_;
    std::optional<long long> target_;
    json::Value values_;
    Clock::time_point since_ = Clock::now();
};

// train_nar_run.NativeNarProgress.
class NarProgress {
public:
    NarProgress(const fs::path & log, long long target, long long resumed, std::string phase)
        : lines_(log), target_(target), phase_(std::move(phase)) {
        values_ = make({{"activity", S("Validating cached training data")}, {"step", I(resumed)}});
    }
    json::Value & values() { return values_; }
    bool events() const { return events_; }
    void activity(const std::string & value) {
        if (value != values_.object["activity"].text) {
            since_ = Clock::now();
            for (const char * key : {"phase_current", "phase_total", "phase_unit"}) values_.object.erase(key);
        }
        values_.object["activity"] = S(value);
    }
    json::Value read() {
        for (const auto & line : lines_.read()) {
            if (line.rfind("TRAIN_EVENT ", 0) == 0) {
                const auto event = json::parse(line.substr(12));
                const auto stage = text(event, "stage");
                events_ = true;
                std::string label;
                if (stage == "cache-validation") label = "Validating cached training data";
                else if (stage == "model-loading") label = "Loading native NAR model";
                else if (stage == "evaluation") {
                    const auto step = integer(member(event, "step"));
                    values_.object["step"] = member(event, "step");
                    label = !step ? "Initial evaluation" : step >= target_ ? "Final evaluation" : "Evaluating checkpoint";
                } else if (stage == "checkpoint-save") {
                    values_.object["step"] = member(event, "step");
                    label = "Saving checkpoint";
                } else {
                    continue;
                }
                activity(label);
                if (stage == "cache-validation" || stage == "evaluation") {
                    values_.object["phase_current"] = member(event, "current");
                    values_.object["phase_total"] = member(event, "total");
                    values_.object["phase_unit"] = S(stage == "cache-validation" ? "files" : "windows");
                }
            } else if (line.rfind("[yue2:train]", 0) == 0 && !events_) {
                activity(phase_ != "fit" ? "Native preflight"
                         : truthy(values_.object["step"]) ? "Loading native optimizer state" : "Initial evaluation");
            } else if (!line.empty() && line[0] == '{') {
                json::Value record;
                try { record = json::parse(line); } catch (const std::exception &) { continue; }
                if (record.find("loss") && record.find("step")) {
                    values_.object["step"] = member(record, "step");
                    activity("Training");
                    if (const auto * value = record.find("seconds")) values_.object["last_update_seconds"] = *value;
                    if (const auto * value = record.find("cache_load_seconds")) values_.object["last_cache_load_seconds"] = *value;
                }
            }
        }
        auto result = values_;
        result.object["activity_elapsed_seconds"] = F(seconds_since(since_));
        return result;
    }

private:
    LogLines lines_;
    long long target_;
    std::string phase_;
    json::Value values_;
    bool events_ = false;
    Clock::time_point since_ = Clock::now();
};

double grouped_loss(const json::Value & values, const std::vector<json::Value> & rows);
json::Value evaluation_history(const fs::path & out, const std::vector<json::Value> & rows, const std::string & expert);
long long prune_checkpoints(const fs::path & out, const std::vector<json::Value> & rows, const std::string & expert);
void select_ar(const fs::path & out, const std::vector<json::Value> & rows, const json::Value & config);
void select_nar(const fs::path & out, const std::vector<json::Value> & rows);
// Everything that changes frozen teacher bits, independent of the adapter family.
json::Value teacher_identity(const json::Value & run_identity, const std::string & device, long long ar_slots, const json::Value & frames,
                             const std::string & precision);
void verify_teacher_cache(const fs::path & folder, const std::string & identity_sha256, const std::vector<json::Value> & train_rows);
std::string prepare_teacher_cache(const fs::path & folder, std::vector<std::string> command, const fs::path & cwd, const fs::path & out,
                                  const std::string & family, const json::Value & resolved, const std::vector<json::Value> & train_rows,
                                  const fs::path & stop_file);
std::pair<fs::path, long long> preflight_sample(const std::vector<json::Value> & rows,
                                                const std::function<fs::path(const json::Value &)> & cache_path, long long frames);

} // namespace yue2::training::fitting_detail
