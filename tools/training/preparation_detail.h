// Helpers the preparation stages share.
#pragma once

#include "server/json.h"
#include "training/files.h"

#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace yue2::training::detail {

namespace json = server::json;

// An error the Python did not catch where it recorded others (an OSError
// such as an existing log): it ends the stage instead of marking an example.
class StageError : public std::exception {
public:
    explicit StageError(std::string message) : message_(std::move(message)) {}
    const char * what() const noexcept override { return message_.c_str(); }
private:
    std::string message_;
};

inline json::Value S(std::string value) { return json::from_string(std::move(value)); }
inline json::Value I(long long value) { return json::from_integer(value); }
inline json::Value F(double value) { return json::from_double(value); }
inline json::Value B(bool value) { return json::from_bool(value); }
inline json::Value P(const std::filesystem::path & path) { return json::from_string(python_path(path)); }

// An object with members in the order given, like a Python dict literal.
inline json::Value make(std::initializer_list<std::pair<const char *, json::Value>> members) {
    auto output = json::object();
    for (const auto & [key, value] : members) output.object[key] = value;
    return output;
}

inline json::Value strings(const std::vector<std::string> & values) {
    auto output = json::array();
    for (const auto & value : values) output.array.push_back(S(value));
    return output;
}

const json::Value & member(const json::Value & object, const char * key);
std::string text(const json::Value & object, const char * key);
// The value of a key, or a fallback (dict.get).
json::Value get(const json::Value & object, const char * key, json::Value fallback);
double number(const json::Value & value);
long long integer(const json::Value & value);
bool truthy(const json::Value & value);

bool stop_requested(const std::optional<std::filesystem::path> & stop_file);
std::optional<std::filesystem::path> config_stop_file(const json::Value & config);
std::string digest(const std::filesystem::path & path);
std::vector<std::int32_t> read_i32(const std::filesystem::path & path);
std::vector<float> read_f32(const std::filesystem::path & path);
std::string executable(const char * name);
// The last characters of a log, as the Python's error messages quote it.
std::string log_tail(const std::filesystem::path & log, std::size_t characters);

// prepare_ar_dataset.execute(): run a native tool to a new log, record its
// command, exit code and duration beside it, and fail with the log's tail.
void execute(const std::vector<std::string> & command, const std::filesystem::path & log, const std::filesystem::path & cwd,
             const json::Value & inference_urls);

// The Python refuses research-only settings this port leaves out.
void reject_research_option(const json::Value & object, const char * key);

} // namespace yue2::training::detail
