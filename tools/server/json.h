#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace yue2::server::json {

enum class Type { null, boolean, number, string, array, object };

struct Value {
    Type type = Type::null;
    bool boolean = false;
    // String payload or the exact JSON number lexeme.
    std::string text;
    std::vector<Value> array;
    std::map<std::string, Value> object;

    const Value * find(std::string_view key) const;
};

Value parse(std::string_view source);
std::string quote(std::string_view value);
std::string stringify(const Value & value);

std::string string(const Value & object, std::string_view key, std::string fallback = {});
bool boolean(const Value & object, std::string_view key, bool fallback);
double number(const Value & object, std::string_view key, double fallback);
std::uint64_t u64(const Value & object, std::string_view key, std::uint64_t fallback);
std::uint32_t u32(const Value & object, std::string_view key, std::uint32_t fallback);

// Values for files the Python training tools also read and compare. A double
// gets Python's repr() lexeme (30.0, 2e-05, 1e+16), so a float stays a float.
Value from_double(double value);
Value from_integer(long long value);
Value from_string(std::string value);
Value from_bool(bool value);
Value object();
Value array();
std::string python_repr(double value);
// A number lexeme with no fraction or exponent, which Python loads as int.
bool is_integer(const Value & value);
// Like json.dumps(value, indent=2): non-ASCII is escaped, so a reader that
// decodes with the locale's encoding still reads what was written. Object
// keys are sorted.
std::string pretty(const Value & value);

} // namespace yue2::server::json
