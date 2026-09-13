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

} // namespace yue2::server::json
