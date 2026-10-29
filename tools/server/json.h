#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace yue2::server::json {

enum class Type { null, boolean, number, string, array, object };

struct Value;

// Object members in insertion order, like a Python dict: a document read and
// written back keeps its key order, and a new one is written in the order its
// members were added.
class Object {
public:
    using Entry = std::pair<std::string, Value>;
    using iterator = std::vector<Entry>::iterator;
    using const_iterator = std::vector<Entry>::const_iterator;

    // Defined after Value, which an entry needs complete.
    iterator begin();
    iterator end();
    const_iterator begin() const;
    const_iterator end() const;
    std::size_t size() const;
    bool empty() const;
    iterator find(std::string_view key);
    const_iterator find(std::string_view key) const;
    std::size_t count(std::string_view key) const;
    Value & at(std::string_view key);
    const Value & at(std::string_view key) const;
    // A missing key is appended; an existing one keeps its place.
    Value & operator[](std::string_view key);
    std::pair<iterator, bool> emplace(std::string key, Value value);
    std::size_t erase(std::string_view key);

private:
    std::vector<Entry> entries_;
};

struct Value {
    Type type = Type::null;
    bool boolean = false;
    // String payload or the exact JSON number lexeme.
    std::string text;
    std::vector<Value> array;
    Object object;

    const Value * find(std::string_view key) const;
};

inline Object::iterator Object::begin() { return entries_.begin(); }
inline Object::iterator Object::end() { return entries_.end(); }
inline Object::const_iterator Object::begin() const { return entries_.begin(); }
inline Object::const_iterator Object::end() const { return entries_.end(); }
inline std::size_t Object::size() const { return entries_.size(); }
inline bool Object::empty() const { return entries_.empty(); }
inline std::size_t Object::count(std::string_view key) const { return find(key) == end() ? 0 : 1; }
inline Object::iterator Object::find(std::string_view key) {
    for (auto it = entries_.begin(); it != entries_.end(); ++it) if (it->first == key) return it;
    return entries_.end();
}
inline Object::const_iterator Object::find(std::string_view key) const {
    for (auto it = entries_.begin(); it != entries_.end(); ++it) if (it->first == key) return it;
    return entries_.end();
}
inline Value & Object::at(std::string_view key) {
    const auto found = find(key);
    if (found == end()) throw std::out_of_range("JSON object has no member " + std::string(key));
    return found->second;
}
inline const Value & Object::at(std::string_view key) const {
    const auto found = find(key);
    if (found == end()) throw std::out_of_range("JSON object has no member " + std::string(key));
    return found->second;
}
inline Value & Object::operator[](std::string_view key) {
    const auto found = find(key);
    if (found != end()) return found->second;
    entries_.emplace_back(std::string(key), Value{});
    return entries_.back().second;
}
inline std::pair<Object::iterator, bool> Object::emplace(std::string key, Value value) {
    const auto found = find(key);
    if (found != end()) return {found, false};
    entries_.emplace_back(std::move(key), std::move(value));
    return {entries_.end() - 1, true};
}
inline std::size_t Object::erase(std::string_view key) {
    const auto found = find(key);
    if (found == end()) return 0;
    entries_.erase(found);
    return 1;
}

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
// decodes with the locale's encoding still reads what was written, and a
// number is written as Python writes what it parsed (2E-5 becomes 2e-05).
std::string pretty(const Value & value);
// Like json.dumps(value, sort_keys=True): one line, ", " and ": " separators.
// Training identities hash these bytes.
std::string python_dumps_sorted(const Value & value);

} // namespace yue2::server::json
