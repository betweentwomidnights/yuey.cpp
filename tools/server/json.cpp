#include "json.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace yue2::server::json {
namespace {

std::runtime_error error(std::size_t offset, const std::string & message) {
    return std::runtime_error("JSON byte " + std::to_string(offset) + ": " + message);
}

void append_utf8(std::string & output, std::uint32_t codepoint) {
    if (codepoint <= 0x7f) {
        output.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7ff) {
        output.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else if (codepoint <= 0xffff) {
        output.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else {
        output.push_back(static_cast<char>(0xf0 | (codepoint >> 18)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    }
}

int hex(char value) {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    if (value >= 'A' && value <= 'F') return value - 'A' + 10;
    return -1;
}

class Parser {
public:
    explicit Parser(std::string_view source) : source_(source) {}

    Value run() {
        auto output = value(0);
        whitespace();
        if (offset_ != source_.size()) throw error(offset_, "trailing data");
        return output;
    }

private:
    void whitespace() {
        while (offset_ < source_.size() &&
               (source_[offset_] == ' ' || source_[offset_] == '\t' ||
                source_[offset_] == '\r' || source_[offset_] == '\n')) {
            ++offset_;
        }
    }

    bool take(char expected) {
        if (offset_ < source_.size() && source_[offset_] == expected) {
            ++offset_;
            return true;
        }
        return false;
    }

    void literal(std::string_view expected) {
        if (source_.substr(offset_, expected.size()) != expected) {
            throw error(offset_, "invalid literal");
        }
        offset_ += expected.size();
    }

    std::uint32_t unicode_unit() {
        if (offset_ + 4 > source_.size()) throw error(offset_, "truncated Unicode escape");
        std::uint32_t output = 0;
        for (int index = 0; index < 4; ++index) {
            const int value = hex(source_[offset_++]);
            if (value < 0) throw error(offset_ - 1, "invalid Unicode escape");
            output = output * 16 + static_cast<std::uint32_t>(value);
        }
        return output;
    }

    std::string string_value() {
        if (!take('"')) throw error(offset_, "expected string");
        std::string output;
        while (offset_ < source_.size()) {
            const unsigned char byte = static_cast<unsigned char>(source_[offset_++]);
            if (byte == '"') return output;
            if (byte < 0x20) throw error(offset_ - 1, "control character in string");
            if (byte != '\\') {
                output.push_back(static_cast<char>(byte));
                continue;
            }
            if (offset_ == source_.size()) throw error(offset_, "truncated string escape");
            const char escaped = source_[offset_++];
            switch (escaped) {
                case '"': output.push_back('"'); break;
                case '\\': output.push_back('\\'); break;
                case '/': output.push_back('/'); break;
                case 'b': output.push_back('\b'); break;
                case 'f': output.push_back('\f'); break;
                case 'n': output.push_back('\n'); break;
                case 'r': output.push_back('\r'); break;
                case 't': output.push_back('\t'); break;
                case 'u': {
                    auto codepoint = unicode_unit();
                    if (codepoint >= 0xd800 && codepoint <= 0xdbff) {
                        if (offset_ + 2 > source_.size() || source_[offset_] != '\\' ||
                            source_[offset_ + 1] != 'u') {
                            throw error(offset_, "unpaired high surrogate");
                        }
                        offset_ += 2;
                        const auto low = unicode_unit();
                        if (low < 0xdc00 || low > 0xdfff) {
                            throw error(offset_ - 4, "invalid low surrogate");
                        }
                        codepoint = 0x10000 + ((codepoint - 0xd800) << 10) + (low - 0xdc00);
                    } else if (codepoint >= 0xdc00 && codepoint <= 0xdfff) {
                        throw error(offset_ - 4, "unpaired low surrogate");
                    }
                    append_utf8(output, codepoint);
                    break;
                }
                default: throw error(offset_ - 1, "invalid string escape");
            }
        }
        throw error(offset_, "unterminated string");
    }

    Value number_value() {
        const auto start = offset_;
        take('-');
        if (take('0')) {
            if (offset_ < source_.size() && source_[offset_] >= '0' && source_[offset_] <= '9') {
                throw error(offset_, "leading zero in number");
            }
        } else {
            if (offset_ == source_.size() || source_[offset_] < '1' || source_[offset_] > '9') {
                throw error(offset_, "invalid number");
            }
            while (offset_ < source_.size() && source_[offset_] >= '0' && source_[offset_] <= '9') ++offset_;
        }
        if (take('.')) {
            const auto digits = offset_;
            while (offset_ < source_.size() && source_[offset_] >= '0' && source_[offset_] <= '9') ++offset_;
            if (digits == offset_) throw error(offset_, "fraction has no digits");
        }
        if (offset_ < source_.size() && (source_[offset_] == 'e' || source_[offset_] == 'E')) {
            ++offset_;
            if (offset_ < source_.size() && (source_[offset_] == '+' || source_[offset_] == '-')) ++offset_;
            const auto digits = offset_;
            while (offset_ < source_.size() && source_[offset_] >= '0' && source_[offset_] <= '9') ++offset_;
            if (digits == offset_) throw error(offset_, "exponent has no digits");
        }
        Value output;
        output.type = Type::number;
        output.text = std::string(source_.substr(start, offset_ - start));
        return output;
    }

    Value value(int depth) {
        if (depth > 64) throw error(offset_, "nesting exceeds 64 levels");
        whitespace();
        if (offset_ == source_.size()) throw error(offset_, "expected value");
        if (source_[offset_] == '"') {
            Value output;
            output.type = Type::string;
            output.text = string_value();
            return output;
        }
        if (source_[offset_] == '{') return object_value(depth + 1);
        if (source_[offset_] == '[') return array_value(depth + 1);
        if (source_[offset_] == 't') {
            literal("true");
            Value output;
            output.type = Type::boolean;
            output.boolean = true;
            return output;
        }
        if (source_[offset_] == 'f') {
            literal("false");
            Value output;
            output.type = Type::boolean;
            return output;
        }
        if (source_[offset_] == 'n') {
            literal("null");
            return {};
        }
        return number_value();
    }

    Value array_value(int depth) {
        ++offset_;
        Value output;
        output.type = Type::array;
        whitespace();
        if (take(']')) return output;
        for (;;) {
            output.array.push_back(value(depth));
            whitespace();
            if (take(']')) return output;
            if (!take(',')) throw error(offset_, "expected ',' or ']'");
        }
    }

    Value object_value(int depth) {
        ++offset_;
        Value output;
        output.type = Type::object;
        whitespace();
        if (take('}')) return output;
        for (;;) {
            whitespace();
            const auto key = string_value();
            whitespace();
            if (!take(':')) throw error(offset_, "expected ':'");
            auto inserted = output.object.emplace(key, value(depth));
            if (!inserted.second) throw error(offset_, "duplicate object key: " + key);
            whitespace();
            if (take('}')) return output;
            if (!take(',')) throw error(offset_, "expected ',' or '}'");
        }
    }

    std::string_view source_;
    std::size_t offset_ = 0;
};

const Value * member(const Value & object, std::string_view key) {
    if (object.type != Type::object) throw std::invalid_argument("JSON value must be an object");
    return object.find(key);
}

} // namespace

const Value * Value::find(std::string_view key) const {
    if (type != Type::object) return nullptr;
    const auto found = object.find(std::string(key));
    return found == object.end() ? nullptr : &found->second;
}

Value parse(std::string_view source) { return Parser(source).run(); }

std::string quote(std::string_view value) {
    std::string output = "\"";
    for (const unsigned char byte : value) {
        switch (byte) {
            case '"': output += "\\\""; break;
            case '\\': output += "\\\\"; break;
            case '\b': output += "\\b"; break;
            case '\f': output += "\\f"; break;
            case '\n': output += "\\n"; break;
            case '\r': output += "\\r"; break;
            case '\t': output += "\\t"; break;
            default:
                if (byte < 0x20) {
                    static constexpr char digits[] = "0123456789abcdef";
                    output += "\\u00";
                    output.push_back(digits[byte >> 4]);
                    output.push_back(digits[byte & 15]);
                } else {
                    output.push_back(static_cast<char>(byte));
                }
        }
    }
    output.push_back('"');
    return output;
}

std::string stringify(const Value & value) {
    switch (value.type) {
        case Type::null: return "null";
        case Type::boolean: return value.boolean ? "true" : "false";
        case Type::number: return value.text;
        case Type::string: return quote(value.text);
        case Type::array: {
            std::string output = "[";
            for (std::size_t index = 0; index < value.array.size(); ++index) {
                if (index) output.push_back(',');
                output += stringify(value.array[index]);
            }
            output.push_back(']');
            return output;
        }
        case Type::object: {
            std::string output = "{";
            bool first = true;
            for (const auto & entry : value.object) {
                if (!first) output.push_back(',');
                first = false;
                output += quote(entry.first) + ":" + stringify(entry.second);
            }
            output.push_back('}');
            return output;
        }
    }
    throw std::logic_error("unknown JSON type");
}

std::string string(const Value & object, std::string_view key, std::string fallback) {
    const auto * value = member(object, key);
    if (!value || value->type == Type::null) return fallback;
    if (value->type != Type::string) throw std::invalid_argument(std::string(key) + " must be a string");
    return value->text;
}

bool boolean(const Value & object, std::string_view key, bool fallback) {
    const auto * value = member(object, key);
    if (!value || value->type == Type::null) return fallback;
    if (value->type != Type::boolean) throw std::invalid_argument(std::string(key) + " must be a boolean");
    return value->boolean;
}

double number(const Value & object, std::string_view key, double fallback) {
    const auto * value = member(object, key);
    if (!value || value->type == Type::null) return fallback;
    if (value->type != Type::number) throw std::invalid_argument(std::string(key) + " must be a number");
    char * end = nullptr;
    errno = 0;
    const auto output = std::strtod(value->text.c_str(), &end);
    if (errno == ERANGE || !end || *end || !std::isfinite(output)) {
        throw std::invalid_argument(std::string(key) + " is outside the finite number range");
    }
    return output;
}

std::uint64_t u64(const Value & object, std::string_view key, std::uint64_t fallback) {
    const auto * value = member(object, key);
    if (!value || value->type == Type::null) return fallback;
    std::string text;
    if (value->type == Type::number || value->type == Type::string) text = value->text;
    else throw std::invalid_argument(std::string(key) + " must be an unsigned integer or string");
    if (text.empty() || text[0] == '-') throw std::invalid_argument(std::string(key) + " must be unsigned");
    std::size_t used = 0;
    unsigned long long output = 0;
    try {
        output = std::stoull(text, &used, 10);
    } catch (const std::exception &) {
        throw std::invalid_argument(std::string(key) + " is outside the uint64 range");
    }
    if (used != text.size()) throw std::invalid_argument(std::string(key) + " must be an integer");
    return static_cast<std::uint64_t>(output);
}

std::uint32_t u32(const Value & object, std::string_view key, std::uint32_t fallback) {
    const auto value = u64(object, key, fallback);
    if (value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(std::string(key) + " is outside the uint32 range");
    }
    return static_cast<std::uint32_t>(value);
}

} // namespace yue2::server::json
