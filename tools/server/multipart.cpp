#include "multipart.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <stdexcept>

namespace yue2::server {
namespace {

std::string lower_ascii(std::string value) {
    for (auto & character : value) {
        character = static_cast<char>(
            std::tolower(static_cast<unsigned char>(character)));
    }
    return value;
}

std::string trim(std::string value) {
    const auto space = [](unsigned char character) { return std::isspace(character) != 0; };
    value.erase(value.begin(), std::find_if(
        value.begin(), value.end(), [&](char character) { return !space(character); }));
    value.erase(std::find_if(
        value.rbegin(), value.rend(), [&](char character) { return !space(character); }).base(),
        value.end());
    return value;
}

std::string parameter(const std::string & header, const std::string & name) {
    const auto lowered = lower_ascii(header);
    const auto needle = name + '=';
    std::size_t offset = 0;
    for (;;) {
        offset = lowered.find(needle, offset);
        if (offset == std::string::npos) return {};
        if (offset == 0 || header[offset - 1] == ';' ||
            std::isspace(static_cast<unsigned char>(header[offset - 1]))) break;
        offset += needle.size();
    }
    auto begin = offset + needle.size();
    if (begin < header.size() && header[begin] == '"') {
        ++begin;
        const auto end = header.find('"', begin);
        if (end == std::string::npos) throw std::invalid_argument("unterminated multipart parameter");
        return header.substr(begin, end - begin);
    }
    const auto end = header.find(';', begin);
    return trim(header.substr(begin, end == std::string::npos ? end : end - begin));
}

MultipartPart parse_part(std::string segment) {
    const auto header_end = segment.find("\r\n\r\n");
    if (header_end == std::string::npos) throw std::invalid_argument("multipart part has no header terminator");
    MultipartPart part;
    part.data = segment.substr(header_end + 4);
    std::istringstream headers(segment.substr(0, header_end));
    std::string line;
    while (std::getline(headers, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const auto colon = line.find(':');
        if (colon == std::string::npos) throw std::invalid_argument("invalid multipart header");
        const auto name = lower_ascii(trim(line.substr(0, colon)));
        const auto value = trim(line.substr(colon + 1));
        if (name == "content-disposition") {
            part.name = parameter(value, "name");
            part.filename = parameter(value, "filename");
        } else if (name == "content-type") {
            part.content_type = value;
        }
    }
    if (part.name.empty()) throw std::invalid_argument("multipart part has no name");
    return part;
}

} // namespace

std::optional<std::string> multipart_boundary(const std::string & content_type) {
    if (lower_ascii(content_type).rfind("multipart/form-data", 0) != 0) return std::nullopt;
    auto value = parameter(content_type, "boundary");
    if (value.empty() || value.size() > 200 || value.find_first_of("\r\n") != std::string::npos) {
        throw std::invalid_argument("invalid multipart boundary");
    }
    return value;
}

std::vector<MultipartPart> parse_multipart(
    const std::string & body,
    const std::string & boundary) {
    const std::string delimiter = "--" + boundary;
    std::vector<MultipartPart> output;
    auto offset = body.find(delimiter);
    if (offset != 0) throw std::invalid_argument("multipart body does not start with its boundary");
    offset += delimiter.size();
    while (offset < body.size()) {
        if (body.compare(offset, 2, "--") == 0) return output;
        if (body.compare(offset, 2, "\r\n") != 0) {
            throw std::invalid_argument("invalid multipart boundary separator");
        }
        offset += 2;
        const auto next = body.find("\r\n" + delimiter, offset);
        if (next == std::string::npos) throw std::invalid_argument("unterminated multipart body");
        output.push_back(parse_part(body.substr(offset, next - offset)));
        offset = next + 2 + delimiter.size();
    }
    throw std::invalid_argument("unterminated multipart body");
}

} // namespace yue2::server
