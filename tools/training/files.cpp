#include "training/files.h"

#include <chrono>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace yue2::training {
namespace fs = std::filesystem;

fs::path from_utf8(const std::string & path) { return fs::u8path(path); }

std::string python_path(const fs::path & path) {
    const auto text = path.u8string();
#ifdef _WIN32
    const auto separator = [](char c) { return c == '\\' || c == '/'; };
    std::string drive, root;
    std::size_t at = 0;
    if (text.size() >= 2 && separator(text[0]) && separator(text[1])) {
        // \\server\share is the drive of a UNC path.
        std::size_t end = 2, parts = 0;
        while (end < text.size() && parts < 2) {
            const auto start = end;
            while (end < text.size() && !separator(text[end])) ++end;
            if (end > start) ++parts;
            if (parts < 2 && end < text.size()) ++end;
        }
        drive = "\\\\" + text.substr(2, end - 2);
        for (auto & c : drive) if (c == '/') c = '\\';
        at = end;
    } else if (text.size() >= 2 && text[1] == ':') {
        drive = text.substr(0, 2);
        at = 2;
    }
    if (at < text.size() && separator(text[at])) root = "\\";
    std::string output = drive + root;
    bool first = true;
    while (at < text.size()) {
        while (at < text.size() && separator(text[at])) ++at;
        const auto start = at;
        while (at < text.size() && !separator(text[at])) ++at;
        const auto part = text.substr(start, at - start);
        if (part.empty() || part == ".") continue;
        if (!first) output += "\\";
        output += part;
        first = false;
    }
#else
    std::string root;
    if (!text.empty() && text[0] == '/') root = text.size() > 1 && text[1] == '/' && (text.size() == 2 || text[2] != '/') ? "//" : "/";
    std::string output = root;
    bool first = true;
    for (std::size_t at = 0; at < text.size();) {
        while (at < text.size() && text[at] == '/') ++at;
        const auto start = at;
        while (at < text.size() && text[at] != '/') ++at;
        const auto part = text.substr(start, at - start);
        if (part.empty() || part == ".") continue;
        if (!first) output += "/";
        output += part;
        first = false;
    }
#endif
    return output.empty() ? "." : output;
}

std::string resolved_path(const fs::path & path) {
    return python_path(fs::weakly_canonical(fs::absolute(path)));
}

std::string read_bytes(const fs::path & path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("Cannot read " + python_path(path));
    return {std::istreambuf_iterator<char>(stream), {}};
}

std::string read_text(const fs::path & path) {
    const auto raw = read_bytes(path);
    std::string text;
    text.reserve(raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i) {
        if (raw[i] == '\r') {
            text.push_back('\n');
            if (i + 1 < raw.size() && raw[i + 1] == '\n') ++i;
        } else {
            text.push_back(raw[i]);
        }
    }
    return text;
}

server::json::Value read_json(const fs::path & path) { return server::json::parse(read_bytes(path)); }

namespace {

std::string platform_newlines(const std::string & text) {
#ifdef _WIN32
    std::string output;
    output.reserve(text.size() + text.size() / 16);
    for (const char c : text) {
        if (c == '\n') output.push_back('\r');
        output.push_back(c);
    }
    return output;
#else
    return text;
#endif
}

void write_bytes(const fs::path & path, const std::string & bytes) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()))) {
        throw std::runtime_error("Cannot write " + python_path(path));
    }
}

} // namespace

void write_text(const fs::path & path, const std::string & text) { write_bytes(path, platform_newlines(text)); }

fs::path with_suffix(const fs::path & path, const std::string & suffix) {
    auto output = path;
    output.replace_extension(fs::u8path(suffix));
    return output;
}

void save(const fs::path & path, const server::json::Value & value) {
    if (path.has_parent_path()) fs::create_directories(path.parent_path());
    const auto temporary = with_suffix(path, path.extension().u8string() + ".partial");
    write_text(temporary, server::json::pretty(value) + "\n");
    // Windows readers can briefly deny replacement of the destination. Keep
    // the old complete JSON visible while retrying; never truncate it in place.
    for (int attempt = 0;; ++attempt) {
#ifdef _WIN32
        if (MoveFileExW(temporary.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING)) return;
        if (attempt == 49) throw std::runtime_error("Cannot replace " + python_path(path));
#else
        std::error_code error;
        fs::rename(temporary, path, error);
        if (!error) return;
        if (attempt == 49) throw std::runtime_error("Cannot replace " + python_path(path));
#endif
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

} // namespace yue2::training
