// Files the training workflow writes the way the Python tools wrote them, so
// a run prepared by either can be resumed by the other: identities hash some
// of these bytes.
#pragma once

#include "server/json.h"

#include <filesystem>
#include <string>

namespace yue2::training {

// str(Path(path)) in Python: separators normalized ("\" on Windows), repeated
// separators and "." parts dropped, no trailing separator. ".." is kept.
std::string python_path(const std::filesystem::path & path);
// str(Path(path).resolve()).
std::string resolved_path(const std::filesystem::path & path);
std::filesystem::path from_utf8(const std::string & path);

// Path.write_text(): "\n" becomes "\r\n" on Windows.
void write_text(const std::filesystem::path & path, const std::string & text);
// Path.read_text(): universal newlines.
std::string read_text(const std::filesystem::path & path);
std::string read_bytes(const std::filesystem::path & path);
server::json::Value read_json(const std::filesystem::path & path);

// prepare_ar_dataset.save(): json.dumps(indent=2) and a newline, written to
// PATH.partial and moved over PATH, retrying while Windows readers hold it.
void save(const std::filesystem::path & path, const server::json::Value & value);

// Path.with_suffix(suffix).
std::filesystem::path with_suffix(const std::filesystem::path & path, const std::string & suffix);

} // namespace yue2::training
