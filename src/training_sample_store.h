#pragma once

#include <cstddef>
#include <filesystem>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace yue2::detail {
// Disk-backed training examples retain at most one decoded sample. Validate
// each example during construction; reject files replaced after validation.
// Model-specific shape/content validation belongs to the caller's inspector.
template <class Sample> class TrainingSampleStore {
  public:
    using Loader = std::function<Sample(const std::string &)>;
    using Inspector = std::function<void(const Sample &)>;
    using Progress = std::function<void(size_t, size_t)>;
    using PathInspector = std::function<void(const std::string &)>;
    TrainingSampleStore(const std::vector<std::string> &paths, bool streaming,
                        Loader loader, Inspector inspect, Progress progress = {}, PathInspector inspect_path = {})
        : paths_(paths), streaming_(streaming), loader_(std::move(loader)) {
        size_t completed = 0;
        for (const auto &path : paths_) {
            stamps_.push_back(stamp(path));
            if (inspect_path) {
                // An equivalent header/content validator can avoid decoding a
                // streaming sample before it is actually consumed.
                inspect_path(path);
                if (!streaming_) samples_.push_back(loader_(path));
            } else {
                auto sample = loader_(path);
                inspect(sample);
                if (!streaming_) samples_.push_back(std::move(sample));
            }
            if (stamp(path) != stamps_.back())
                throw std::runtime_error("training sample changed while loading: " + path);
            if (progress) progress(++completed, paths_.size());
        }
    }
    const Sample &get(size_t index) {
        if (!streaming_) return samples_.at(index);
        const auto &path = paths_.at(index);
        if (stamp(path) != stamps_.at(index))
            throw std::runtime_error("training sample changed after validation: " + path);
        if (!current_ || current_index_ != index) {
            current_.reset();
            current_.emplace(loader_(path));
            current_index_ = index;
            if (stamp(path) != stamps_.at(index))
                throw std::runtime_error("training sample changed while loading: " + path);
        }
        return *current_;
    }
    size_t size() const { return paths_.size(); }
    bool empty() const { return paths_.empty(); }

  private:
    using Stamp = std::pair<std::uintmax_t, std::filesystem::file_time_type>;
    static Stamp stamp(const std::string &path) {
        return {std::filesystem::file_size(path), std::filesystem::last_write_time(path)};
    }
    std::vector<std::string> paths_;
    std::vector<Stamp> stamps_;
    bool streaming_;
    Loader loader_;
    std::vector<Sample> samples_;
    std::optional<Sample> current_;
    size_t current_index_ = 0;
};
} // namespace yue2::detail
