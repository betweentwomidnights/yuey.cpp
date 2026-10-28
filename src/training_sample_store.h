#pragma once

#include <cstddef>
#include <filesystem>
#include <functional>
#include <future>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace yue2::detail {
// Disk-backed training examples retain at most one decoded sample, plus one
// prefetched sample. Validate each example during construction; reject files
// replaced after validation. A prefetch performs the same before/after-load
// stamp checks on its background thread, and get() checks the stamp again
// when it consumes the result. Model-specific shape/content validation
// belongs to the caller's inspector. The loader must be safe to call from a
// background thread while the caller uses the store.
template <class Sample> class TrainingSampleStore {
  public:
    using Loader = std::function<Sample(const std::string &)>;
    using Inspector = std::function<void(const Sample &)>;
    using Progress = std::function<void(size_t, size_t)>;
    using PathInspector = std::function<void(const std::string &)>;
    // Deferred stores never touch the files at construction: inspect_path
    // supplies every identity fact and the loader verifies each file itself
    // when consumed (rolling caches) or there is no file (resident inputs).
    TrainingSampleStore(const std::vector<std::string> &paths, bool streaming,
                        Loader loader, Inspector inspect, Progress progress = {}, PathInspector inspect_path = {},
                        bool deferred = false)
        : paths_(paths), streaming_(streaming), deferred_(deferred), loader_(std::move(loader)) {
        if (deferred_ && (!streaming_ || !inspect_path))
            throw std::runtime_error("deferred samples require streaming and a path inspector");
        size_t completed = 0;
        for (const auto &path : paths_) {
            if (deferred_) {
                inspect_path(path);
                stamps_.emplace_back();
                if (progress) progress(++completed, paths_.size());
                continue;
            }
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
    ~TrainingSampleStore() {
        if (pending_.valid()) pending_.wait();
    }
    TrainingSampleStore(const TrainingSampleStore &) = delete;
    TrainingSampleStore &operator=(const TrainingSampleStore &) = delete;
    const Sample &get(size_t index) {
        if (!streaming_) return samples_.at(index);
        const auto &path = paths_.at(index);
        if (!deferred_ && stamp(path) != stamps_.at(index))
            throw std::runtime_error("training sample changed after validation: " + path);
        if (!current_ || current_index_ != index) {
            current_.reset();
            if (pending_.valid() && pending_index_ == index) {
                current_.emplace(pending_.get());
            } else {
                current_.emplace(loader_(path));
            }
            current_index_ = index;
            if (!deferred_ && stamp(path) != stamps_.at(index))
                throw std::runtime_error("training sample changed while loading: " + path);
        }
        return *current_;
    }
    // Streaming only: start loading a sample the caller will get() next. A
    // pending load for another index is waited for and discarded.
    void prefetch(size_t index) {
        if (!streaming_ || index >= paths_.size()) return;
        if ((current_ && current_index_ == index) || (pending_.valid() && pending_index_ == index))
            return;
        if (pending_.valid()) pending_.wait();
        pending_index_ = index;
        pending_ = std::async(std::launch::async, [this, index] {
            const auto &path = paths_[index];
            if (!deferred_ && stamp(path) != stamps_[index])
                throw std::runtime_error("training sample changed after validation: " + path);
            auto sample = loader_(path);
            if (!deferred_ && stamp(path) != stamps_[index])
                throw std::runtime_error("training sample changed while loading: " + path);
            return sample;
        });
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
    bool deferred_;
    Loader loader_;
    std::vector<Sample> samples_;
    std::optional<Sample> current_;
    size_t current_index_ = 0;
    // Last member: a running prefetch may use everything above.
    std::future<Sample> pending_;
    size_t pending_index_ = 0;
};
} // namespace yue2::detail
