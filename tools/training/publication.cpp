#include "training/publication.h"

#include "training/gguf_file.h"
#include "training/preparation.h"
#include "training/preparation_detail.h"
#include "training/sha256.h"

#include <cstring>
#include <fstream>
#include <set>

namespace yue2::training {
namespace fs = std::filesystem;
using namespace detail;
namespace {

// The GGUF gguf-py's GGUFWriter writes: metadata in the order it was added,
// then tensor infos, each tensor's data padded to 32 bytes.
class GgufWriter {
public:
    explicit GgufWriter(std::string architecture) { string("general.architecture", std::move(architecture)); }
    void string(const std::string & key, const std::string & value) {
        begin(key, GGUF_TYPE_STRING);
        text(metadata_, value);
    }
    void u32(const std::string & key, std::uint32_t value) {
        begin(key, GGUF_TYPE_UINT32);
        raw(metadata_, &value, 4);
    }
    void f32(const std::string & key, float value) {
        begin(key, GGUF_TYPE_FLOAT32);
        raw(metadata_, &value, 4);
    }
    void tensor(const std::string & name, int dims, const int64_t * ne, std::string data) {
        tensors_.push_back({name, std::vector<int64_t>(ne, ne + dims), std::move(data)});
    }
    void write(const fs::path & path) const {
        std::string out = "GGUF";
        const std::uint32_t version = 3;
        const std::uint64_t tensor_count = tensors_.size(), kv_count = keys_;
        raw(out, &version, 4);
        raw(out, &tensor_count, 8);
        raw(out, &kv_count, 8);
        out += metadata_;
        std::uint64_t offset = 0;
        for (const auto & t : tensors_) {
            text(out, t.name);
            const auto dims = static_cast<std::uint32_t>(t.ne.size());
            raw(out, &dims, 4);
            for (const auto extent : t.ne) { const auto value = static_cast<std::uint64_t>(extent); raw(out, &value, 8); }
            const std::uint32_t type = GGML_TYPE_F32;
            raw(out, &type, 4);
            raw(out, &offset, 8);
            offset += padded(t.data.size());
        }
        out.append(padded(out.size()) - out.size(), '\0');
        for (const auto & t : tensors_) {
            out += t.data;
            out.append(padded(t.data.size()) - t.data.size(), '\0');
        }
        std::ofstream stream(path, std::ios::binary | std::ios::trunc);
        if (!stream.write(out.data(), static_cast<std::streamsize>(out.size()))) throw std::runtime_error("Cannot write " + python_path(path));
    }

private:
    struct Tensor { std::string name; std::vector<int64_t> ne; std::string data; };
    static std::size_t padded(std::size_t size) { return (size + 31) / 32 * 32; }
    static void raw(std::string & out, const void * data, std::size_t size) { out.append(static_cast<const char *>(data), size); }
    static void text(std::string & out, const std::string & value) {
        const std::uint64_t size = value.size();
        raw(out, &size, 8);
        out += value;
    }
    void begin(const std::string & key, gguf_type type) {
        text(metadata_, key);
        const auto value = static_cast<std::uint32_t>(type);
        raw(metadata_, &value, 4);
        ++keys_;
    }
    std::string metadata_;
    std::uint64_t keys_ = 0;
    std::vector<Tensor> tensors_;
};

} // namespace

json::Value publish(const fs::path & run_path) {
    const auto run = from_utf8(resolved_path(run_path));
    const auto runtime = read_json(run / "runtime.json");
    const auto request = read_json(run / "request.json");
    const auto state = read_json(run / "state.json");
    const auto & ar_record = member(state, "ar_selection");
    const auto & nar_record = member(state, "nar_selection");
    const std::vector<fs::path> sources{from_utf8(text(ar_record, "adapter")), from_utf8(text(nar_record, "adapter"))};
    const std::vector<const json::Value *> records{&ar_record, &nar_record};
    for (std::size_t i = 0; i < 2; ++i) {
        if (digest(sources[i]) != text(*records[i], "adapter_sha256")) throw std::invalid_argument("Selected checkpoint changed");
    }
    const GgufFile ar(sources[0]), nar(sources[1]);
    const auto base = ar.string("yue2.adapter.base_sha256");
    const auto rank = ar.u32("yue2.adapter.rank");
    const auto alpha = ar.f32("yue2.adapter.alpha");
    if (base != nar.string("yue2.adapter.base_sha256") || rank != nar.u32("yue2.adapter.rank") || alpha != nar.f32("yue2.adapter.alpha")) {
        throw std::invalid_argument("Expert pair base/rank/scaling differs");
    }
    for (const auto & name : ar.tensors()) if (name.find(".nar_") != std::string::npos) throw std::invalid_argument("Adapter contains targets outside its intended expert");
    for (const auto & name : nar.tensors()) if (name.find(".nar_") == std::string::npos) throw std::invalid_argument("Adapter contains targets outside its intended expert");
    const auto folder = from_utf8(resolved_path(from_utf8(text(runtime, "models_dir"))));
    fs::create_directories(folder);
    const auto target = folder / (text(request, "id") + "-F32-LoRA.gguf");
    const auto sidecar = with_suffix(target, ".training.json");
    if (fs::exists(sidecar)) {
        const auto previous = read_json(sidecar);
        if (text(previous, "run") != python_path(run) || digest(target) != text(previous, "sha256")) {
            throw std::invalid_argument("Published name already belongs to another artifact");
        }
        return previous;
    }
    if (fs::exists(target)) throw std::invalid_argument("Adapter name already exists; inspect the unregistered export");
    const auto temporary = with_suffix(target, ".gguf.partial");
    if (fs::exists(temporary)) throw std::invalid_argument("Incomplete publication exists; inspect before retrying");
    const auto family = ar.string("yue2.adapter.type");
    const auto & settings = member(state, "settings");
    GgufWriter writer("yue2_lora");
    writer.string("general.name", text(state, "name"));
    writer.string("yue2.component", "generation-adapter");
    writer.string("yue2.adapter.type", family == "dora" ? "mixed_lora_dora" : "lora");
    writer.string("yue2.adapter.base_sha256", base);
    writer.string("yue2.training.trigger", text(settings, "trigger"));
    writer.string("yue2.training.ar_source_sha256", text(ar_record, "adapter_sha256"));
    writer.string("yue2.training.nar_source_sha256", text(nar_record, "adapter_sha256"));
    for (const auto & key : ar.keys()) {
        if (key.rfind("yue2.adapter.dora.", 0) != 0) continue;
        switch (ar.type(key.c_str())) {
            case GGUF_TYPE_STRING: writer.string(key, ar.string(key.c_str())); break;
            case GGUF_TYPE_UINT32: writer.u32(key, ar.u32(key.c_str())); break;
            case GGUF_TYPE_FLOAT32: writer.f32(key, ar.f32(key.c_str())); break;
            default: throw std::invalid_argument("Unsupported DoRA metadata type: " + key);
        }
    }
    writer.u32("yue2.adapter.rank", rank);
    writer.f32("yue2.adapter.alpha", alpha);
    std::set<std::string> seen, targets;
    std::vector<std::pair<std::string, std::string>> expected;
    for (const auto * reader : {&ar, &nar}) {
        for (const auto & name : reader->tensors()) {
            const auto & tensor = reader->tensor(name);
            if (seen.count(name) || tensor.type != GGML_TYPE_F32) throw std::invalid_argument("Duplicate or unsupported adapter tensor");
            seen.insert(name);
            targets.insert(name.substr(0, name.rfind('.')));
            auto data = reader->payload(name);
            expected.emplace_back(name, data);
            writer.tensor(name, ggml_n_dims(&tensor), tensor.ne, std::move(data));
        }
    }
    writer.u32("yue2.adapter.target_count", static_cast<std::uint32_t>(targets.size()));
    writer.u32("yue2.adapter.replacement_count", 0);
    writer.write(temporary);
    {
        // Verify the pair byte for byte before making it visible to /loras.
        const GgufFile merged(temporary);
        for (const auto & [name, data] : expected) {
            if (merged.payload(name) != data) throw std::invalid_argument("Export tensor verification failed");
        }
    }
    const auto prompts = read_json(run / "data" / text(settings, "family") / "prompts.json");
    const auto companion = text(member(runtime, "models"), "companion");
    auto result = make({{"format", S("yue2-trained-style-v1")}, {"name", member(state, "name")}, {"run", P(run)}, {"adapter", P(target)},
        {"sha256", S(digest(temporary))}, {"trigger", member(settings, "trigger")}, {"instrumental", B(true)},
        {"prompts", member(member(prompts, "dice"), "instrumental")}, {"base_sha256", S(base)},
        {"ar_sha256", member(ar_record, "adapter_sha256")}, {"nar_sha256", member(nar_record, "adapter_sha256")},
        {"companion", make({{"name", S(from_utf8(companion).filename().u8string())}, {"path", S(companion)},
                            {"sha256", S(digest(from_utf8(companion)))}, {"strength", I(1)}, {"order", S("before-style")}})},
        {"settings", settings}, {"selection", S("Recording-balanced heldout semantic CE and flow MSE; user publishes after review")}});
    fs::rename(temporary, target);
    save(sidecar, result);
    return result;
}

long long release_caches(const fs::path & run_path) {
    const auto run = from_utf8(resolved_path(run_path));
    const auto state = read_json(run / "state.json");
    const auto family = text(member(state, "settings"), "family");
    std::set<std::string> keep;
    for (const char * key : {"ar_selection", "nar_selection"}) keep.insert(resolved_path(from_utf8(text(member(state, key), "adapter"))));
    std::vector<fs::path> files;
    const auto gguf_under = [](const fs::path & folder, std::vector<fs::path> & into) {
        if (!fs::is_directory(folder)) return;
        for (const auto & entry : fs::recursive_directory_iterator(folder)) {
            if (entry.is_regular_file() && entry.path().extension() == ".gguf") into.push_back(entry.path());
        }
    };
    gguf_under(run / "nar-conditioning", files);
    if (fs::is_directory(run / "data" / "examples")) {
        for (const auto & entry : fs::directory_iterator(run / "data" / "examples")) {
            if (fs::is_regular_file(entry.path() / "source.wav")) files.push_back(entry.path() / "source.wav");
        }
    }
    for (const auto & folder : {run / "data" / from_utf8(family), run / "nar-run"}) {
        std::vector<fs::path> found;
        gguf_under(folder, found);
        for (const auto & path : found) if (!keep.count(resolved_path(path))) files.push_back(path);
    }
    // Best effort: a file that cannot be removed now stays for a later pass.
    long long released = 0;
    for (const auto & path : files) {
        std::error_code error;
        const auto size = fs::file_size(path, error);
        if (error) continue;
        if (fs::remove(path, error) && !error) released += static_cast<long long>(size);
    }
    return released;
}

} // namespace yue2::training
