#include "training/workflow.h"

#include "training/dataset.h"
#include "training/process.h"
#include "training/sha256.h"
#include "training/unicode.h"
#include "yue2/audio.h"

#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <stdexcept>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace yue2::training {
namespace fs = std::filesystem;
namespace {

constexpr double gib = 1073741824.0;
// One frozen KL teacher row per target frame: MUSIC_END + 32768 codec classes, F32.
constexpr long long teacher_row_bytes = (32768 + 1) * 4;
// Measured CUDA AR fitter host working set on the Koan corpus, rank 32
// (docs/training-history.md, recipe 4): a 3.1 GiB transient while loading
// weights, then 1.13 GiB steady including two 30-second windows. CPU fitting
// also holds the weights themselves; that addition is derived, not measured.
constexpr double ar_host_load_gib = 3.1;
constexpr double ar_host_steady_gib = 0.95;
constexpr long long ar_streamed_windows = 2;
// Rank-32 adapters train all seven projections in 28 blocks: 34,865,152 F32
// parameters per expert. A checkpoint is the adapter plus a state holding the
// factors and both AdamW moments, 16 bytes per parameter. Pruned fits keep at
// most step zero, heldout-best and latest, plus one being written. Probe
// diagnostics are bounded by 1 GiB per expert. Derived, not measured.
constexpr double parameters_per_rank = 34865152.0 / 32;
constexpr long long checkpoints_kept = 4;
constexpr long long diagnostics_gib = 1;
// Rolling conditioning keeps at most one segment of training windows plus the
// heldout set (during an evaluating segment) on disk at once.
constexpr long long rolling_window_updates = 25;

const std::vector<std::string> teacher_storage{"cache", "online"};
const std::vector<std::string> conditioning_storage{"cache", "rolling", "resident"};
const std::vector<std::string> request_settings{
    "id", "dataset", "name", "trigger", "window_seconds", "ar_steps", "nar_steps", "rank", "family", "ar_lr", "nar_lr",
    "kl_weight", "seed", "eval_every", "evaluation_scope", "keep_caches", "teacher_storage", "conditioning_storage"};

std::string utf8(const fs::path & path) { return path.u8string(); }

std::string read_text(const fs::path & path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("Cannot read " + utf8(path));
    return {std::istreambuf_iterator<char>(stream), {}};
}

json::Value read_json(const fs::path & path) { return json::parse(read_text(path)); }

double number_of(const json::Value & value) { return std::strtod(value.text.c_str(), nullptr); }

long long integer_of(const json::Value & object, const char * key) {
    const auto * value = object.find(key);
    if (!value || value->type != json::Type::number) throw std::invalid_argument(std::string("Row is missing ") + key);
    return json::is_integer(*value) ? std::strtoll(value->text.c_str(), nullptr, 10)
                                    : static_cast<long long>(number_of(*value));
}

std::string text_of(const json::Value & object, const char * key, const std::string & fallback = "") {
    const auto * value = object.find(key);
    return value && value->type == json::Type::string ? value->text : fallback;
}

bool has(const json::Value & object, const char * key) { return object.find(key) != nullptr; }

std::string lower_ascii(std::string text) {
    for (auto & c : text) c = static_cast<char>(c >= 'A' && c <= 'Z' ? c + 32 : c);
    return text;
}

// Python's str() of a number: the bounds in settings messages.
std::string bound(double value, bool integer) {
    return integer ? std::to_string(static_cast<long long>(value)) : json::python_repr(value);
}

// integer_bounds: the bounds are written as integers in training_workflow.py.
json::Value number_setting(const json::Value & request, const char * key, const json::Value & fallback,
                           double minimum, double maximum, bool integer, bool integer_bounds = true) {
    const auto * found = request.find(key);
    const auto & value = found ? *found : fallback;
    const bool bounds_are_integers = integer || integer_bounds;
    if (value.type != json::Type::number) {
        throw std::invalid_argument(std::string(key) + " must be between " + bound(minimum, bounds_are_integers) + " and " +
                                    bound(maximum, bounds_are_integers));
    }
    const double number = number_of(value);
    if (!std::isfinite(number) || number < minimum || number > maximum) {
        throw std::invalid_argument(std::string(key) + " must be between " + bound(minimum, bounds_are_integers) + " and " +
                                    bound(maximum, bounds_are_integers));
    }
    if (!integer) return value;
    if (number != std::floor(number)) throw std::invalid_argument(std::string(key) + " must be an integer");
    return json::from_integer(static_cast<long long>(number));
}

bool option_true(const json::Value & options, const char * key) {
    const auto * value = options.find(key);
    return value && value->type == json::Type::boolean && value->boolean;
}

long long window_bytes(const json::Value & row) {
    return (integer_of(row, "prefix_tokens") + integer_of(row, "frames") + 1) * 114688 + integer_of(row, "frames") * 256;
}

std::vector<const json::Value *> train_rows(const std::vector<json::Value> & eligible) {
    std::vector<const json::Value *> output;
    for (const auto & row : eligible) if (text_of(row, "split") == "train") output.push_back(&row);
    return output;
}

long long teacher_bytes(const std::vector<json::Value> & eligible, const json::Value & options) {
    const auto * weight = options.find("kl_weight");
    if (!weight || !(number_of(*weight) > 0)) return 0;
    long long total = 0;
    for (const auto * row : train_rows(eligible)) total += integer_of(*row, "frames") * teacher_row_bytes;
    return total;
}

#ifdef _WIN32
std::string fold_case(const fs::path & name) {
    const auto wide = name.wstring();
    if (wide.empty()) return {};
    std::wstring folded(wide.size() * 2 + 8, L'\0');
    const int size = LCMapStringEx(LOCALE_NAME_INVARIANT, LCMAP_LOWERCASE, wide.c_str(), static_cast<int>(wide.size()),
                                   folded.data(), static_cast<int>(folded.size()), nullptr, nullptr, 0);
    folded.resize(size > 0 ? static_cast<std::size_t>(size) : 0);
    return fs::path(folded).u8string();
}
#endif

std::string sort_key(const fs::path & path) {
#ifdef _WIN32
    return fold_case(path.filename());
#else
    return path.filename().string();
#endif
}

} // namespace

std::vector<fs::path> sorted_entries(const fs::path & folder) {
    std::vector<std::pair<std::string, fs::path>> entries;
    for (const auto & entry : fs::directory_iterator(folder)) entries.emplace_back(sort_key(entry.path()), entry.path());
    std::sort(entries.begin(), entries.end(), [](const auto & a, const auto & b) { return a.first < b.first; });
    std::vector<fs::path> output;
    for (auto & entry : entries) output.push_back(std::move(entry.second));
    return output;
}

json::Value settings(const json::Value & request) {
    if (request.type != json::Type::object) throw std::invalid_argument("Training request must be an object");
    const auto * keep = request.find("keep_caches");
    if (keep && keep->type != json::Type::boolean) throw std::invalid_argument("keep_caches must be true or false");
    const auto family = request.find("family") ? text_of(request, "family", "\x01") : "dora";
    if (family != "lora" && family != "dora") throw std::invalid_argument("AR family must be lora or dora");
    std::string trigger;
    if (const auto * value = request.find("trigger")) {
        trigger = value->type == json::Type::string ? value->text : "";
    } else {
        const auto * id = request.find("id");
        if (!id || id->type != json::Type::string) throw std::invalid_argument("Training request needs an id");
        trigger = id->text + "_style";
    }
    const bool trigger_ok = !trigger.empty() && trigger.size() <= 80 &&
        trigger.find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-") == std::string::npos;
    if (!trigger_ok) throw std::invalid_argument("Trigger must be one word using letters, numbers, hyphens or underscores");

    auto options = json::object();
    auto & o = options.object;
    // Rank 1 is not a useful adapter, and GGML's CUDA OUT_PROD rejects its shape.
    o["window_seconds"] = number_setting(request, "window_seconds", json::from_integer(30), 2, 60, true);
    o["rank"] = number_setting(request, "rank", json::from_integer(32), 2, 128, true);
    o["ar_steps"] = number_setting(request, "ar_steps", json::from_integer(800), 1, 10000, true);
    o["nar_steps"] = number_setting(request, "nar_steps", json::from_integer(400), 1, 10000, true);
    o["ar_lr"] = number_setting(request, "ar_lr", json::from_double(2e-5), 1e-7, .001, false, false);
    o["nar_lr"] = number_setting(request, "nar_lr", json::from_double(3e-5), 1e-7, .001, false, false);
    o["kl_weight"] = number_setting(request, "kl_weight", json::from_double(.2), 0, 5, false);
    o["seed"] = number_setting(request, "seed", json::from_integer(831001), 0, 4294967295.0, true);
    o["eval_every"] = number_setting(request, "eval_every", json::from_integer(100), 1, 1000, true);
    o["family"] = json::from_string(family);
    o["trigger"] = json::from_string(trigger);
    o["keep_caches"] = json::from_bool(keep && keep->boolean);

    long long version = 1;
    if (const auto * value = request.find("recipe_version")) {
        if (!json::is_integer(*value)) throw std::invalid_argument("Unsupported training recipe version");
        version = std::strtoll(value->text.c_str(), nullptr, 10);
    }
    if (version < 1 || version > 5) throw std::invalid_argument("Unsupported training recipe version");
    if (version >= 2) {
        const auto scope = request.find("evaluation_scope") ? text_of(request, "evaluation_scope", "\x01") : "heldout";
        if (scope != "heldout" && scope != "full") throw std::invalid_argument("Evaluation scope must be heldout or full");
        o["evaluation_scope"] = json::from_string(scope);
    }
    if (version >= 3) o["cache_fingerprint"] = json::from_string("bytes-v2");
    // v4 streams AR KL teachers from SHA-256-indexed F32 files: fitter host RAM
    // no longer grows with the number of training windows; resume skips them.
    if (version == 4) o["teacher_cache"] = json::from_string("f32");
    // v5 chooses where frozen teachers and NAR conditioning live. The defaults
    // reproduce v4 exactly; every alternative is bit-identical by construction
    // and verified against it, so storage trades disk for time or VRAM only.
    const bool storage = has(request, "teacher_storage") || has(request, "conditioning_storage");
    if (storage && version < 5) throw std::invalid_argument("Storage modes require recipe version 5");
    if (version >= 5) {
        const auto teacher = has(request, "teacher_storage") ? text_of(request, "teacher_storage", "\x01") : "cache";
        if (std::find(teacher_storage.begin(), teacher_storage.end(), teacher) == teacher_storage.end()) {
            throw std::invalid_argument("teacher_storage must be cache or online");
        }
        const auto conditioning = has(request, "conditioning_storage") ? text_of(request, "conditioning_storage", "\x01") : "cache";
        if (std::find(conditioning_storage.begin(), conditioning_storage.end(), conditioning) == conditioning_storage.end()) {
            throw std::invalid_argument("conditioning_storage must be cache, rolling, resident");
        }
        if (teacher == "cache") o["teacher_cache"] = json::from_string("f32");
        else o["teacher_online"] = json::from_bool(true);
        o["conditioning_storage"] = json::from_string(conditioning);
    }
    return options;
}

std::vector<json::Value> eligible_rows(const json::Value & dataset) {
    const auto * rows = dataset.find("rows");
    if (!rows || rows->type != json::Type::array) throw std::invalid_argument("Training dataset has no rows");
    std::vector<json::Value> output;
    for (const auto & row : rows->array) {
        const auto * eligible = row.find("eligible");
        if (!eligible || eligible->type != json::Type::boolean || eligible->boolean) output.push_back(row);
    }
    return output;
}

json::Value ar_resource_estimate(const std::vector<json::Value> & eligible, const json::Value & options, const std::string & device,
                                 std::uint64_t model_bytes) {
    long long largest = 0;
    for (const auto & row : eligible) largest = std::max(largest, integer_of(row, "frames"));
    const long long window = largest * teacher_row_bytes;
    const long long teachers = teacher_bytes(eligible, options);
    const bool cached = has(options, "teacher_cache");
    // Online teachers hold one window's distribution, like the streamed cache.
    const bool streamed = cached || option_true(options, "teacher_online");
    double steady = ar_host_steady_gib * gib;
    steady += static_cast<double>(ar_streamed_windows * window);
    steady += static_cast<double>(streamed ? 0 : teachers);
    const double ram = std::max(ar_host_load_gib * gib, steady) + static_cast<double>(device == "cpu" ? model_bytes : 0);
    long long disk = 0;
    if (cached && teachers) {
        for (const auto * row : train_rows(eligible)) {
            disk += integer_of(*row, "frames") * teacher_row_bytes + (integer_of(*row, "prefix_tokens") + integer_of(*row, "frames")) * 4 + 4096;
        }
    }
    auto output = json::object();
    output.object["estimated_ar_host_ram_gib"] = json::from_double(ram / gib);
    output.object["estimated_teacher_disk_gib"] = json::from_double(static_cast<double>(disk) / gib);
    return output;
}

double checkpoint_allowance(const json::Value & options, int experts) {
    const double per_expert = static_cast<double>(checkpoints_kept * 16) * parameters_per_rank * static_cast<double>(integer_of(options, "rank")) +
                              static_cast<double>(diagnostics_gib) * gib;
    return experts * per_expert;
}

long long conditioning_bytes(const std::vector<json::Value> & eligible, const std::string & storage, const std::string & evaluation_scope) {
    if (storage == "resident") return 0;
    long long all = 0, largest = 0, evaluated = 0;
    for (const auto & row : eligible) {
        const auto bytes = window_bytes(row);
        all += bytes;
        largest = std::max(largest, bytes);
        if (evaluation_scope != "heldout" || text_of(row, "split") != "train") evaluated += bytes;
    }
    // Never more than keeping every window.
    if (storage == "rolling") return std::min(rolling_window_updates * largest + evaluated, all);
    return all;
}

ExpertBytes expert_bytes(const fs::path & base) {
    gguf_init_params params{true, nullptr};
    auto * context = gguf_init_from_file(utf8(base).c_str(), params);
    if (!context) throw std::runtime_error("Cannot read base model: " + utf8(base));
    ExpertBytes output;
    const auto starts = [](const std::string & name, const char * prefix) { return name.rfind(prefix, 0) == 0; };
    for (int64_t i = 0; i < gguf_get_n_tensors(context); ++i) {
        const std::string name = gguf_get_tensor_name(context, i);
        const auto bytes = static_cast<std::uint64_t>(gguf_get_tensor_size(context, i));
        const bool nar_layer = name.find(".nar_") != std::string::npos;
        if (nar_layer || name == "model.norm.weight" || name == "latent_pos_embed.pe" || starts(name, "vae2llm.") ||
            starts(name, "llm2vae.") || starts(name, "time_embedder.")) {
            output.nar += bytes;
        }
        if (name == "model.embed_tokens.weight" || name == "model.norm.weight" || (starts(name, "model.layers.") && !nar_layer)) {
            output.ar += bytes;
        }
    }
    gguf_free(context);
    return output;
}

json::Value storage_options(const std::vector<json::Value> & eligible, const json::Value & options, const ExpertBytes & experts) {
    const auto teacher = teacher_bytes(eligible, options);
    const auto scope = text_of(options, "evaluation_scope", "full");
    auto teachers = json::object();
    auto cache = json::object();
    cache.object["disk_gib"] = json::from_double(static_cast<double>(teacher) / gib);
    teachers.object["cache"] = cache;
    auto online = json::object();
    online.object["disk_gib"] = json::from_integer(0);
    teachers.object["online"] = online;
    auto conditioning = json::object();
    for (const auto & mode : conditioning_storage) {
        auto entry = json::object();
        entry.object["disk_gib"] = json::from_double(static_cast<double>(conditioning_bytes(eligible, mode, scope)) / gib);
        entry.object["model_vram_gib"] = json::from_double(static_cast<double>(experts.nar + (mode == "resident" ? experts.ar : 0)) / gib);
        conditioning.object[mode] = entry;
    }
    auto output = json::object();
    output.object["teacher_storage"] = teachers;
    output.object["conditioning_storage"] = conditioning;
    return output;
}

json::Value inventory(const fs::path & folder, long long window_seconds, long long seed) {
    struct Row { json::Value value; std::string sha256; };
    std::vector<Row> rows;
    std::vector<std::string> seen;
    for (const auto & audio : sorted_entries(folder)) {
        const auto suffix = lower_ascii(audio.extension().u8string());
        if (suffix != ".wav" && suffix != ".flac") continue;
        const auto name = audio.filename().u8string();
        auto sidecar = audio;
        sidecar.replace_extension(".txt");
        if (!fs::is_regular_file(sidecar)) throw std::invalid_argument("Missing caption: " + sidecar.filename().u8string());
        // read_text(encoding='utf-8-sig'): no BOM, universal newlines.
        auto raw = read_text(sidecar);
        if (raw.rfind("\xEF\xBB\xBF", 0) == 0) raw.erase(0, 3);
        std::string text;
        for (std::size_t i = 0; i < raw.size(); ++i) {
            if (raw[i] == '\r') {
                text.push_back('\n');
                if (i + 1 < raw.size() && raw[i + 1] == '\n') ++i;
            } else {
                text.push_back(raw[i]);
            }
        }
        try {
            text = strip(text);
        } catch (const std::invalid_argument &) {
            throw std::invalid_argument("Caption is not UTF-8: " + sidecar.filename().u8string());
        }
        const auto meta = caption_metadata(text);
        const auto bytes = read_text(audio);
        const auto info = audio::audio_info(reinterpret_cast<const std::uint8_t *>(bytes.data()), bytes.size());
        if (meta.style.empty()) throw std::invalid_argument("Caption has no style description: " + name);
        const double seconds = static_cast<double>(info.frames) / info.sample_rate;
        if (seconds < window_seconds) throw std::invalid_argument("Recording shorter than training window: " + name);
        const auto sha = sha256_hex(bytes);
        if (std::find(seen.begin(), seen.end(), sha) != seen.end()) throw std::invalid_argument("Duplicate recording: " + name);
        seen.push_back(sha);
        auto row = json::object();
        row.object["path"] = json::from_string(utf8(fs::canonical(audio)));
        row.object["sha256"] = json::from_string(sha);
        row.object["caption"] = json::from_string(text);
        row.object["caption_sha256"] = json::from_string(sha256_file(sidecar));
        row.object["seconds"] = json::from_double(seconds);
        row.object["bpm"] = json::from_double(meta.bpm);
        row.object["key"] = json::from_string(meta.key);
        row.object["split"] = json::from_string("train");
        rows.push_back({row, sha});
    }
    if (rows.size() < 2) throw std::invalid_argument("Provide at least two distinct WAV/FLAC recordings with TXT captions");
    // Recording-level deterministic holdouts; never split neighboring windows.
    std::vector<std::pair<std::string, std::size_t>> order;
    for (std::size_t i = 0; i < rows.size(); ++i) order.emplace_back(sha256_hex(std::to_string(seed) + rows[i].sha256), i);
    std::stable_sort(order.begin(), order.end(), [](const auto & a, const auto & b) { return a.first < b.first; });
    const double share = static_cast<double>(rows.size()) * .15;
    const double floor = std::floor(share);
    auto holdouts = static_cast<std::size_t>(floor);
    if (share - floor > .5 || (share - floor == .5 && holdouts % 2)) ++holdouts;
    holdouts = std::max<std::size_t>(1, holdouts);
    for (std::size_t i = 0; i < holdouts; ++i) rows[order[i].second].value.object["split"] = json::from_string("validation");
    auto output = json::array();
    for (auto & row : rows) output.array.push_back(std::move(row.value));
    return output;
}

json::Value review(const fs::path & run) {
    const auto request = read_json(run / "request.json");
    const auto runtime = read_json(run / "runtime.json");
    const auto options = settings(request);
    const auto data = read_json(run / "data" / "dataset.json");
    const auto * rows = data.find("rows");
    if (!rows || rows->type != json::Type::array) throw std::invalid_argument("Training dataset has no rows");
    const auto eligible = eligible_rows(data);
    const auto * models = runtime.find("models");
    if (!models) throw std::invalid_argument("Training runtime has no models");
    const fs::path base = fs::u8path(text_of(*models, "base")), companion = fs::u8path(text_of(*models, "companion"));
    const auto weights = fs::file_size(base) + fs::file_size(companion);
    auto estimate = ar_resource_estimate(eligible, options, text_of(runtime, "device", "cuda"), weights);
    // The teacher cache is released before NAR conditioning is written.
    const auto conditioning = conditioning_bytes(eligible, text_of(options, "conditioning_storage", "cache"),
                                                 text_of(options, "evaluation_scope", "full"));
    const double teacher = number_of(estimate.object["estimated_teacher_disk_gib"]) * gib;
    const double peak = std::max(teacher, static_cast<double>(conditioning)) + checkpoint_allowance(options);
    long long train = 0, validation = 0;
    for (const auto & row : eligible) {
        train += text_of(row, "split") == "train";
        validation += text_of(row, "split") == "validation";
    }
    auto output = estimate;
    output.object["examples"] = json::from_integer(static_cast<long long>(eligible.size()));
    output.object["excluded"] = json::from_integer(static_cast<long long>(rows->array.size() - eligible.size()));
    output.object["train_examples"] = json::from_integer(train);
    output.object["validation_examples"] = json::from_integer(validation);
    output.object["estimated_conditioning_gib"] = json::from_double(static_cast<double>(conditioning) / gib);
    output.object["estimated_peak_disk_gib"] = json::from_double(peak / gib);
    output.object["storage_options"] = storage_options(eligible, options, expert_bytes(base));
    return output;
}

json::Value capabilities(const std::optional<fs::path> & runtime) {
    std::optional<json::Value> native;
    if (runtime) {
        const fs::path bin = fs::u8path(text_of(read_json(*runtime), "bin"));
#ifdef _WIN32
        const std::string suffix = ".exe";
#else
        const std::string suffix;
#endif
        const auto help = [&](const char * name) { return capture_output(bin / (name + suffix), {"--help"}, 30).value_or(""); };
        const auto ar = help("yue2-ar-train"), nar = help("yue2-train");
        native = json::object();
        native->object["teacher_online"] = json::from_bool(ar.find("--teacher-online") != std::string::npos);
        native->object["rolling_conditioning"] = json::from_bool(nar.find("--rolling-caches") != std::string::npos);
        native->object["resident_conditioning"] = json::from_bool(nar.find("--resident-conditioning") != std::string::npos);
    }
    const auto supported = [&](const char * key) { return native ? native->object.at(key) : json::Value{}; };
    const auto mode = [](json::Value supported_value, const char * trade_off) {
        auto entry = json::object();
        entry.object["supported"] = std::move(supported_value);
        entry.object["trade_off"] = json::from_string(trade_off);
        return entry;
    };
    auto teacher = json::object();
    teacher.object["cache"] = mode(json::from_bool(true), "Fastest AR updates; about 94 MiB of disk per 30-second training window until AR selection");
    teacher.object["online"] = mode(supported("teacher_online"), "No teacher disk; one extra common-model forward per AR update");
    auto conditioning = json::object();
    conditioning.object["cache"] = mode(json::from_bool(true), "Fastest NAR updates; about 160 MiB of disk per 30-second window until publish");
    conditioning.object["rolling"] = mode(supported("rolling_conditioning"),
        "Disk bounded to one segment plus heldout windows; regenerates each window before use (about 1.2 s each on an RTX 5070 Laptop)");
    conditioning.object["resident"] = mode(supported("resident_conditioning"),
        "No conditioning disk; the NAR fitter also holds the frozen AR expert (3.3 GiB BF16 beside the 2.7 GiB NAR expert) and its "
        "prefill. With 30-second windows an 8 GB laptop GPU spills into shared memory and an update takes about 190 s instead of 1.5 s");
    auto storage = json::object();
    storage.object["teacher_storage"] = teacher;
    storage.object["conditioning_storage"] = conditioning;
    auto output = json::object();
    output.object["schema_version"] = json::from_integer(1);
    auto recipes = json::array();
    for (int recipe = 1; recipe <= 5; ++recipe) recipes.array.push_back(json::from_integer(recipe));
    output.object["recipes"] = recipes;
    output.object["default_recipe"] = json::from_integer(5);
    auto settings_list = json::array();
    for (const auto & key : request_settings) settings_list.array.push_back(json::from_string(key));
    output.object["request_settings"] = settings_list;
    output.object["storage"] = storage;
    output.object["native"] = native ? *native : json::Value{};
    return output;
}

json::Value runtime_identity(const fs::path & executable) {
    std::vector<fs::path> files{executable};
    for (const auto & path : sorted_entries(executable.parent_path())) {
        const auto name = path.filename().u8string();
        const auto suffix = lower_ascii(path.extension().u8string());
        if (fs::is_regular_file(path) && (suffix == ".dll" || suffix == ".dylib" || name.find(".so") != std::string::npos)) {
            files.push_back(path);
        }
    }
    const auto digests = sha256_files(files);
    auto output = json::object();
    auto program = json::object();
    program.object[executable.filename().u8string()] = json::from_string(digests[0]);
    output.object["executable"] = program;
    auto libraries = json::object();
    for (std::size_t i = 1; i < files.size(); ++i) libraries.object[files[i].filename().u8string()] = json::from_string(digests[i]);
    output.object["shared_libraries"] = libraries;
    auto environment = json::object();
    for (const char * key : {"GGML_CUDA_TF32", "NVIDIA_TF32_OVERRIDE", "CUBLAS_WORKSPACE_CONFIG"}) {
        const char * value = std::getenv(key);
        environment.object[key] = value ? json::from_string(value) : json::Value{};
    }
    output.object["precision_environment"] = environment;
    return output;
}

} // namespace yue2::training
