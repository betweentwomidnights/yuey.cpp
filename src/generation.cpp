#include "yue2/generation.h"
#include "yue2/tokenizer.h"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace yue2 {
namespace {

std::string trim(std::string value) {
    const auto first = std::find_if_not(value.begin(), value.end(), [](unsigned char c) {
        return std::isspace(c) != 0;
    });
    const auto last = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char c) {
        return std::isspace(c) != 0;
    }).base();
    if (first >= last) return {};
    return std::string(first, last);
}

void validate_abc_ids(const std::vector<std::int32_t> & ids) {
    for (const auto id : ids) {
        if (id < 0 || id >= kEodToken) {
            throw std::invalid_argument(
                "YuE2 ABC IDs must remain inside the ordinary text vocabulary");
        }
    }
}

void append(std::vector<std::int32_t> & destination,
            const std::vector<std::int32_t> & source) {
    destination.insert(destination.end(), source.begin(), source.end());
}

struct MetadataFile {
    std::unique_ptr<gguf_context, decltype(&gguf_free)> gguf{nullptr, gguf_free};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> tensors{nullptr, ggml_free};
};

std::runtime_error package_error(const std::string & message) {
    return std::runtime_error("[yue2:generation] " + message);
}

MetadataFile open_metadata(const std::string & path) {
    MetadataFile result;
    ggml_context * context = nullptr;
    const gguf_init_params params = {true, &context};
    result.gguf.reset(gguf_init_from_file(path.c_str(), params));
    result.tensors.reset(context);
    if (!result.gguf || !result.tensors) {
        throw package_error("failed to open GGUF metadata: " + path);
    }
    return result;
}

int require_key(const MetadataFile & file, const char * key, gguf_type type) {
    const int index = gguf_find_key(file.gguf.get(), key);
    if (index < 0) throw package_error("missing metadata key: " + std::string(key));
    if (gguf_get_kv_type(file.gguf.get(), index) != type) {
        throw package_error("metadata type mismatch: " + std::string(key));
    }
    return index;
}

std::string require_string(const MetadataFile & file, const char * key) {
    return gguf_get_val_str(file.gguf.get(), require_key(file, key, GGUF_TYPE_STRING));
}

std::string optional_string(const MetadataFile & file, const char * key) {
    const int index = gguf_find_key(file.gguf.get(), key);
    if (index < 0) return {};
    if (gguf_get_kv_type(file.gguf.get(), index) != GGUF_TYPE_STRING) {
        throw package_error("metadata type mismatch: " + std::string(key));
    }
    return gguf_get_val_str(file.gguf.get(), index);
}

std::uint32_t require_u32(const MetadataFile & file, const char * key) {
    return gguf_get_val_u32(file.gguf.get(), require_key(file, key, GGUF_TYPE_UINT32));
}

float require_f32(const MetadataFile & file, const char * key) {
    return gguf_get_val_f32(file.gguf.get(), require_key(file, key, GGUF_TYPE_FLOAT32));
}

bool require_bool(const MetadataFile & file, const char * key) {
    return gguf_get_val_bool(file.gguf.get(), require_key(file, key, GGUF_TYPE_BOOL));
}

void require_equal(std::uint32_t actual, std::uint32_t expected, const char * label) {
    if (actual != expected) {
        throw package_error(
            std::string("unsupported ") + label + " " + std::to_string(actual) +
            "; expected " + std::to_string(expected));
    }
}

ggml_tensor * require_tensor(const MetadataFile & file, const char * name) {
    auto * tensor = ggml_get_tensor(file.tensors.get(), name);
    if (!tensor) throw package_error("missing tensor: " + std::string(name));
    return tensor;
}

void require_shape(
    const MetadataFile & file,
    const char * name,
    const std::vector<std::int64_t> & ggml_shape) {
    const auto * tensor = require_tensor(file, name);
    const int dimensions = ggml_n_dims(tensor);
    if (dimensions != static_cast<int>(ggml_shape.size())) {
        throw package_error("tensor rank mismatch: " + std::string(name));
    }
    for (int dimension = 0; dimension < dimensions; ++dimension) {
        if (tensor->ne[dimension] != ggml_shape[static_cast<std::size_t>(dimension)]) {
            throw package_error("tensor shape mismatch: " + std::string(name));
        }
    }
}

std::string homogeneous_storage_type(const MetadataFile & file) {
    int type = -1;
    for (auto * tensor = ggml_get_first_tensor(file.tensors.get()); tensor;
         tensor = ggml_get_next_tensor(file.tensors.get(), tensor)) {
        if (type < 0) type = tensor->type;
        if (type != tensor->type) return "mixed";
    }
    switch (type) {
        case GGML_TYPE_F32: return "f32";
        case GGML_TYPE_F16: return "f16";
        case GGML_TYPE_BF16: return "bf16";
        default: return type < 0 ? "empty" : ggml_type_name(static_cast<ggml_type>(type));
    }
}

// Quantized packages mix storage types by design, so their recorded encoding
// (e.g. Q4_K_M) describes them better than "mixed".
std::string storage_type(const MetadataFile & file) {
    auto encoding = optional_string(file, "yue2.quantization.encoding");
    if (!encoding.empty()) return encoding;
    return homogeneous_storage_type(file);
}

std::size_t tensor_count(const MetadataFile & file) {
    return static_cast<std::size_t>(gguf_get_n_tensors(file.gguf.get()));
}

std::uint64_t checked_file_size(const std::string & path) {
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    if (error) throw package_error("could not determine file size: " + path);
    return size;
}

void validate_model(const MetadataFile & file, GenerationPackageInfo & info) {
    if (require_string(file, "general.architecture") != "yue2" ||
        require_string(file, "yue2.component") != "generation") {
        throw package_error("not a yue2.cpp YuE2 generation GGUF");
    }
    if (require_string(file, "yue2.tensor_names") != "huggingface-native") {
        throw package_error("unsupported YuE2 tensor naming scheme");
    }
    auto & config = info.model;
    config.vocab_size = require_u32(file, "yue2.vocab_size");
    config.context_length = require_u32(file, "yue2.context_length");
    config.embedding_length = require_u32(file, "yue2.embedding_length");
    config.block_count = require_u32(file, "yue2.block_count");
    config.attention_head_count = require_u32(file, "yue2.attention.head_count");
    config.attention_head_count_kv = require_u32(file, "yue2.attention.head_count_kv");
    config.latent_dim = require_u32(file, "yue2.latent_dim");
    config.timestep_shift = require_f32(file, "yue2.timestep_shift");
    require_equal(config.vocab_size, kGenerationVocabSize, "vocabulary size");
    require_equal(config.context_length, 24576, "context length");
    require_equal(config.embedding_length, 2048, "embedding length");
    require_equal(config.block_count, 28, "block count");
    require_equal(config.attention_head_count, 16, "attention head count");
    require_equal(config.attention_head_count_kv, 8, "KV head count");
    require_equal(config.latent_dim, 64, "latent dimension");
    if (config.timestep_shift != 1.0F) throw package_error("unsupported timestep shift");

    info.model_tensor_count = tensor_count(file);
    if (info.model_tensor_count != 628) {
        throw package_error("generation tensor count mismatch");
    }
    require_shape(file, "model.embed_tokens.weight", {2048, kGenerationVocabSize});
    require_shape(file, "lm_head.weight", {2048, kGenerationVocabSize});
    require_shape(file, "latent_pos_embed.pe", {2048, 24576});
    require_shape(file, "llm2vae.weight", {2048, 64});
    require_shape(file, "model.layers.0.self_attn.q_proj.weight", {2048, 2048});
    require_shape(file, "model.layers.27.nar_self_attn.q_proj.weight", {2048, 2048});
    info.model_storage_type = storage_type(file);
    info.model_checkpoint_sha256 = optional_string(file, "yue2.checkpoint.sha256");
}

void validate_vae(const MetadataFile & file, GenerationPackageInfo & info) {
    if (require_string(file, "general.architecture") != "yue2_vae" ||
        require_string(file, "yue2.component") != "vae") {
        throw package_error("not a yue2.cpp YuE2 VAE GGUF");
    }
    if (!require_bool(file, "yue2.vae.weight_norm_folded")) {
        throw package_error("YuE2 VAE weight norm must be folded");
    }
    auto & config = info.vae;
    config.sample_rate = require_u32(file, "yue2.vae.sample_rate");
    config.audio_channels = require_u32(file, "yue2.vae.audio_channels");
    config.downsampling_ratio = require_u32(file, "yue2.vae.downsampling_ratio");
    config.latent_dim = require_u32(file, "yue2.vae.latent_dim");
    config.decode_core_frames = require_u32(file, "yue2.vae.decode_core_frames");
    config.decode_halo_frames = require_u32(file, "yue2.vae.decode_halo_frames");
    require_equal(config.sample_rate, 48000, "VAE sample rate");
    require_equal(config.audio_channels, 2, "VAE channel count");
    require_equal(config.downsampling_ratio, 1920, "VAE downsampling ratio");
    require_equal(config.latent_dim, 64, "VAE latent dimension");
    require_equal(config.decode_core_frames, 1024, "VAE decode core frames");
    require_equal(config.decode_halo_frames, 16, "VAE decode halo frames");

    info.vae_tensor_count = tensor_count(file);
    if (info.vae_tensor_count != 347) throw package_error("VAE tensor count mismatch");
    require_shape(file, "encoder.layers.0.weight", {7, 2, 64});
    require_shape(file, "encoder.layers.8.weight", {3, 2048, 128});
    require_shape(file, "decoder.layers.0.weight", {7, 64, 2048});
    require_shape(file, "decoder.layers.8.weight", {7, 64, 2});
    for (std::int64_t index = 0; index < gguf_get_n_tensors(file.gguf.get()); ++index) {
        const std::string name = gguf_get_tensor_name(file.gguf.get(), index);
        if (name.size() >= 9 &&
            (name.compare(name.size() - 9, 9, ".weight_g") == 0 ||
             name.compare(name.size() - 9, 9, ".weight_v") == 0)) {
            throw package_error("unfolded VAE weight norm tensor: " + name);
        }
    }
    info.vae_storage_type = storage_type(file);
    info.vae_checkpoint_sha256 = optional_string(file, "yue2.checkpoint.sha256");
}

bool abc_field_or_comment(const std::string & line) {
    const auto value = trim(line);
    return value.empty() || value.front() == '%' ||
        (value.size() >= 2 && std::isalpha(static_cast<unsigned char>(value[0])) &&
         value[1] == ':');
}

bool abc_pitch(char value) {
    return (value >= 'A' && value <= 'G') || (value >= 'a' && value <= 'g');
}

std::size_t abc_quoted_end(const std::string & line, std::size_t offset, char closing) {
    const auto found = line.find(closing, offset + 1);
    return found == std::string::npos ? line.size() : found + 1;
}

bool abc_has_pitch(const std::vector<std::string> & lines) {
    for (const auto & line : lines) {
        if (abc_field_or_comment(line)) continue;
        for (std::size_t offset = 0; offset < line.size();) {
            if (line[offset] == '"') {
                offset = abc_quoted_end(line, offset, '"');
                continue;
            }
            if (line[offset] == '[') {
                offset = abc_quoted_end(line, offset, ']');
                continue;
            }
            auto pitch = offset;
            while (pitch < line.size() &&
                   (line[pitch] == '^' || line[pitch] == '_' || line[pitch] == '=')) {
                ++pitch;
            }
            if (pitch < line.size() && abc_pitch(line[pitch])) return true;
            ++offset;
        }
    }
    return false;
}

std::string transform_abc_music_line(
    const std::string & line,
    bool rest_notes,
    bool remove_chords) {
    if (abc_field_or_comment(line)) return line;
    std::string output;
    output.reserve(line.size());
    for (std::size_t offset = 0; offset < line.size();) {
        if (line[offset] == '"') {
            const auto end = abc_quoted_end(line, offset, '"');
            if (!remove_chords) output.append(line, offset, end - offset);
            offset = end;
            continue;
        }
        if (line[offset] == '[') {
            const auto end = abc_quoted_end(line, offset, ']');
            output.append(line, offset, end - offset);
            offset = end;
            continue;
        }
        auto pitch = offset;
        while (pitch < line.size() &&
               (line[pitch] == '^' || line[pitch] == '_' || line[pitch] == '=')) {
            ++pitch;
        }
        if (pitch < line.size() && abc_pitch(line[pitch])) {
            if (rest_notes) {
                output.push_back('z');
                offset = pitch + 1;
                while (offset < line.size() &&
                       (line[offset] == ',' || line[offset] == '\'')) {
                    ++offset;
                }
                continue;
            }
        }
        if (rest_notes && line[offset] == '-') {
            ++offset;
            continue;
        }
        output.push_back(line[offset++]);
    }
    return output;
}

std::vector<std::string> transform_abc_music(
    const std::vector<std::string> & lines,
    bool rest_notes,
    bool remove_chords) {
    std::vector<std::string> result;
    result.reserve(lines.size());
    for (const auto & line : lines) {
        result.push_back(transform_abc_music_line(line, rest_notes, remove_chords));
    }
    return result;
}

std::vector<std::string> abc_lines(const std::string & abc) {
    std::vector<std::string> result;
    std::size_t offset = 0;
    while (offset < abc.size()) {
        const auto end = abc.find('\n', offset);
        auto line = abc.substr(offset, end == std::string::npos ? end : end - offset);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        result.push_back(std::move(line));
        if (end == std::string::npos) break;
        offset = end + 1;
    }
    return result;
}

void append_lines(std::string & output, const std::vector<std::string> & lines) {
    for (const auto & line : lines) output += line + '\n';
}

} // namespace

std::string normalize_abc_key(const std::string & input) {
    auto value = trim(input);
    if (value.empty()) throw std::invalid_argument("planning key is required");

    const unsigned char root = static_cast<unsigned char>(value.front());
    if (std::toupper(root) < 'A' || std::toupper(root) > 'G') {
        throw std::invalid_argument("planning key must start with A through G");
    }
    std::string output(1, static_cast<char>(std::toupper(root)));
    std::size_t offset = 1;
    if (offset < value.size() && (value[offset] == '#' || value[offset] == 'b')) {
        output.push_back(value[offset++]);
    }

    std::string quality;
    for (; offset < value.size(); ++offset) {
        const unsigned char c = static_cast<unsigned char>(value[offset]);
        if (std::isspace(c) || c == ':' || c == '-' || c == '_') continue;
        if (!std::isalpha(c)) {
            throw std::invalid_argument("planning key must be a major or minor key name");
        }
        quality.push_back(static_cast<char>(std::tolower(c)));
    }
    if (quality.empty() || quality == "major" || quality == "maj") return output;
    if (quality == "m" || quality == "minor" || quality == "min") return output + 'm';
    throw std::invalid_argument("planning key must use major or minor quality");
}

std::string make_planning_abc_prefix(const PlanningHeader & header) {
    if (header.bpm < 20 || header.bpm > 400) {
        throw std::invalid_argument("planning bpm must be in [20, 400]");
    }
    if (header.meter_numerator == 0 || header.meter_numerator > 32) {
        throw std::invalid_argument("planning meter numerator must be in [1, 32]");
    }
    const auto denominator = header.meter_denominator;
    if (denominator == 0 || denominator > 32 || (denominator & (denominator - 1)) != 0) {
        throw std::invalid_argument("planning meter denominator must be a power of two up to 32");
    }
    const auto key = normalize_abc_key(header.key);
    return "X:1\nT:\nM:" + std::to_string(header.meter_numerator) + '/' +
        std::to_string(header.meter_denominator) +
        "\nL:1/32\nQ:1/4=" + std::to_string(header.bpm) +
        "\nV: Vocal clef=treble name=\"Vocal Melody\" snm=\"Vocal\""
        "\nV: Ins clef=treble name=\"Ins Melody\" snm=\"Inst.\""
        "\nK:" + key + "\n% intro\n";
}

std::string make_instrumental_abc(const std::string & abc) {
    const auto lines = abc_lines(abc);
    std::string output;
    output.reserve(abc.size());
    bool transformed = false;
    std::size_t offset = 0;
    while (offset < lines.size()) {
        if (trim(lines[offset]) != "V: Vocal") {
            output += lines[offset++] + '\n';
            continue;
        }

        const auto vocal_selector = offset++;
        const auto vocal_begin = offset;
        while (offset < lines.size() && trim(lines[offset]) != "V: Ins" &&
               trim(lines[offset]) != "V: Vocal") {
            ++offset;
        }
        if (offset >= lines.size() || trim(lines[offset]) != "V: Ins") {
            throw std::invalid_argument(
                "YuE2 instrumental mode requires paired Vocal and Ins blocks");
        }
        const auto ins_selector = offset++;
        const auto ins_begin = offset;
        while (offset < lines.size()) {
            const auto value = trim(lines[offset]);
            if (value == "V: Vocal" || value == "V: Ins" ||
                (!value.empty() && value.front() == '%')) {
                break;
            }
            ++offset;
        }

        const std::vector<std::string> vocal(
            lines.begin() + static_cast<std::ptrdiff_t>(vocal_begin),
            lines.begin() + static_cast<std::ptrdiff_t>(ins_selector));
        const std::vector<std::string> ins(
            lines.begin() + static_cast<std::ptrdiff_t>(ins_begin),
            lines.begin() + static_cast<std::ptrdiff_t>(offset));
        output += lines[vocal_selector] + '\n';
        append_lines(output, transform_abc_music(vocal, true, false));
        output += lines[ins_selector] + '\n';
        append_lines(output, abc_has_pitch(vocal)
            ? transform_abc_music(vocal, false, true)
            : ins);
        transformed = true;
    }
    if (!transformed) {
        throw std::invalid_argument(
            "YuE2 instrumental mode requires native Vocal and Ins score blocks");
    }
    if (abc.empty() || abc.back() != '\n') output.pop_back();
    return output;
}

const char * generation_instruction(SymbolicMode mode) noexcept {
    switch (mode) {
        case SymbolicMode::off:
            return "Generate music with codec tokens from the given conditions.";
        case SymbolicMode::melody:
            return "Generate a melody-only ABC transcription without chord symbols, then "
                   "generate music with codec tokens from the given conditions.";
        case SymbolicMode::full:
            return "Generate a chord-annotated ABC transcription, then generate music with "
                   "codec tokens from the given conditions.";
    }
    return "";
}

float generation_guidance(const SongRequest & request) {
    const float value = request.guidance_scale.value_or(
        request.symbolic_mode == SymbolicMode::off ? 1.01F : 1.0F);
    if (!std::isfinite(value) || value < 0.0F || value > 20.0F) {
        throw std::invalid_argument("YuE2 guidance scale must be finite and in [0,20]");
    }
    return value;
}

std::string generation_request_text(const SongRequest & request) {
    return std::string(generation_instruction(request.symbolic_mode)) +
        "\n[Tags]\n" + request.style + "\n[Lyrics]\n" + request.lyrics + "\n";
}

std::vector<std::int32_t> make_positive_prefix(
    const SongRequest & request,
    const TextTokenizer & tokenizer,
    const std::optional<std::vector<std::int32_t>> & abc_ids) {
    std::vector<std::int32_t> output{kEodToken};
    append(output, tokenizer.encode(generation_request_text(request)));
    if (request.symbolic_mode == SymbolicMode::off) {
        output.insert(output.end(), {kAbcStartToken, kAbcEndToken, kMusicStartToken});
        return output;
    }

    output.push_back(kAbcStartToken);
    if (!abc_ids && !request.abc) return output;
    const auto ids = abc_ids ? *abc_ids : tokenizer.encode(*request.abc);
    validate_abc_ids(ids);
    append(output, ids);
    output.insert(output.end(), {kAbcEndToken, kMusicStartToken});
    return output;
}

std::vector<std::int32_t> make_negative_prefix(
    const SongRequest & request,
    const TextTokenizer & tokenizer,
    const std::optional<std::vector<std::int32_t>> & abc_ids) {
    std::vector<std::int32_t> output{kEodToken};
    append(output, tokenizer.encode(generation_instruction(request.symbolic_mode)));
    if (request.symbolic_mode == SymbolicMode::off) {
        output.push_back(kMusicStartToken);
        return output;
    }
    if (!abc_ids) {
        throw std::invalid_argument(
            "YuE2 symbolic guidance requires the exact positive-branch ABC IDs");
    }
    validate_abc_ids(*abc_ids);
    output.push_back(kAbcStartToken);
    append(output, *abc_ids);
    output.insert(output.end(), {kAbcEndToken, kMusicStartToken});
    return output;
}

GenerationPackageInfo inspect_generation_package(
    const std::string & model_gguf_path,
    const std::string & vae_gguf_path) {
    GenerationPackageInfo info;
    auto model = open_metadata(model_gguf_path);
    auto vae = open_metadata(vae_gguf_path);
    validate_model(model, info);
    validate_vae(vae, info);
    info.model_file_bytes = checked_file_size(model_gguf_path);
    info.vae_file_bytes = checked_file_size(vae_gguf_path);
    return info;
}

} // namespace yue2
