#include "yue2/generation.h"
#include "yue2/tokenizer.h"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <limits>
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

std::string rest_abc_music_line(const std::string & line) {
    if (abc_field_or_comment(line)) return line;
    std::string output;
    output.reserve(line.size());
    for (std::size_t offset = 0; offset < line.size();) {
        if (line[offset] == '"') {
            const auto end = abc_quoted_end(line, offset, '"');
            output.append(line, offset, end - offset);
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
            output.push_back('z');
            offset = pitch + 1;
            while (offset < line.size() &&
                   (line[offset] == ',' || line[offset] == '\'')) {
                ++offset;
            }
            continue;
        }
        if (line[offset] == '-') {
            ++offset;
            continue;
        }
        output.push_back(line[offset++]);
    }
    return output;
}

std::vector<std::string> rest_abc_music(const std::vector<std::string> & lines) {
    std::vector<std::string> result;
    result.reserve(lines.size());
    for (const auto & line : lines) {
        result.push_back(rest_abc_music_line(line));
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

struct NativeBar {
    std::string vocal;
    std::string instrumental;
    std::string section;
    std::vector<std::string> vocal_fields;
    std::vector<std::string> instrumental_fields;
};

struct NativeScore {
    std::vector<std::string> header;
    std::vector<NativeBar> bars;
    bool trailing_newline = false;
};

std::string section_name(const std::string & line) {
    const auto value = trim(line);
    return value.empty() || value.front() != '%' ? std::string{} : trim(value.substr(1));
}

struct VoiceMusic {
    std::vector<std::string> fields;
    std::vector<std::string> bars;
};

std::uint32_t parse_positive_u32(const std::string & text);

void append_expanded_bar(std::vector<std::string> & bars, const std::string & input) {
    auto body = trim(input);
    if (body.empty() || body == "|") return;
    if (body.back() != '|') {
        bars.push_back(std::move(body));
        return;
    }
    std::size_t offset = 0;
    while (offset < body.size() && body[offset] == '"') {
        offset = abc_quoted_end(body, offset, '"');
    }
    if (offset >= body.size() || body[offset] != 'Z') {
        bars.push_back(std::move(body));
        return;
    }
    const auto digits = offset + 1;
    auto end = digits;
    while (end < body.size() && std::isdigit(static_cast<unsigned char>(body[end]))) ++end;
    if (end + 1 != body.size() || body[end] != '|' || end == digits) {
        bars.push_back(std::move(body));
        return;
    }
    const auto count = parse_positive_u32(body.substr(digits, end - digits));
    const auto prefix = body.substr(0, offset);
    for (std::uint32_t index = 0; index < count; ++index) {
        bars.push_back((index == 0 ? prefix : std::string{}) + "Z|");
    }
}

VoiceMusic split_voice_music(const std::vector<std::string> & lines) {
    VoiceMusic result;
    std::string pending;
    for (const auto & raw : lines) {
        const auto line = trim(raw);
        if (line.empty()) continue;
        if (abc_field_or_comment(line)) {
            result.fields.push_back(line);
            continue;
        }
        pending += line;
        std::size_t begin = 0;
        bool quoted = false;
        for (std::size_t offset = 0; offset < pending.size(); ++offset) {
            if (pending[offset] == '"') quoted = !quoted;
            if (!quoted && pending[offset] == '|') {
                append_expanded_bar(
                    result.bars, pending.substr(begin, offset - begin + 1));
                begin = offset + 1;
            }
        }
        pending.erase(0, begin);
    }
    if (!trim(pending).empty()) {
        throw std::invalid_argument("YuE2 ABC score has music outside a complete bar");
    }
    return result;
}

NativeScore parse_native_score(const std::string & abc, bool allow_empty_score = false) {
    const auto lines = abc_lines(abc);
    NativeScore score;
    score.trailing_newline = !abc.empty() && abc.back() == '\n';
    std::size_t offset = 0;
    std::string pending_section;
    while (offset < lines.size() && trim(lines[offset]) != "V: Vocal") {
        const auto section = section_name(lines[offset]);
        if (!section.empty()) pending_section = section;
        score.header.push_back(lines[offset++]);
    }
    const bool has_lanes = offset < lines.size();
    if (!has_lanes && !allow_empty_score) {
        throw std::invalid_argument("YuE2 score fitting requires native Vocal and Ins lanes");
    }
    if (!pending_section.empty()) {
        while (!score.header.empty() && trim(score.header.back()).empty()) score.header.pop_back();
        if (!score.header.empty() && !section_name(score.header.back()).empty()) score.header.pop_back();
    }
    if (!has_lanes) return score;

    while (offset < lines.size()) {
        while (offset < lines.size() && trim(lines[offset]) != "V: Vocal") {
            const auto section = section_name(lines[offset]);
            if (!section.empty()) pending_section = section;
            else if (!trim(lines[offset]).empty()) {
                throw std::invalid_argument("unexpected text between YuE2 score blocks");
            }
            ++offset;
        }
        if (offset == lines.size()) break;
        ++offset;
        std::vector<std::string> vocal_lines;
        while (offset < lines.size() && trim(lines[offset]) != "V: Ins") {
            if (trim(lines[offset]) == "V: Vocal" || !section_name(lines[offset]).empty()) {
                throw std::invalid_argument("YuE2 score has an unpaired Vocal block");
            }
            vocal_lines.push_back(lines[offset++]);
        }
        if (offset == lines.size()) {
            throw std::invalid_argument("YuE2 score has an unpaired Vocal block");
        }
        ++offset;
        std::vector<std::string> instrumental_lines;
        while (offset < lines.size() && trim(lines[offset]) != "V: Vocal" &&
               section_name(lines[offset]).empty()) {
            instrumental_lines.push_back(lines[offset++]);
        }
        const auto vocal = split_voice_music(vocal_lines);
        const auto instrumental = split_voice_music(instrumental_lines);
        if (vocal.bars.empty() || vocal.bars.size() != instrumental.bars.size()) {
            throw std::invalid_argument(
                "YuE2 score fitting requires equal nonempty Vocal and Ins bars");
        }
        for (std::size_t index = 0; index < vocal.bars.size(); ++index) {
            NativeBar bar;
            bar.vocal = vocal.bars[index];
            bar.instrumental = instrumental.bars[index];
            bar.section = pending_section.empty() ? "section" : pending_section;
            if (index == 0) {
                bar.vocal_fields = vocal.fields;
                bar.instrumental_fields = instrumental.fields;
            }
            score.bars.push_back(std::move(bar));
        }
    }
    if (score.bars.empty() && !allow_empty_score) {
        throw std::invalid_argument("YuE2 score contains no complete bars");
    }
    return score;
}

void append_bar_lines(std::string & output, const std::vector<NativeBar> & bars,
                      std::size_t begin, std::size_t end, bool vocal) {
    std::size_t on_line = 0;
    for (std::size_t index = begin; index < end; ++index) {
        const auto & fields = vocal ? bars[index].vocal_fields : bars[index].instrumental_fields;
        if (!fields.empty() && on_line != 0) {
            output += '\n';
            on_line = 0;
        }
        for (const auto & field : fields) output += field + '\n';
        output += vocal ? bars[index].vocal : bars[index].instrumental;
        if (++on_line == 4 || index + 1 == end) {
            output += '\n';
            on_line = 0;
        }
    }
}

std::uint32_t parse_positive_u32(const std::string & text) {
    std::size_t used = 0;
    const auto value = std::stoul(text, &used);
    if (used != text.size() || value == 0 || value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("invalid positive ABC numeric field");
    }
    return static_cast<std::uint32_t>(value);
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

std::string make_vocal_rest_abc(const std::string & abc) {
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
                "YuE2 vocal-rest experiment requires paired Vocal and Ins blocks");
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
        append_lines(output, rest_abc_music(vocal));
        output += lines[ins_selector] + '\n';
        append_lines(output, ins);
        transformed = true;
    }
    if (!transformed) {
        throw std::invalid_argument(
            "YuE2 vocal-rest experiment requires native Vocal and Ins score blocks");
    }
    if (abc.empty() || abc.back() != '\n') output.pop_back();
    return output;
}

std::string make_instrumental_lyrics(const std::string & abc) {
    std::vector<std::string> sections;
    for (const auto & raw_line : abc_lines(abc)) {
        const auto line = trim(raw_line);
        if (line.empty() || line.front() != '%') continue;
        auto section = trim(line.substr(1));
        if (section.empty() || section.size() > 48) continue;

        bool valid = true;
        bool capitalize = true;
        for (auto & c : section) {
            const auto value = static_cast<unsigned char>(c);
            if (std::isalnum(value)) {
                c = static_cast<char>(capitalize ? std::toupper(value) : std::tolower(value));
                capitalize = false;
            } else if (c == ' ' || c == '-' || c == '_') {
                if (c == '_') c = ' ';
                capitalize = true;
            } else {
                valid = false;
                break;
            }
        }
        if (valid) sections.push_back(std::move(section));
    }

    if (sections.empty()) {
        sections = {"Intro", "Verse", "Chorus", "Verse", "Chorus", "Outro"};
    }
    std::string output;
    for (const auto & section : sections) {
        if (!output.empty()) output += "\n\n";
        output += '[' + section + ']';
    }
    return output;
}

AbcScoreInfo inspect_abc_score(const std::string & abc, bool allow_empty_score) {
    AbcScoreInfo result;
    for (const auto & raw_line : abc_lines(abc)) {
        const auto line = trim(raw_line);
        if (line.rfind("M:", 0) == 0) {
            const auto slash = line.find('/', 2);
            if (slash != std::string::npos) {
                result.meter_numerator = parse_positive_u32(line.substr(2, slash - 2));
                result.meter_denominator = parse_positive_u32(line.substr(slash + 1));
            }
        } else if (line.rfind("Q:", 0) == 0) {
            const auto equals = line.rfind('=');
            if (equals != std::string::npos && equals + 1 < line.size()) {
                result.bpm = parse_positive_u32(line.substr(equals + 1));
            }
        }
    }
    const auto score = parse_native_score(abc, allow_empty_score);
    if (score.bars.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("YuE2 ABC score contains too many bars");
    }
    result.bars = static_cast<std::uint32_t>(score.bars.size());
    if (result.bpm == 0) {
        throw std::invalid_argument("YuE2 score alignment requires a Q tempo field");
    }
    result.duration_seconds = static_cast<double>(result.bars) *
        static_cast<double>(result.meter_numerator) * 60.0 * 4.0 /
        (static_cast<double>(result.bpm) * static_cast<double>(result.meter_denominator));
    return result;
}

std::string fit_abc_score_to_bars(
    const std::string & abc,
    std::uint32_t target_bars,
    std::uint32_t outro_bars) {
    if (target_bars == 0 || outro_bars == 0 || outro_bars > target_bars) {
        throw std::invalid_argument(
            "YuE2 score fitting requires target bars and outro bars within the target");
    }
    const auto score = parse_native_score(abc);
    if (target_bars > score.bars.size()) {
        throw std::invalid_argument(
            "YuE2 target bars exceed the completed plan; generate a longer plan first");
    }
    if (target_bars == score.bars.size()) return abc;

    const auto head_count = static_cast<std::size_t>(target_bars - outro_bars);
    const auto tail_begin = score.bars.size() - outro_bars;
    std::vector<NativeBar> selected;
    selected.reserve(target_bars);
    selected.insert(selected.end(), score.bars.begin(), score.bars.begin() + head_count);
    selected.insert(selected.end(), score.bars.begin() + tail_begin, score.bars.end());
    for (std::size_t index = head_count; index < selected.size(); ++index) {
        selected[index].section = "outro";
    }

    std::string output;
    append_lines(output, score.header);
    for (std::size_t begin = 0; begin < selected.size();) {
        std::size_t end = begin + 1;
        while (end < selected.size() && selected[end].section == selected[begin].section) ++end;
        output += "% " + selected[begin].section + '\n';
        output += "V: Vocal\n";
        append_bar_lines(output, selected, begin, end, true);
        output += "V: Ins\n";
        append_bar_lines(output, selected, begin, end, false);
        begin = end;
    }
    if (!score.trailing_newline && !output.empty()) output.pop_back();
    return output;
}

std::uint32_t score_aligned_semantic_budget(const AbcScoreInfo & score) {
    if (score.bars == 0 || !std::isfinite(score.duration_seconds) ||
        score.duration_seconds <= 0.0) {
        throw std::invalid_argument("YuE2 semantic alignment requires a timed score");
    }
    // Short scores have proportionally more learned intro/outro overhead. This
    // covers the observed 8-bar 1.70x case while retaining ample headroom for
    // full arrangements. MUSIC_END normally stops generation before the cap.
    const double seconds = std::max(
        score.duration_seconds * 1.75,
        score.duration_seconds + 30.0);
    const double tokens = std::ceil(seconds * 25.0);
    if (tokens > static_cast<double>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::invalid_argument("YuE2 score-aligned semantic budget is too large");
    }
    return static_cast<std::uint32_t>(tokens);
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
