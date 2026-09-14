#include "yue2/runtime_info.h"

#include "gguf_model.h"

#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <memory>
#include <string_view>

namespace yue2 {
namespace {

namespace fs = std::filesystem;

constexpr std::uint64_t gib(double value) {
  return static_cast<std::uint64_t>(value * 1024.0 * 1024.0 * 1024.0);
}

struct TierSpec {
  const char *encoding;
  const char *label;
  std::uint64_t estimated_model_bytes;
  std::uint64_t recommended_vram_bytes;
};

// The model byte estimates come from the conversion/quantization plan. The
// VRAM thresholds leave room for VAE weights, CFG KV state, graphs, and a
// useful request length. We will replace estimates with measured peaks as the
// quant listening matrix is completed.
constexpr std::array<TierSpec, 6> kTierSpecs = {{
    {"BF16", "Full quality", gib(6.76), gib(16.0)},
    {"F16", "Full quality (F16)", gib(6.76), gib(16.0)},
    {"Q8_0", "High quality", gib(3.9), gib(12.0)},
    {"Q5_K_M", "Balanced", gib(3.0), gib(10.0)},
    // An advertised 8 GB GPU reports about 7.9 GiB. Keep this threshold in
    // binary bytes without rejecting that hardware on a units technicality.
    {"Q4_K_M", "Compact", gib(2.4), gib(7.5)},
    {"F32", "Reference (CPU)", gib(13.52), gib(24.0)},
}};

bool starts_with(std::string_view value, std::string_view prefix) {
  return value.size() >= prefix.size() &&
         value.compare(0, prefix.size(), prefix) == 0;
}

bool ends_with(std::string_view value, std::string_view suffix) {
  return value.size() >= suffix.size() &&
         value.compare(value.size() - suffix.size(), suffix.size(), suffix) ==
             0;
}

std::optional<std::string> gguf_string(const gguf_context *context,
                                       const char *key) {
  const auto index = gguf_find_key(context, key);
  if (index < 0 || gguf_get_kv_type(context, index) != GGUF_TYPE_STRING)
    return std::nullopt;
  return std::string(gguf_get_val_str(context, index));
}

std::optional<std::uint32_t> gguf_u32(const gguf_context *context,
                                      const char *key) {
  const auto index = gguf_find_key(context, key);
  if (index < 0 || gguf_get_kv_type(context, index) != GGUF_TYPE_UINT32)
    return std::nullopt;
  return gguf_get_val_u32(context, index);
}

std::string encoding_from_file_type(std::uint32_t type) {
  switch (type) {
  case 0:
    return "F32";
  case 1:
    return "F16";
  case 7:
    return "Q8_0";
  case 15:
    return "Q4_K_M";
  case 17:
    return "Q5_K_M";
  case 32:
    return "BF16";
  default:
    return {};
  }
}

std::string encoding_from_filename(std::string_view name) {
  for (const auto &spec : kTierSpecs) {
    const std::string suffix = std::string("-") + spec.encoding + ".gguf";
    if (ends_with(name, suffix))
      return spec.encoding;
  }
  return {};
}

std::string component_from_filename(std::string_view name) {
  if (ends_with(name, "-LoRA.gguf"))
    return "generation-adapter";
  if (starts_with(name, "yue2-vae-"))
    return "vae";
  if (starts_with(name, "sheetsage2-mert2-"))
    return "transcription";
  if (starts_with(name, "yue2-"))
    return "generation";
  return {};
}

std::vector<fs::path> candidate_ggufs(const fs::path &directory) {
  std::vector<fs::path> result;
  std::error_code error;
  if (!fs::is_directory(directory, error))
    return result;
  const auto collect = [&](const fs::path &folder) {
    for (const auto &entry : fs::directory_iterator(folder, error)) {
      const auto name = entry.path().filename().string();
      // Failed or deliberately retained converter outputs use this suffix.
      // Do not pass them to GGML during model discovery: gguf_init reports
      // malformed tensor tables directly to stderr before returning nullptr.
      if (entry.is_regular_file(error) && entry.path().extension() == ".gguf" &&
          !ends_with(name, ".invalid.gguf")) {
        result.push_back(entry.path());
      }
    }
  };
  collect(directory);
  for (const auto &entry : fs::directory_iterator(directory, error)) {
    if (entry.is_directory(error))
      collect(entry.path());
  }
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

void add_tokenizer_if_present(std::vector<ModelFileInfo> &files,
                              const fs::path &path) {
  std::error_code error;
  if (!fs::is_regular_file(path, error))
    return;
  const auto canonical = fs::weakly_canonical(path, error);
  const auto normalized = error ? path.lexically_normal() : canonical;
  for (const auto &file : files) {
    if (fs::path(file.path).lexically_normal() == normalized)
      return;
  }
  files.push_back({path.filename().string(), normalized.string(), "tokenizer",
                   "", static_cast<std::uint64_t>(fs::file_size(path, error)),
                   false});
}

DeviceType convert_device_type(enum ggml_backend_dev_type type) {
  switch (type) {
  case GGML_BACKEND_DEVICE_TYPE_CPU:
    return DeviceType::cpu;
  case GGML_BACKEND_DEVICE_TYPE_GPU:
    return DeviceType::gpu;
  case GGML_BACKEND_DEVICE_TYPE_IGPU:
    return DeviceType::integrated_gpu;
  case GGML_BACKEND_DEVICE_TYPE_ACCEL:
    return DeviceType::accelerator;
  case GGML_BACKEND_DEVICE_TYPE_META:
    return DeviceType::meta;
  }
  return DeviceType::unknown;
}

} // namespace

const char *device_type_name(DeviceType type) noexcept {
  switch (type) {
  case DeviceType::cpu:
    return "cpu";
  case DeviceType::gpu:
    return "gpu";
  case DeviceType::integrated_gpu:
    return "integrated_gpu";
  case DeviceType::accelerator:
    return "accelerator";
  case DeviceType::meta:
    return "meta";
  case DeviceType::unknown:
    return "unknown";
  }
  return "unknown";
}

std::vector<DeviceInfo> available_devices() {
  detail::load_dynamic_backends_once();
  std::vector<DeviceInfo> result;
  result.reserve(ggml_backend_dev_count());
  for (std::size_t index = 0; index < ggml_backend_dev_count(); ++index) {
    auto *device = ggml_backend_dev_get(index);
    if (!device)
      continue;
    ggml_backend_dev_props properties{};
    ggml_backend_dev_get_props(device, &properties);
    auto *registration = ggml_backend_dev_backend_reg(device);
    DeviceInfo info;
    info.index = index;
    info.name = properties.name ? properties.name : "";
    info.description = properties.description ? properties.description : "";
    info.backend = registration && ggml_backend_reg_name(registration)
                       ? ggml_backend_reg_name(registration)
                       : "";
    info.id = properties.device_id ? properties.device_id : "";
    info.type = convert_device_type(properties.type);
    info.memory_free_bytes = properties.memory_free;
    info.memory_total_bytes = properties.memory_total;
    result.push_back(std::move(info));
  }
  return result;
}

std::vector<ModelFileInfo>
inspect_model_files(const std::string &models_directory) {
  const fs::path directory(models_directory);
  std::vector<ModelFileInfo> result;
  for (const auto &path : candidate_ggufs(directory)) {
    gguf_init_params parameters{true, nullptr};
    std::unique_ptr<gguf_context, decltype(&gguf_free)> context(
        gguf_init_from_file(path.string().c_str(), parameters), gguf_free);
    if (!context)
      continue;

    ModelFileInfo file;
    file.name = path.filename().string();
    std::error_code error;
    const auto canonical = fs::weakly_canonical(path, error);
    file.path = (error ? path.lexically_normal() : canonical).string();
    file.size_bytes = static_cast<std::uint64_t>(fs::file_size(path, error));

    const auto component = gguf_string(context.get(), "yue2.component");
    if (component) {
      file.component = *component;
      file.metadata_classified = true;
    } else {
      file.component = component_from_filename(file.name);
    }
    if (const auto quantization =
            gguf_string(context.get(), "yue2.quantization.encoding")) {
      file.encoding = *quantization;
    } else if (const auto type = gguf_u32(context.get(), "general.file_type")) {
      file.encoding = encoding_from_file_type(*type);
    }
    if (file.encoding.empty())
      file.encoding = encoding_from_filename(file.name);
    if (!file.component.empty())
      result.push_back(std::move(file));
  }

  std::vector<fs::path> roots = {directory};
  for (const auto &file : result) {
    if (file.component == "generation")
      roots.push_back(fs::path(file.path).parent_path());
  }
  for (const auto &root : roots) {
    add_tokenizer_if_present(result, root / "sidecars" / "yue2-qwen.tiktoken");
    add_tokenizer_if_present(result, root / "qwen.tiktoken");
  }
  std::sort(result.begin(), result.end(),
            [](const auto &left, const auto &right) {
              if (left.component != right.component)
                return left.component < right.component;
              if (left.encoding != right.encoding)
                return left.encoding < right.encoding;
              return left.path < right.path;
            });
  return result;
}

std::optional<ModelFileInfo>
find_model_file(const std::vector<ModelFileInfo> &files,
                const std::string &component,
                const std::vector<std::string> &encoding_preference) {
  if (encoding_preference.empty()) {
    const auto found =
        std::find_if(files.begin(), files.end(), [&](const auto &file) {
          return file.component == component;
        });
    return found == files.end() ? std::nullopt
                                : std::optional<ModelFileInfo>(*found);
  }
  for (const auto &encoding : encoding_preference) {
    const auto found =
        std::find_if(files.begin(), files.end(), [&](const auto &file) {
          return file.component == component && file.encoding == encoding;
        });
    if (found != files.end())
      return *found;
  }
  return std::nullopt;
}

std::string recommended_quantization(std::uint64_t memory) {
  if (memory >= gib(16.0))
    return "BF16";
  if (memory >= gib(12.0))
    return "Q8_0";
  if (memory >= gib(10.0))
    return "Q5_K_M";
  return "Q4_K_M";
}

std::vector<QuantizationTierInfo>
quantization_tiers(const std::vector<ModelFileInfo> &files,
                   std::uint64_t total_memory, std::uint64_t free_memory) {
  const auto recommendation = recommended_quantization(total_memory);
  std::vector<QuantizationTierInfo> result;
  result.reserve(kTierSpecs.size());
  for (const auto &spec : kTierSpecs) {
    QuantizationTierInfo tier;
    tier.encoding = spec.encoding;
    tier.label = spec.label;
    tier.estimated_model_bytes = spec.estimated_model_bytes;
    tier.recommended_vram_bytes = spec.recommended_vram_bytes;
    tier.fits_total_memory = total_memory >= spec.recommended_vram_bytes;
    tier.fits_free_memory = free_memory >= spec.recommended_vram_bytes;
    tier.recommended = tier.encoding == recommendation;
    if (const auto installed =
            find_model_file(files, "generation", {tier.encoding})) {
      tier.installed = true;
      tier.model_path = installed->path;
      tier.installed_bytes = installed->size_bytes;
    }
    result.push_back(std::move(tier));
  }
  return result;
}

} // namespace yue2
