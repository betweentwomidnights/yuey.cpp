#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace yue2 {

// Local runtime information used by the standalone UI and by embedders such as
// gary4juce. Device discovery deliberately goes through GGML so CUDA, Vulkan,
// Metal, HIP, SYCL, and future shared backends report through one contract.
enum class DeviceType {
  cpu,
  gpu,
  integrated_gpu,
  accelerator,
  meta,
  unknown,
};

struct DeviceInfo {
  std::size_t index = 0;
  std::string name;
  std::string description;
  std::string backend;
  std::string id;
  DeviceType type = DeviceType::unknown;
  std::uint64_t memory_free_bytes = 0;
  std::uint64_t memory_total_bytes = 0;
};

const char *device_type_name(DeviceType type) noexcept;
std::vector<DeviceInfo> available_devices();

// A metadata-only model entry. New GGUFs are classified by yue2.component and
// yue2.quantization.encoding/general.file_type; the filename fallback keeps
// pre-v1.0 local conversions usable.
struct ModelFileInfo {
  std::string name;
  std::string path;
  std::string component;
  std::string encoding;
  std::uint64_t size_bytes = 0;
  bool metadata_classified = false;
};

std::vector<ModelFileInfo>
inspect_model_files(const std::string &models_directory);

std::optional<ModelFileInfo>
find_model_file(const std::vector<ModelFileInfo> &files,
                const std::string &component,
                const std::vector<std::string> &encoding_preference = {});

struct QuantizationTierInfo {
  std::string encoding;
  std::string label;
  std::uint64_t estimated_model_bytes = 0;
  std::uint64_t recommended_vram_bytes = 0;
  bool installed = false;
  std::string model_path;
  std::uint64_t installed_bytes = 0;
  bool fits_total_memory = false;
  bool fits_free_memory = false;
  bool recommended = false;
};

// Hardware recommendations are intentionally conservative and represent a
// normal generation working set, not merely the bytes occupied by weights.
// Unknown or sub-8-GiB accelerators fall back to the smallest published tier
// and report fits_* = false so the UI can explain the constraint.
std::string
recommended_quantization(std::uint64_t accelerator_total_memory_bytes);

std::vector<QuantizationTierInfo>
quantization_tiers(const std::vector<ModelFileInfo> &files,
                   std::uint64_t accelerator_total_memory_bytes,
                   std::uint64_t accelerator_free_memory_bytes);

} // namespace yue2
