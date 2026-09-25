#include "yue2/runtime_info.h"

#include "gguf.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

namespace {

namespace fs = std::filesystem;

constexpr std::uint64_t gib(std::uint64_t value) {
  return value * 1024ULL * 1024ULL * 1024ULL;
}

void write_gguf(const fs::path &path, const char *component,
                const char *encoding, std::uint32_t file_type,
                bool include_component = true) {
  auto *file = gguf_init_empty();
  assert(file);
  gguf_set_val_str(file, "general.architecture", "yue2");
  if (include_component)
    gguf_set_val_str(file, "yue2.component", component);
  if (encoding && *encoding)
    gguf_set_val_str(file, "yue2.quantization.encoding", encoding);
  gguf_set_val_u32(file, "general.file_type", file_type);
  assert(gguf_write_to_file(file, path.string().c_str(), false));
  gguf_free(file);
}

} // namespace

int main() {
  assert(std::string(yue2::device_type_name(yue2::DeviceType::gpu)) == "gpu");
  assert(yue2::recommended_quantization(24 * gib(1)) == "BF16");
  assert(yue2::recommended_quantization(16 * gib(1)) == "BF16");
  assert(yue2::recommended_quantization(12 * gib(1)) == "Q8_0");
  assert(yue2::recommended_quantization(10 * gib(1)) == "Q5_K_M");
  assert(yue2::recommended_quantization(8 * gib(1)) == "Q4_K_M");
  assert(yue2::recommended_quantization(0) == "Q4_K_M");

  const auto nonce =
      std::chrono::steady_clock::now().time_since_epoch().count();
  const auto root = fs::temp_directory_path() /
                    ("yue2-runtime-info-" + std::to_string(nonce));
  fs::create_directories(root / "package" / "sidecars");
  write_gguf(root / "package" / "renamed-generation.gguf", "generation",
             "Q5_K_M", 17);
  write_gguf(root / "yue2-3.6B-v1.0-Q4_K_M.gguf", "generation", "", 15, false);
  write_gguf(root / "yue2-vae-v1.0-F16.gguf", "vae", "", 1);
  // Existing converter artifacts used lowercase precision names and omitted
  // general.file_type, so discovery must classify their encoding by filename.
  write_gguf(root / "sheetsage2-mert2-f16.gguf", "transcription", "",
             999);
  write_gguf(root / "yue2-failed-conversion.invalid.gguf", "generation",
             "BF16", 32);
  {
    std::ofstream tokenizer(root / "package" / "sidecars" /
                            "yue2-qwen.tiktoken");
    tokenizer << "tokenizer fixture";
  }

  const auto files = yue2::inspect_model_files(root.string());
  const auto q5 = yue2::find_model_file(files, "generation", {"Q5_K_M"});
  const auto q4 = yue2::find_model_file(files, "generation", {"Q4_K_M"});
  const auto vae = yue2::find_model_file(files, "vae", {"F16"});
  const auto transcription =
      yue2::find_model_file(files, "transcription", {"F16"});
  const auto tokenizer = yue2::find_model_file(files, "tokenizer");
  assert(std::none_of(files.begin(), files.end(), [](const auto &file) {
    return file.name == "yue2-failed-conversion.invalid.gguf";
  }));
  assert(q5 && q5->metadata_classified &&
         q5->name == "renamed-generation.gguf");
  assert(q4 && !q4->metadata_classified);
  assert(vae && transcription && tokenizer);

  const auto tiers = yue2::quantization_tiers(files, 10 * gib(1), 9 * gib(1));
  bool saw_q5 = false;
  bool saw_q4 = false;
  for (const auto &tier : tiers) {
    if (tier.encoding == "Q5_K_M") {
      saw_q5 = true;
      assert(tier.installed && tier.recommended && tier.fits_total_memory &&
             !tier.fits_free_memory);
    }
    if (tier.encoding == "Q4_K_M") {
      saw_q4 = true;
      assert(tier.installed && !tier.recommended && tier.fits_total_memory &&
             tier.fits_free_memory);
    }
  }
  assert(saw_q5 && saw_q4);

  const auto laptop_tiers =
      yue2::quantization_tiers(files, gib(8) - 64 * 1024 * 1024, gib(7));
  for (const auto &tier : laptop_tiers) {
    if (tier.encoding == "Q4_K_M") {
      assert(tier.recommended && tier.fits_total_memory &&
             !tier.fits_free_memory);
    }
  }

  // The published layout, as models.cmd and models.sh download it: every
  // file flat in one folder, tokenizer included.
  const auto flat = root / "flat";
  fs::create_directories(flat);
  write_gguf(flat / "yue2-3.6B-v1.0-Q4_K_M.gguf", "generation", "Q4_K_M", 15);
  {
    std::ofstream published(flat / "yue2-qwen.tiktoken");
    published << "tokenizer fixture";
  }
  const auto flat_files = yue2::inspect_model_files(flat.string());
  const auto flat_tokenizer = yue2::find_model_file(flat_files, "tokenizer");
  assert(flat_tokenizer && flat_tokenizer->name == "yue2-qwen.tiktoken");

  // This call must remain safe on a CPU-only build; actual device presence is
  // backend/platform dependent and therefore is not asserted.
  (void)yue2::available_devices();

  std::error_code error;
  fs::remove_all(root, error);
  assert(!error);
  return 0;
}
