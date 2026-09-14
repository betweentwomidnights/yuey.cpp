#include "yue2/generation_pipeline.h"

#include "yue2/tokenizer.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <utility>

namespace yue2 {
namespace {

VaeRuntimeOptions resolve_vae_options(const GenerationPipelineOptions & options) {
    auto vae = options.vae;
    if (vae.device.empty()) vae.device = options.autoregressive.device;
    if (vae.threads == 0) vae.threads = options.autoregressive.threads;
    return vae;
}

void validate_request(const SongRequest & request) {
    if (request.instrumental && request.symbolic_mode == SymbolicMode::off) {
        throw std::invalid_argument(
            "YuE2 instrumental mode requires melody or full symbolic planning");
    }
    if (request.instrumental && !request.lyrics.empty()) {
        throw std::invalid_argument(
            "YuE2 instrumental mode is mutually exclusive with lyrics");
    }
    if (request.instrumental && request.experimental_vocal_rest) {
        throw std::invalid_argument(
            "YuE2 instrumental mode already includes the vocal-rest intervention");
    }
    if (request.experimental_vocal_rest && request.symbolic_mode == SymbolicMode::off) {
        throw std::invalid_argument(
            "YuE2 vocal-rest experiment requires melody or full symbolic planning");
    }
    if (request.experimental_vocal_rest && !request.lyrics.empty()) {
        throw std::invalid_argument(
            "YuE2 vocal-rest experiment requires empty lyrics");
    }
    if (request.abc &&
        (request.symbolic_mode == SymbolicMode::off || request.abc->empty())) {
        throw std::invalid_argument(
            "YuE2 external ABC must be nonempty and requires melody or full mode");
    }
    if (request.abc_prefix &&
        (request.symbolic_mode == SymbolicMode::off || request.abc_prefix->empty())) {
        throw std::invalid_argument(
            "YuE2 ABC prefix must be nonempty and requires melody or full mode");
    }
    if (request.abc && request.abc_prefix) {
        throw std::invalid_argument(
            "YuE2 ABC prefix and complete external ABC are mutually exclusive");
    }
    if (request.abc_prefix && request.abc_prefix->back() != '\n') {
        throw std::invalid_argument("YuE2 ABC prefix must end with a newline");
    }
    (void)generation_guidance(request);
}

void check_cancelled(const GenerationControl & control) {
    if (control.should_cancel && control.should_cancel()) {
        throw std::runtime_error("YuE2 generation cancelled");
    }
}

AutoregressiveControl ar_control(
    const GenerationControl & control,
    GenerationStage stage) {
    AutoregressiveControl result;
    result.should_cancel = control.should_cancel;
    if (control.on_progress) {
        result.on_progress = [&control, stage](std::uint32_t current, std::uint32_t total) {
            control.on_progress(stage, current, total);
        };
    }
    return result;
}

} // namespace

class GenerationPipeline::Impl {
public:
    Impl(
        const std::string & model_path,
        const std::string & vae_path,
        const std::string & tokenizer_path,
        GenerationPipelineOptions options)
        : options(std::move(options)),
          tokenizer(tokenizer_path),
          autoregressive(model_path, this->options.autoregressive),
          vae(vae_path, resolve_vae_options(this->options)) {
        if (this->options.flow.context_length > 24576) {
            throw std::invalid_argument("YuE2 flow context exceeds generation context");
        }
    }

    GeneratedSong generate(
        const SongRequest & request,
        const GenerationRunOptions & run_options,
        const GenerationControl & control) {
        validate_request(request);
        check_cancelled(control);
        auto effective = request;
        if (effective.instrumental) {
            effective.style = "Instrumental, no vocals, no singing, no humming. " +
                effective.style;
            effective.lyrics = effective.abc
                ? make_instrumental_lyrics(*effective.abc)
                : make_instrumental_lyrics({});
        }
        GeneratedSong result;
        if (effective.symbolic_mode != SymbolicMode::off) {
            if (effective.abc) {
                result.abc = *effective.abc;
                result.abc_token_ids = tokenizer.encode(result.abc);
            } else {
                auto initial = make_positive_prefix(effective, tokenizer);
                std::vector<std::int32_t> seeded_abc_ids;
                if (effective.abc_prefix) {
                    seeded_abc_ids = tokenizer.encode(*effective.abc_prefix);
                    initial.insert(initial.end(), seeded_abc_ids.begin(), seeded_abc_ids.end());
                }
                const auto planned = autoregressive.generate(
                    initial, run_options.generation.abc,
                    AutoregressivePhase::abc, effective.seed,
                    ar_control(control, GenerationStage::abc));
                result.abc_token_ids = std::move(seeded_abc_ids);
                result.abc_token_ids.insert(
                    result.abc_token_ids.end(), planned.tokens.begin(), planned.tokens.end());
                result.abc = tokenizer.decode(result.abc_token_ids);
                result.abc_truncated = !planned.reached_end;
                if (control.on_progress) {
                    const auto completed = static_cast<std::uint32_t>(
                        planned.tokens.size() + (planned.reached_end ? 1 : 0));
                    control.on_progress(
                        GenerationStage::abc, completed, std::max(1U, completed));
                }
            }
            if (effective.abc && control.on_progress) {
                control.on_progress(GenerationStage::abc, 1, 1);
            }
            if (effective.experimental_vocal_rest || effective.instrumental) {
                result.abc = make_vocal_rest_abc(result.abc);
                result.abc_token_ids = tokenizer.encode(result.abc);
            }
            if (effective.instrumental) {
                effective.lyrics = make_instrumental_lyrics(result.abc);
            }
        }

        const std::optional<std::vector<std::int32_t>> abc_ids =
            effective.symbolic_mode == SymbolicMode::off
            ? std::nullopt
            : std::optional<std::vector<std::int32_t>>(result.abc_token_ids);
        const auto positive = make_positive_prefix(effective, tokenizer, abc_ids);
        const auto guidance = generation_guidance(effective);
        AutoregressiveResult semantic;
        if (guidance == 1.0F) {
            semantic = autoregressive.generate(
                positive, run_options.generation.semantic,
                AutoregressivePhase::semantic, effective.seed,
                ar_control(control, GenerationStage::semantic));
        } else {
            const auto negative = make_negative_prefix(effective, tokenizer, abc_ids);
            semantic = autoregressive.generate_cfg(
                positive, negative, run_options.generation.semantic,
                AutoregressivePhase::semantic, guidance, effective.seed,
                ar_control(control, GenerationStage::semantic));
        }
        result.semantic_truncated = !semantic.reached_end;
        if (control.on_progress) {
            const auto completed = static_cast<std::uint32_t>(
                semantic.tokens.size() + (semantic.reached_end ? 1 : 0));
            control.on_progress(
                GenerationStage::semantic, completed, std::max(1U, completed));
        }
        result.semantic_codec_ids.reserve(semantic.tokens.size());
        for (const auto token : semantic.tokens) {
            if (token < kCodecOffset || token >= kCodecOffset + kCodecSize) {
                throw std::logic_error("YuE2 semantic sampler emitted a non-codec token");
            }
            result.semantic_codec_ids.push_back(token - kCodecOffset);
        }
        if (result.semantic_codec_ids.empty()) {
            throw std::runtime_error("YuE2 semantic generation produced no audio codes");
        }
        result.latents = autoregressive.synthesize_latents(
            positive, result.semantic_codec_ids, effective.seed, run_options.flow,
            ar_control(control, GenerationStage::flow));
        VaeDecodeControl decode_control;
        decode_control.should_cancel = control.should_cancel;
        if (control.on_progress) {
            decode_control.on_progress = [&control](std::uint32_t current, std::uint32_t total) {
                control.on_progress(GenerationStage::decode, current, total);
            };
        }
        result.audio = vae.decode(result.latents, decode_control);
        if (control.on_progress) control.on_progress(GenerationStage::complete, 1, 1);
        return result;
    }

    GenerationPipelineOptions options;
    TextTokenizer tokenizer;
    AutoregressiveModel autoregressive;
    VaeDecoder vae;
};

GenerationPipeline::GenerationPipeline(
    const std::string & model_gguf_path,
    const std::string & vae_gguf_path,
    const std::string & qwen_tiktoken_path,
    const GenerationPipelineOptions & options)
    : impl_(std::make_unique<Impl>(
          model_gguf_path, vae_gguf_path, qwen_tiktoken_path, options)) {}

GenerationPipeline::~GenerationPipeline() = default;
GenerationPipeline::GenerationPipeline(GenerationPipeline &&) noexcept = default;
GenerationPipeline & GenerationPipeline::operator=(GenerationPipeline &&) noexcept = default;

GeneratedSong GenerationPipeline::generate(const SongRequest & request) {
    return impl_->generate(
        request, {impl_->options.generation, impl_->options.flow}, {});
}

GeneratedSong GenerationPipeline::generate(
    const SongRequest & request,
    const GenerationRunOptions & options) {
    if (options.flow.context_length > 24576) {
        throw std::invalid_argument("YuE2 flow context exceeds generation context");
    }
    return impl_->generate(request, options, {});
}

GeneratedSong GenerationPipeline::generate(
    const SongRequest & request,
    const GenerationRunOptions & options,
    const GenerationControl & control) {
    if (options.flow.context_length > 24576) {
        throw std::invalid_argument("YuE2 flow context exceeds generation context");
    }
    return impl_->generate(request, options, control);
}

} // namespace yue2
