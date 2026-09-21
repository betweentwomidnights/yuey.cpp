#include "yue2/generation_pipeline.h"

#include "yue2/tokenizer.h"

#include <algorithm>
#include <cmath>
#include <limits>
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
    if (request.semantic_prefix.size() >= 24576) {
        throw std::invalid_argument("YuE2 semantic continuation prefix is too long");
    }
    for (const auto token : request.semantic_prefix) {
        if (token < 0 || token >= kCodecSize) {
            throw std::invalid_argument("YuE2 semantic continuation token is out of range");
        }
    }
    if (request.target_bars != 0 && request.ending_mode != EndingMode::outro) {
        throw std::invalid_argument(
            "YuE2 target bars require ending mode outro");
    }
    if (request.ending_mode == EndingMode::outro && request.target_bars == 0) {
        throw std::invalid_argument(
            "YuE2 outro ending requires a target bar count");
    }
    if (request.target_bars != 0 && request.symbolic_mode == SymbolicMode::off) {
        throw std::invalid_argument(
            "YuE2 target bars require melody or full symbolic planning");
    }
    if (request.target_bars != 0 &&
        (request.outro_bars == 0 || request.outro_bars > request.target_bars)) {
        throw std::invalid_argument(
            "YuE2 outro bars must be within the target bar count");
    }
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
          vae_path(vae_path),
          vae_options(resolve_vae_options(this->options)) {
        if (this->options.flow.context_length > 24576) {
            throw std::invalid_argument("YuE2 flow context exceeds generation context");
        }
    }

    SongRequest effective_request(const SongRequest & request) const {
        auto effective = request;
        if (effective.instrumental) {
            effective.style = "Instrumental, no vocals, no singing, no humming. " +
                effective.style;
            effective.lyrics = effective.abc
                ? make_instrumental_lyrics(*effective.abc)
                : make_instrumental_lyrics({});
        }
        return effective;
    }

    GeneratedPlan plan_effective(
        const SongRequest & effective,
        const GenerationRunOptions & run_options,
        const GenerationControl & control) {
        if (effective.symbolic_mode == SymbolicMode::off) {
            throw std::invalid_argument(
                "YuE2 planner-only mode requires melody or full symbolic planning");
        }
        GeneratedPlan result;
        if (effective.abc) {
            result.abc = *effective.abc;
            result.abc_token_ids = tokenizer.encode(result.abc);
            if (control.on_progress) {
                control.on_progress(GenerationStage::abc, 1, 1);
            }
        } else {
            auto initial = make_positive_prefix(effective, tokenizer);
            std::vector<std::int32_t> seeded_abc_ids;
            std::uint32_t minimum_complete_bars = effective.target_bars;
            if (effective.abc_prefix) {
                // A planning prefix is a header with no bars yet; a score
                // continuation prefix is a complete transcription. Both arrive
                // here, so an empty score is a legitimate answer rather than a
                // fault, and a planning prefix then keeps the old one-bar floor.
                const auto prefix_score = inspect_abc_score(*effective.abc_prefix, true);
                if (effective.target_bars == 0) {
                    if (prefix_score.bars == std::numeric_limits<std::uint32_t>::max()) {
                        throw std::invalid_argument(
                            "YuE2 continuation score has too many bars to extend");
                    }
                    // A complete transcribed prefix already satisfies a natural
                    // plan's old one-bar floor. Require one newly completed
                    // paired Voice/Ins bar so score continuation cannot stop
                    // at its input or a malformed partial continuation tail.
                    minimum_complete_bars = prefix_score.bars + 1;
                }
                seeded_abc_ids = tokenizer.encode(*effective.abc_prefix);
                initial.insert(initial.end(), seeded_abc_ids.begin(), seeded_abc_ids.end());
            }
            auto planning_control = ar_control(control, GenerationStage::abc);
            planning_control.allow_stop = [this, &seeded_abc_ids, minimum_complete_bars](
                                              const std::vector<std::int32_t> & generated) {
                try {
                    auto ids = seeded_abc_ids;
                    ids.insert(ids.end(), generated.begin(), generated.end());
                    const auto score = inspect_abc_score(tokenizer.decode(ids));
                    // ABC_END must land after a structurally complete bar. A
                    // natural continuation must also extend the source score.
                    return score.bars >= std::max(1U, minimum_complete_bars);
                } catch (const std::exception &) {
                    // Keep sampling until both native lanes end on a barline.
                    return false;
                }
            };
            const auto planned = autoregressive.generate(
                initial, run_options.generation.abc,
                AutoregressivePhase::abc, effective.seed,
                planning_control);
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
        if (effective.target_bars != 0) {
            result.abc = fit_abc_score_to_bars(
                result.abc, effective.target_bars, effective.outro_bars);
            result.abc_token_ids = tokenizer.encode(result.abc);
        }
        if (effective.experimental_vocal_rest || effective.instrumental) {
            result.abc = make_vocal_rest_abc(result.abc);
            result.abc_token_ids = tokenizer.encode(result.abc);
        }
        const auto score = inspect_abc_score(result.abc);
        result.score_bars = score.bars;
        result.score_duration_seconds = score.duration_seconds;
        result.midi_exports = serialize_yue2_abc_midis(result.abc);
        return result;
    }

    GeneratedPlan plan(
        const SongRequest & request,
        const GenerationRunOptions & run_options,
        const GenerationControl & control) {
        validate_request(request);
        check_cancelled(control);
        return plan_effective(effective_request(request), run_options, control);
    }

    GeneratedSong generate(
        const SongRequest & request,
        const GenerationRunOptions & run_options,
        const GenerationControl & control) {
        validate_request(request);
        check_cancelled(control);
        auto effective = effective_request(request);
        GeneratedSong result;
        if (effective.symbolic_mode != SymbolicMode::off) {
            auto plan = plan_effective(effective, run_options, control);
            result.abc = std::move(plan.abc);
            result.abc_token_ids = std::move(plan.abc_token_ids);
            result.abc_truncated = plan.abc_truncated;
            result.score_bars = plan.score_bars;
            result.score_duration_seconds = plan.score_duration_seconds;
            result.midi_exports = std::move(plan.midi_exports);
            if (effective.instrumental) {
                effective.lyrics = make_instrumental_lyrics(result.abc);
            }
        }

        const std::optional<std::vector<std::int32_t>> abc_ids =
            effective.symbolic_mode == SymbolicMode::off
            ? std::nullopt
            : std::optional<std::vector<std::int32_t>>(result.abc_token_ids);
        const auto generation_prefix = make_positive_prefix(effective, tokenizer, abc_ids);
        auto positive = generation_prefix;
        positive.reserve(positive.size() + effective.semantic_prefix.size());
        for (const auto code : effective.semantic_prefix) {
            positive.push_back(kCodecOffset + code);
        }
        if (positive.size() >= 24576) {
            throw std::invalid_argument(
                "YuE2 score and audio continuation prefix leave no semantic room");
        }
        const auto guidance = generation_guidance(effective);
        auto semantic_sampling = run_options.generation.semantic;
        if (!run_options.semantic_budget_explicit && result.score_bars != 0) {
            AbcScoreInfo score;
            score.bars = result.score_bars;
            score.duration_seconds = result.score_duration_seconds;
            auto total_budget = score_aligned_semantic_budget(score);
            const auto total_minimum = static_cast<std::uint32_t>(
                std::ceil(score.duration_seconds * 25.0));
            const auto prefix_frames = static_cast<std::uint32_t>(effective.semantic_prefix.size());
            auto budget = total_budget > prefix_frames ? total_budget - prefix_frames : 0;
            const auto minimum = total_minimum > prefix_frames
                ? total_minimum - prefix_frames : 1U;
            if (!effective.semantic_prefix.empty()) {
                // With a real-audio prefix the stock AR can otherwise treat
                // the continuation as a fresh song and run far beyond the
                // completed score. Keep enough room for two seconds of decay,
                // while still requiring every planned continuation frame.
                constexpr std::uint32_t continuation_tail_frames = 50;
                const auto aligned_budget = minimum >
                        std::numeric_limits<std::uint32_t>::max() - continuation_tail_frames
                    ? std::numeric_limits<std::uint32_t>::max()
                    : minimum + continuation_tail_frames;
                budget = std::min(budget, aligned_budget);
            }
            const auto positive_capacity = positive.size() < 24576
                ? 24576 - positive.size()
                : 0;
            budget = static_cast<std::uint32_t>(std::min<std::size_t>(budget, positive_capacity));
            if (budget < minimum) {
                throw std::invalid_argument(
                    "YuE2 score is too long to align inside the model context");
            }
            semantic_sampling.max_tokens = budget;
            semantic_sampling.min_tokens = effective.semantic_prefix.empty()
                ? std::max(semantic_sampling.min_tokens, minimum)
                : minimum;
        }
        result.semantic_budget = semantic_sampling.max_tokens;
        AutoregressiveResult semantic;
        if (guidance == 1.0F) {
            semantic = autoregressive.generate(
                positive, semantic_sampling,
                AutoregressivePhase::semantic, effective.seed,
                ar_control(control, GenerationStage::semantic));
        } else {
            auto negative = make_negative_prefix(effective, tokenizer, abc_ids);
            negative.reserve(negative.size() + effective.semantic_prefix.size());
            for (const auto code : effective.semantic_prefix) {
                negative.push_back(kCodecOffset + code);
            }
            if (negative.size() >= 24576) {
                throw std::invalid_argument(
                    "YuE2 negative score prefix leaves no semantic room in the model context");
            }
            if (negative.size() + semantic_sampling.max_tokens > 24576) {
                semantic_sampling.max_tokens = static_cast<std::uint32_t>(24576 - negative.size());
                if (semantic_sampling.max_tokens < semantic_sampling.min_tokens) {
                    throw std::invalid_argument(
                        "YuE2 guided score is too long to align inside the model context");
                }
                semantic_sampling.min_tokens = std::min(
                    semantic_sampling.min_tokens, semantic_sampling.max_tokens);
                result.semantic_budget = semantic_sampling.max_tokens;
            }
            semantic = autoregressive.generate_cfg(
                positive, negative, semantic_sampling,
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
        result.semantic_prefix_frames = static_cast<std::uint32_t>(effective.semantic_prefix.size());
        result.semantic_codec_ids = effective.semantic_prefix;
        result.semantic_codec_ids.reserve(
            result.semantic_codec_ids.size() + semantic.tokens.size());
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
            generation_prefix, result.semantic_codec_ids, effective.seed, run_options.flow,
            ar_control(control, GenerationStage::flow));
        VaeDecodeControl decode_control;
        decode_control.should_cancel = control.should_cancel;
        if (control.on_progress) {
            decode_control.on_progress = [&control](std::uint32_t current, std::uint32_t total) {
                control.on_progress(GenerationStage::decode, current, total);
            };
        }
        if (!vae) vae = std::make_unique<VaeDecoder>(vae_path, vae_options);
        result.audio = vae->decode(result.latents, decode_control);
        if (control.on_progress) control.on_progress(GenerationStage::complete, 1, 1);
        return result;
    }

    GenerationPipelineOptions options;
    TextTokenizer tokenizer;
    AutoregressiveModel autoregressive;
    std::string vae_path;
    VaeRuntimeOptions vae_options;
    std::unique_ptr<VaeDecoder> vae;
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

GeneratedPlan GenerationPipeline::plan(const SongRequest & request) {
    return impl_->plan(
        request,
        {impl_->options.generation, impl_->options.flow,
         impl_->options.semantic_budget_explicit},
        {});
}

GeneratedPlan GenerationPipeline::plan(
    const SongRequest & request,
    const GenerationRunOptions & options,
    const GenerationControl & control) {
    if (options.flow.context_length > 24576) {
        throw std::invalid_argument("YuE2 flow context exceeds generation context");
    }
    return impl_->plan(request, options, control);
}

GeneratedSong GenerationPipeline::generate(const SongRequest & request) {
    return impl_->generate(
        request,
        {impl_->options.generation, impl_->options.flow,
         impl_->options.semantic_budget_explicit},
        {});
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
