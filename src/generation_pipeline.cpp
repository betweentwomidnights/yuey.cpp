#include "yue2/generation_pipeline.h"

#include "yue2/tokenizer.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
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

// Stage wall-clock reporting, so a slow run can be attributed rather than
// guessed at. Enabled with YUE2_DEBUG_TIMING.
class StageTimer {
public:
    explicit StageTimer(const char * name)
        : name_(name), started_(std::chrono::steady_clock::now()) {}
    ~StageTimer() {
        static const bool report = std::getenv("YUE2_DEBUG_TIMING") != nullptr;
        if (!report) return;
        const auto ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started_).count();
        std::fprintf(stderr, "[yue2] stage %-9s %8.0f ms\n", name_, ms);
    }
private:
    const char * name_;
    std::chrono::steady_clock::time_point started_;
};

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
          model_path(model_path),
          autoregressive(std::make_unique<AutoregressiveModel>(
              model_path, this->options.autoregressive)),
          vae_path(vae_path),
          vae_options(resolve_vae_options(this->options)) {
        if (this->options.flow.context_length > 24576) {
            throw std::invalid_argument("YuE2 flow context exceeds generation context");
        }
    }

    // The generator, reloaded if a previous frugal call freed it. Adapters
    // are part of the stored options, so a reload binds the same ones.
    AutoregressiveModel & ar() {
        if (!autoregressive) {
            autoregressive = std::make_unique<AutoregressiveModel>(
                model_path, options.autoregressive);
        }
        return *autoregressive;
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
        const bool planner_chose_the_score = !effective.abc.has_value();
        // Bars the caller supplied as a continuation prefix. A planning header
        // contributes none, so this separates "yuey wrote all of this" from
        // "yuey extended something the caller already had".
        std::uint32_t supplied_prefix_bars = 0;
        double supplied_prefix_seconds = 0.0;
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
                supplied_prefix_bars = prefix_score.bars;
                supplied_prefix_seconds = prefix_score.duration_seconds;
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
            // Planning runs until the model composes an ending or hits the
            // token limit, and only then is the score cut to what we keep. On
            // a continuation that is the worst case: the prefix is a whole
            // transcribed score, so the model has plenty to continue and keeps
            // going, and natural length trims afterwards rather than bounding
            // anything. Stop once the plan is comfortably past what the fit
            // will keep.
            //
            // The stop can only land where the score parses, which is a
            // balanced Vocal/Ins block boundary, so it is block-granular. That
            // is what made an exact bar limit a no-op once
            // (eb19c73, reverted by 417671a); here overshooting by a block is
            // the point rather than the flaw.
            std::uint32_t planning_stop_bars = 0;
            double planning_stop_seconds = 0.0;
            if (effective.planning_overrun > 0.0) {
                if (effective.target_bars != 0) {
                    const double wanted =
                        static_cast<double>(effective.target_bars) * effective.planning_overrun;
                    planning_stop_bars = wanted >= static_cast<double>(
                        std::numeric_limits<std::uint32_t>::max())
                        ? std::numeric_limits<std::uint32_t>::max()
                        : static_cast<std::uint32_t>(wanted);
                } else if (effective.natural_max_seconds > 0.0) {
                    // A continuation keeps its prefix whatever the ceiling
                    // says, so the plan has to clear the prefix before the
                    // ceiling means anything.
                    planning_stop_seconds = supplied_prefix_seconds +
                        effective.natural_max_seconds * effective.planning_overrun;
                }
            }
            const auto planning_loop_bars = effective.planning_loop_bars;
            if (planning_stop_bars != 0 || planning_stop_seconds > 0.0 ||
                planning_loop_bars != 0) {
                planning_control.force_stop =
                    [this, &seeded_abc_ids, planning_stop_bars, planning_stop_seconds,
                     planning_loop_bars](
                        const std::vector<std::int32_t> & generated) {
                        // A score only parses just after a barline closing a
                        // balanced Vocal/Ins block. Check the tail before
                        // decoding the whole growing plan: doing a full decode
                        // on every token makes this check quadratic in length.
                        if (generated.size() < 64) return false;
                        try {
                            bool barline = false;
                            for (std::size_t index = generated.size(); index > 0;) {
                                const auto tail = tokenizer.decode({generated[--index]});
                                auto end = tail.size();
                                while (end > 0 && std::isspace(
                                    static_cast<unsigned char>(tail[end - 1]))) {
                                    --end;
                                }
                                if (end == 0) continue;
                                barline = tail[end - 1] == '|';
                                break;
                            }
                            if (!barline) return false;
                        } catch (const std::exception &) {
                            return false;
                        }
                        try {
                            auto ids = seeded_abc_ids;
                            ids.insert(ids.end(), generated.begin(), generated.end());
                            const auto score = inspect_abc_score(tokenizer.decode(ids), true);
                            // A planner repeating itself has stopped composing.
                            // Cutting that short loses nothing, so it does not
                            // wait for the overrun margin.
                            if (planning_loop_bars != 0 &&
                                score.repeated_tail_bars >= planning_loop_bars) {
                                return true;
                            }
                            if (planning_stop_bars != 0 && score.bars >= planning_stop_bars) {
                                return true;
                            }
                            return planning_stop_seconds > 0.0 &&
                                score.duration_seconds >= planning_stop_seconds;
                        } catch (const std::exception &) {
                            // Mid-bar, so not a boundary we may stop on.
                            return false;
                        }
                    };
            }
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
            StageTimer plan_timer("plan");
            const auto planned = ar().generate(
                initial, run_options.generation.abc,
                AutoregressivePhase::abc, effective.seed,
                planning_control);
            result.abc_token_ids = std::move(seeded_abc_ids);
            result.abc_token_ids.insert(
                result.abc_token_ids.end(), planned.tokens.begin(), planned.tokens.end());
            result.abc = tokenizer.decode(result.abc_token_ids);
            result.abc_truncated = !planned.reached_end;
            // Two ways a plan arrives unusable, both recovered the same way:
            // keep the longest prefix that still renders.
            //
            // Planning can reach its token budget part way through a bar, and a
            // full symbolic plan is dense enough that a long instrumental one
            // can run out of room. But a plan that ended cleanly can equally
            // hold a bar whose notes do not fill the meter, and nothing on this
            // path looks at bar durations, so the first thing to notice was the
            // MIDI exporter - which failed the whole job after the entire plan
            // had been paid for. Two of five instrumental planning runs died
            // that way, each after more than four minutes.
            auto complete = trim_to_complete_score(result.abc);
            if (complete != result.abc) {
                result.abc_repaired = !result.abc_truncated;
                result.abc = std::move(complete);
                result.abc_token_ids = tokenizer.encode(result.abc);
            }
            if (control.on_progress) {
                const auto completed = static_cast<std::uint32_t>(
                    planned.tokens.size() + (planned.reached_end ? 1 : 0));
                control.on_progress(
                    GenerationStage::abc, completed, std::max(1U, completed));
            }
        }
        if (std::getenv("YUE2_DEBUG_TIMING") != nullptr) {
            std::fprintf(stderr, "[yue2] planned %u bars, keeping %u, abc_truncated=%d, abc_repaired=%d\n",
                         inspect_abc_score(result.abc, true).bars,
                         effective.target_bars, result.abc_truncated ? 1 : 0,
                         result.abc_repaired ? 1 : 0);
        }
        if (effective.target_bars != 0) {
            // A plan can come back shorter than what was asked for: planning
            // hits its token limit, or part of it would not render and is
            // trimmed. Handing back a shorter score beats failing a job that
            // already cost minutes, so the target is clamped to what the plan
            // actually holds. A continuation is where this bites, because its
            // prefix is a whole transcribed score and the target is that plus
            // the bars to add. abc_truncated already says the plan was cut
            // short, and the server reports it.
            const auto available = inspect_abc_score(result.abc, true).bars;
            const auto target = std::min(effective.target_bars, available);
            if (target != 0) {
                result.abc = fit_abc_score_to_bars(
                    result.abc, target, std::min(effective.outro_bars, target));
                result.abc_token_ids = tokenizer.encode(result.abc);
            }
        } else if (planner_chose_the_score) {
            // Natural length is the one path with no bound but the context
            // window: the score sets the semantic floor, so a plan that runs
            // long forces a render that runs long. A supplied score is the
            // caller's own decision and is left alone. A continuation is held
            // too, but by what it adds: its prefix bars are retained whatever
            // the ceiling says, so the result never comes back shorter than the
            // audio it extends.
            auto held = fit_natural_plan_to_ceiling(
                result.abc, effective.natural_max_seconds, effective.outro_bars,
                supplied_prefix_bars);
            if (held != result.abc) {
                result.abc = std::move(held);
                result.abc_token_ids = tokenizer.encode(result.abc);
            }
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
            result.abc_repaired = plan.abc_repaired;
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
        // Whether the budget was cut to the length the caller's own score asks
        // for. Reaching that cap is the render finishing, not running out of
        // room, so it must not be reported as truncation.
        bool aligned_to_supplied_score = false;
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
            // Whose decision the length was. A score the caller supplied - a
            // cover's transcription of their own audio, or an abc they wrote -
            // already says how long the result should be, so the budget is that
            // plus a decay tail. A score yuey planned is its own, and the
            // margin there is real headroom for it to finish a phrase.
            //
            // score_aligned_semantic_budget adds a flat 30s because generation
            // routinely spends the whole budget instead of stopping at
            // MUSIC_END, which makes the margin the delivered length rather
            // than headroom. On a long score that is a tolerable overshoot; on
            // a short one it dominates, and a 16 second clip sent to /cover
            // came back as 46 seconds of audio.
            if (!effective.semantic_prefix.empty() || effective.abc.has_value()) {
                // With a real-audio prefix the stock AR can otherwise treat
                // the continuation as a fresh song and run far beyond the
                // completed score. Keep enough room for two seconds of decay,
                // while still requiring every planned continuation frame.
                constexpr std::uint32_t continuation_tail_frames = 50;
                const auto aligned_budget = minimum >
                        std::numeric_limits<std::uint32_t>::max() - continuation_tail_frames
                    ? std::numeric_limits<std::uint32_t>::max()
                    : minimum + continuation_tail_frames;
                if (aligned_budget < budget) aligned_to_supplied_score = true;
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
            StageTimer semantic_timer("semantic");
            semantic = ar().generate(
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
            StageTimer semantic_timer("semantic");
            semantic = ar().generate_cfg(
                positive, negative, semantic_sampling,
                AutoregressivePhase::semantic, guidance, effective.seed,
                ar_control(control, GenerationStage::semantic));
        }
        // Stopping at a budget cut to the caller's own score is the render
        // being the length they asked for, so it is not truncation. Without
        // this every short cover warns: the model rarely ends on its own
        // inside a score it did not choose, and the warning would fire on the
        // ordinary case rather than the surprising one.
        result.semantic_truncated = !semantic.reached_end && !aligned_to_supplied_score;
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
        {
            StageTimer flow_timer("flow");
            result.latents = ar().synthesize_latents(
                generation_prefix, result.semantic_codec_ids, effective.seed, run_options.flow,
                ar_control(control, GenerationStage::flow));
        }
        // The generator's weights and graph buffers are ~3 GB on the Q4 tier,
        // and the decode needs its own window buffer on top: together they are
        // what ran an 8 GB card out of memory at the end of a long song.
        if (!run_options.keep_models) autoregressive.reset();
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
    std::string model_path;
    std::unique_ptr<AutoregressiveModel> autoregressive;
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
