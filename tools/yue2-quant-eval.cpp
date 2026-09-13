// yue2-quant-eval: measure what quantization changes in YuE2 generation, one
// stage at a time, against a reference GGUF (normally BF16). Only one
// generation model is resident at a time, so a reference and several tiers
// can be compared in one run.
//
//   AR      teacher-forced semantic logits over the codec vocabulary: KL
//           divergence from the reference, top-1 agreement, top-100 overlap
//   flow    latents from identical prefix, codes and noise, decoded by one
//           shared VAE: relative RMS, cosine, log-spectral distance
//   render  optional seeded end-to-end songs, written as WAVs for listening
//
// Sampled continuations diverge after the first differing token, so comparing
// same-seed renders sample by sample says nothing about quality. The teacher-
// forced AR and fixed-noise flow stages are the numerical comparison; renders
// are for ears.
#include "yue2/audio.h"
#include "yue2/autoregressive.h"
#include "yue2/generation.h"
#include "yue2/generation_pipeline.h"
#include "yue2/tokenizer.h"
#include "yue2/vae.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

constexpr std::size_t kTopK = 100;

constexpr const char * kDefaultStyle = "indie pop, warm female vocal, acoustic guitar, soft drums";
constexpr const char * kDefaultLyrics =
    "[Verse]\nMorning light across the kitchen floor\nHumming something from the night before\n"
    "[Chorus]\nStay a while, stay a while\n";
constexpr const char * kDefaultAbc =
    "X:1\nL:1/8\nM:4/4\nK:C\nC2 E2 G2 E2|F2 A2 c4|B2 G2 E2 D2|C8|\n";

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

std::string read_text(const std::string & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot read " + path);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

class Reader {
public:
    explicit Reader(const std::string & path) : path_(path) {
        std::ifstream input(path, std::ios::binary);
        if (!input) throw std::runtime_error("cannot read fixture " + path);
        bytes_ = {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    }
    void magic(const char tag[4], std::uint8_t version) {
        require(8);
        const std::uint8_t expected[8] = {
            static_cast<std::uint8_t>(tag[0]), static_cast<std::uint8_t>(tag[1]),
            static_cast<std::uint8_t>(tag[2]), static_cast<std::uint8_t>(tag[3]), version, 0, 0, 0};
        if (std::memcmp(bytes_.data(), expected, 8) != 0) {
            throw std::runtime_error("unexpected fixture header in " + path_);
        }
        offset_ = 8;
    }
    std::uint32_t u32() {
        require(4);
        std::uint32_t output = 0;
        for (int byte = 0; byte < 4; ++byte) {
            output |= static_cast<std::uint32_t>(bytes_[offset_ + byte]) << (8 * byte);
        }
        offset_ += 4;
        return output;
    }
    std::int32_t i32() { return static_cast<std::int32_t>(u32()); }
    std::vector<std::int32_t> ids() {
        std::vector<std::int32_t> output(u32());
        for (auto & value : output) value = i32();
        return output;
    }
    std::vector<float> floats() {
        std::vector<float> output(u32());
        for (auto & value : output) {
            const auto bits = u32();
            std::memcpy(&value, &bits, sizeof(value));
        }
        return output;
    }
    bool done() const noexcept { return offset_ == bytes_.size(); }

private:
    void require(std::size_t count) const {
        if (count > bytes_.size() - offset_) throw std::runtime_error("truncated fixture " + path_);
    }
    std::string path_;
    std::vector<std::uint8_t> bytes_;
    std::size_t offset_ = 0;
};

// Official static-cache AR fixture written by tools/dump_yue2_ar_reference.py.
struct ArFixture {
    std::vector<std::int32_t> ids;
    std::vector<float> logits;
    std::int32_t next_token = 0;
    std::vector<float> next_logits;
};

ArFixture read_ar_fixture(const std::string & path) {
    Reader reader(path);
    reader.magic("Y2AR", 3);
    ArFixture fixture;
    fixture.ids = reader.ids();
    fixture.logits = reader.floats();
    fixture.next_token = reader.i32();
    fixture.next_logits = reader.floats();
    (void)reader.ids(); // greedy continuation, exercised by yue2-autoregressive-test
    if (fixture.logits.size() != yue2::kGenerationVocabSize ||
        fixture.next_logits.size() != yue2::kGenerationVocabSize) {
        throw std::runtime_error("unexpected AR fixture vocabulary size");
    }
    return fixture;
}

// Official midpoint-flow fixture written by tools/dump_yue2_flow_reference.py.
struct FlowFixture {
    std::uint32_t steps = 0;
    std::vector<std::int32_t> prefix;
    std::vector<std::int32_t> codec;
    std::vector<float> noise;
    std::vector<float> latents;
    std::vector<float> audio;
};

FlowFixture read_flow_fixture(const std::string & path) {
    Reader reader(path);
    reader.magic("Y2FL", 2);
    FlowFixture fixture;
    fixture.steps = reader.u32();
    fixture.prefix = reader.ids();
    fixture.codec = reader.ids();
    fixture.noise = reader.floats();
    fixture.latents = reader.floats();
    fixture.audio = reader.floats();
    if (!reader.done()) throw std::runtime_error("trailing data in flow fixture " + path);
    return fixture;
}

struct Difference {
    double relative_rms = 0.0;
    double cosine = 0.0;
    double max_abs = 0.0;
};

Difference difference(const std::vector<float> & reference, const std::vector<float> & candidate) {
    if (reference.size() != candidate.size() || reference.empty()) {
        throw std::runtime_error("cannot compare tensors of different sizes");
    }
    double error = 0.0;
    double norm_reference = 0.0;
    double norm_candidate = 0.0;
    double dot = 0.0;
    Difference result;
    for (std::size_t index = 0; index < reference.size(); ++index) {
        const double a = reference[index];
        const double b = candidate[index];
        error += (a - b) * (a - b);
        norm_reference += a * a;
        norm_candidate += b * b;
        dot += a * b;
        result.max_abs = std::max(result.max_abs, std::abs(a - b));
    }
    result.relative_rms = norm_reference > 0.0 ? std::sqrt(error / norm_reference) : std::sqrt(error);
    const double denominator = std::sqrt(norm_reference * norm_candidate);
    result.cosine = denominator > 0.0 ? dot / denominator : 1.0;
    return result;
}

void log_softmax(const float * logits, std::size_t count, std::vector<double> & output) {
    output.resize(count);
    const double maximum = *std::max_element(logits, logits + count);
    double sum = 0.0;
    for (std::size_t index = 0; index < count; ++index) sum += std::exp(logits[index] - maximum);
    const double normalizer = maximum + std::log(sum);
    for (std::size_t index = 0; index < count; ++index) output[index] = logits[index] - normalizer;
}

std::vector<std::uint32_t> top_indices(const float * values, std::size_t count, std::size_t k) {
    std::vector<std::uint32_t> indices(count);
    std::iota(indices.begin(), indices.end(), 0U);
    if (k < count) {
        std::nth_element(indices.begin(), indices.begin() + static_cast<std::ptrdiff_t>(k), indices.end(),
            [values](std::uint32_t a, std::uint32_t b) { return values[a] > values[b]; });
        indices.resize(k);
    }
    std::sort(indices.begin(), indices.end());
    return indices;
}

struct Divergence {
    double kl = 0.0;       // KL(reference || candidate), nats
    bool top1 = false;
    double overlap = 0.0;  // |top-k(reference) ∩ top-k(candidate)| / k
};

Divergence divergence(const float * reference, const float * candidate, std::size_t count) {
    std::vector<double> p;
    std::vector<double> q;
    log_softmax(reference, count, p);
    log_softmax(candidate, count, q);
    Divergence result;
    for (std::size_t index = 0; index < count; ++index) {
        const double probability = std::exp(p[index]);
        if (probability > 0.0) result.kl += probability * (p[index] - q[index]);
    }
    result.top1 = std::max_element(reference, reference + count) - reference ==
        std::max_element(candidate, candidate + count) - candidate;
    const auto a = top_indices(reference, count, kTopK);
    const auto b = top_indices(candidate, count, kTopK);
    std::vector<std::uint32_t> shared;
    std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(shared));
    result.overlap = static_cast<double>(shared.size()) / static_cast<double>(a.size());
    return result;
}

void fft(std::vector<std::complex<double>> & values) {
    const std::size_t n = values.size();
    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(values[i], values[j]);
    }
    for (std::size_t length = 2; length <= n; length <<= 1) {
        const double angle = -2.0 * 3.14159265358979323846 / static_cast<double>(length);
        const std::complex<double> step(std::cos(angle), std::sin(angle));
        for (std::size_t start = 0; start < n; start += length) {
            std::complex<double> w(1.0, 0.0);
            for (std::size_t k = 0; k < length / 2; ++k) {
                const auto u = values[start + k];
                const auto v = values[start + k + length / 2] * w;
                values[start + k] = u + v;
                values[start + k + length / 2] = u - v;
                w *= step;
            }
        }
    }
}

// Mean per-frame RMS difference of log magnitude spectra (dB) of the mono mix,
// 1024-sample Hann frames with a 256-sample hop. Unlike waveform error it is
// insensitive to phase and sensitive to timbre and noise-floor changes.
double log_spectral_distance(
    const std::vector<float> & reference,
    const std::vector<float> & candidate,
    std::size_t channels) {
    constexpr std::size_t kFrame = 1024;
    constexpr std::size_t kHop = 256;
    const std::size_t frames = std::min(reference.size(), candidate.size()) / channels;
    if (frames < kFrame) return std::nan("");
    std::vector<double> window(kFrame);
    for (std::size_t index = 0; index < kFrame; ++index) {
        window[index] = 0.5 - 0.5 * std::cos(2.0 * 3.14159265358979323846 * index / kFrame);
    }
    std::vector<std::complex<double>> a(kFrame);
    std::vector<std::complex<double>> b(kFrame);
    double total = 0.0;
    std::size_t count = 0;
    for (std::size_t start = 0; start + kFrame <= frames; start += kHop) {
        for (std::size_t index = 0; index < kFrame; ++index) {
            double left = 0.0;
            double right = 0.0;
            for (std::size_t channel = 0; channel < channels; ++channel) {
                left += reference[(start + index) * channels + channel];
                right += candidate[(start + index) * channels + channel];
            }
            a[index] = window[index] * left / static_cast<double>(channels);
            b[index] = window[index] * right / static_cast<double>(channels);
        }
        fft(a);
        fft(b);
        double sum = 0.0;
        for (std::size_t bin = 1; bin <= kFrame / 2; ++bin) {
            const double db = 20.0 * std::log10((std::abs(a[bin]) + 1.0e-6) / (std::abs(b[bin]) + 1.0e-6));
            sum += db * db;
        }
        total += std::sqrt(sum / (kFrame / 2));
        ++count;
    }
    return total / static_cast<double>(count);
}

struct Settings {
    std::string reference;
    std::vector<std::string> candidates;
    std::string vae;
    std::string tokenizer;
    std::string device;
    std::string out;
    std::string style = kDefaultStyle;
    std::string lyrics = kDefaultLyrics;
    std::string abc = kDefaultAbc;
    yue2::SymbolicMode symbolic_mode = yue2::SymbolicMode::melody;
    std::uint64_t seed = 831001;
    std::uint32_t tokens = 64;
    std::uint32_t ode_steps = 32;
    std::uint32_t render_tokens = 0;
    std::string ar_fixture;
    std::string flow_fixture;
};

struct Inputs {
    yue2::SongRequest request;
    std::vector<std::int32_t> prefix;
    std::vector<float> noise;
    std::optional<ArFixture> ar;
    std::optional<FlowFixture> flow;
};

struct ModelRun {
    std::string path;
    std::optional<Divergence> fixture_prefill;
    std::optional<Divergence> fixture_decode;
    std::vector<std::int32_t> semantic_tokens;
    std::vector<float> window_logits;
    std::vector<float> latents;
    std::vector<float> audio;
    std::vector<std::int32_t> render_codes;
};

void print_divergence(const char * label, const Divergence & value) {
    std::printf("  %-28s kl=%.6f top1=%s top%zu_overlap=%.3f\n", label, value.kl,
        value.top1 ? "match" : "DIFFER", kTopK, value.overlap);
}

void print_difference(const char * label, const Difference & value) {
    std::printf("  %-28s rel_rms=%.6f cos=%.6f max_abs=%.6f\n", label, value.relative_rms,
        value.cosine, value.max_abs);
}

ModelRun evaluate(
    const std::string & path,
    const Settings & settings,
    const Inputs & inputs,
    const std::vector<std::int32_t> * reference_tokens,
    yue2::VaeDecoder & vae) {
    ModelRun run;
    run.path = path;
    std::printf("\n--- %s ---\n", path.c_str());
    {
        auto start = Clock::now();
        yue2::AutoregressiveOptions options;
        options.device = settings.device;
        yue2::AutoregressiveModel model(path, options);
        std::printf("  load %.2fs\n", seconds_since(start));

        if (inputs.ar) {
            const auto & fixture = *inputs.ar;
            auto session = model.create_session(fixture.ids.size() + 1);
            const auto prefill = session->append(fixture.ids);
            const auto decode = session->append({fixture.next_token});
            run.fixture_prefill = divergence(fixture.logits.data(), prefill.data(), prefill.size());
            run.fixture_decode = divergence(fixture.next_logits.data(), decode.data(), decode.size());
            print_divergence("official AR prefill", *run.fixture_prefill);
            print_divergence("official AR decode", *run.fixture_decode);
        }

        start = Clock::now();
        if (reference_tokens) {
            run.semantic_tokens = *reference_tokens;
        } else {
            auto sampling = yue2::GenerationDefaults{}.semantic;
            sampling.min_tokens = settings.tokens;
            sampling.max_tokens = settings.tokens;
            run.semantic_tokens = model.generate(
                inputs.prefix, sampling, yue2::AutoregressivePhase::semantic, settings.seed).tokens;
            if (run.semantic_tokens.size() != settings.tokens) {
                throw std::runtime_error("reference produced fewer semantic tokens than requested");
            }
        }
        const std::size_t steps = run.semantic_tokens.size();
        auto session = model.create_session(inputs.prefix.size() + steps);
        auto logits = session->append(inputs.prefix);
        run.window_logits.resize(steps * static_cast<std::size_t>(yue2::kCodecSize));
        for (std::size_t step = 0; step < steps; ++step) {
            std::copy_n(logits.begin() + yue2::kCodecOffset, yue2::kCodecSize,
                run.window_logits.begin() + static_cast<std::ptrdiff_t>(step * yue2::kCodecSize));
            if (step + 1 < steps) logits = session->append({run.semantic_tokens[step]});
        }
        std::printf("  AR %zu teacher-forced steps %.2fs\n", steps, seconds_since(start));

        start = Clock::now();
        yue2::FlowOptions flow;
        std::vector<std::int32_t> codes;
        if (inputs.flow) {
            flow.ode_steps = inputs.flow->steps;
            run.latents = model.synthesize_latents(
                inputs.flow->prefix, inputs.flow->codec, inputs.flow->noise, flow);
        } else {
            flow.ode_steps = settings.ode_steps;
            codes.reserve(steps);
            for (const auto token : run.semantic_tokens) codes.push_back(token - yue2::kCodecOffset);
            run.latents = model.synthesize_latents(inputs.prefix, codes, inputs.noise, flow);
        }
        std::printf("  flow %u steps %.2fs\n", flow.ode_steps, seconds_since(start));
    }

    run.audio = vae.decode(run.latents).interleaved_samples;
    if (inputs.flow) {
        print_difference("official flow latents", difference(inputs.flow->latents, run.latents));
        if (inputs.flow->audio.size() == run.audio.size()) {
            print_difference("official flow audio", difference(inputs.flow->audio, run.audio));
        }
    }

    const auto stem = fs::path(path).stem().string();
    if (!settings.out.empty()) {
        yue2::audio::write_wav_float(fs::path(settings.out) / (stem + "-flow.wav"), run.audio, 48000, 2);
    }
    if (settings.render_tokens > 0) {
        const auto start = Clock::now();
        yue2::GenerationPipelineOptions options;
        options.autoregressive.device = settings.device;
        options.generation.semantic.max_tokens = settings.render_tokens;
        options.generation.semantic.min_tokens = std::min(options.generation.semantic.min_tokens, settings.render_tokens);
        yue2::GenerationPipeline pipeline(path, settings.vae, settings.tokenizer, options);
        const auto song = pipeline.generate(inputs.request);
        run.render_codes = song.semantic_codec_ids;
        std::printf("  render %zu semantic frames %.2fs\n", run.render_codes.size(), seconds_since(start));
        if (!settings.out.empty()) {
            yue2::audio::write_wav_float(
                fs::path(settings.out) / (stem + "-render.wav"), song.audio.interleaved_samples,
                song.audio.sample_rate, song.audio.channels);
        }
    }
    return run;
}

void compare_runs(const ModelRun & reference, const ModelRun & candidate) {
    std::printf("\n=== %s vs %s ===\n", candidate.path.c_str(), reference.path.c_str());
    const std::size_t steps = reference.semantic_tokens.size();
    double kl_sum = 0.0;
    double kl_max = 0.0;
    double overlap_sum = 0.0;
    std::size_t top1 = 0;
    for (std::size_t step = 0; step < steps; ++step) {
        const auto offset = step * static_cast<std::size_t>(yue2::kCodecSize);
        const auto value = divergence(
            reference.window_logits.data() + offset, candidate.window_logits.data() + offset,
            yue2::kCodecSize);
        kl_sum += value.kl;
        kl_max = std::max(kl_max, value.kl);
        overlap_sum += value.overlap;
        top1 += value.top1 ? 1 : 0;
    }
    std::printf("  %-28s mean_kl=%.6f max_kl=%.6f top1=%.1f%% top%zu_overlap=%.3f\n",
        "AR codec logits", kl_sum / steps, kl_max, 100.0 * top1 / steps, kTopK, overlap_sum / steps);
    print_difference("flow latents", difference(reference.latents, candidate.latents));
    print_difference("flow audio", difference(reference.audio, candidate.audio));
    std::printf("  %-28s lsd=%.3f dB\n", "flow audio spectrum",
        log_spectral_distance(reference.audio, candidate.audio, 2));
    if (!reference.render_codes.empty() || !candidate.render_codes.empty()) {
        const std::size_t shared = std::min(reference.render_codes.size(), candidate.render_codes.size());
        std::size_t diverged = shared;
        for (std::size_t index = 0; index < shared; ++index) {
            if (reference.render_codes[index] != candidate.render_codes[index]) {
                diverged = index;
                break;
            }
        }
        std::printf("  %-28s frames=%zu/%zu first_divergence=%zu (expected early; listen instead)\n",
            "render", candidate.render_codes.size(), reference.render_codes.size(), diverged);
    }
}

void usage() {
    std::fprintf(stderr,
        "usage: yue2-quant-eval --ref REFERENCE.gguf --quant CANDIDATE.gguf [--quant ...]\n"
        "                       --vae VAE.gguf --tokenizer QWEN.TIKTOKEN [options]\n"
        "  --device NAME        cpu, cuda, ... (default: best accelerator)\n"
        "  --seed N             semantic sampling and flow noise seed (831001)\n"
        "  --tokens N           teacher-forced semantic steps and flow frames (64)\n"
        "  --ode-steps N        flow midpoint steps (32)\n"
        "  --style TEXT --lyrics TEXT --lyrics-file PATH --abc FILE --symbolic melody|full\n"
        "  --ar-fixture FILE    also score each model against an official Y2AR fixture\n"
        "  --flow-fixture FILE  run flow on an official Y2FL fixture and score against it\n"
        "  --render-tokens N    seeded end-to-end render of N semantic frames per model\n"
        "  --out DIR            write <model>-flow.wav and <model>-render.wav\n");
}

Settings parse(int argc, char ** argv) {
    Settings settings;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        const auto value = [&]() -> std::string {
            if (index + 1 >= argc) throw std::invalid_argument(argument + " requires a value");
            return argv[++index];
        };
        const auto number = [&]() -> std::uint64_t {
            const auto text = value();
            char * end = nullptr;
            const auto parsed = std::strtoull(text.c_str(), &end, 10);
            if (!end || *end != '\0' || text.empty()) throw std::invalid_argument(argument + " expects an integer");
            return parsed;
        };
        if (argument == "--ref") settings.reference = value();
        else if (argument == "--quant") settings.candidates.push_back(value());
        else if (argument == "--vae") settings.vae = value();
        else if (argument == "--tokenizer") settings.tokenizer = value();
        else if (argument == "--device") settings.device = value();
        else if (argument == "--out") settings.out = value();
        else if (argument == "--style") settings.style = value();
        else if (argument == "--lyrics") settings.lyrics = value();
        else if (argument == "--lyrics-file") settings.lyrics = read_text(value());
        else if (argument == "--abc") settings.abc = read_text(value());
        else if (argument == "--seed") settings.seed = number();
        else if (argument == "--tokens") settings.tokens = static_cast<std::uint32_t>(number());
        else if (argument == "--ode-steps") settings.ode_steps = static_cast<std::uint32_t>(number());
        else if (argument == "--render-tokens") settings.render_tokens = static_cast<std::uint32_t>(number());
        else if (argument == "--ar-fixture") settings.ar_fixture = value();
        else if (argument == "--flow-fixture") settings.flow_fixture = value();
        else if (argument == "--symbolic") {
            const auto mode = value();
            if (mode == "melody") settings.symbolic_mode = yue2::SymbolicMode::melody;
            else if (mode == "full") settings.symbolic_mode = yue2::SymbolicMode::full;
            else throw std::invalid_argument("--symbolic expects melody or full");
        } else if (argument == "-h" || argument == "--help") {
            usage();
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option " + argument);
        }
    }
    if (settings.reference.empty() || settings.candidates.empty() || settings.vae.empty() ||
        settings.tokenizer.empty()) {
        usage();
        throw std::invalid_argument("--ref, --quant, --vae and --tokenizer are required");
    }
    if (settings.tokens < 1 || settings.ode_steps < 1) {
        throw std::invalid_argument("--tokens and --ode-steps must be positive");
    }
    return settings;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const auto settings = parse(argc, argv);
        if (!settings.out.empty()) fs::create_directories(settings.out);

        Inputs inputs;
        inputs.request.style = settings.style;
        inputs.request.lyrics = settings.lyrics;
        inputs.request.symbolic_mode = settings.symbolic_mode;
        inputs.request.abc = settings.abc;
        inputs.request.seed = settings.seed;
        {
            const yue2::TextTokenizer tokenizer(settings.tokenizer);
            inputs.prefix = yue2::make_positive_prefix(
                inputs.request, tokenizer, tokenizer.encode(settings.abc));
        }
        std::mt19937_64 generator(settings.seed);
        std::normal_distribution<float> normal(0.0F, 1.0F);
        inputs.noise.resize(static_cast<std::size_t>(settings.tokens) * 64);
        for (auto & value : inputs.noise) value = normal(generator);
        if (!settings.ar_fixture.empty()) inputs.ar = read_ar_fixture(settings.ar_fixture);
        if (!settings.flow_fixture.empty()) inputs.flow = read_flow_fixture(settings.flow_fixture);
        std::printf("prompt %zu tokens, %u semantic steps, seed %llu\n", inputs.prefix.size(),
            settings.tokens, static_cast<unsigned long long>(settings.seed));

        yue2::VaeRuntimeOptions vae_options;
        vae_options.device = settings.device;
        yue2::VaeDecoder vae(settings.vae, vae_options);

        const auto reference = evaluate(settings.reference, settings, inputs, nullptr, vae);
        for (const auto & candidate_path : settings.candidates) {
            const auto candidate = evaluate(
                candidate_path, settings, inputs, &reference.semantic_tokens, vae);
            compare_runs(reference, candidate);
        }
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "yue2-quant-eval: %s\n", error.what());
        return 1;
    }
}
