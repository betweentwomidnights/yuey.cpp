#include "gguf_model.h"
#include "training_sampler.h"
#include "training_noise.h"
#include "training_sample_store.h"
#include "training_cache_bytes.h"
#include "server/json.h"
#include "yue2/nar_training.h"
#include "yue2/tokenizer.h"
#include <array>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream>

namespace {
void training_event(const char *stage, size_t current, size_t total, uint64_t step = 0) {
    std::cout << "TRAIN_EVENT {\"stage\":\"" << stage << "\",\"current\":" << current
              << ",\"total\":" << total << ",\"step\":" << step << "}" << std::endl;
}
template <class T> std::vector<T> read(const std::string &path) {
    const auto size = std::filesystem::file_size(path);
    if (!size || size % sizeof(T))
        throw std::runtime_error("invalid raw input: " + path);
    std::vector<T> data(size / sizeof(T));
    std::ifstream file(path, std::ios::binary);
    if (!file.read(reinterpret_cast<char *>(data.data()), size))
        throw std::runtime_error("short input: " + path);
    return data;
}
std::vector<float> noise(std::mt19937 &rng, size_t count) {
    std::normal_distribution<float> normal;
    std::vector<float> result(count);
    for (auto &value : result)
        value = normal(rng);
    return result;
}
std::string fixed_identity(const std::vector<yue2::LoraAdapterSpec> &fixed) {
    std::string result;
    for (const auto &spec : fixed)
        result += std::filesystem::absolute(spec.path).lexically_normal().string() + "=" +
                  std::to_string(spec.strength) + "\n";
    return result;
}
yue2::NarTrainingSample load(const std::string &path, const std::string &model,
                             const std::vector<yue2::LoraAdapterSpec> &fixed,
                             const std::vector<yue2::LoraAdapterSpec> &conditioning) {
    auto cache = yue2::detail::load_gguf_raw(path.c_str(), "cpu", 1);
    if (cache.string("general.architecture") != "yue2_nar_sample")
        throw std::runtime_error("not a NAR sample cache");
    if (!std::filesystem::equivalent(cache.string("yue2.training.model"), model) ||
        cache.string("yue2.training.fixed_adapters") != fixed_identity(fixed) ||
        cache.optional_string("yue2.training.conditioning_adapters").value_or("") !=
            fixed_identity(conditioning))
        throw std::runtime_error(
            "sample was prepared with different base weights or fixed adapters");
    yue2::NarTrainingSample sample;
    sample.conditioning.token_count = cache.u32("yue2.training.ar_tokens");
    auto values = [&](const char *name) {
        auto *t = cache.get(name);
        std::vector<float> v(ggml_nelements(t));
        if(t->type==GGML_TYPE_F32)ggml_backend_tensor_get(t,v.data(),0,v.size()*4);
        else if(t->type==GGML_TYPE_F16 && (std::string(name)=="keys"||std::string(name)=="values")) {
            std::vector<ggml_fp16_t> half(v.size());
            ggml_backend_tensor_get(t,half.data(),0,half.size()*sizeof(half[0]));
            ggml_fp16_to_fp32_row(half.data(),v.data(),v.size());
        } else throw std::runtime_error("clean targets must be F32; conditioning must be F32 or F16");
        return v;
    };
    sample.clean_latents = values("clean");
    sample.conditioning.keys = values("keys");
    sample.conditioning.values = values("values");
    return sample;
}
void prepare(const std::map<std::string, std::string> &args,
             const std::vector<yue2::LoraAdapterSpec> &fixed,
             const std::vector<yue2::LoraAdapterSpec> &conditioning_adapters,
             std::unique_ptr<yue2::AutoregressiveModel> &resident) {
    auto required = [&](const char *name) {
        auto it = args.find(name);
        if (it == args.end())
            throw std::runtime_error(std::string("missing ") + name);
        return it->second;
    };
    const auto output = required("--cache-out");
    if (std::filesystem::exists(output))
        throw std::runtime_error("cache output exists");
    yue2::SongRequest request;
    request.style = args.count("--prompt") ? args.at("--prompt") : "instrumental electronic music";
    request.symbolic_mode = yue2::SymbolicMode::off;
    std::vector<int32_t> prefix;
    if (args.count("--prefix")) {
        if (args.count("--abc") || args.count("--prompt"))
            throw std::runtime_error("--prefix cannot be combined with --abc/--prompt");
        prefix = read<int32_t>(args.at("--prefix"));
        if (prefix.back() != yue2::kMusicStartToken)
            throw std::runtime_error("exact prefix must end in MUSIC_START");
        request.style = "exact token prefix";
    } else {
        yue2::TextTokenizer tokenizer(required("--tokenizer"));
        std::vector<int32_t> abc;
        if (args.count("--abc")) {
            const auto bytes = read<char>(args.at("--abc"));
            request.abc = std::string(bytes.begin(), bytes.end());
            request.symbolic_mode = yue2::SymbolicMode::full;
            abc = tokenizer.encode(*request.abc);
        }
        prefix = yue2::make_positive_prefix(request, tokenizer, abc);
    }
    yue2::AutoregressiveOptions runtime;
    runtime.device = args.count("--device") ? args.at("--device") : "cpu";
    runtime.threads = args.count("--threads") ? std::stoi(args.at("--threads")) : 4;
    runtime.flash_attention = args.count("--cache-flash-attn")!=0;
    runtime.conditioning_only=args.count("--conditioning-only")!=0;
    if(args.count("--conditioning-batch")) {
        const auto batch=std::stoull(args.at("--conditioning-batch"));
        if(batch>24576)throw std::runtime_error("conditioning-batch must be in [0,24576]");
        runtime.conditioning_batch_tokens=static_cast<uint32_t>(batch);
    }
    if(runtime.conditioning_only && (args.count("--mint-frames")||args.count("--generate-latents")||args.count("--reference")))
        throw std::runtime_error("conditioning-only preparation requires supplied tokens and latents");
    const auto kv_type=args.count("--cache-kv-type")?args.at("--cache-kv-type"):"f32";
    if(kv_type!="f32"&&kv_type!="f16")throw std::runtime_error("cache-kv-type must be f32 or f16");
    runtime.lora_adapters = fixed;
    runtime.lora_adapters.insert(runtime.lora_adapters.end(), conditioning_adapters.begin(),
                                 conditioning_adapters.end());
    if ((args.count("--mint-frames") || args.count("--generate-latents") ||
         args.count("--reference")) &&
        runtime.device != "cuda")
        throw std::runtime_error("minting/flow diagnostics currently require --device cuda; CPU "
                                 "supports frozen conditioning caches from supplied latents");
    if (!resident) resident = std::make_unique<yue2::AutoregressiveModel>(required("--model"), runtime);
    auto &model = *resident;
    std::vector<int32_t> codes;
    std::vector<float> clean;
    if (args.count("--mint-frames")) {
        const int count = std::stoi(args.at("--mint-frames"));
        if (count < 2 || count > 1024)
            throw std::runtime_error("mint-frames must be in [2,1024]");
        yue2::GenerationSampling sampling;
        sampling.min_tokens = count;
        sampling.max_tokens = count;
        auto generated =
            model.generate(prefix, sampling, yue2::AutoregressivePhase::semantic, 831001);
        for (auto token : generated.tokens) {
            if (token < yue2::kCodecOffset || token >= yue2::kCodecOffset + yue2::kCodecSize)
                throw std::runtime_error("unexpected minted token");
            codes.push_back(token - yue2::kCodecOffset);
        }
        clean = model.synthesize_latents(prefix, codes, 831001);
    } else {
        codes = read<int32_t>(required("--tokens"));
        clean = args.count("--generate-latents") ? model.synthesize_latents(prefix, codes, 831001)
                                                 : read<float>(required("--latents"));
    }
    if (clean.size() != codes.size() * 64)
        throw std::runtime_error("tokens/latents are not frame aligned");
    const auto conditioning = model.freeze_nar_conditioning(prefix, codes);
    auto *ctx = ggml_init({ggml_tensor_overhead() * 16, nullptr, true});
    auto *file = gguf_init_empty();
    if (!ctx || !file)
        throw std::runtime_error("cache context allocation failed");
    gguf_set_val_str(file, "general.architecture", "yue2_nar_sample");
    gguf_set_val_u32(file, "yue2.training.ar_tokens", conditioning.token_count);
    gguf_set_val_str(file, "yue2.training.model",
                     std::filesystem::absolute(required("--model")).string().c_str());
    gguf_set_val_str(file, "yue2.training.fixed_adapters", fixed_identity(fixed).c_str());
    gguf_set_val_str(file, "yue2.training.conditioning_adapters",
                     fixed_identity(conditioning_adapters).c_str());
    gguf_set_val_str(file, "yue2.training.prompt", request.style.c_str());
    gguf_set_val_str(file,"yue2.training.cache.kv_type",kv_type.c_str());
    if (args.count("--cache-identity"))
        gguf_set_val_str(file, "yue2.training.cache.identity", args.at("--cache-identity").c_str());
    gguf_set_val_bool(file,"yue2.training.cache.conditioning_only",runtime.conditioning_only);
    gguf_set_val_u32(file,"yue2.training.cache.prefill_batch",runtime.conditioning_batch_tokens);
    gguf_set_val_bool(file,"yue2.training.cache.flash_attention",runtime.flash_attention);
    auto add = [&](const char *name, ggml_type type, const void *data, int64_t a, int64_t b = 1,
                   int64_t c = 1, int64_t d = 1) {
        auto *t = ggml_new_tensor_4d(ctx, type, a, b, c, d);
        ggml_set_name(t, name);
        t->data = const_cast<void *>(data);
        gguf_add_tensor(file, t);
    };
    add("clean", GGML_TYPE_F32, clean.data(), 64, codes.size());
    std::vector<ggml_fp16_t> half_keys,half_values;
    if(kv_type=="f16") {
        auto exact_half=[](const std::vector<float> &source) {
            std::vector<ggml_fp16_t> half(source.size());
            ggml_fp32_to_fp16_row(source.data(),half.data(),source.size());
            for(size_t i=0;i<source.size();++i) {
                const float roundtrip=ggml_fp16_to_fp32(half[i]);
                if(!std::isfinite(source[i])||std::memcmp(&roundtrip,&source[i],sizeof(float)))
                    throw std::runtime_error("conditioning cannot be represented losslessly in F16");
            }
            return half;
        };
        half_keys=exact_half(conditioning.keys);half_values=exact_half(conditioning.values);
        add("keys",GGML_TYPE_F16,half_keys.data(),128,conditioning.token_count,8,28);
        add("values",GGML_TYPE_F16,half_values.data(),128,conditioning.token_count,8,28);
    } else {
        add("keys", GGML_TYPE_F32, conditioning.keys.data(), 128, conditioning.token_count, 8, 28);
        add("values", GGML_TYPE_F32, conditioning.values.data(), 128, conditioning.token_count, 8, 28);
    }
    add("prefix", GGML_TYPE_I32, prefix.data(), prefix.size());
    add("codes", GGML_TYPE_I32, codes.data(), codes.size());
    std::vector<float> reference;
    if (args.count("--reference")) {
        std::mt19937 rng(831001);
        auto e = noise(rng, clean.size());
        std::vector<float> mixed(clean.size());
        for (size_t i = 0; i < clean.size(); ++i)
            mixed[i] = 0.5F * (clean[i] + e[i]);
        reference = model.flow_velocity(prefix, codes, mixed, 0.5F);
        add("reference_velocity", GGML_TYPE_F32, reference.data(), 64, codes.size());
    }
    const auto parent = std::filesystem::path(output).parent_path();
    if (!parent.empty())
        std::filesystem::create_directories(parent);
    const auto tmp = output + ".partial";
    if (!gguf_write_to_file(file, tmp.c_str(), false))
        throw std::runtime_error("cannot write sample cache");
    gguf_free(file);
    ggml_free(ctx);
    std::filesystem::rename(tmp, output);
    std::cout << "cached " << codes.size() << " frames and " << conditioning.token_count
              << " AR tokens to " << output << '\n';
}
} // namespace

int main(int argc, char **argv) {
    try {
        std::map<std::string, std::string> args;
        std::vector<std::string> samples, validations, retentions;
        std::vector<yue2::LoraAdapterSpec> fixed, conditioning;
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            if (key == "--help") {
                std::cout << "Experimental native YuE2 NAR LoRA trainer\n"
                             "Prepare: --prepare --model BASE.gguf --tokenizer qwen.tiktoken "
                             "--prompt "
                             "TEXT\n"
                             "  (--tokens codes.i32 --latents means.f32 | --mint-frames N) "
                             "--cache-out "
                             "sample.gguf\n"
                             "  --tokens codes.i32 --generate-latents renders known tokens "
                             "as the "
                             "target. Minting/flow diagnostics require CUDA.\n"
                             "  [--abc score.abc] [--reference] [--device cpu|cuda] "
                             "[--fixed-lora "
                             "COMPANION]\n"
                             "  --prefix prefix.i32 uses exact conditioning instead of "
                             "tokenizer/prompt/ABC.\n"
                             "  --conditioning-lora AR.gguf (repeat) freezes AR adapters "
                             "into caches.\n"
                             "  [--conditioning-only] omit unused NAR/head weights; supplied tokens/latents only.\n"
                             "  [--cache-kv-type f32|f16] F16 preserves original cache values losslessly.\n"
                             "  [--conditioning-batch N --cache-flash-attn] bounded causal prefill and fused GPU attention.\n"
                             "  --prepare-batch PLAN.json reuses one AR-only model for an array of\n"
                             "  {prefix,tokens,latents,cache-out,cache-identity}; --stop-file stops between items.\n"
                             "Train: --model BASE.gguf --sample sample.gguf (repeat) --out "
                             "DIR "
                             "--identity HASH\n"
                             "  [--validation sample.gguf (repeat)] [--fixed-lora COMPANION "
                             "(repeat)]\n"
                             "  [--retention sample.gguf (repeat)] evaluation only, separate "
                             "from heldout selection.\n"
                             "  [--device cuda|cpu] [--rank 8] [--steps 100] [--lr 0.0001] "
                             "[--seed "
                             "831001]\n"
                             "  [--save-every 10] [--eval-every 10] [--resume state.gguf] "
                             "[--eval-scope full|heldout] [--probe-ar-slots N (probe only)] "
                             "[--validate-only (cache audit, no model load)] "
                             "[--cache-fingerprint decoded-v1|bytes-v2] "
                             "[--fixed-noise]\n"
                             "  [--f32-projections] [--conditioning-lora AR.gguf (repeat)]\n"
                             "  [--dora] full-gradient row DoRA; requires F32 projections "
                             "and no fixed NAR adapter.\n"
                             "  [--shuffle-samples] visits each sample once per shuffled "
                             "epoch.\n"
                             "  [--dataset prepared.json --group-samples] recording-balanced "
                             "windows; manifest replaces explicit sample lists.\n"
                             "  [--stream-samples] loads caches on demand to bound host RAM.\n"
                             "  [--stop-file PATH] checkpoints after the current update.\n"
                             "  [--noise-schedule uniform|sigmoid] default uniform; separate "
                             "time RNG preserves paired example/Gaussian streams.\n"
                             "  [--probe-only --diagnostics probe.gguf]\n"
                             "Adapters are deltas: retain the same fixed companion during "
                             "inference.\n";
                return 0;
            }
            if (key == "--prepare" || key == "--reference" || key == "--probe-only" || key == "--validate-only" ||
                key == "--fixed-noise" || key == "--generate-latents" ||
                key == "--f32-projections" || key == "--shuffle-samples" || key == "--dora" ||
                key == "--group-samples" || key == "--stream-samples" || key == "--conditioning-only" || key == "--cache-flash-attn") {
                args[key] = "1";
                continue;
            }
            if (i + 1 == argc)
                throw std::runtime_error("missing value for " + key);
            const std::string value = argv[++i];
            if (key == "--sample")
                samples.push_back(value);
            else if (key == "--validation")
                validations.push_back(value);
            else if (key == "--retention")
                retentions.push_back(value);
            else if (key == "--fixed-lora")
                fixed.push_back({value, 1});
            else if (key == "--conditioning-lora")
                conditioning.push_back({value, 1});
            else if (key == "--model" || key == "--tokenizer" || key == "--tokens" ||
                     key == "--latents" || key == "--prompt" || key == "--cache-out" ||
                     key == "--mint-frames" || key == "--device" || key == "--threads" ||
                     key == "--out" || key == "--identity" || key == "--rank" || key == "--steps" ||
                     key == "--lr" || key == "--seed" || key == "--save-every" ||
                     key == "--eval-every" || key == "--resume" || key == "--diagnostics" ||
                     key == "--abc" || key == "--prefix" || key == "--noise-schedule" ||
                     key == "--dataset" || key == "--stop-file" || key == "--cache-kv-type" || key == "--conditioning-batch" ||
                     key == "--prepare-batch" || key == "--cache-identity" ||
                     key == "--eval-scope" || key == "--probe-ar-slots" || key == "--cache-fingerprint") {
                if (!args.emplace(key, value).second)
                    throw std::runtime_error("duplicate argument " + key);
            } else
                throw std::runtime_error("unknown option " + key);
        }
        if (args.count("--prepare-batch")) {
            namespace json = yue2::server::json;
            if (!args.count("--conditioning-only"))
                throw std::runtime_error("batch preparation requires conditioning-only mode");
            for (auto key : {"--prefix", "--tokens", "--latents", "--cache-out", "--cache-identity",
                             "--mint-frames", "--generate-latents", "--reference", "--abc", "--prompt"})
                if (args.count(key)) throw std::runtime_error("batch inputs must be supplied in the plan");
            const auto bytes = read<char>(args.at("--prepare-batch"));
            const auto plan = json::parse(std::string(bytes.begin(), bytes.end()));
            if (plan.type != json::Type::array || plan.array.empty())
                throw std::runtime_error("batch plan must be a nonempty array");
            std::unique_ptr<yue2::AutoregressiveModel> resident;
            for (size_t i = 0; i < plan.array.size(); ++i) {
                if (args.count("--stop-file") && std::filesystem::exists(args.at("--stop-file"))) break;
                auto item_args = args;
                for (auto key : {"prefix", "tokens", "latents", "cache-out", "cache-identity"}) {
                    const auto value = json::string(plan.array[i], key);
                    if (value.empty()) throw std::runtime_error("missing batch input: " + std::string(key));
                    item_args["--" + std::string(key)] = value;
                }
                const auto started = std::chrono::steady_clock::now();
                prepare(item_args, fixed, conditioning, resident);
                std::cout << "CACHE_EVENT {\"index\":" << i << ",\"path\":"
                          << json::quote(item_args.at("--cache-out")) << ",\"seconds\":"
                          << std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()
                          << "}" << std::endl;
            }
            return 0;
        }
        if (args.count("--prepare")) {
            std::unique_ptr<yue2::AutoregressiveModel> resident;
            prepare(args, fixed, conditioning, resident);
            return 0;
        }
        std::vector<size_t> groups;
        if (args.count("--dataset")) {
            if (!samples.empty() || !validations.empty() || !retentions.empty())
                throw std::runtime_error("dataset and explicit samples are mutually exclusive");
            namespace json = yue2::server::json;
            const std::filesystem::path manifest(args.at("--dataset"));
            const auto bytes = read<char>(manifest.string());
            const auto document = json::parse(std::string(bytes.begin(), bytes.end()));
            if (json::string(document, "format") != "yue2-nar-prepared-dataset-v1" ||
                !json::boolean(document, "completed", false))
                throw std::runtime_error("incomplete or unsupported NAR dataset");
            const auto *rows = document.find("rows");
            if (!rows || rows->type != json::Type::array)
                throw std::runtime_error("dataset rows are required");
            std::map<std::string, size_t> group_ids;
            for (const auto &row : rows->array) {
                if (!json::boolean(row, "eligible", true)) continue;
                std::filesystem::path path(json::string(row, "cache"));
                if (path.empty()) throw std::runtime_error("dataset cache path is missing");
                if (!path.is_absolute()) path = manifest.parent_path() / path;
                const auto split = json::string(row, "split");
                if (split == "train") {
                    if (args.count("--group-samples")) {
                        const auto group = json::string(row, "group", json::string(row, "source_sha256"));
                        if (group.empty()) throw std::runtime_error("sample group is missing");
                        groups.push_back(group_ids.emplace(group, group_ids.size()).first->second);
                    }
                    samples.push_back(path.string());
                } else if (split == "validation") validations.push_back(path.string());
                else if (split == "retention") retentions.push_back(path.string());
                else throw std::runtime_error("unknown dataset split");
            }
        }
        if (args.count("--group-samples") &&
            (groups.empty() || !args.count("--shuffle-samples")))
            throw std::runtime_error("group sampling requires a dataset and --shuffle-samples");
        if (!args.count("--model") || !args.count("--out") || !args.count("--identity") ||
            samples.empty())
            throw std::runtime_error("training requires --model, --sample, --out and "
                                     "--identity (see --help)");
        yue2::NarTrainingOptions options;
        options.device = args.count("--device") ? args.at("--device") : "cuda";
        options.fixed_adapters = fixed;
        options.conditioning_adapters = conditioning;
        options.f32_projections = args.count("--f32-projections") != 0;
        options.dora = args.count("--dora") != 0;
        options.run_identity = args.at("--identity");
        if (args.count("--rank"))
            options.rank = std::stoi(args.at("--rank"));
        if (args.count("--lr"))
            options.learning_rate = std::stof(args.at("--lr"));
        if (args.count("--seed"))
            options.seed = static_cast<uint32_t>(std::stoul(args.at("--seed")));
        if (args.count("--threads"))
            options.threads = std::stoi(args.at("--threads"));
        const int steps = args.count("--steps") ? std::stoi(args.at("--steps")) : 100;
        const int every = args.count("--save-every") ? std::stoi(args.at("--save-every")) : 10;
        const int eval_every = args.count("--eval-every") ? std::stoi(args.at("--eval-every")) : 10;
        if (steps < 1 || every < 1 || eval_every < 1 || options.run_identity.empty())
            throw std::runtime_error("steps/intervals/identity must be positive/nonempty");
        const auto eval_scope = args.count("--eval-scope") ? args.at("--eval-scope") : "full";
        if (eval_scope != "full" && eval_scope != "heldout")
            throw std::runtime_error("--eval-scope must be full or heldout");
        if (eval_scope == "heldout" && validations.empty())
            throw std::runtime_error("heldout evaluation requires validation samples");
        if (args.count("--probe-ar-slots") && !args.count("--probe-only"))
            throw std::runtime_error("--probe-ar-slots is only valid for a preflight probe");
        const auto fingerprint = args.count("--cache-fingerprint") ? args.at("--cache-fingerprint") : "decoded-v1";
        if (fingerprint!="decoded-v1" && fingerprint!="bytes-v2")
            throw std::runtime_error("unsupported cache fingerprint policy");
        if (fingerprint=="bytes-v2" && !args.count("--dataset"))
            throw std::runtime_error("byte cache fingerprints require an explicit dataset");
        uint32_t ar = 0;
        size_t frames = 0;
        uint64_t content_hash = 14695981039346656037ULL;
        auto mix = [&](uint64_t value) {
            for (int byte = 0; byte < 8; ++byte) {
                content_hash ^= (value >> (8 * byte)) & 255;
                content_hash *= 1099511628211ULL;
            }
        };
        auto inspect = [&](const yue2::NarTrainingSample &s) {
            if (!frames) frames = s.clean_latents.size() / 64;
            if (!frames || s.clean_latents.size() != frames * 64 || !s.conditioning.token_count ||
                s.conditioning.keys.size() != size_t(s.conditioning.token_count) * 128 * 8 * 28 ||
                s.conditioning.values.size() != s.conditioning.keys.size())
                throw std::runtime_error("invalid or differently shaped training cache");
            ar = std::max(ar, s.conditioning.token_count);
            if (args.count("--dataset")) {
                mix(s.conditioning.token_count);
                for (const auto *v : {&s.clean_latents, &s.conditioning.keys, &s.conditioning.values}) {
                    mix(v->size());
                    for (float value : *v) {
                        uint32_t bits; std::memcpy(&bits, &value, sizeof(bits)); mix(bits);
                    }
                }
            }
        };
        auto loader = [&](const std::string &path) { return load(path, args.at("--model"), fixed, conditioning); };
        using Store = yue2::detail::TrainingSampleStore<yue2::NarTrainingSample>;
        const bool streaming = args.count("--stream-samples") != 0;
        const auto validation_start = std::chrono::steady_clock::now();
        size_t validated = 0;
        const auto cache_count = samples.size() + validations.size() + retentions.size();
        training_event("cache-validation", 0, cache_count);
        auto validation_progress = [&](size_t, size_t) {
            training_event("cache-validation", ++validated, cache_count);
        };
        Store::PathInspector inspect_path;
        if (fingerprint=="bytes-v2") inspect_path=[&](const std::string &path) {
            const auto header=yue2::detail::inspect_nar_cache_header(path,args.at("--model"),
                fixed_identity(fixed),fixed_identity(conditioning));
            if (!frames) frames=header.frames;
            if (frames!=header.frames) throw std::runtime_error("differently shaped training caches");
            ar=std::max(ar,header.tokens);
            mix(header.bytes);mix(yue2::detail::training_file_fingerprint(path));
        };
        mix(samples.size()); Store train(samples, streaming, loader, inspect, validation_progress, inspect_path);
        mix(validations.size()); Store validation(validations, streaming, loader, inspect, validation_progress, inspect_path);
        mix(retentions.size()); Store retention(retentions, streaming, loader, inspect, validation_progress, inspect_path);
        if (args.count("--probe-ar-slots")) {
            const auto capacity = std::stoul(args.at("--probe-ar-slots"));
            if (capacity < ar || capacity > 24576)
                throw std::runtime_error("probe AR capacity is smaller than a sample or exceeds 24576");
            ar = static_cast<uint32_t>(capacity);
        }
        if (args.count("--dataset")) {
            mix(groups.size()); for (auto group : groups) mix(group);
            std::ostringstream signature; signature << std::hex << content_hash;
            options.run_identity += (fingerprint=="bytes-v2" ? ":nar-cache-bytes-fnv1a64-v2:" : ":nar-cache-fnv1a64-v1:") + signature.str();
        }
        if (args.count("--validate-only")) {
            std::cout << "CACHE_AUDIT {\"files\":" << cache_count << ",\"frames\":" << frames
                      << ",\"fingerprint\":" << yue2::server::json::quote(fingerprint)
                      << ",\"ar_slots\":" << ar << ",\"seconds\":"
                      << std::chrono::duration<double>(std::chrono::steady_clock::now()-validation_start).count()
                      << ",\"identity\":" << yue2::server::json::quote(options.run_identity) << "}" << std::endl;
            return 0;
        }
        const auto directory = std::filesystem::path(args.at("--out"));
        if (std::filesystem::exists(directory) && !args.count("--resume"))
            throw std::runtime_error(
                "output directory exists; choose a new run or explicit resume");
        std::filesystem::create_directories(directory);
        training_event("model-loading", 0, 1);
        yue2::NarTrainer trainer(args.at("--model"), static_cast<uint32_t>(frames), ar, options);
        training_event("model-loading", 1, 1);
        std::mt19937 rng(options.seed);
        std::mt19937 time_rng(options.seed ^ 0x6e617274U);
        const auto schedule = yue2::detail::noise_schedule(
            args.count("--noise-schedule") ? args.at("--noise-schedule") : "uniform");
        const bool sigmoid = schedule == yue2::detail::NoiseSchedule::Sigmoid;
        if (sigmoid && args.count("--fixed-noise"))
            throw std::runtime_error("--fixed-noise cannot be combined with sigmoid sampling");
        const bool shuffle = args.count("--shuffle-samples") != 0;
        if (shuffle && args.count("--fixed-noise"))
            throw std::runtime_error("--shuffle-samples cannot be combined with --fixed-noise");
        yue2::detail::TrainingSampler sampler(train.size(), true, options.seed, groups);
        if (args.count("--resume")) {
            auto saved = trainer.load_state(args.at("--resume"));
            const std::string time_marker = "nar-sigmoid-v1\n";
            const bool saved_sigmoid = saved.rfind(time_marker, 0) == 0;
            if (saved_sigmoid != sigmoid)
                throw std::runtime_error("checkpoint noise schedule differs");
            if (saved_sigmoid) saved.erase(0, time_marker.size());
            const auto marker = std::string("nar-shuffle-v1\n");
            const bool saved_shuffle = saved.rfind(marker, 0) == 0;
            if (saved_shuffle != shuffle)
                throw std::runtime_error("checkpoint sample-selection mode differs");
            std::istringstream state(saved_shuffle ? saved.substr(marker.size()) : saved);
            if (!(state >> rng))
                throw std::runtime_error("invalid checkpoint RNG state");
            if (sigmoid && !(state >> time_rng))
                throw std::runtime_error("invalid checkpoint time RNG state");
            if (saved_shuffle)
                sampler.load(std::string(std::istreambuf_iterator<char>(state), {}));
        }
        auto evaluate = [&](const yue2::NarTrainingSample &s) {
            std::mt19937 heldout(42);
            const auto e = noise(heldout, frames * 64);
            return std::array<float, 3>{trainer.evaluate(s, e, .2F),
                                        trainer.evaluate(s, e, .5F),
                                        trainer.evaluate(s, e, .8F)};
        };
        std::ofstream history(directory / "history.jsonl", std::ios::app);
        auto evaluate_all = [&](uint64_t step) {
            size_t completed = 0;
            const auto total = (eval_scope == "full" ? train.size() : 0) + validation.size() + retention.size();
            training_event("evaluation", 0, total, step);
            double training = 0, heldout = 0;
            std::array<double, 3> heldout_by_sigma{};
            std::vector<float> heldout_values;
            for (size_t i = 0; eval_scope == "full" && i < train.size(); ++i) {
                const auto values = evaluate(train.get(i));
                training += (values[0] + values[1] + values[2]) / 3;
                training_event("evaluation", ++completed, total, step);
            }
            for (size_t i = 0; i < validation.size(); ++i) {
                const auto values = evaluate(validation.get(i));
                const auto value = (values[0] + values[1] + values[2]) / 3;
                for (size_t i = 0; i < 3; ++i) heldout_by_sigma[i] += values[i];
                heldout += value;
                heldout_values.push_back(value);
                training_event("evaluation", ++completed, total, step);
            }
            std::ostringstream line;
            line << "{\"step\":" << step << ",\"evaluation_scope\":\"" << eval_scope << "\"";
            if (eval_scope == "full") line << ",\"train_eval\":" << training / train.size();
            if (!validation.empty()) {
                line << ",\"heldout_eval\":" << heldout / validation.size();
                line << ",\"heldout_per_sample\":[";
                for (size_t i = 0; i < heldout_values.size(); ++i) {
                    if (i)
                        line << ',';
                    line << heldout_values[i];
                }
                line << ']';
                line << ",\"heldout_per_sigma\":[";
                for (size_t i = 0; i < 3; ++i) {
                    if (i) line << ',';
                    line << heldout_by_sigma[i] / validation.size();
                }
                line << "],\"evaluation_sigmas\":[0.2,0.5,0.8]";
            }
            if (!retention.empty()) {
                double retention_sum = 0;
                std::array<double, 3> by_sigma{};
                std::vector<float> per_sample;
                for (size_t i = 0; i < retention.size(); ++i) {
                    const auto values = evaluate(retention.get(i));
                    const float mean = (values[0] + values[1] + values[2]) / 3;
                    retention_sum += mean;
                    per_sample.push_back(mean);
                    for (size_t i = 0; i < 3; ++i) by_sigma[i] += values[i];
                    training_event("evaluation", ++completed, total, step);
                }
                line << ",\"retention_eval\":" << retention_sum / retention.size();
                line << ",\"retention_per_sample\":[";
                for (size_t i = 0; i < per_sample.size(); ++i) {
                    if (i) line << ',';
                    line << per_sample[i];
                }
                line << "],\"retention_per_sigma\":[";
                for (size_t i = 0; i < 3; ++i) {
                    if (i) line << ',';
                    line << by_sigma[i] / retention.size();
                }
                line << ']';
            }
            line << "}";
            history << line.str() << '\n';
            history.flush();
            std::cout << line.str() << std::endl;
        };
        if (args.count("--probe-only")) {
            auto e = noise(rng, frames * 64);
            const auto r = trainer.step(train.get(0), e, .5F, false);
            const auto path = args.count("--diagnostics") ? args.at("--diagnostics")
                                                          : (directory / "probe.gguf").string();
            trainer.save_diagnostics(path);
            trainer.save_adapter((directory / "initial.gguf").string());
            std::cout << "probe loss=" << r.loss << " grad_norm=" << r.gradient_norm << std::endl;
            return 0;
        }
        if (!args.count("--resume")) {
            trainer.save_adapter((directory / "step-0.gguf").string());
            evaluate_all(0);
        }
        for (int i = 0; i < steps; ++i) {
            size_t at = 0;
            std::vector<float> e;
            float sigma = .5F;
            if (args.count("--fixed-noise")) {
                std::mt19937 fixed_rng(options.seed);
                e = noise(fixed_rng, frames * 64);
            } else {
                at = shuffle ? sampler.next()
                             : std::uniform_int_distribution<size_t>(0, train.size() - 1)(rng);
                e = noise(rng, frames * 64);
                sigma = yue2::detail::training_noise_time(schedule, rng, time_rng);
            }
            const auto load_start=std::chrono::steady_clock::now();
            const auto &sample=train.get(at);
            const auto load_done=std::chrono::steady_clock::now();
            auto result = trainer.step(sample, e, sigma);
            const auto update_done=std::chrono::steady_clock::now();
            std::ostringstream line;
            line << "{\"step\":" << result.optimizer_step << ",\"loss\":" << result.loss
                 << ",\"grad_norm\":" << result.gradient_norm << ",\"sample_index\":" << at
                 << ",\"sigma\":" << sigma
                 << ",\"cache_load_seconds\":" << std::chrono::duration<double>(load_done-load_start).count()
                 << ",\"update_seconds\":" << std::chrono::duration<double>(update_done-load_done).count()
                 << ",\"seconds\":" << std::chrono::duration<double>(update_done-load_start).count();
            if (!groups.empty()) line << ",\"sample_group\":" << groups[at];
            line << "}";
            history << line.str() << '\n';
            history.flush();
            std::cout << line.str() << std::endl;
            const bool requested_stop = args.count("--stop-file") && std::filesystem::exists(args.at("--stop-file"));
            if (result.optimizer_step % eval_every == 0 || i == steps - 1)
                evaluate_all(result.optimizer_step);
            if (result.optimizer_step % every == 0 || i == steps - 1 || requested_stop) {
                training_event("checkpoint-save", 0, 1, result.optimizer_step);
                const auto stem = "step-" + std::to_string(result.optimizer_step);
                trainer.save_adapter((directory / (stem + ".gguf")).string());
                std::ostringstream state;
                if (sigmoid)
                    state << "nar-sigmoid-v1\n";
                if (shuffle)
                    state << "nar-shuffle-v1\n";
                state << rng;
                if (sigmoid)
                    state << '\n' << time_rng;
                if (shuffle)
                    state << '\n' << sampler.save();
                trainer.save_state((directory / (stem + ".state.gguf")).string(), state.str());
                training_event("checkpoint-save", 1, 1, result.optimizer_step);
            }
            if (requested_stop) {
                std::cout << "requested_stop step=" << result.optimizer_step << std::endl;
                break;
            }
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "yue2-train: " << error.what() << '\n';
        return 1;
    }
}
