#include "training_sampler.h"
#include "yue2/ar_training.h"
#include "yue2/tokenizer.h"
#include "server/json.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>

namespace {
template <class T> std::vector<T> read(const std::string &path) {
    const auto bytes = std::filesystem::file_size(path);
    if (!bytes || bytes % sizeof(T))
        throw std::runtime_error("invalid input: " + path);
    std::vector<T> result(bytes / sizeof(T));
    std::ifstream file(path, std::ios::binary);
    if (!file.read(reinterpret_cast<char *>(result.data()), bytes))
        throw std::runtime_error("short input: " + path);
    return result;
}
std::string text(const std::string &path) {
    auto bytes = read<char>(path);
    return {bytes.begin(), bytes.end()};
}
void parent(const std::filesystem::path &path) {
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path());
}
void write_prefix(const std::map<std::string, std::string> &args) {
    yue2::SongRequest request;
    request.symbolic_mode = yue2::SymbolicMode::off;
    request.style = args.count("--prompt") ? args.at("--prompt") : "Instrumental music.";
    if (args.count("--abc")) {
        request.abc = text(args.at("--abc"));
        request.symbolic_mode = yue2::SymbolicMode::full;
    }
    if (args.count("--instrumental")) {
        if (!request.abc)
            throw std::runtime_error("instrumental prefix requires a used score");
        // Use the already transformed/used score exported by generation.
        request.style = "Instrumental, no vocals, no singing, no humming. " + request.style;
        request.lyrics = yue2::make_instrumental_lyrics(*request.abc);
    }
    if (args.count("--lyrics-file")) {
        if (args.count("--instrumental"))
            throw std::runtime_error("instrumental supplies lyrics");
        request.lyrics = text(args.at("--lyrics-file"));
    }
    yue2::TextTokenizer tokenizer(args.at("--tokenizer"));
    const auto prefix = yue2::make_positive_prefix(request, tokenizer);
    const std::filesystem::path path(args.at("--prefix-out"));
    if (std::filesystem::exists(path))
        throw std::runtime_error("prefix output exists");
    parent(path);
    std::ofstream out(path, std::ios::binary);
    if (!out.write(reinterpret_cast<const char *>(prefix.data()), prefix.size() * 4))
        throw std::runtime_error("cannot write prefix");
    std::cout << "prefix_tokens=" << prefix.size() << '\n';
}
} // namespace

int main(int argc, char **argv) {
    try {
        std::map<std::string, std::string> args;
        std::map<std::string, std::vector<std::string>> repeated;
        std::vector<yue2::LoraAdapterSpec> fixed;
        const std::set<std::string> repeat_keys{"--sample-prefix",     "--sample-tokens",
                                                "--validation-prefix", "--validation-tokens",
                                                "--retention-prefix",  "--retention-tokens"};
        const std::set<std::string> flags{"--predict-end", "--probe-only", "--instrumental",
                                          "--shuffle-samples", "--dora", "--group-samples"};
        const std::set<std::string> values{"--model",
                                           "--prefix",
                                           "--tokens",
                                           "--out",
                                           "--identity",
                                           "--device",
                                           "--threads",
                                           "--rank",
                                           "--lr",
                                           "--weight-decay",
                                           "--clip",
                                           "--seed",
                                           "--steps",
                                           "--save-every",
                                           "--resume",
                                           "--diagnostics",
                                           "--prefix-out",
                                           "--tokenizer",
                                           "--prompt",
                                           "--abc",
                                           "--lyrics-file",
                                           "--max-prefix",
                                           "--eval-every",
                                           "--kl-weight",
                                           "--early-stop-patience",
                                           "--early-stop-min-steps", "--dataset", "--stop-file", "--eval-scope"};
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            if (key == "--help") {
                std::cout << "Experimental semantic AR LoRA/DoRA trainer (fixed frames, "
                             "masked variable prefixes).\n"
                             "--model BF16.gguf --prefix prefix.i32 --tokens codes.i32 "
                             "--out NEW_DIR\n"
                             "--identity CONTENT_HASH [--fixed-lora PATH (repeatable)] "
                             "[--device cuda|cpu]\n"
                             "[--rank 8 --lr .0001 --weight-decay .01 --clip 1 --seed "
                             "831001]\n"
                             "[--steps 1 --save-every 1] [--resume STATE] "
                             "[--diagnostics NEW.gguf]\n"
                             "Repeat --sample-prefix PATH and --sample-tokens PATH in "
                             "matching order.\n"
                             "Or --dataset PREPARED.json [--group-samples] for track-balanced windows.\n"
                             "Repeat --validation-prefix/--validation-tokens and "
                             "--retention-prefix/--retention-tokens\n"
                             "for evaluation only. [--max-prefix N --eval-every N --eval-scope full|heldout]\n"
                             "[--kl-weight .2 --shuffle-samples --early-stop-patience "
                             "N --early-stop-min-steps N]\n"
                             "[--dora] [--stop-file PATH (save and stop after an update)]\n"
                             "[--probe-only] [--predict-end (only for genuine complete "
                             "sequences)]\n"
                             "Prefix preparation: --prefix-out NEW.i32 --tokenizer "
                             "PATH [--abc USED.abc]\n"
                             "[--prompt TEXT --instrumental | --lyrics-file PATH]\n"
                             "Shared AR weights affect planning; use a separate "
                             "common-weight planning pass.\n";
                return 0;
            }
            if (flags.count(key)) {
                if (!args.emplace(key, "1").second)
                    throw std::runtime_error("duplicate " + key);
            } else {
                if (key != "--fixed-lora" && !values.count(key) && !repeat_keys.count(key))
                    throw std::runtime_error("unknown option " + key);
                if (++i == argc)
                    throw std::runtime_error("missing value for " + key);
                if (key == "--fixed-lora")
                    fixed.push_back({argv[i], 1});
                else if (repeat_keys.count(key))
                    repeated[key].push_back(argv[i]);
                else if (!args.emplace(key, argv[i]).second)
                    throw std::runtime_error("duplicate " + key);
            }
        }
        auto require = [&](const char *key) -> const std::string & {
            if (!args.count(key) || args.at(key).empty())
                throw std::runtime_error(std::string("missing ") + key);
            return args.at(key);
        };
        if (args.count("--prefix-out")) {
            require("--tokenizer");
            write_prefix(args);
            return 0;
        }
        auto load_samples = [&](const std::string &p, const std::string &t) {
            std::vector<yue2::ArTrainingSample> result;
            if (repeated[p].size() != repeated[t].size())
                throw std::runtime_error("prefix/token list length mismatch: " + p);
            for (size_t i = 0; i < repeated[p].size(); ++i)
                result.push_back({read<int32_t>(repeated[p][i]), read<int32_t>(repeated[t][i])});
            return result;
        };
        auto samples = load_samples("--sample-prefix", "--sample-tokens");
        auto validation = load_samples("--validation-prefix", "--validation-tokens");
        auto retention = load_samples("--retention-prefix", "--retention-tokens");
        std::vector<size_t> groups;
        if(args.count("--dataset")) {
            if(!samples.empty()||!validation.empty()||!retention.empty()||args.count("--prefix")||args.count("--tokens"))throw std::runtime_error("dataset and explicit examples are mutually exclusive");
            namespace json=yue2::server::json;
            const std::filesystem::path manifest(args.at("--dataset"));
            const auto document=json::parse(text(manifest.string()));
            if(json::string(document,"format")!="yue2-ar-prepared-dataset-v1"||!json::boolean(document,"completed",false))throw std::runtime_error("incomplete or unsupported dataset manifest");
            const auto *rows=document.find("rows");
            if(!rows||rows->type!=json::Type::array)throw std::runtime_error("dataset rows are required");
            std::map<std::string,size_t> group_ids;
            for(const auto &row:rows->array) {
                if(!json::boolean(row,"eligible",true))continue;
                auto path=[&](const char *key){std::filesystem::path p(json::string(row,key));if(p.empty())throw std::runtime_error("missing dataset path");return p.is_absolute()?p:manifest.parent_path()/p;};
                yue2::ArTrainingSample sample{read<int32_t>(path("prefix").string()),read<int32_t>(path("codes").string())};
                const auto split=json::string(row,"split");
                if(split=="train") {
                    const auto name=json::string(row,"group",json::string(row,"source_sha256"));
                    if(args.count("--group-samples")) {
                        if(name.empty())throw std::runtime_error("grouped example needs a source/group identity");
                        auto found=group_ids.emplace(name,group_ids.size());groups.push_back(found.first->second);
                    }
                    samples.push_back(std::move(sample));
                } else if(split=="validation")validation.push_back(std::move(sample));
                else if(split=="retention")retention.push_back(std::move(sample));
                else throw std::runtime_error("unknown dataset split");
            }
            if(samples.empty())throw std::runtime_error("dataset has no training examples");
        } else if (samples.empty())
            samples.push_back(
                {read<int32_t>(require("--prefix")), read<int32_t>(require("--tokens"))});
        else if (args.count("--prefix") || args.count("--tokens"))
            throw std::runtime_error("single and multi-sample options are mutually exclusive");
        if(args.count("--group-samples")&&groups.empty())throw std::runtime_error("group sampling requires a dataset manifest");
        size_t max_prefix = 0;
        const auto frames = samples.front().codes.size();
        for (const auto *group : {&samples, &validation, &retention})
            for (const auto &sample : *group) {
                if (sample.codes.size() != frames || sample.prefix.empty())
                    throw std::runtime_error(
                        "all examples must share a nonempty fixed frame count");
                max_prefix = std::max(max_prefix, sample.prefix.size());
            }
        if (args.count("--max-prefix")) {
            const auto requested = std::stoul(args.at("--max-prefix"));
            if (requested < max_prefix)
                throw std::runtime_error("max-prefix is smaller than a sample prefix");
            max_prefix = requested;
        }
        yue2::ArTrainingOptions options;
        options.run_identity = require("--identity");
        if(args.count("--dataset")) {
            // Compatibility signature over actual loaded streams, not just the
            // manifest's claimed hashes or filenames. The job owner separately
            // supplies SHA-256 identities for model/runtime/source provenance.
            uint64_t hash=14695981039346656037ULL;
            auto mix=[&](uint64_t value){for(int byte=0;byte<8;++byte){hash^=(value>>(8*byte))&255;hash*=1099511628211ULL;}};
            for(const auto *set:{&samples,&validation,&retention}) {
                mix(set->size());
                for(const auto &sample:*set) {
                    mix(sample.prefix.size());for(auto id:sample.prefix)mix(uint32_t(id));
                    mix(sample.codes.size());for(auto id:sample.codes)mix(uint32_t(id));
                }
            }
            mix(groups.size());for(auto group:groups)mix(group);
            std::ostringstream signature;signature<<std::hex<<hash;
            options.run_identity+=":examples-fnv1a64-v1:"+signature.str();
        }
        options.fixed_adapters = fixed;
        options.dora=args.count("--dora")!=0;
        if (args.count("--device"))
            options.device = args.at("--device");
        if (args.count("--threads"))
            options.threads = std::stoi(args.at("--threads"));
        if (args.count("--rank"))
            options.rank = std::stoi(args.at("--rank"));
        if (args.count("--lr"))
            options.learning_rate = std::stof(args.at("--lr"));
        if (args.count("--weight-decay"))
            options.weight_decay = std::stof(args.at("--weight-decay"));
        if (args.count("--clip"))
            options.gradient_clip = std::stof(args.at("--clip"));
        if (args.count("--kl-weight"))
            options.kl_weight = std::stof(args.at("--kl-weight"));
        if (args.count("--seed"))
            options.seed = std::stoul(args.at("--seed"));
        const int steps = args.count("--steps") ? std::stoi(args.at("--steps")) : 1;
        const int save_every = args.count("--save-every") ? std::stoi(args.at("--save-every")) : 1;
        const int eval_every =
            args.count("--eval-every") ? std::stoi(args.at("--eval-every")) : save_every;
        const int patience =
            args.count("--early-stop-patience") ? std::stoi(args.at("--early-stop-patience")) : 0;
        const int min_stop =
            args.count("--early-stop-min-steps") ? std::stoi(args.at("--early-stop-min-steps")) : 0;
        if (steps < 0 || save_every < 1 || eval_every < 1)
            throw std::runtime_error("invalid step count");
        const auto eval_scope = args.count("--eval-scope") ? args.at("--eval-scope") : "full";
        if (eval_scope != "full" && eval_scope != "heldout")
            throw std::runtime_error("--eval-scope must be full or heldout");
        if (eval_scope == "heldout" && validation.empty())
            throw std::runtime_error("heldout evaluation requires validation examples");
        if (patience < 0 || min_stop < 0 || (patience && validation.empty()))
            throw std::runtime_error("invalid early stopping configuration");
        const std::filesystem::path out(require("--out"));
        // Entire invocation uses a new directory, including resumed invocations.
        if (std::filesystem::exists(out))
            throw std::runtime_error("output directory exists");
        std::filesystem::create_directories(out);
        yue2::ArTrainer trainer(require("--model"), uint32_t(max_prefix), uint32_t(frames),
                                args.count("--predict-end") != 0, options);
        if (options.kl_weight > 0) {
            const size_t teacher_count=args.count("--probe-only")?1:samples.size();
            for (size_t i = 0; i < teacher_count; ++i) {
                std::cout << "prepare_teacher=" << i + 1 << '/' << teacher_count << std::endl;
                trainer.prepare_teacher(samples[i]);
            }
        }
        yue2::detail::TrainingSampler sampler(samples.size(), args.count("--shuffle-samples") != 0,
                                              options.seed,groups);
        bool restored_stopping = false;
        double restored_best = 0;
        int restored_stale = 0;
        if (args.count("--resume")) {
            const auto state = trainer.load_state(args.at("--resume"));
            if (state.rfind("ar-cli-v1\n", 0) == 0) {
                std::istringstream input(state);
                std::string line;
                std::getline(input, line);
                std::getline(input, line);
                std::istringstream values(line);
                int saved_patience, saved_min;
                if (!(values >> restored_best >> restored_stale >> saved_patience >> saved_min) ||
                    saved_patience != patience || saved_min != min_stop || restored_stale < 0 ||
                    !std::isfinite(restored_best))
                    throw std::runtime_error("incompatible early stopping state");
                sampler.load(std::string(std::istreambuf_iterator<char>(input), {}));
                restored_stopping = true;
            } else {
                if (patience)
                    throw std::runtime_error("legacy state has no early stopping history");
                sampler.load(state);
            }
        }
        std::ofstream log(out / "history.jsonl");
        std::ofstream metrics(out / "metrics.jsonl");
        std::ofstream checkpoint_times;
        if(args.count("--dataset"))checkpoint_times.open(out/"checkpoint-times.jsonl");
        auto evaluate = [&](uint64_t step) {
            double held_mean = 0;
            metrics.precision(10);
            metrics << "{\"step\":" << step;
            metrics << ",\"evaluation_scope\":\"" << eval_scope << "\"";
            for (const auto &entry :
                 {std::make_pair("train", &samples), std::make_pair("heldout", &validation),
                  std::make_pair("retention", &retention)}) {
                if (eval_scope == "heldout" && std::string(entry.first) == "train") continue;
                double sum = 0;
                metrics << ",\"" << entry.first << "_per_sample\":[";
                for (size_t i = 0; i < entry.second->size(); ++i) {
                    const auto loss = trainer.evaluate((*entry.second)[i]);
                    sum += loss;
                    if (i)
                        metrics << ',';
                    metrics << loss;
                }
                metrics << "],\"" << entry.first
                        << "_mean\":" << (entry.second->empty() ? 0 : sum / entry.second->size());
                if (std::string(entry.first) == "heldout" && !entry.second->empty())
                    held_mean = sum / entry.second->size();
            }
            metrics << "}\n";
            metrics.flush();
            return held_mean;
        };
        const auto before = trainer.step(samples.front(), false);
        if (args.count("--diagnostics"))
            trainer.save_diagnostics(args.at("--diagnostics"));
        std::cout << "initial_loss=" << before.loss << " step=" << before.optimizer_step
                  << std::endl;
        if (args.count("--probe-only")) {
            trainer.save_adapter(
                (out / ("step-" + std::to_string(before.optimizer_step) + ".gguf")).string());
            return 0;
        }
        const auto initial_held = evaluate(before.optimizer_step);
        double best_held = restored_stopping ? restored_best : (patience ? initial_held : 0);
        int stale = restored_stopping ? restored_stale : 0;
        auto cli_state = [&]() {
            std::ostringstream stream;
            stream << "ar-cli-v1\n"
                   << std::setprecision(17) << best_held << ' ' << stale << ' ' << patience << ' '
                   << min_stop << '\n'
                   << sampler.save();
            return stream.str();
        };
        for (int i = 1; i <= steps; ++i) {
            const auto start = std::chrono::steady_clock::now();
            const auto sample_index = sampler.next();
            const auto result = trainer.step(samples[sample_index]);
            const auto seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            log.precision(10);
            log << "{\"step\":" << result.optimizer_step << ",\"loss\":" << result.loss
                << ",\"gradient_norm\":" << result.gradient_norm
                << ",\"sample_index\":" << sample_index
                << ",\"cross_entropy\":" << result.cross_entropy
                << ",\"teacher_kl\":" << result.teacher_kl << ",\"seconds\":" << seconds
                << ",\"inputs_seconds\":" << result.inputs_seconds
                << ",\"optimizer_seconds\":" << result.optimizer_seconds;
            if(!groups.empty())log<<",\"sample_group\":"<<groups[sample_index]<<",\"target_tokens\":"<<samples[sample_index].codes.size();
            log<<"}\n";
            log.flush();
            std::cout << "step=" << result.optimizer_step << " loss=" << result.loss
                      << " grad=" << result.gradient_norm << " seconds=" << seconds << std::endl;
            const bool requested_stop=args.count("--stop-file")&&std::filesystem::exists(args.at("--stop-file"));
            bool stop = requested_stop;
            if (result.optimizer_step % eval_every == 0 || i == steps) {
                const auto held = evaluate(result.optimizer_step);
                if (patience && result.optimizer_step % eval_every == 0) {
                    if (held < best_held - 1e-4) {
                        best_held = held;
                        stale = 0;
                    } else
                        ++stale;
                    stop = stop || (stale >= patience && result.optimizer_step >= uint64_t(min_stop));
                }
            }
            if (result.optimizer_step % save_every == 0 || i == steps || stop) {
                const auto stem = out / ("step-" + std::to_string(result.optimizer_step));
                const auto checkpoint_start=std::chrono::steady_clock::now();
                trainer.save_adapter(stem.string() + ".gguf");
                const auto adapter_done=std::chrono::steady_clock::now();
                trainer.save_state(stem.string() + ".state.gguf", cli_state());
                if(checkpoint_times.is_open()) {
                    const auto state_done=std::chrono::steady_clock::now();
                    checkpoint_times<<"{\"step\":"<<result.optimizer_step
                        <<",\"adapter_seconds\":"<<std::chrono::duration<double>(adapter_done-checkpoint_start).count()
                        <<",\"state_seconds\":"<<std::chrono::duration<double>(state_done-adapter_done).count()<<"}\n";
                    checkpoint_times.flush();
                }
            }
            if (stop) {
                std::cout << (requested_stop?"requested_stop step=":"early_stop step=") << result.optimizer_step
                          << " best_held=" << best_held << std::endl;
                break;
            }
        }
        std::cout << "final_loss=" << trainer.evaluate(samples.front()) << std::endl;
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "yue2-ar-train: " << e.what() << '\n';
        return 2;
    }
}
