// Native training workflow pieces that need no models: hashing, Python-style
// JSON, captions, windows, scores, settings and audio headers. Behaviour of the
// fits and the workflow is in training_native_test.cpp.
#include "server/flac.h"
#include "training/dataset.h"
#include "training/files.h"
#include "training/preparation.h"
#include "training/process.h"
#include "training/sha256.h"
#include "training/workflow.h"
#include "yue2/audio.h"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace {
namespace json = yue2::server::json;
namespace training = yue2::training;

template <class F> std::string rejection(F && call) {
    try {
        call();
    } catch (const std::invalid_argument & error) {
        return error.what();
    }
    return {};
}

void check_sha256() {
    assert(training::sha256_hex("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    assert(training::sha256_hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    assert(training::sha256_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
           "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    // A million bytes, fed in uneven pieces across block boundaries.
    training::Sha256 hash;
    const std::string piece(997, 'a');
    std::size_t fed = 0;
    for (; fed + piece.size() <= 1000000; fed += piece.size()) hash.update(piece);
    hash.update(std::string(1000000 - fed, 'a'));
    assert(hash.hex() == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

void check_json() {
    assert(json::python_repr(30) == "30.0");
    assert(json::python_repr(2e-5) == "2e-05");
    assert(json::python_repr(0.2) == "0.2");
    assert(json::python_repr(1e16) == "1e+16");
    assert(json::python_repr(1e15) == "1000000000000000.0");
    assert(json::python_repr(0.0001) == "0.0001");
    assert(json::python_repr(-0.0) == "-0.0");
    assert(json::python_repr(0.1 + 0.2) == "0.30000000000000004");
    assert(json::python_repr(1.5e-300) == "1.5e-300");
    auto value = json::object();
    value.object["caption"] = json::from_string("B\xE2\x99\xAD major \xF0\x9F\x8E\xB9\n");
    value.object["empty"] = json::array();
    assert(json::pretty(value) == "{\n  \"caption\": \"B\\u266d major \\ud83c\\udfb9\\n\",\n  \"empty\": []\n}");
}

void check_captions() {
    const auto meta = training::caption_metadata("neurofunk, dark electronica, 88 bpm, B\xE2\x99\xAD major");
    assert(meta.bpm == 88 && meta.key == "Bb" && meta.style == "neurofunk, dark electronica");
    assert(training::caption_metadata("lofi, 72.5BPM, f# MINOR").key == "F#m");
    // Word boundaries: "x120 bpm" has no tempo, and "minority" is not a mode.
    assert(rejection([] { training::caption_metadata("x120 bpm, C major"); }) == "Caption needs explicit tempo and key");
    assert(rejection([] { training::caption_metadata("120 bpm, C minority"); }) == "Caption needs explicit tempo and key");
    assert(rejection([] { training::caption_metadata("piano, 500 bpm, C major"); }) == "Unsupported caption tempo");
    assert(training::strip("\xE3\x80\x80 piano \t\n") == "piano");
}

void check_windows_and_scores() {
    const auto starts = training::window_starts(1129, 750, 85);
    assert(starts.size() >= 2 && starts.front() == 0 && starts.back() + 750 > 1100);
    assert(rejection([] { training::window_starts(749, 750, 85); }) == "Recording shorter than training window");
    assert(training::chord_name("E:maj/3") == "E/G#");
    assert(training::chord_name("B:min7/b7") == "Bm7/A");
    assert(training::abc_pitch(60) == "=C" && training::abc_pitch(48) == "=C," && training::abc_pitch(84) == "=c'");
    auto events = json::parse(R"([{"time_seconds":0,"chord":"C:maj","notes":[{"pitch":60,"end_time_seconds":0.5,"track":0}]}])");
    const auto score = training::score_from_events(events, json::from_integer(2), 120, "C");
    assert(score.abc.find("\"C\"=C4z12|") != std::string::npos);
    assert(json::stringify(score.stats.object.at("retained_monophonic_notes")) == "1");
}

void check_settings() {
    const auto defaults = training::settings(json::parse(R"({"id":"koan","recipe_version":5})"));
    // Python's dict order, so a written state.json matches byte for byte.
    assert(json::stringify(defaults) ==
           R"({"window_seconds":30,"rank":32,"ar_steps":800,"nar_steps":400,"ar_lr":2e-05,"nar_lr":3e-05,"kl_weight":0.2,)"
           R"("seed":831001,"eval_every":100,"family":"dora","trigger":"koan_style","keep_caches":false,)"
           R"("evaluation_scope":"heldout","cache_fingerprint":"bytes-v2","teacher_cache":"f32","conditioning_storage":"cache"})");
    // "z" holds U+00E9 as UTF-8 bytes; the dump escapes it.
    auto nested = json::parse("{\"b\":[1,{\"z\":\"\xC3\xA9\",\"a\":null}],\"a\":true}");
    assert(json::python_dumps_sorted(nested) == "{\"a\": true, \"b\": [1, {\"a\": null, \"z\": \"\\u00e9\"}]}");
    assert(rejection([] { training::settings(json::parse(R"({"id":"a","rank":1})")); }) == "rank must be between 2 and 128");
    assert(rejection([] { training::settings(json::parse(R"({"id":"a","ar_lr":1})")); }) == "ar_lr must be between 1e-07 and 0.001");
    assert(rejection([] { training::settings(json::parse(R"({"id":"a","teacher_storage":"online"})")); }) ==
           "Storage modes require recipe version 5");
}

void check_processes_and_folders() {
#ifdef _WIN32
    const auto echoed = training::capture_output("C:\\Windows\\System32\\cmd.exe", {"/c", "echo yuey"}, 30);
#else
    const auto echoed = training::capture_output("/bin/sh", {"-c", "echo yuey; echo hidden >&2"}, 30);
#endif
    assert(echoed && echoed->rfind("yuey", 0) == 0 && echoed->find("hidden") == std::string::npos);
    assert(!training::capture_output("/no/such/program", {}, 30));
#ifndef _WIN32
    assert(!training::capture_output("/bin/sh", {"-c", "sleep 5"}, 1));
#endif
    const auto folder = std::filesystem::temp_directory_path() / "yue2-training-workflow-test";
    std::filesystem::remove_all(folder);
    std::filesystem::create_directories(folder);
    for (const char * name : {"b.wav", "_c.wav", "A.wav"}) std::ofstream(folder / name) << name;
    std::string order;
    for (const auto & path : training::sorted_entries(folder)) order += path.filename().string() + " ";
#ifdef _WIN32
    // Python compares lowercased names on Windows: "_" (0x5f) sorts before "a".
    assert(order == "_c.wav A.wav b.wav ");
#else
    assert(order == "A.wav _c.wav b.wav ");
#endif
    std::filesystem::remove_all(folder);
}

void check_audio_info() {
    const auto wav = yue2::audio::encode_wav_float(std::vector<float>(4800 * 2), 48000, 2);
    const auto info = yue2::audio::audio_info(wav.data(), wav.size());
    assert(info.sample_rate == 48000 && info.channels == 2 && info.frames == 4800);
}

// Semantic preparation reads FLAC recordings as it reads their WAV.
void check_flac_reads_like_wav() {
    const std::uint32_t rate = 44100, channels = 2;
    std::vector<std::int16_t> pcm(rate * channels);
    for (std::size_t i = 0; i < pcm.size(); ++i) pcm[i] = static_cast<std::int16_t>((i * 7919 + i / 3) % 60000 - 30000);
    const auto le = [](std::string & out, std::uint32_t value, int bytes) {
        for (int b = 0; b < bytes; ++b) out.push_back(static_cast<char>((value >> (8 * b)) & 0xff));
    };
    const auto data_bytes = static_cast<std::uint32_t>(pcm.size() * 2);
    std::string wav = "RIFF";
    le(wav, 36 + data_bytes, 4);
    wav += "WAVEfmt ";
    le(wav, 16, 4); le(wav, 1, 2); le(wav, channels, 2); le(wav, rate, 4); le(wav, rate * channels * 2, 4);
    le(wav, channels * 2, 2); le(wav, 16, 2);
    wav += "data";
    le(wav, data_bytes, 4);
    wav.append(reinterpret_cast<const char *>(pcm.data()), data_bytes);
    const auto flac = yue2::server::encode_flac_pcm16(pcm, rate, channels);
    const auto folder = std::filesystem::temp_directory_path();
    std::ofstream(folder / "yue2-semantic-test.wav", std::ios::binary) << wav;
    std::ofstream(folder / "yue2-semantic-test.flac", std::ios::binary)
        .write(reinterpret_cast<const char *>(flac.data()), static_cast<std::streamsize>(flac.size()));
    const auto from_wav = yue2::audio::read_audio_mono_24k(folder / "yue2-semantic-test.wav");
    const auto from_flac = yue2::audio::read_audio_mono_24k(folder / "yue2-semantic-test.flac");
    assert(from_wav.sample_rate == 24000 && from_flac.sample_rate == 24000);
    assert(from_wav.samples == from_flac.samples);
    assert(from_wav.samples == yue2::audio::read_wav_mono_24k(folder / "yue2-semantic-test.wav").samples);
    std::filesystem::remove(folder / "yue2-semantic-test.wav");
    std::filesystem::remove(folder / "yue2-semantic-test.flac");
}

void check_files() {
#ifdef _WIN32
    assert(training::python_path("C:/runs//koan/./data/") == "C:\\runs\\koan\\data");
    assert(training::python_path("..\\a/b") == "..\\a\\b");
    assert(training::python_path("\\\\host\\share/x") == "\\\\host\\share\\x");
#else
    assert(training::python_path("/runs//koan/./data/") == "/runs/koan/data");
    assert(training::python_path("../a/b") == "../a/b");
#endif
    assert(training::python_path("") == ".");
    assert(training::python_equal(json::parse(R"({"a":1,"b":[2.0,true]})"), json::parse(R"({"b":[2,1],"a":1.0})")));
    assert(!training::python_equal(json::parse(R"({"a":"1"})"), json::parse(R"({"a":1})")));
    assert(!training::python_equal(json::parse(R"([1,2])"), json::parse(R"([2,1])")));
    // save() writes json.dumps(indent=2) and a newline in text mode.
    const auto path = std::filesystem::temp_directory_path() / "yue2-training-save-test.json";
    training::save(path, json::parse(R"({"b":1,"a":[]})"));
#ifdef _WIN32
    assert(training::read_bytes(path) == "{\r\n  \"b\": 1,\r\n  \"a\": []\r\n}\r\n");
#else
    assert(training::read_bytes(path) == "{\n  \"b\": 1,\n  \"a\": []\n}\n");
#endif
    assert(!std::filesystem::exists(training::with_suffix(path, ".json.partial")));
    std::filesystem::remove(path);
}

} // namespace

int main() {
    check_sha256();
    check_json();
    check_captions();
    check_windows_and_scores();
    check_settings();
    check_processes_and_folders();
    check_audio_info();
    check_files();
    check_flac_reads_like_wav();
    return 0;
}
