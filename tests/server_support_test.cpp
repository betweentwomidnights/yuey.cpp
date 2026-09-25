#include "server/base64.h"
#include "server/json.h"
#include "server/policy.h"
#include "server/prompts.h"
#include "server/multipart.h"

#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string encode(const std::string & text) {
    return yue2::server::base64_encode(
        reinterpret_cast<const std::uint8_t *>(text.data()), text.size());
}

bool decode_rejected(const char * text) {
    try {
        (void)yue2::server::base64_decode(text);
    } catch (const std::invalid_argument &) {
        return true;
    }
    return false;
}

void check_base64() {
    using yue2::server::base64_decode;
    assert(encode("").empty());
    assert(encode("f") == "Zg==");
    assert(encode("fo") == "Zm8=");
    assert(encode("foo") == "Zm9v");
    assert(encode("foobar") == "Zm9vYmFy");

    std::vector<std::uint8_t> every_byte(256);
    for (int value = 0; value < 256; ++value) every_byte[value] = static_cast<std::uint8_t>(value);
    const auto encoded = yue2::server::base64_encode(every_byte.data(), every_byte.size());
    assert(base64_decode(encoded) == every_byte);

    const auto wrapped = base64_decode("Zm9v\r\nYmFy\n");
    assert(std::string(wrapped.begin(), wrapped.end()) == "foobar");
    const auto uri = base64_decode("data:audio/wav;base64,Zm8=");
    assert(std::string(uri.begin(), uri.end()) == "fo");

    assert(decode_rejected("Zm9"));
    assert(decode_rejected("Zm$v"));
    assert(decode_rejected("Z==="));
    assert(decode_rejected("Zg==Zg=="));
    assert(decode_rejected("data:audio/wav,Zm8="));
}

} // namespace

void check_generation_policy() {
    // /health and /props both advertise how the server bounds a generation.
    // They were two hand-maintained concatenations and drifted: the planning
    // bounds reached /health only, so a UI discovering capabilities through
    // /props could not see them. Both now splice this, and every field the
    // policy carries has to appear.
    yue2::server::GenerationPolicy policy;
    policy.natural_max_seconds = 96.0;
    policy.planning_overrun = 1.5;
    policy.planning_loop_bars = 12;
    const auto body = yue2::server::generation_policy_json(policy);

    for (const char * key : {"natural_max_seconds", "planning_overrun", "planning_loop_bars"}) {
        assert(body.find(std::string("\"") + key + "\":") != std::string::npos);
    }
    // Spliceable into a larger object: no braces, no leading comma.
    assert(body.front() == '"');
    assert(body.find('{') == std::string::npos);
    assert(body.find('}') == std::string::npos);

    const auto parsed = yue2::server::json::parse("{" + body + "}");
    assert(yue2::server::json::number(parsed, "natural_max_seconds", 0.0) == 96.0);
    assert(yue2::server::json::number(parsed, "planning_overrun", 0.0) == 1.5);
    assert(yue2::server::json::u32(parsed, "planning_loop_bars", 0) == 12);

    // A zero disables either bound and still has to be reported, not omitted.
    yue2::server::GenerationPolicy off;
    off.planning_overrun = 0.0;
    off.planning_loop_bars = 0;
    const auto disabled = yue2::server::generation_policy_json(off);
    assert(disabled.find("\"planning_overrun\":") != std::string::npos);
    assert(disabled.find("\"planning_loop_bars\":") != std::string::npos);
}

void check_dice_prompts() {
    const auto & pool = yue2::server::dice_prompts();
    assert(pool.instrumental.size() >= 32);
    assert(pool.vocal.size() >= 24);

    // The rules the pool is written to, enforced rather than trusted. Tempo and
    // length are set by the caller through a planning header, a score's Q:
    // field or a bar count, so a prompt claiming either can only contradict
    // them, and the score wins.
    auto contains = [](const std::string & hay, const char * needle) {
        return hay.find(needle) != std::string::npos;
    };
    // Digits are fine and often the point: 808, 303 and 909 name the machines
    // rather than a speed. What must not appear is a claim about tempo or
    // length, since the caller sets both and the score wins the disagreement.
    //
    // Whole words only. "tempo" sits inside both "downtempo" and
    // "contemporary", and neither is making a claim about speed.
    auto whole_word = [](const std::string & hay, const std::string & needle) {
        auto is_letter = [](char c) {
            return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
        };
        for (std::size_t at = hay.find(needle); at != std::string::npos;
             at = hay.find(needle, at + 1)) {
            const bool left = at > 0 && is_letter(hay[at - 1]);
            const std::size_t after = at + needle.size();
            const bool right = after < hay.size() && is_letter(hay[after]);
            if (!left && !right) return true;
        }
        return false;
    };
    auto claims_time = [&whole_word](const std::string & text) {
        for (const char * word : {"bpm", "tempo", "second", "seconds",
                                  "minute", "minutes", "hour", "hours"}) {
            if (whole_word(text, word)) return true;
        }
        return false;
    };

    std::vector<std::string> all;
    all.insert(all.end(), pool.instrumental.begin(), pool.instrumental.end());
    all.insert(all.end(), pool.vocal.begin(), pool.vocal.end());
    for (const auto & prompt : all) {
        assert(!prompt.empty());
        assert(prompt.size() < 200);
        assert(!claims_time(prompt));
        assert(prompt.front() != ' ' && prompt.back() != ' ');
        assert(prompt.back() != ',');
    }

    // An instrumental job already has "Instrumental, no vocals, no singing, no
    // humming." prefixed to its style, so a prompt reaching for a voice would
    // be arguing with its own preamble.
    // Whole words again: "pulsing" ends in "sing", and a chorus pedal is not a
    // choir, so "chorus" stays allowed.
    for (const auto & prompt : pool.instrumental) {
        for (const char * word : {"vocal", "vocals", "vocalist", "voice", "voices",
                                  "sing", "sings", "singing", "sung", "choir",
                                  "choirs", "lyric", "lyrics"}) {
            assert(!whole_word(prompt, word));
        }
    }

    // With instrumental off yuey sings whether or not the prompt names a
    // singer, so a vocal-bucket prompt needs no voice in it. Keep a real share
    // of them that way, or every roll reads as a vocal brief and the model
    // never gets to pick the voice itself.
    std::size_t voiceless = 0;
    for (const auto & prompt : pool.vocal) {
        bool names_a_voice = false;
        for (const char * word : {"vocal", "vocals", "voice", "voices", "singer", "sing",
                                  "singing", "sung", "choir", "male", "female", "falsetto",
                                  "soprano", "tenor", "baritone", "rap", "rapped",
                                  "toasting", "chant", "chanted", "harmonies", "harmony"}) {
            if (whole_word(prompt, word)) names_a_voice = true;
        }
        if (!names_a_voice) ++voiceless;
    }
    assert(voiceless * 4 >= pool.vocal.size());

    // Duplicates would make the dice land on the same prompt more often than
    // it looks like it should.
    for (std::size_t i = 0; i < all.size(); ++i)
        for (std::size_t j = i + 1; j < all.size(); ++j)
            assert(all[i] != all[j]);

    const auto body = yue2::server::dice_prompts_json();
    const auto parsed = yue2::server::json::parse(body);
    const auto * dice = parsed.find("dice");
    assert(dice != nullptr && dice->type == yue2::server::json::Type::object);
    for (const char * bucket : {"instrumental", "vocal"}) {
        const auto * arr = dice->find(bucket);
        assert(arr != nullptr && arr->type == yue2::server::json::Type::array);
        assert(!arr->array.empty());
    }
}

int main() {
    check_dice_prompts();
    check_generation_policy();
    using namespace yue2::server;
    const auto parsed = json::parse(
        "{\"text\":\"line\\n\\ud83c\\udfb5\",\"seed\":\"18446744073709551615\","
        "\"value\":1.25e2,\"yes\":true,\"items\":[null,3]}");
    assert(parsed.type == json::Type::object);
    assert(json::string(parsed, "text") == "line\n\xf0\x9f\x8e\xb5");
    assert(json::u64(parsed, "seed", 0) == UINT64_MAX);
    assert(json::number(parsed, "value", 0) == 125.0);
    assert(json::boolean(parsed, "yes", false));
    assert(json::parse(json::stringify(parsed)).object.size() == parsed.object.size());
    assert(json::quote("a\nb") == "\"a\\nb\"");

    bool duplicate_rejected = false;
    try {
        (void)json::parse("{\"a\":1,\"a\":2}");
    } catch (const std::runtime_error &) {
        duplicate_rejected = true;
    }
    assert(duplicate_rejected);

    const std::string boundary = "test-boundary";
    const std::string binary = std::string("RIFF\0payload\r\n", 15);
    const std::string body =
        "--test-boundary\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n"
        "yue2-transcription\r\n"
        "--test-boundary\r\nContent-Disposition: form-data; name=\"file\"; filename=\"x.wav\"\r\n"
        "Content-Type: audio/wav\r\n\r\n" + binary +
        "\r\n--test-boundary--\r\n";
    const auto extracted = multipart_boundary(
        "multipart/form-data; boundary=\"test-boundary\"");
    assert(extracted && *extracted == boundary);
    const auto parts = parse_multipart(body, boundary);
    assert(parts.size() == 2);
    assert(parts[0].name == "model" && parts[0].data == "yue2-transcription");
    assert(parts[1].name == "file" && parts[1].filename == "x.wav");
    assert(parts[1].content_type == "audio/wav" && parts[1].data == binary);

    check_base64();
    return 0;
}
