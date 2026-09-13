#include "server/base64.h"
#include "server/json.h"
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

int main() {
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
