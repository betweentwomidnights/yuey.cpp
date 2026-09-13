#include "server/json.h"
#include "server/multipart.h"

#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>

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
    return 0;
}
