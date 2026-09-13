#include "yue2/transcription.h"

#include <cassert>
#include <iostream>
#include <string>

int main() {
    yue2::TranscriptionResult result;
    result.duration_seconds = 1.25;
    result.tokens = {1, 4, 11, 3, 2};
    yue2::ScoreEvent event;
    event.subbeat = 7;
    event.time_seconds = 0.5;
    event.has_timestamp = true;
    event.structure = "verse\n\"A\"";
    event.notes.push_back({60, 1, 4, 8, 0.75});
    result.events.push_back(event);
    result.warnings.push_back("a\\b");
    const auto json = yue2::serialize_transcription_json(result);
    assert(json.find("\"duration_seconds\": 1.25") != std::string::npos);
    assert(json.find("\"structure\": \"verse\\n\\\"A\\\"\"") != std::string::npos);
    assert(json.find("\"pitch\": 60") != std::string::npos);
    assert(json.find("\"warnings\": [\"a\\\\b\"]") != std::string::npos);
    std::cout << "transcription JSON serialization and escaping: ok\n";
    return 0;
}
