// Dataset planning for AR training: caption metadata, window placement and
// re-engraving transcribed events as ABC. Ported from the Python ar_dataset.py
// and checked equal to it before it was removed (docs/training-history.md);
// its rules are kept so runs it prepared resume unchanged.
#pragma once

#include "server/json.h"

#include <string>
#include <vector>

namespace yue2::training {

struct CaptionMetadata {
    double bpm = 0;
    std::string key;   // "Bb", "F#m"
    std::string style; // the caption without its tempo and key parts
};

// Tempo and key are required ("88 bpm", "B♭ major"). Matching follows Python's
// re for str patterns: Unicode word boundaries and whitespace, ASCII digits
// and ASCII case folding.
CaptionMetadata caption_metadata(const std::string & caption);

// Python's round() of a float: half to even.
long long round_half_even(double value);

// Python's str.strip(): Unicode whitespace at both ends.
std::string strip(const std::string & text);

// Bar-aligned window starts covering a recording; end windows may overlap.
std::vector<long long> window_starts(long long total_frames, long long frames, double bpm);

std::string chord_name(const std::string & text);
std::string abc_pitch(int pitch);

struct EngravedScore {
    std::string abc;
    server::json::Value stats;
};

// Re-engrave transcriber events (yue2-transcribe's events array) at absolute
// times on a 12-tick quarter grid, keeping both melody voices. seconds is a
// JSON number: the stats report it with the type it was given.
EngravedScore score_from_events(const server::json::Value & events, const server::json::Value & seconds, double bpm,
                                const std::string & key);

} // namespace yue2::training
