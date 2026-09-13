#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

namespace yue2::sheetsage2 {

struct TokenLayout {
    static constexpr std::int32_t pad = 0;
    static constexpr std::int32_t sos = 1;
    static constexpr std::int32_t eos = 2;
    static constexpr std::int32_t out = 3;

    static constexpr std::int32_t prompt_begin = 4;
    static constexpr std::int32_t prompt_end = 260;
    static constexpr std::int32_t subbeat_shift_begin = 260;
    static constexpr std::int32_t subbeat_shift_end = 517;
    static constexpr std::int32_t time_begin = 517;
    static constexpr std::int32_t time_end = 30517;
    static constexpr std::int32_t meter_begin = 30517;
    static constexpr std::int32_t meter_end = 30709;
    static constexpr std::int32_t eighth_position_begin = 30709;
    static constexpr std::int32_t eighth_position_end = 30965;
    static constexpr std::int32_t structure_begin = 30965;
    static constexpr std::int32_t structure_end = 30988;
    static constexpr std::int32_t key_begin = 30988;
    static constexpr std::int32_t key_end = 31012;
    static constexpr std::int32_t chord_majmin_begin = 31012;
    static constexpr std::int32_t chord_majmin_end = 31037;
    static constexpr std::int32_t chord_full_begin = 31037;
    static constexpr std::int32_t chord_full_end = 31398;
    static constexpr std::int32_t pitch_begin = 31398;
    static constexpr std::int32_t pitch_end = 31654;
    static constexpr std::int32_t duration_begin = 31654;
    static constexpr std::int32_t duration_end = 31678;
    static constexpr std::int32_t vocab_size = 31678;

    static constexpr std::string_view fingerprint = "5ba3325af0344c7f";

    enum class Type {
        special,
        prompt,
        subbeat_shift,
        time,
        meter,
        eighth_position,
        structure,
        key,
        chord_majmin,
        chord_full,
        pitch,
        duration,
        invalid,
    };

    static constexpr Type type(std::int32_t token) noexcept {
        if (token >= 0 && token < prompt_begin) return Type::special;
        if (token >= prompt_begin && token < prompt_end) return Type::prompt;
        if (token >= subbeat_shift_begin && token < subbeat_shift_end) return Type::subbeat_shift;
        if (token >= time_begin && token < time_end) return Type::time;
        if (token >= meter_begin && token < meter_end) return Type::meter;
        if (token >= eighth_position_begin && token < eighth_position_end) return Type::eighth_position;
        if (token >= structure_begin && token < structure_end) return Type::structure;
        if (token >= key_begin && token < key_end) return Type::key;
        if (token >= chord_majmin_begin && token < chord_majmin_end) return Type::chord_majmin;
        if (token >= chord_full_begin && token < chord_full_end) return Type::chord_full;
        if (token >= pitch_begin && token < pitch_end) return Type::pitch;
        if (token >= duration_begin && token < duration_end) return Type::duration;
        return Type::invalid;
    }
};

class PromptGrammarState {
public:
    std::vector<std::uint8_t> allowed() const;
    bool update(std::int32_t token);
    std::int64_t generated_events() const noexcept { return generated_events_; }

private:
    enum class Incomplete { none, rhythm_after_meter, melody_after_pitch };
    void allow_field_starts(std::vector<std::uint8_t> & mask) const;

    std::int64_t generated_events_ = 0;
    bool in_shift_ = true;
    std::int32_t shift_run_ = 0;
    std::int32_t payload_count_ = 0;
    std::int32_t last_field_index_ = -1;
    Incomplete incomplete_ = Incomplete::none;
};

} // namespace yue2::sheetsage2

