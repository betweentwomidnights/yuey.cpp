#include "yue2/sheetsage2_tokens.h"

#include <algorithm>
#include <stdexcept>

namespace yue2::sheetsage2 {
namespace {

constexpr std::int32_t timestamp_field = 0;
constexpr std::int32_t rhythm_field = 1;
constexpr std::int32_t structure_field = 2;
constexpr std::int32_t key_field = 3;
constexpr std::int32_t chord_field = 4;
constexpr std::int32_t melody_field = 5;

void allow_range(std::vector<std::uint8_t> & mask, std::int32_t begin, std::int32_t end) {
    std::fill(mask.begin() + begin, mask.begin() + end, std::uint8_t{1});
}

} // namespace

void PromptGrammarState::allow_field_starts(std::vector<std::uint8_t> & mask) const {
    if (last_field_index_ < timestamp_field) allow_range(mask, TokenLayout::time_begin, TokenLayout::time_end);
    if (last_field_index_ < rhythm_field) {
        allow_range(mask, TokenLayout::meter_begin, TokenLayout::meter_end);
        allow_range(mask, TokenLayout::eighth_position_begin, TokenLayout::eighth_position_end);
    }
    if (last_field_index_ < structure_field) allow_range(mask, TokenLayout::structure_begin, TokenLayout::structure_end);
    if (last_field_index_ < key_field) allow_range(mask, TokenLayout::key_begin, TokenLayout::key_end);
    if (last_field_index_ < chord_field) allow_range(mask, TokenLayout::chord_full_begin, TokenLayout::chord_full_end);
    if (last_field_index_ <= melody_field) allow_range(mask, TokenLayout::pitch_begin, TokenLayout::pitch_end);
}

std::vector<std::uint8_t> PromptGrammarState::allowed() const {
    std::vector<std::uint8_t> mask(TokenLayout::vocab_size, 0);
    if (payload_count_ > 0) mask[TokenLayout::eos] = 1;
    if ((payload_count_ > 0 || in_shift_) && shift_run_ < 4) {
        allow_range(mask, TokenLayout::subbeat_shift_begin, TokenLayout::subbeat_shift_end);
    }
    if (incomplete_ == Incomplete::rhythm_after_meter) {
        allow_range(mask, TokenLayout::eighth_position_begin, TokenLayout::eighth_position_end);
        return mask;
    }
    if (incomplete_ == Incomplete::melody_after_pitch) {
        allow_range(mask, TokenLayout::duration_begin, TokenLayout::duration_end);
        allow_range(mask, TokenLayout::pitch_begin, TokenLayout::pitch_end);
        return mask;
    }
    allow_field_starts(mask);
    return mask;
}

bool PromptGrammarState::update(std::int32_t token) {
    const auto type = TokenLayout::type(token);
    if (token == TokenLayout::eos) return true;
    if (type == TokenLayout::Type::subbeat_shift) {
        if (!in_shift_ && payload_count_ > 0) {
            ++generated_events_;
            payload_count_ = 0;
            last_field_index_ = -1;
            incomplete_ = Incomplete::none;
        }
        in_shift_ = true;
        ++shift_run_;
        return false;
    }

    in_shift_ = false;
    shift_run_ = 0;
    ++payload_count_;
    switch (type) {
        case TokenLayout::Type::time:
            last_field_index_ = timestamp_field;
            incomplete_ = Incomplete::none;
            break;
        case TokenLayout::Type::meter:
            last_field_index_ = rhythm_field;
            incomplete_ = Incomplete::rhythm_after_meter;
            break;
        case TokenLayout::Type::eighth_position:
            last_field_index_ = rhythm_field;
            incomplete_ = Incomplete::none;
            break;
        case TokenLayout::Type::structure:
            last_field_index_ = structure_field;
            incomplete_ = Incomplete::none;
            break;
        case TokenLayout::Type::key:
            last_field_index_ = key_field;
            incomplete_ = Incomplete::none;
            break;
        case TokenLayout::Type::chord_full:
            last_field_index_ = chord_field;
            incomplete_ = Incomplete::none;
            break;
        case TokenLayout::Type::pitch:
            last_field_index_ = melody_field;
            incomplete_ = Incomplete::melody_after_pitch;
            break;
        case TokenLayout::Type::duration:
            last_field_index_ = melody_field;
            incomplete_ = Incomplete::none;
            break;
        default:
            throw std::runtime_error("unexpected SheetSage2 prompt token");
    }
    return false;
}

} // namespace yue2::sheetsage2

