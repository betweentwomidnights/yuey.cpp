#include "server/flac.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace yue2::server {
namespace {

constexpr std::uint32_t kBlockSize = 4096;
constexpr int kMaxFixedOrder = 4;
constexpr int kMaxPartitionOrder = 8;
constexpr unsigned kMaxRiceParameter = 14;  // 15 is the escape code

class BitWriter {
public:
    void write(std::uint64_t value, int bits) {
        for (int bit = bits - 1; bit >= 0; --bit) put((value >> bit) & 1U);
    }
    void write_signed(std::int64_t value, int bits) {
        write(static_cast<std::uint64_t>(value) & ((bits == 64) ? ~0ULL : ((1ULL << bits) - 1)), bits);
    }
    void write_unary_zeros(std::uint64_t zeros) {
        for (std::uint64_t i = 0; i < zeros; ++i) put(0);
        put(1);
    }
    void align() {
        while (filled_ != 0) put(0);
    }
    std::vector<std::uint8_t> & bytes() { return bytes_; }

private:
    void put(std::uint64_t bit) {
        current_ = static_cast<std::uint8_t>((current_ << 1) | bit);
        if (++filled_ == 8) {
            bytes_.push_back(current_);
            current_ = 0;
            filled_ = 0;
        }
    }

    std::vector<std::uint8_t> bytes_;
    std::uint8_t current_ = 0;
    int filled_ = 0;
};

std::uint8_t crc8(const std::uint8_t * data, std::size_t size) {
    std::uint8_t crc = 0;
    for (std::size_t i = 0; i < size; ++i) {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit) {
            crc = static_cast<std::uint8_t>((crc & 0x80) ? (crc << 1) ^ 0x07 : crc << 1);
        }
    }
    return crc;
}

std::uint16_t crc16(const std::uint8_t * data, std::size_t size) {
    std::uint16_t crc = 0;
    for (std::size_t i = 0; i < size; ++i) {
        crc ^= static_cast<std::uint16_t>(data[i] << 8);
        for (int bit = 0; bit < 8; ++bit) {
            crc = static_cast<std::uint16_t>((crc & 0x8000) ? (crc << 1) ^ 0x8005 : crc << 1);
        }
    }
    return crc;
}

// FLAC's frame number: UTF-8's variable-length scheme, extended to 36 bits.
void write_frame_number(BitWriter & out, std::uint64_t value) {
    if (value < 0x80) {
        out.write(value, 8);
        return;
    }
    int continuation = 1;
    while (value >= (1ULL << (5 * continuation + 6))) ++continuation;
    const int lead_bits = 6 - continuation;
    const std::uint64_t lead_marker = (0xFFULL << (7 - continuation)) & 0xFF;
    out.write(lead_marker | (value >> (6 * continuation)) & ((1ULL << lead_bits) - 1), 8);
    for (int i = continuation - 1; i >= 0; --i) {
        out.write(0x80 | ((value >> (6 * i)) & 0x3F), 8);
    }
}

std::uint64_t fold(std::int64_t value) {
    return value >= 0 ? static_cast<std::uint64_t>(value) << 1
                      : (static_cast<std::uint64_t>(-(value + 1)) << 1) | 1ULL;
}

std::int64_t fixed_residual(const std::vector<std::int64_t> & x, std::size_t i, int order) {
    switch (order) {
        case 0: return x[i];
        case 1: return x[i] - x[i - 1];
        case 2: return x[i] - 2 * x[i - 1] + x[i - 2];
        case 3: return x[i] - 3 * x[i - 1] + 3 * x[i - 2] - x[i - 3];
        default: return x[i] - 4 * x[i - 1] + 6 * x[i - 2] - 4 * x[i - 3] + x[i - 4];
    }
}

struct Plan {
    int partition_order = 0;
    std::vector<unsigned> parameters;
    std::uint64_t bits = std::numeric_limits<std::uint64_t>::max();
};

// Rice coding cost of `count` folded residuals summing to `sum`, with
// parameter k. sum >> k slightly overstates sum(u >> k), which only ever errs
// toward a larger estimate; the encoding itself is exact either way.
std::uint64_t rice_cost(std::uint64_t sum, std::uint64_t count, unsigned k) {
    return count * (k + 1ULL) + (sum >> k);
}

unsigned rice_parameter(std::uint64_t sum, std::uint64_t count, std::uint64_t & cost) {
    unsigned guess = 0;
    if (count > 0) {
        for (std::uint64_t mean = sum / count; mean > 1 && guess < kMaxRiceParameter; mean >>= 1) ++guess;
    }
    unsigned best = guess;
    cost = rice_cost(sum, count, guess);
    for (const unsigned k : {guess > 0 ? guess - 1 : guess, std::min(guess + 1, kMaxRiceParameter)}) {
        const auto c = rice_cost(sum, count, k);
        if (c < cost) {
            cost = c;
            best = k;
        }
    }
    return best;
}

// The cheapest partitioned-Rice layout for these folded residuals, where the
// first partition is short by `order` warm-up samples. Sums are taken once at
// the finest partitioning and merged pairwise on the way to coarser ones.
Plan plan_rice(const std::vector<std::uint64_t> & folded, std::uint32_t block, int order) {
    int finest = 0;
    while (finest < kMaxPartitionOrder && block % (1U << (finest + 1)) == 0 &&
           (block >> (finest + 1)) > static_cast<std::uint32_t>(order)) {
        ++finest;
    }
    if ((block >> finest) <= static_cast<std::uint32_t>(order)) return {};

    std::vector<std::uint64_t> sums(1U << finest, 0), counts(1U << finest, 0);
    const std::uint32_t length = block >> finest;
    std::size_t index = 0;
    for (std::size_t p = 0; p < sums.size(); ++p) {
        counts[p] = p == 0 ? length - order : length;
        for (std::uint64_t i = 0; i < counts[p]; ++i) sums[p] += folded[index++];
    }

    Plan best;
    for (int partition_order = finest; partition_order >= 0; --partition_order) {
        Plan plan;
        plan.partition_order = partition_order;
        plan.bits = 2 + 4;  // coding method, partition order
        for (std::size_t p = 0; p < sums.size(); ++p) {
            std::uint64_t cost = 0;
            plan.parameters.push_back(rice_parameter(sums[p], counts[p], cost));
            plan.bits += 4 + cost;
        }
        if (plan.bits < best.bits) best = std::move(plan);
        if (partition_order > 0) {
            for (std::size_t p = 0; p < sums.size() / 2; ++p) {
                sums[p] = sums[2 * p] + sums[2 * p + 1];
                counts[p] = counts[2 * p] + counts[2 * p + 1];
            }
            sums.resize(sums.size() / 2);
            counts.resize(counts.size() / 2);
        }
    }
    return best;
}

struct Subframe {
    enum class Kind { constant, verbatim, fixed } kind = Kind::verbatim;
    int order = 0;
    Plan plan;
    std::vector<std::uint64_t> folded;
    std::uint64_t bits = 0;
};

Subframe choose_subframe(const std::vector<std::int64_t> & x, int bps) {
    const auto block = static_cast<std::uint32_t>(x.size());
    Subframe best;
    best.kind = Subframe::Kind::verbatim;
    best.bits = 8 + static_cast<std::uint64_t>(block) * bps;

    if (std::all_of(x.begin(), x.end(), [&](std::int64_t v) { return v == x[0]; })) {
        best.kind = Subframe::Kind::constant;
        best.bits = 8 + bps;
        return best;
    }
    for (int order = 0; order <= kMaxFixedOrder && static_cast<std::uint32_t>(order) < block; ++order) {
        std::vector<std::uint64_t> folded;
        folded.reserve(block - order);
        for (std::size_t i = order; i < block; ++i) folded.push_back(fold(fixed_residual(x, i, order)));
        auto plan = plan_rice(folded, block, order);
        if (plan.parameters.empty()) continue;
        const std::uint64_t bits = 8 + static_cast<std::uint64_t>(order) * bps + plan.bits;
        if (bits < best.bits) {
            best.kind = Subframe::Kind::fixed;
            best.order = order;
            best.plan = std::move(plan);
            best.folded = std::move(folded);
            best.bits = bits;
        }
    }
    return best;
}

void write_subframe(BitWriter & out, const Subframe & s, const std::vector<std::int64_t> & x, int bps) {
    switch (s.kind) {
        case Subframe::Kind::constant:
            out.write(0b00000000, 8);
            out.write_signed(x[0], bps);
            return;
        case Subframe::Kind::verbatim:
            out.write(0b00000010, 8);
            for (const auto v : x) out.write_signed(v, bps);
            return;
        case Subframe::Kind::fixed:
            out.write(static_cast<std::uint64_t>(0b00010000 | (s.order << 1)), 8);
            for (int i = 0; i < s.order; ++i) out.write_signed(x[static_cast<std::size_t>(i)], bps);
            out.write(0, 2);  // Rice, 4-bit parameters
            out.write(static_cast<std::uint64_t>(s.plan.partition_order), 4);
            {
                const auto block = static_cast<std::uint32_t>(x.size());
                const std::uint32_t length = block >> s.plan.partition_order;
                std::size_t index = 0;
                for (std::size_t p = 0; p < s.plan.parameters.size(); ++p) {
                    const unsigned k = s.plan.parameters[p];
                    out.write(k, 4);
                    const std::size_t count = p == 0 ? length - s.order : length;
                    for (std::size_t i = 0; i < count; ++i, ++index) {
                        const std::uint64_t u = s.folded[index];
                        out.write_unary_zeros(u >> k);
                        if (k > 0) out.write(u & ((1ULL << k) - 1), static_cast<int>(k));
                    }
                }
            }
            return;
    }
}

int sample_rate_code(std::uint32_t rate) {
    switch (rate) {
        case 88200: return 0b0001;
        case 176400: return 0b0010;
        case 192000: return 0b0011;
        case 8000: return 0b0100;
        case 16000: return 0b0101;
        case 22050: return 0b0110;
        case 24000: return 0b0111;
        case 32000: return 0b1000;
        case 44100: return 0b1001;
        case 48000: return 0b1010;
        case 96000: return 0b1011;
        default: return 0b0000;  // from STREAMINFO
    }
}

} // namespace

std::vector<std::uint8_t> encode_flac_pcm16(
    const std::vector<std::int16_t> & interleaved,
    std::uint32_t sample_rate,
    std::uint32_t channels) {
    if (channels < 1 || channels > 2) throw std::invalid_argument("FLAC encoder supports mono and stereo");
    if (sample_rate == 0 || sample_rate >= (1U << 20)) throw std::invalid_argument("FLAC sample rate out of range");
    if (interleaved.size() % channels != 0) throw std::invalid_argument("FLAC input is not whole frames");
    const std::uint64_t total = interleaved.size() / channels;
    if (total >= (1ULL << 36)) throw std::invalid_argument("FLAC input is too long");

    std::vector<std::uint8_t> output = {'f', 'L', 'a', 'C'};
    const std::size_t streaminfo_at = output.size();
    output.resize(output.size() + 4 + 34);

    std::uint32_t min_frame = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t max_frame = 0;
    std::uint64_t frame_number = 0;
    for (std::uint64_t first = 0; first < total; first += kBlockSize, ++frame_number) {
        const auto block = static_cast<std::uint32_t>(std::min<std::uint64_t>(kBlockSize, total - first));
        std::array<std::vector<std::int64_t>, 2> ch;
        for (std::uint32_t c = 0; c < channels; ++c) {
            ch[c].resize(block);
            for (std::uint32_t i = 0; i < block; ++i) {
                ch[c][i] = interleaved[static_cast<std::size_t>((first + i) * channels + c)];
            }
        }

        // Stereo: pick whichever pair of channels is cheapest to code.
        int assignment = static_cast<int>(channels - 1);  // independent
        std::array<std::vector<std::int64_t>, 2> coded = ch;
        std::array<int, 2> bps = {16, 16};
        std::array<Subframe, 2> subframes;
        if (channels == 2) {
            std::vector<std::int64_t> mid(block), side(block);
            for (std::uint32_t i = 0; i < block; ++i) {
                side[i] = ch[0][i] - ch[1][i];
                mid[i] = (ch[0][i] + ch[1][i]) >> 1;
            }
            const auto left = choose_subframe(ch[0], 16);
            const auto right = choose_subframe(ch[1], 16);
            const auto s = choose_subframe(side, 17);
            const auto m = choose_subframe(mid, 16);
            const std::array<std::uint64_t, 4> cost = {
                left.bits + right.bits, left.bits + s.bits, s.bits + right.bits, m.bits + s.bits};
            const auto pick = static_cast<int>(std::min_element(cost.begin(), cost.end()) - cost.begin());
            switch (pick) {
                case 0: assignment = 0b0001; subframes = {left, right}; break;
                case 1: assignment = 0b1000; coded = {ch[0], side}; bps = {16, 17}; subframes = {left, s}; break;
                case 2: assignment = 0b1001; coded = {side, ch[1]}; bps = {17, 16}; subframes = {s, right}; break;
                default: assignment = 0b1010; coded = {mid, side}; bps = {16, 17}; subframes = {m, s}; break;
            }
        } else {
            subframes[0] = choose_subframe(ch[0], 16);
        }

        BitWriter frame;
        frame.write(0x3FFE, 14);
        frame.write(0, 1);  // reserved
        frame.write(0, 1);  // fixed block size
        const bool full = block == kBlockSize;
        frame.write(full ? 0b1100 : 0b0111, 4);
        frame.write(static_cast<std::uint64_t>(sample_rate_code(sample_rate)), 4);
        frame.write(static_cast<std::uint64_t>(assignment), 4);
        frame.write(0b100, 3);  // 16 bits per sample
        frame.write(0, 1);      // reserved
        write_frame_number(frame, frame_number);
        if (!full) frame.write(block - 1, 16);
        frame.write(crc8(frame.bytes().data(), frame.bytes().size()), 8);

        for (std::uint32_t c = 0; c < channels; ++c) write_subframe(frame, subframes[c], coded[c], bps[c]);
        frame.align();
        const auto footer = crc16(frame.bytes().data(), frame.bytes().size());
        frame.write(footer, 16);

        const auto size = static_cast<std::uint32_t>(frame.bytes().size());
        min_frame = std::min(min_frame, size);
        max_frame = std::max(max_frame, size);
        output.insert(output.end(), frame.bytes().begin(), frame.bytes().end());
    }
    if (total == 0) min_frame = 0;

    BitWriter info;
    info.write(1, 1);   // last metadata block
    info.write(0, 7);   // STREAMINFO
    info.write(34, 24);
    const std::uint32_t declared_block = total < kBlockSize ? static_cast<std::uint32_t>(std::max<std::uint64_t>(total, 16)) : kBlockSize;
    info.write(declared_block, 16);
    info.write(declared_block, 16);
    info.write(min_frame, 24);
    info.write(max_frame, 24);
    info.write(sample_rate, 20);
    info.write(channels - 1, 3);
    info.write(16 - 1, 5);
    info.write(total, 36);
    for (int i = 0; i < 16; ++i) info.write(0, 8);  // MD5 not computed
    std::copy(info.bytes().begin(), info.bytes().end(), output.begin() + static_cast<std::ptrdiff_t>(streaminfo_at));
    return output;
}

} // namespace yue2::server
