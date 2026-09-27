// Round-trips the FLAC encoder through an independent decoder for the parts of
// the format it writes: STREAMINFO, fixed-size frames with CRCs, constant,
// verbatim and fixed subframes, partitioned Rice residuals, and all four
// stereo decorrelations. Sample-exact recovery is the only acceptable result.
#include "server/flac.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

class BitReader {
public:
    BitReader(const std::vector<std::uint8_t> & data, std::size_t offset) : data_(data), bit_(offset * 8) {}
    std::uint64_t read(int bits) {
        std::uint64_t value = 0;
        for (int i = 0; i < bits; ++i) {
            if (bit_ / 8 >= data_.size()) throw std::runtime_error("read past end");
            value = (value << 1) | ((data_[bit_ / 8] >> (7 - bit_ % 8)) & 1U);
            ++bit_;
        }
        return value;
    }
    std::int64_t read_signed(int bits) {
        const auto raw = read(bits);
        return (raw & (1ULL << (bits - 1))) ? static_cast<std::int64_t>(raw) - (1LL << bits)
                                            : static_cast<std::int64_t>(raw);
    }
    std::uint64_t read_unary() {
        std::uint64_t zeros = 0;
        while (read(1) == 0) ++zeros;
        return zeros;
    }
    void align() { bit_ = (bit_ + 7) / 8 * 8; }
    std::size_t byte() const { return bit_ / 8; }

private:
    const std::vector<std::uint8_t> & data_;
    std::size_t bit_;
};

std::uint8_t crc8(const std::uint8_t * data, std::size_t size) {
    std::uint8_t crc = 0;
    for (std::size_t i = 0; i < size; ++i) {
        crc ^= data[i];
        for (int b = 0; b < 8; ++b) crc = static_cast<std::uint8_t>((crc & 0x80) ? (crc << 1) ^ 0x07 : crc << 1);
    }
    return crc;
}

std::uint16_t crc16(const std::uint8_t * data, std::size_t size) {
    std::uint16_t crc = 0;
    for (std::size_t i = 0; i < size; ++i) {
        crc ^= static_cast<std::uint16_t>(data[i] << 8);
        for (int b = 0; b < 8; ++b) crc = static_cast<std::uint16_t>((crc & 0x8000) ? (crc << 1) ^ 0x8005 : crc << 1);
    }
    return crc;
}

std::vector<std::int64_t> read_subframe(BitReader & in, std::uint32_t block, int bps) {
    assert(in.read(1) == 0);
    const auto type = in.read(6);
    assert(in.read(1) == 0);  // no wasted bits
    std::vector<std::int64_t> x(block);
    if (type == 0) {
        const auto v = in.read_signed(bps);
        for (auto & s : x) s = v;
    } else if (type == 1) {
        for (auto & s : x) s = in.read_signed(bps);
    } else {
        assert((type & 0b111000) == 0b001000);
        const int order = static_cast<int>(type & 0b111);
        assert(order <= 4);
        for (int i = 0; i < order; ++i) x[i] = in.read_signed(bps);
        assert(in.read(2) == 0);
        const int partition_order = static_cast<int>(in.read(4));
        const std::uint32_t length = block >> partition_order;
        std::size_t i = order;
        for (std::uint32_t p = 0; p < (1U << partition_order); ++p) {
            const auto k = static_cast<int>(in.read(4));
            assert(k != 15);
            const std::uint32_t count = p == 0 ? length - order : length;
            for (std::uint32_t n = 0; n < count; ++n, ++i) {
                const std::uint64_t u = (in.read_unary() << k) | (k ? in.read(k) : 0);
                const std::int64_t r = (u & 1) ? -static_cast<std::int64_t>(u >> 1) - 1 : static_cast<std::int64_t>(u >> 1);
                std::int64_t prediction = 0;
                switch (order) {
                    case 0: prediction = 0; break;
                    case 1: prediction = x[i - 1]; break;
                    case 2: prediction = 2 * x[i - 1] - x[i - 2]; break;
                    case 3: prediction = 3 * x[i - 1] - 3 * x[i - 2] + x[i - 3]; break;
                    default: prediction = 4 * x[i - 1] - 6 * x[i - 2] + 4 * x[i - 3] - x[i - 4]; break;
                }
                x[i] = prediction + r;
            }
        }
        assert(i == block);
    }
    return x;
}

struct Decoded {
    std::uint32_t rate = 0;
    std::uint32_t channels = 0;
    std::uint64_t total = 0;
    std::vector<std::int16_t> samples;
    std::vector<int> assignments;
};

Decoded decode(const std::vector<std::uint8_t> & flac) {
    assert(flac.size() >= 42 && flac[0] == 'f' && flac[1] == 'L' && flac[2] == 'a' && flac[3] == 'C');
    BitReader info(flac, 4);
    assert(info.read(1) == 1 && info.read(7) == 0 && info.read(24) == 34);
    info.read(16);
    info.read(16);
    info.read(24);
    info.read(24);
    Decoded out;
    out.rate = static_cast<std::uint32_t>(info.read(20));
    out.channels = static_cast<std::uint32_t>(info.read(3)) + 1;
    assert(info.read(5) == 15);
    out.total = info.read(36);

    std::size_t at = 4 + 4 + 34;
    while (out.samples.size() < out.total * out.channels) {
        BitReader in(flac, at);
        assert(in.read(14) == 0x3FFE);
        in.read(2);
        const auto size_code = in.read(4);
        in.read(4);
        const auto assignment = static_cast<int>(in.read(4));
        assert(in.read(3) == 0b100);
        in.read(1);
        auto first = in.read(8);
        for (std::uint64_t mask = 0x80; first & mask && mask > 0x20; mask >>= 1) in.read(8);
        std::uint32_t block = 4096;
        if (size_code == 0b0111) block = static_cast<std::uint32_t>(in.read(16)) + 1;
        else assert(size_code == 0b1100);
        const auto header_end = in.byte();
        assert(in.read(8) == crc8(flac.data() + at, header_end - at));

        std::vector<std::vector<std::int64_t>> ch;
        for (std::uint32_t c = 0; c < out.channels; ++c) {
            const bool side = (assignment == 0b1000 && c == 1) || (assignment == 0b1001 && c == 0) ||
                              (assignment == 0b1010 && c == 1);
            ch.push_back(read_subframe(in, block, side ? 17 : 16));
        }
        in.align();
        const auto body_end = in.byte();
        assert(in.read(16) == crc16(flac.data() + at, body_end - at));
        at = in.byte();
        out.assignments.push_back(assignment);

        for (std::uint32_t i = 0; i < block; ++i) {
            std::int64_t l = ch[0][i], r = out.channels == 2 ? ch[1][i] : 0;
            if (assignment == 0b1000) r = l - ch[1][i];
            else if (assignment == 0b1001) { r = ch[1][i]; l = ch[0][i] + r; }
            else if (assignment == 0b1010) {
                const std::int64_t side = ch[1][i];
                const std::int64_t mid = (ch[0][i] << 1) | (side & 1);
                l = (mid + side) >> 1;
                r = (mid - side) >> 1;
            }
            out.samples.push_back(static_cast<std::int16_t>(l));
            if (out.channels == 2) out.samples.push_back(static_cast<std::int16_t>(r));
        }
    }
    assert(at == flac.size());
    return out;
}

void round_trip(const std::vector<std::int16_t> & pcm, std::uint32_t channels, std::uint32_t rate = 48000) {
    const auto flac = yue2::server::encode_flac_pcm16(pcm, rate, channels);
    const auto back = decode(flac);
    assert(back.rate == rate && back.channels == channels);
    assert(back.total == pcm.size() / channels);
    assert(back.samples == pcm);
}

} // namespace

int main() {
    std::mt19937 random(1234);
    const std::uint32_t frames = 4096 * 3 + 123;  // a short last block

    std::vector<std::int16_t> silence(frames * 2, 0);
    round_trip(silence, 2);
    // Silence costs a constant subframe per channel: a few bytes a frame.
    assert(yue2::server::encode_flac_pcm16(silence, 48000, 2).size() < 200);

    std::vector<std::int16_t> sine(frames * 2);
    for (std::uint32_t i = 0; i < frames; ++i) {
        const auto v = static_cast<std::int16_t>(std::lrint(12000 * std::sin(i * 0.031)));
        sine[2 * i] = v;
        sine[2 * i + 1] = static_cast<std::int16_t>(v / 2 + 7);
    }
    round_trip(sine, 2);
    // A smooth signal compresses well below PCM's two bytes a sample.
    assert(yue2::server::encode_flac_pcm16(sine, 48000, 2).size() < sine.size());

    std::uniform_int_distribution<int> any(-32768, 32767);
    std::vector<std::int16_t> noise(frames * 2);
    for (auto & s : noise) s = static_cast<std::int16_t>(any(random));
    round_trip(noise, 2);

    // Full-scale opposites stress the 17-bit side channel.
    std::vector<std::int16_t> extremes(frames * 2);
    for (std::uint32_t i = 0; i < frames; ++i) {
        extremes[2 * i] = (i / 7) % 2 ? 32767 : -32768;
        extremes[2 * i + 1] = (i / 7) % 2 ? -32768 : 32767;
    }
    round_trip(extremes, 2);

    std::vector<std::int16_t> walk(frames * 2);
    std::int32_t a = 0, b = 0;
    std::uniform_int_distribution<int> step(-300, 300);
    for (std::uint32_t i = 0; i < frames; ++i) {
        a = std::max(-32768, std::min(32767, a + step(random)));
        b = std::max(-32768, std::min(32767, b + step(random)));
        walk[2 * i] = static_cast<std::int16_t>(a);
        walk[2 * i + 1] = static_cast<std::int16_t>(b);
    }
    round_trip(walk, 2);

    std::vector<std::int16_t> mono(frames);
    for (std::uint32_t i = 0; i < frames; ++i) mono[i] = static_cast<std::int16_t>(std::lrint(9000 * std::sin(i * 0.2)));
    round_trip(mono, 1);
    round_trip(std::vector<std::int16_t>(200, 5), 1, 44100);
    round_trip({}, 2);

    // Identical channels leave the side channel silent, so a side-based
    // decorrelation has to win.
    std::vector<std::int16_t> dual(frames * 2);
    for (std::uint32_t i = 0; i < frames; ++i) dual[2 * i] = dual[2 * i + 1] = sine[2 * i];
    const auto dual_flac = yue2::server::encode_flac_pcm16(dual, 48000, 2);
    for (const int assignment : decode(dual_flac).assignments) assert(assignment != 0b0001);
    round_trip(dual, 2);
    return 0;
}
