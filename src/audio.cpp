#include "yue2/audio.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace yue2::audio {
namespace {

std::uint16_t u16le(const std::uint8_t * p) {
    return static_cast<std::uint16_t>(p[0]) |
           static_cast<std::uint16_t>(p[1] << 8);
}

std::uint32_t u32le(const std::uint8_t * p) {
    return static_cast<std::uint32_t>(p[0]) |
           (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) |
           (static_cast<std::uint32_t>(p[3]) << 24);
}

std::int32_t s24le(const std::uint8_t * p) {
    std::uint32_t value = static_cast<std::uint32_t>(p[0]) |
                          (static_cast<std::uint32_t>(p[1]) << 8) |
                          (static_cast<std::uint32_t>(p[2]) << 16);
    if ((value & 0x00800000U) != 0) value |= 0xff000000U;
    return static_cast<std::int32_t>(value);
}

std::vector<std::uint8_t> read_file(const std::filesystem::path & path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("cannot open audio file: " + path.string());
    const auto end = stream.tellg();
    if (end < 0) throw std::runtime_error("cannot determine audio size: " + path.string());
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(end));
    stream.seekg(0);
    if (!bytes.empty() && !stream.read(reinterpret_cast<char *>(bytes.data()), end)) {
        throw std::runtime_error("cannot read audio file: " + path.string());
    }
    return bytes;
}

double bessel_i0(double x) {
    double sum = 1.0;
    double term = 1.0;
    const double y = x * x * 0.25;
    for (int k = 1; k < 30; ++k) {
        term *= y / static_cast<double>(k * k);
        sum += term;
        if (term < sum * 1e-15) break;
    }
    return sum;
}

} // namespace

MonoAudio decode_wav_mono(const std::uint8_t * bytes, std::size_t byte_count) {
    constexpr const char * label = "memory WAV";
    if (!bytes || byte_count < 12 || std::memcmp(bytes, "RIFF", 4) != 0 ||
        std::memcmp(bytes + 8, "WAVE", 4) != 0) {
        throw std::runtime_error(std::string(label) + " is not a RIFF/WAVE file");
    }

    std::uint16_t format = 0;
    std::uint16_t channels = 0;
    std::uint16_t bits = 0;
    std::uint32_t rate = 0;
    const std::uint8_t * payload = nullptr;
    std::size_t payload_size = 0;

    for (std::size_t offset = 12; offset + 8 <= byte_count;) {
        const auto * id = bytes + offset;
        const std::uint32_t declared = u32le(id + 4);
        offset += 8;
        if (declared > byte_count - offset) {
            throw std::runtime_error(std::string(label) + ": truncated WAV chunk");
        }
        if (std::memcmp(id, "fmt ", 4) == 0) {
            if (declared < 16) throw std::runtime_error(std::string(label) + ": invalid fmt chunk");
            format = u16le(bytes + offset);
            channels = u16le(bytes + offset + 2);
            rate = u32le(bytes + offset + 4);
            bits = u16le(bytes + offset + 14);
            if (format == 0xfffe && declared >= 40) {
                const auto subformat = u16le(bytes + offset + 24);
                if (subformat == 1 || subformat == 3) format = subformat;
            }
        } else if (std::memcmp(id, "data", 4) == 0 && payload == nullptr) {
            payload = bytes + offset;
            payload_size = declared;
        }
        offset += declared + (declared & 1U);
    }

    if (channels == 0 || rate == 0 || payload == nullptr || payload_size == 0) {
        throw std::runtime_error(std::string(label) + ": WAV is missing format or audio data");
    }
    const bool pcm = format == 1;
    const bool ieee = format == 3;
    if ((!pcm && !ieee) || (pcm && bits != 16 && bits != 24 && bits != 32) ||
        (ieee && bits != 32 && bits != 64)) {
        throw std::runtime_error(std::string(label) + ": unsupported WAV encoding");
    }

    const std::size_t bytes_per_sample = bits / 8;
    const std::size_t bytes_per_frame = bytes_per_sample * channels;
    if (bytes_per_frame == 0 || payload_size % bytes_per_frame != 0) {
        throw std::runtime_error(std::string(label) + ": invalid WAV data length");
    }
    const std::size_t frames = payload_size / bytes_per_frame;
    MonoAudio result;
    result.sample_rate = static_cast<std::int32_t>(rate);
    result.samples.resize(frames);

    for (std::size_t frame = 0; frame < frames; ++frame) {
        double mixed = 0.0;
        for (std::uint16_t channel = 0; channel < channels; ++channel) {
            const auto * p = payload + (frame * channels + channel) * bytes_per_sample;
            double value = 0.0;
            if (ieee && bits == 32) {
                float sample = 0.0F;
                std::memcpy(&sample, p, sizeof(sample));
                value = sample;
            } else if (ieee) {
                std::memcpy(&value, p, sizeof(value));
            } else if (bits == 16) {
                value = static_cast<std::int16_t>(u16le(p)) / 32768.0;
            } else if (bits == 24) {
                value = s24le(p) / 8388608.0;
            } else {
                std::int32_t sample = 0;
                const auto raw = u32le(p);
                std::memcpy(&sample, &raw, sizeof(sample));
                value = sample / 2147483648.0;
            }
            if (!std::isfinite(value)) {
                throw std::runtime_error(std::string(label) + ": non-finite audio sample");
            }
            mixed += value;
        }
        result.samples[frame] = static_cast<float>(mixed / channels);
    }
    if (result.samples.size() < 1025) {
        throw std::runtime_error(std::string(label) + ": audio must contain at least 1025 samples");
    }
    return result;
}

MonoAudio read_wav_mono(const std::filesystem::path & path) {
    const auto bytes = read_file(path);
    try {
        return decode_wav_mono(bytes.data(), bytes.size());
    } catch (const std::runtime_error & error) {
        throw std::runtime_error(path.string() + ": " + error.what());
    }
}

std::vector<float> resample_sinc(
    const std::vector<float> & input,
    std::int32_t input_rate,
    std::int32_t output_rate) {
    if (input.empty() || input_rate <= 0 || output_rate <= 0) {
        throw std::invalid_argument("invalid resampler input");
    }
    if (input_rate == output_rate) return input;

    constexpr int taps = 64;
    constexpr int phases = 256;
    constexpr int half = taps / 2;
    constexpr double pi = 3.14159265358979323846;
    constexpr double beta = 9.0;
    const double ratio = static_cast<double>(output_rate) / input_rate;
    const double cutoff = 0.5 * std::min(ratio, 1.0);
    const double inverse_i0 = 1.0 / bessel_i0(beta);

    std::array<std::array<float, taps>, phases + 1> table{};
    for (int phase = 0; phase <= phases; ++phase) {
        const double fraction = static_cast<double>(phase) / phases;
        for (int tap = 0; tap < taps; ++tap) {
            const double distance = fraction + half - 1 - tap;
            const double sinc = std::abs(distance) < 1e-12
                ? 2.0 * cutoff
                : std::sin(2.0 * pi * cutoff * distance) / (pi * distance);
            const double position = distance / half;
            const double window = std::abs(position) > 1.0
                ? 0.0
                : bessel_i0(beta * std::sqrt(std::max(0.0, 1.0 - position * position))) * inverse_i0;
            table[phase][tap] = static_cast<float>(sinc * window);
        }
    }

    const auto output_count = static_cast<std::size_t>(std::floor(input.size() * ratio));
    std::vector<float> output(output_count);
    for (std::size_t i = 0; i < output_count; ++i) {
        const double center = static_cast<double>(i) / ratio;
        const auto center_index = static_cast<std::int64_t>(std::floor(center));
        const double phase_position = (center - center_index) * phases;
        const auto phase = std::min(static_cast<int>(phase_position), phases - 1);
        const float mix = static_cast<float>(phase_position - phase);
        const auto base = center_index - half + 1;
        double sum = 0.0;
        double weight = 0.0;
        for (int tap = 0; tap < taps; ++tap) {
            const float coefficient = table[phase][tap] + mix * (table[phase + 1][tap] - table[phase][tap]);
            const auto source = std::clamp<std::int64_t>(base + tap, 0, static_cast<std::int64_t>(input.size() - 1));
            sum += input[static_cast<std::size_t>(source)] * coefficient;
            weight += coefficient;
        }
        output[i] = std::abs(weight) > 1e-12 ? static_cast<float>(sum / weight) : 0.0F;
    }
    return output;
}

MonoAudio read_wav_mono_24k(const std::filesystem::path & path) {
    auto audio = read_wav_mono(path);
    if (audio.sample_rate != transcription_sample_rate) {
        audio.samples = resample_sinc(audio.samples, audio.sample_rate, transcription_sample_rate);
        audio.sample_rate = transcription_sample_rate;
    }
    return audio;
}

void write_wav_float(
    const std::filesystem::path & path,
    const std::vector<float> & interleaved_samples,
    std::int32_t sample_rate,
    std::int32_t channels) {
    const auto bytes = encode_wav_float(interleaved_samples, sample_rate, channels);
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    std::ofstream stream(path, std::ios::binary);
    if (!stream || !stream.write(
            reinterpret_cast<const char *>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()))) {
        throw std::runtime_error("cannot write audio file: " + path.string());
    }
}

std::vector<std::uint8_t> encode_wav_float(
    const std::vector<float> & interleaved_samples,
    std::int32_t sample_rate,
    std::int32_t channels) {
    if (interleaved_samples.empty() || sample_rate <= 0 || channels <= 0 ||
        interleaved_samples.size() % static_cast<std::size_t>(channels) != 0) {
        throw std::invalid_argument("invalid WAV output");
    }
    constexpr std::uint64_t header_bytes = 36;
    const auto data_bytes = static_cast<std::uint64_t>(interleaved_samples.size()) * sizeof(float);
    if (data_bytes > std::numeric_limits<std::uint32_t>::max() - header_bytes) {
        throw std::invalid_argument("WAV output exceeds the RIFF size limit");
    }
    for (const auto sample : interleaved_samples) {
        if (!std::isfinite(sample)) throw std::invalid_argument("non-finite WAV output sample");
    }
    const auto channel_bytes = static_cast<std::uint64_t>(channels) * sizeof(float);
    const auto byte_rate = static_cast<std::uint64_t>(sample_rate) * channel_bytes;
    if (channels > std::numeric_limits<std::uint16_t>::max() ||
        channel_bytes > std::numeric_limits<std::uint16_t>::max() ||
        byte_rate > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("WAV format values exceed RIFF limits");
    }
    const auto channel_count = static_cast<std::uint16_t>(channels);
    const auto block_align = static_cast<std::uint16_t>(channel_bytes);
    std::vector<std::uint8_t> output;
    output.reserve(static_cast<std::size_t>(44 + data_bytes));
    const auto append = [&output](const char * value, std::size_t size) {
        output.insert(output.end(), value, value + size);
    };
    const auto append_u16 = [&output](std::uint16_t value) {
        output.push_back(static_cast<std::uint8_t>(value));
        output.push_back(static_cast<std::uint8_t>(value >> 8));
    };
    const auto append_u32 = [&append_u16](std::uint32_t value) {
        append_u16(static_cast<std::uint16_t>(value));
        append_u16(static_cast<std::uint16_t>(value >> 16));
    };
    append("RIFF", 4);
    append_u32(static_cast<std::uint32_t>(header_bytes + data_bytes));
    append("WAVE", 4);
    append("fmt ", 4);
    append_u32(16);
    append_u16(3); // WAVE_FORMAT_IEEE_FLOAT
    append_u16(channel_count);
    append_u32(static_cast<std::uint32_t>(sample_rate));
    append_u32(static_cast<std::uint32_t>(byte_rate));
    append_u16(block_align);
    append_u16(32);
    append("data", 4);
    append_u32(static_cast<std::uint32_t>(data_bytes));
    for (const auto sample : interleaved_samples) {
        std::uint32_t bits = 0;
        static_assert(sizeof(bits) == sizeof(sample), "unexpected float size");
        std::memcpy(&bits, &sample, sizeof(bits));
        append_u32(bits);
    }
    return output;
}

} // namespace yue2::audio
