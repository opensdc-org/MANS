#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "../../mans_defs.h"

namespace mans::nv::adm::reference {

constexpr std::size_t kWarpSize = 32;
constexpr std::size_t kChunk1D = 16;
constexpr std::size_t kBlock1D = kWarpSize * kChunk1D;
constexpr std::size_t kTileX = 16;
constexpr std::size_t kTileY = 16;
constexpr std::size_t kTileZ = 16;
constexpr std::uint64_t kAdmThreshold = 3500;

inline bool should_use_adm(std::uint64_t min_value,
                           std::uint64_t max_value,
                           std::size_t count) {
    return count != 0 && (max_value - min_value) < kAdmThreshold;
}

inline std::size_t max_adm_signal_bytes_per_element() {
    // range < 3500 permits at most ceil(3499 / 126) = 28 bits.
    return (28 + 7) / 8;
}

struct BlockShape {
    std::size_t x0 = 0;
    std::size_t y0 = 0;
    std::size_t z0 = 0;
    std::size_t sx = 0;
    std::size_t sy = 0;
    std::size_t sz = 0;

    std::size_t elements() const { return sx * sy * sz; }
};

inline std::size_t checked_add(std::size_t a, std::size_t b, const char* what) {
    if (a > std::numeric_limits<std::size_t>::max() - b) {
        throw std::runtime_error(std::string("ADM size overflow: ") + what);
    }
    return a + b;
}

inline std::size_t checked_mul(std::size_t a, std::size_t b, const char* what) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::runtime_error(std::string("ADM size overflow: ") + what);
    }
    return a * b;
}

inline std::size_t ceil_div(std::size_t value, std::size_t divisor) {
    return value == 0 ? 0 : (value - 1) / divisor + 1;
}

inline std::size_t effective_ny(const mans::MansParams& params) {
    return params.dims >= 2 ? params.ny : 1;
}

inline std::size_t effective_nz(const mans::MansParams& params) {
    return params.dims == 3 ? params.nz : 1;
}

inline std::size_t element_count(const mans::MansParams& params) {
    if (params.dims < 1 || params.dims > 3 || params.nx == 0 ||
        (params.dims >= 2 && params.ny == 0) ||
        (params.dims == 3 && params.nz == 0)) {
        throw std::runtime_error("Invalid ADM dimensions");
    }
    return checked_mul(
        checked_mul(static_cast<std::size_t>(params.nx), effective_ny(params), "geometry"),
        effective_nz(params),
        "geometry");
}

inline std::size_t block_count(std::size_t num_elements, const mans::MansParams& params) {
    if (params.dims == 1) {
        return ceil_div(num_elements, kBlock1D);
    }
    const std::size_t gx = ceil_div(params.nx, kTileX);
    const std::size_t gy = ceil_div(params.ny, kTileY);
    const std::size_t count = checked_mul(gx, gy, "block count");
    return params.dims == 2 ? count : checked_mul(count, ceil_div(params.nz, kTileZ), "block count");
}

inline BlockShape block_shape(std::size_t block, const mans::MansParams& params) {
    if (params.dims == 1) {
        const std::size_t x0 = block * kBlock1D;
        const std::size_t count = element_count(params);
        return {x0, 0, 0, x0 < count ? std::min(kBlock1D, count - x0) : 0, 1, 1};
    }

    const std::size_t gx = ceil_div(params.nx, kTileX);
    const std::size_t gy = ceil_div(params.ny, kTileY);
    const std::size_t bx = block % gx;
    const std::size_t t = block / gx;
    const std::size_t by = t % gy;
    const std::size_t bz = params.dims == 3 ? t / gy : 0;
    const std::size_t x0 = bx * kTileX;
    const std::size_t y0 = by * kTileY;
    const std::size_t z0 = bz * kTileZ;
    return {
        x0,
        y0,
        z0,
        x0 < params.nx ? std::min(kTileX, static_cast<std::size_t>(params.nx) - x0) : 0,
        y0 < params.ny ? std::min(kTileY, static_cast<std::size_t>(params.ny) - y0) : 0,
        params.dims == 3
            ? (z0 < params.nz ? std::min(kTileZ, static_cast<std::size_t>(params.nz) - z0) : 0)
            : 1};
}

inline std::size_t flat_index(const BlockShape& shape,
                              std::size_t local,
                              const mans::MansParams& params) {
    const std::size_t plane = shape.sx * shape.sy;
    const std::size_t lz = params.dims == 3 ? local / plane : 0;
    const std::size_t rem = params.dims == 3 ? local - lz * plane : local;
    const std::size_t ly = rem / shape.sx;
    const std::size_t lx = rem - ly * shape.sx;
    return (shape.x0 + lx) +
           (shape.y0 + ly) * static_cast<std::size_t>(params.nx) +
           (shape.z0 + lz) * static_cast<std::size_t>(params.nx) * effective_ny(params);
}

inline void write_le16(std::uint8_t* dst, std::uint16_t value) {
    dst[0] = static_cast<std::uint8_t>(value & 0xffu);
    dst[1] = static_cast<std::uint8_t>(value >> 8);
}

inline void write_le32(std::uint8_t* dst, std::uint32_t value) {
    for (int i = 0; i < 4; ++i) {
        dst[i] = static_cast<std::uint8_t>(value >> (8 * i));
    }
}

inline std::uint16_t read_le16(const std::uint8_t* src) {
    return static_cast<std::uint16_t>(src[0]) |
           static_cast<std::uint16_t>(static_cast<std::uint16_t>(src[1]) << 8);
}

inline std::uint32_t read_le32(const std::uint8_t* src) {
    return static_cast<std::uint32_t>(src[0]) |
           (static_cast<std::uint32_t>(src[1]) << 8) |
           (static_cast<std::uint32_t>(src[2]) << 16) |
           (static_cast<std::uint32_t>(src[3]) << 24);
}

inline std::size_t flags_size(std::size_t blocks) {
    return ceil_div(blocks, std::size_t{8});
}

template <typename T>
inline void write_raw(std::uint8_t* dst, T value) {
    if constexpr (sizeof(T) == 2) {
        write_le16(dst, static_cast<std::uint16_t>(value));
    } else {
        write_le32(dst, static_cast<std::uint32_t>(value));
    }
}

template <typename T>
inline T read_raw(const std::uint8_t* src) {
    if constexpr (sizeof(T) == 2) {
        return static_cast<T>(read_le16(src));
    } else {
        return static_cast<T>(read_le32(src));
    }
}

template <typename T>
inline std::uint8_t make_code(T value, T center, std::size_t& signal_bits) {
    const std::uint64_t diff = value >= center
        ? static_cast<std::uint64_t>(value - center)
        : static_cast<std::uint64_t>(center - value);
    const std::size_t output_bits = value == center ? 1 : (diff + 125) / 126;
    const std::uint64_t residual =
        diff + 126 - static_cast<std::uint64_t>(output_bits) * 126;
    signal_bits += output_bits;
    if (value == center) {
        return 1;
    }
    return static_cast<std::uint8_t>(residual * 2 + (value > center ? 0 : 1));
}

template <typename T>
inline T decode_value(std::uint8_t code, std::uint8_t signal, T center) {
    const std::uint64_t base = (code & 1u) ? (code - 1u) / 2u : code / 2u;
    const std::uint64_t diff = base + static_cast<std::uint64_t>(signal) * 126;
    const std::uint64_t center_value = static_cast<std::uint64_t>(center);
    const std::uint64_t max_value = static_cast<std::uint64_t>(std::numeric_limits<T>::max());
    if ((code & 1u) != 0) {
        if (diff > center_value) {
            throw std::runtime_error("ADM value underflow");
        }
        return static_cast<T>(center_value - diff);
    }
    if (diff > max_value - center_value) {
        throw std::runtime_error("ADM value overflow");
    }
    return static_cast<T>(center_value + diff);
}

template <typename T>
std::vector<std::uint8_t> encode(const T* input,
                                 std::size_t num_elements,
                                 const mans::MansParams& params) {
    static_assert(std::is_same_v<T, std::uint16_t> || std::is_same_v<T, std::uint32_t>);
    if (!input || num_elements == 0) {
        throw std::runtime_error("ADM input is empty");
    }
    if (element_count(params) != num_elements) {
        throw std::runtime_error("ADM geometry does not match input length");
    }

    const std::size_t blocks = block_count(num_elements, params);
    const std::size_t offsets_bytes = checked_mul(blocks + 1, sizeof(std::int32_t), "offsets");
    const std::size_t centers_bytes = checked_mul(blocks, sizeof(T), "centers");
    const std::size_t flags_bytes = flags_size(blocks);
    const std::size_t codes_bytes = num_elements;
    std::vector<std::int32_t> offsets(blocks + 1, 0);
    std::vector<T> centers(blocks, 0);
    std::vector<std::uint8_t> flags(flags_bytes, 0xff);
    std::vector<std::uint8_t> codes(num_elements, 0);
    std::vector<std::uint8_t> signals;

    for (std::size_t block = 0; block < blocks; ++block) {
        const BlockShape shape = block_shape(block, params);
        const std::size_t block_elements = shape.elements();
        const std::size_t per_lane = params.dims == 1 ? kChunk1D : ceil_div(block_elements, kWarpSize);
        T min_value = std::numeric_limits<T>::max();
        T max_value = 0;
        std::uint64_t sum = 0;
        for (std::size_t local = 0; local < block_elements; ++local) {
            const T value = input[flat_index(shape, local, params)];
            min_value = std::min(min_value, value);
            max_value = std::max(max_value, value);
            sum += static_cast<std::uint64_t>(value);
        }

        const bool use_adm = should_use_adm(static_cast<std::uint64_t>(min_value),
                                             static_cast<std::uint64_t>(max_value),
                                             block_elements);
        if (!use_adm) {
            flags[block >> 3] &= static_cast<std::uint8_t>(~(1u << (7u - (block & 7u))));
        } else {
            centers[block] = static_cast<T>(sum / block_elements);
        }

        std::vector<std::vector<std::uint8_t>> lane_data(kWarpSize);
        std::size_t signal_length = 0;
        for (std::size_t lane = 0; lane < kWarpSize; ++lane) {
            const std::size_t first = lane * per_lane;
            const std::size_t last = std::min(first + per_lane, block_elements);
            std::size_t signal_bits = 0;
            if (use_adm) {
                for (std::size_t local = first; local < last; ++local) {
                    const std::size_t q = flat_index(shape, local, params);
                    codes[q] = make_code(input[q], centers[block], signal_bits);
                }
                lane_data[lane].assign((signal_bits + 7) / 8, 0);
                std::size_t bit = 0;
                for (std::size_t local = first; local < last; ++local) {
                    const std::size_t q = flat_index(shape, local, params);
                    const T value = input[q];
                    const std::uint64_t diff = value >= centers[block]
                        ? static_cast<std::uint64_t>(value - centers[block])
                        : static_cast<std::uint64_t>(centers[block] - value);
                    const std::size_t count = value == centers[block] ? 1 : (diff + 125) / 126;
                    lane_data[lane][bit / 8] |= static_cast<std::uint8_t>(1u << (7u - (bit & 7u)));
                    bit += count;
                }
                signal_length = std::max(signal_length, lane_data[lane].size());
            } else {
                // CPU ADM reserves a fixed per-lane RAW region, including tail lanes.
                lane_data[lane].assign(per_lane * sizeof(T), 0);
                for (std::size_t local = first; local < last; ++local) {
                    write_raw(lane_data[lane].data() + (local - first) * sizeof(T),
                              input[flat_index(shape, local, params)]);
                }
                signal_length = per_lane * sizeof(T);
            }
        }

        if (signal_length > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
            throw std::runtime_error("ADM signal length exceeds 32-bit range");
        }
        offsets[block + 1] = static_cast<std::int32_t>(offsets[block] + signal_length);
        const std::size_t old_size = signals.size();
        signals.resize(checked_add(old_size, checked_mul(signal_length, kWarpSize, "signals"), "signals"), 0);
        for (std::size_t lane = 0; lane < kWarpSize; ++lane) {
            if (use_adm) {
                lane_data[lane].resize(signal_length, 0);
                const std::size_t actual_bits = [&]() {
                    std::size_t bits = 0;
                    const std::size_t first = lane * per_lane;
                    const std::size_t last = std::min(first + per_lane, block_elements);
                    for (std::size_t local = first; local < last; ++local) {
                        const T value = input[flat_index(shape, local, params)];
                        const std::uint64_t diff = value >= centers[block]
                            ? static_cast<std::uint64_t>(value - centers[block])
                            : static_cast<std::uint64_t>(centers[block] - value);
                        bits += value == centers[block] ? 1 : (diff + 125) / 126;
                    }
                    return bits;
                }();
                if (actual_bits < signal_length * 8) {
                    const std::size_t byte = actual_bits / 8;
                    lane_data[lane][byte] |= static_cast<std::uint8_t>(0xffu >> (actual_bits & 7u));
                }
            }
            std::memcpy(signals.data() + old_size + lane * signal_length,
                        lane_data[lane].data(), lane_data[lane].size());
        }
    }

    const std::size_t total = checked_add(
        checked_add(checked_add(checked_add(offsets_bytes, centers_bytes, "payload"),
                                flags_bytes, "payload"),
                    codes_bytes, "payload"),
        signals.size(), "payload");
    std::vector<std::uint8_t> output(total, 0);
    std::memcpy(output.data(), offsets.data(), offsets_bytes);
    for (std::size_t i = 0; i < blocks; ++i) {
        write_raw(output.data() + offsets_bytes + i * sizeof(T), centers[i]);
    }
    std::memcpy(output.data() + offsets_bytes + centers_bytes, flags.data(), flags_bytes);
    std::memcpy(output.data() + offsets_bytes + centers_bytes + flags_bytes, codes.data(), codes_bytes);
    std::memcpy(output.data() + offsets_bytes + centers_bytes + flags_bytes + codes_bytes,
                signals.data(), signals.size());
    return output;
}

template <typename T>
void decode(const std::uint8_t* input,
            std::size_t input_size,
            T* output,
            std::size_t num_elements,
            const mans::MansParams& params) {
    static_assert(std::is_same_v<T, std::uint16_t> || std::is_same_v<T, std::uint32_t>);
    if (!input || !output || num_elements == 0) {
        throw std::runtime_error("ADM input/output is empty");
    }
    if (element_count(params) != num_elements) {
        throw std::runtime_error("ADM geometry does not match output length");
    }

    const std::size_t blocks = block_count(num_elements, params);
    const std::size_t offsets_bytes = checked_mul(blocks + 1, sizeof(std::int32_t), "offsets");
    const std::size_t centers_bytes = checked_mul(blocks, sizeof(T), "centers");
    const std::size_t flags_bytes = flags_size(blocks);
    const std::size_t fixed = checked_add(checked_add(checked_add(offsets_bytes, centers_bytes, "payload"),
                                                     flags_bytes, "payload"),
                                         num_elements, "payload");
    if (input_size < fixed) {
        throw std::runtime_error("Truncated ADM payload");
    }

    std::vector<std::int32_t> offsets(blocks + 1, 0);
    std::memcpy(offsets.data(), input, offsets_bytes);
    if (offsets.front() != 0) {
        throw std::runtime_error("Invalid ADM offsets");
    }
    for (std::size_t i = 0; i < blocks; ++i) {
        if (offsets[i] < 0 || offsets[i + 1] < offsets[i]) {
            throw std::runtime_error("Invalid ADM offsets");
        }
    }
    const std::size_t signal_bytes = checked_mul(static_cast<std::size_t>(offsets.back()), kWarpSize, "signals");
    if (checked_add(fixed, signal_bytes, "payload") != input_size) {
        throw std::runtime_error("ADM signal payload length mismatch");
    }

    std::vector<T> centers(blocks, 0);
    for (std::size_t i = 0; i < blocks; ++i) {
        centers[i] = read_raw<T>(input + offsets_bytes + i * sizeof(T));
    }
    const auto* flags = input + offsets_bytes + centers_bytes;
    const auto* codes = flags + flags_bytes;
    const auto* signals = codes + num_elements;

    for (std::size_t block = 0; block < blocks; ++block) {
        const BlockShape shape = block_shape(block, params);
        const std::size_t block_elements = shape.elements();
        const std::size_t per_lane = params.dims == 1 ? kChunk1D : ceil_div(block_elements, kWarpSize);
        const std::size_t lane_length = static_cast<std::size_t>(offsets[block + 1] - offsets[block]);
        const bool use_adm = (flags[block >> 3] & (1u << (7u - (block & 7u)))) != 0;

        for (std::size_t lane = 0; lane < kWarpSize; ++lane) {
            const std::size_t first = lane * per_lane;
            const std::size_t last = first < block_elements
                ? std::min(first + per_lane, block_elements)
                : first;
            const auto* lane_signal = signals + offsets[block] * kWarpSize + lane * lane_length;
            if (!use_adm) {
                if (lane_length < (last - first) * sizeof(T)) {
                    throw std::runtime_error("Invalid RAW ADM block length");
                }
                for (std::size_t local = first; local < last; ++local) {
                    output[flat_index(shape, local, params)] =
                        read_raw<T>(lane_signal + (local - first) * sizeof(T));
                }
                continue;
            }

            std::vector<std::uint8_t> lane_values(last - first, 0);
            int symbol = -1;
            for (std::size_t byte = 0; byte < lane_length && symbol < static_cast<int>(lane_values.size()); ++byte) {
                for (int bit = 7; bit >= 0 && symbol < static_cast<int>(lane_values.size()); --bit) {
                    if (lane_signal[byte] & (1u << bit)) {
                        ++symbol;
                    } else if (symbol >= 0) {
                        ++lane_values[static_cast<std::size_t>(symbol)];
                    } else {
                        throw std::runtime_error("Invalid ADM signal stream");
                    }
                }
            }
            if (symbol < static_cast<int>(lane_values.size()) - 1) {
                throw std::runtime_error("Invalid ADM signal stream");
            }
            for (std::size_t local = first; local < last; ++local) {
                const std::size_t q = flat_index(shape, local, params);
                output[q] = decode_value(codes[q], lane_values[local - first], centers[block]);
            }
        }
    }
}

template <typename T>
std::size_t max_payload_bytes(std::size_t num_elements, const mans::MansParams& params) {
    const std::size_t blocks = block_count(num_elements, params);
    const std::size_t fixed = checked_add(
        checked_add(checked_add(checked_mul(blocks + 1, sizeof(std::int32_t), "offsets"),
                                checked_mul(blocks, sizeof(T), "centers"), "payload"),
                    flags_size(blocks), "payload"),
        num_elements, "payload");
    const std::size_t max_block_elements = params.dims == 1
        ? kBlock1D
        : params.dims == 2 ? kTileX * kTileY : kTileX * kTileY * kTileZ;
    const std::size_t max_lane_elements = ceil_div(max_block_elements, kWarpSize);
    const std::size_t max_signal_per_lane = max_lane_elements *
        max_adm_signal_bytes_per_element();
    return checked_add(fixed, checked_mul(checked_mul(blocks, kWarpSize, "signals"),
                                          max_signal_per_lane, "signals"), "payload");
}

} // namespace mans::nv::adm::reference
