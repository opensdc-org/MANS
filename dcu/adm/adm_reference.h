#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "../../mans_defs.h"

namespace mans::dcu::adm::reference {

constexpr std::size_t kWarpSize = 32;
constexpr std::size_t kChunk1D = 16;
constexpr std::size_t kBlock1D = kWarpSize * kChunk1D;
constexpr std::size_t kTileX = 16;
constexpr std::size_t kTileY = 16;
constexpr std::size_t kTileZ = 16;
constexpr std::uint64_t kAdmThreshold = 3500;

inline std::size_t ceil_div(std::size_t value, std::size_t divisor) {
    return value == 0 ? 0 : (value - 1) / divisor + 1;
}

inline std::size_t effective_ny(const mans::MansParams& params) {
    return params.dims >= 2 ? params.ny : 1;
}

inline std::size_t effective_nz(const mans::MansParams& params) {
    return params.dims == 3 ? params.nz : 1;
}

inline std::size_t checked_mul(std::size_t a, std::size_t b, const char* what) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::runtime_error(std::string("DCU ADM size overflow: ") + what);
    }
    return a * b;
}

inline std::size_t checked_add(std::size_t a, std::size_t b, const char* what) {
    if (a > std::numeric_limits<std::size_t>::max() - b) {
        throw std::runtime_error(std::string("DCU ADM size overflow: ") + what);
    }
    return a + b;
}

inline std::size_t element_count(const mans::MansParams& params) {
    if (params.dims < 1 || params.dims > 3 || params.nx == 0 ||
        (params.dims >= 2 && params.ny == 0) ||
        (params.dims == 3 && params.nz == 0)) {
        throw std::runtime_error("Invalid DCU ADM dimensions");
    }
    return checked_mul(checked_mul(static_cast<std::size_t>(params.nx),
                                   effective_ny(params), "geometry"),
                      effective_nz(params), "geometry");
}

inline std::size_t block_count(std::size_t elements, const mans::MansParams& params) {
    (void)elements;
    if (params.dims == 1) {
        return ceil_div(elements, kBlock1D);
    }
    const std::size_t gx = ceil_div(params.nx, kTileX);
    const std::size_t gy = ceil_div(params.ny, kTileY);
    const std::size_t xy = checked_mul(gx, gy, "block count");
    return params.dims == 2 ? xy : checked_mul(xy, ceil_div(params.nz, kTileZ), "block count");
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
    return {x0, y0, z0,
            x0 < params.nx ? std::min(kTileX, static_cast<std::size_t>(params.nx) - x0) : 0,
            y0 < params.ny ? std::min(kTileY, static_cast<std::size_t>(params.ny) - y0) : 0,
            params.dims == 3 ? (z0 < params.nz ? std::min(kTileZ, static_cast<std::size_t>(params.nz) - z0) : 0) : 1};
}

inline std::size_t flat_index(const BlockShape& shape, std::size_t local,
                              const mans::MansParams& params) {
    const std::size_t plane = shape.sx * shape.sy;
    const std::size_t lz = params.dims == 3 ? local / plane : 0;
    const std::size_t rem = params.dims == 3 ? local - lz * plane : local;
    const std::size_t ly = rem / shape.sx;
    const std::size_t lx = rem - ly * shape.sx;
    return (shape.x0 + lx) + (shape.y0 + ly) * static_cast<std::size_t>(params.nx) +
           (shape.z0 + lz) * static_cast<std::size_t>(params.nx) * effective_ny(params);
}

inline std::size_t flags_size(std::size_t blocks) { return ceil_div(blocks, std::size_t{8}); }

inline void write_le16(std::uint8_t* p, std::uint16_t v) {
    p[0] = static_cast<std::uint8_t>(v);
    p[1] = static_cast<std::uint8_t>(v >> 8);
}

inline void write_le32(std::uint8_t* p, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) p[i] = static_cast<std::uint8_t>(v >> (8 * i));
}

inline std::uint16_t read_le16(const std::uint8_t* p) {
    return static_cast<std::uint16_t>(p[0]) | static_cast<std::uint16_t>(p[1] << 8);
}

inline std::uint32_t read_le32(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0]) |
           (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) |
           (static_cast<std::uint32_t>(p[3]) << 24);
}

template <typename T>
inline void write_raw(std::uint8_t* p, T value) {
    if constexpr (sizeof(T) == 2) write_le16(p, static_cast<std::uint16_t>(value));
    else write_le32(p, static_cast<std::uint32_t>(value));
}

template <typename T>
inline T read_raw(const std::uint8_t* p) {
    if constexpr (sizeof(T) == 2) return static_cast<T>(read_le16(p));
    else return static_cast<T>(read_le32(p));
}

inline std::size_t max_signal_bytes_per_element() { return 4; }

template <typename T>
inline std::size_t max_payload_bytes(std::size_t elements, const mans::MansParams& params) {
    const std::size_t blocks = block_count(elements, params);
    const std::size_t fixed = checked_add(
        checked_add(checked_add(checked_mul(blocks + 1, sizeof(std::int32_t), "offsets"),
                                checked_mul(blocks, sizeof(T), "centers"), "payload"),
                    flags_size(blocks), "payload"),
        elements, "payload");
    const std::size_t max_block = params.dims == 1 ? kBlock1D :
        params.dims == 2 ? kTileX * kTileY : kTileX * kTileY * kTileZ;
    const std::size_t lane_elements = ceil_div(max_block, kWarpSize);
    const std::size_t per_block = checked_mul(
        checked_mul(kWarpSize, lane_elements * max_signal_bytes_per_element(), "signals"),
        blocks, "signals");
    return checked_add(fixed, per_block, "payload");
}

} // namespace mans::dcu::adm::reference
