#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "../../mans_defs.h"

namespace mans::nv::adm::kernel {

constexpr int kWarpSize = 32;
constexpr int kChunk1D = 16;
constexpr int kBlock1D = kWarpSize * kChunk1D;
constexpr int kTile = 16;
constexpr unsigned long long kThreshold = 3500;

template <typename T>
__device__ inline T load_value(const T* input, std::size_t index) {
    return input[index];
}

template <typename T>
__device__ inline void store_le(std::uint8_t* dst, T value) {
    using U = std::conditional_t<sizeof(T) == 2, std::uint16_t, std::uint32_t>;
    const U v = static_cast<U>(value);
    for (int i = 0; i < static_cast<int>(sizeof(T)); ++i) {
        dst[i] = static_cast<std::uint8_t>(v >> (8 * i));
    }
}

template <typename T>
__device__ inline std::size_t flat_index(std::size_t local,
                                          std::size_t x0,
                                          std::size_t y0,
                                          std::size_t z0,
                                          std::size_t sx,
                                          std::size_t sy,
                                          const mans::MansParams& params) {
    const std::size_t plane = sx * sy;
    const std::size_t lz = params.dims == 3 ? local / plane : 0;
    const std::size_t rem = params.dims == 3 ? local - lz * plane : local;
    const std::size_t ly = rem / sx;
    const std::size_t lx = rem - ly * sx;
    return (x0 + lx) +
           (y0 + ly) * static_cast<std::size_t>(params.nx) +
           (z0 + lz) * static_cast<std::size_t>(params.nx) *
               (params.dims >= 2 ? static_cast<std::size_t>(params.ny) : 1);
}

template <typename T>
__global__ void encode(const T* input,
                       std::size_t num_elements,
                       mans::MansParams params,
                       std::uint8_t* codes,
                       std::uint32_t* flags_words,
                       T* centers,
                       std::uint32_t* signal_lengths,
                       std::uint8_t* lane_signals,
                       std::size_t lane_signal_stride) {
    __shared__ unsigned long long sum;
    __shared__ unsigned long long count;
    __shared__ unsigned long long min_value;
    __shared__ unsigned long long max_value;
    __shared__ unsigned int max_signal_bytes;

    const int lane = threadIdx.x;
    const std::size_t block = static_cast<std::size_t>(blockIdx.x);
    const int dims = static_cast<int>(params.dims);
    const std::size_t nx = params.nx;
    const std::size_t ny = dims >= 2 ? params.ny : 1;
    const std::size_t nz = dims == 3 ? params.nz : 1;

    std::size_t x0 = 0;
    std::size_t y0 = 0;
    std::size_t z0 = 0;
    std::size_t sx = 0;
    std::size_t sy = 1;
    std::size_t sz = 1;
    if (dims == 1) {
        x0 = block * kBlock1D;
        sx = x0 < num_elements ? min(static_cast<std::size_t>(kBlock1D), num_elements - x0) : 0;
    } else {
        const std::size_t gx = (nx + kTile - 1) / kTile;
        const std::size_t gy = (ny + kTile - 1) / kTile;
        const std::size_t bx = block % gx;
        const std::size_t t = block / gx;
        const std::size_t by = t % gy;
        const std::size_t bz = dims == 3 ? t / gy : 0;
        x0 = bx * kTile;
        y0 = by * kTile;
        z0 = bz * kTile;
        sx = x0 < nx ? min(static_cast<std::size_t>(kTile), nx - x0) : 0;
        sy = y0 < ny ? min(static_cast<std::size_t>(kTile), ny - y0) : 0;
        sz = dims == 3 ? (z0 < nz ? min(static_cast<std::size_t>(kTile), nz - z0) : 0) : 1;
    }

    const std::size_t block_elements = sx * sy * sz;
    const std::size_t per_lane = dims == 1
        ? static_cast<std::size_t>(kChunk1D)
        : (block_elements + kWarpSize - 1) / kWarpSize;
    const std::size_t first = static_cast<std::size_t>(lane) * per_lane;
    const std::size_t last = min(first + per_lane, block_elements);

    if (lane == 0) {
        sum = 0;
        count = 0;
        min_value = ~0ull;
        max_value = 0;
        max_signal_bytes = 0;
    }
    __syncthreads();

    unsigned long long local_sum = 0;
    unsigned long long local_count = 0;
    unsigned long long local_min = ~0ull;
    unsigned long long local_max = 0;
    for (std::size_t local = first; local < last; ++local) {
        const T value = load_value(input, flat_index<T>(local, x0, y0, z0, sx, sy, params));
        const unsigned long long wide = static_cast<unsigned long long>(value);
        local_sum += wide;
        ++local_count;
        local_min = min(local_min, wide);
        local_max = max(local_max, wide);
    }
    atomicAdd(&sum, local_sum);
    atomicAdd(&count, local_count);
    if (local_count != 0) {
        atomicMin(&min_value, local_min);
        atomicMax(&max_value, local_max);
    }
    __syncthreads();

    const bool use_adm = count != 0 && (max_value - min_value) < kThreshold;
    const T center = use_adm ? static_cast<T>(sum / count) : static_cast<T>(0);
    if (lane == 0) {
        centers[block] = center;
        if (use_adm) {
            atomicOr(&flags_words[block >> 5], 1u << (31 - (block & 31)));
        }
    }
    __syncthreads();

    std::uint8_t* signal = lane_signals + (block * kWarpSize + static_cast<std::size_t>(lane)) * lane_signal_stride;
    for (std::size_t i = 0; i < lane_signal_stride; ++i) {
        signal[i] = 0;
    }

    std::size_t signal_bits = 0;
    for (std::size_t local = first; local < last; ++local) {
        const std::size_t q = flat_index<T>(local, x0, y0, z0, sx, sy, params);
        const T value = load_value(input, q);
        if (!use_adm) {
            store_le(signal + (local - first) * sizeof(T), value);
            continue;
        }

        const unsigned long long value_wide = static_cast<unsigned long long>(value);
        const unsigned long long center_wide = static_cast<unsigned long long>(center);
        const unsigned long long diff = value_wide >= center_wide
            ? value_wide - center_wide
            : center_wide - value_wide;
        const std::size_t output_bits = value == center ? 1 : (diff + 125) / 126;
        const unsigned long long residual =
            diff + 126 - static_cast<unsigned long long>(output_bits) * 126;
        codes[q] = value == center
            ? 1
            : static_cast<std::uint8_t>(residual * 2 + (value > center ? 0 : 1));
        signal[signal_bits >> 3] |= static_cast<std::uint8_t>(1u << (7 - (signal_bits & 7)));
        signal_bits += output_bits;
    }

    const unsigned int signal_bytes = static_cast<unsigned int>((signal_bits + 7) / 8);
    atomicMax(&max_signal_bytes, signal_bytes);
    __syncthreads();

    if (use_adm && signal_bits < static_cast<std::size_t>(max_signal_bytes) * 8) {
        const std::size_t byte = signal_bits >> 3;
        const unsigned int shift = static_cast<unsigned int>(signal_bits & 7);
        signal[byte] |= static_cast<std::uint8_t>(0xffu >> shift);
    }
    if (lane == 0) {
        signal_lengths[block] = use_adm
            ? max_signal_bytes
            : static_cast<std::uint32_t>(per_lane * sizeof(T));
    }
}

} // namespace mans::nv::adm::kernel
