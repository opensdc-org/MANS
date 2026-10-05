#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "../../mans_defs.h"

namespace mans::dcu::adm::kernel {

constexpr int kWarpSize = 32;
constexpr int kChunk1D = 16;
constexpr int kBlock1D = kWarpSize * kChunk1D;
constexpr int kTile = 16;
constexpr unsigned long long kThreshold = 3500;

__device__ inline std::size_t flat_index(std::size_t local, std::size_t x0,
                                          std::size_t y0, std::size_t z0,
                                          std::size_t sx, std::size_t sy,
                                          const mans::MansParams& params) {
    const std::size_t plane = sx * sy;
    const std::size_t lz = params.dims == 3 ? local / plane : 0;
    const std::size_t rem = params.dims == 3 ? local - lz * plane : local;
    const std::size_t ly = rem / sx;
    const std::size_t lx = rem - ly * sx;
    return (x0 + lx) + (y0 + ly) * static_cast<std::size_t>(params.nx) +
           (z0 + lz) * static_cast<std::size_t>(params.nx) *
               (params.dims >= 2 ? static_cast<std::size_t>(params.ny) : 1);
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
__device__ inline T load_le(const std::uint8_t* src) {
    using U = std::conditional_t<sizeof(T) == 2, std::uint16_t, std::uint32_t>;
    U value = 0;
    for (int i = 0; i < static_cast<int>(sizeof(T)); ++i) {
        value |= static_cast<U>(src[i]) << (8 * i);
    }
    return static_cast<T>(value);
}

__device__ inline std::int32_t load_i32_le(const std::uint8_t* src) {
    const std::uint32_t value = static_cast<std::uint32_t>(src[0]) |
        (static_cast<std::uint32_t>(src[1]) << 8) |
        (static_cast<std::uint32_t>(src[2]) << 16) |
        (static_cast<std::uint32_t>(src[3]) << 24);
    return static_cast<std::int32_t>(value);
}

template <typename T>
__global__ void encode(const T* input, std::size_t num_elements,
                       mans::MansParams params, std::uint8_t* codes,
                       std::uint32_t* flags_words, T* centers,
                       std::uint32_t* signal_lengths, std::uint8_t* lane_signals,
                       std::size_t lane_signal_stride) {
    // A 32-thread block is deliberate: the wire contract has exactly 32 lanes.
    // Reductions use shared memory so they do not depend on native wave size.
    __shared__ unsigned long long sums[kWarpSize];
    __shared__ unsigned long long counts[kWarpSize];
    __shared__ unsigned long long mins[kWarpSize];
    __shared__ unsigned long long maxs[kWarpSize];
    __shared__ unsigned long long total_sum;
    __shared__ unsigned long long total_count;
    __shared__ unsigned long long total_min;
    __shared__ unsigned long long total_max;
    __shared__ unsigned int max_signal_bytes;
    __shared__ unsigned int use_adm;
    __shared__ T center;

    const int lane = static_cast<int>(threadIdx.x);
    const std::size_t block = static_cast<std::size_t>(blockIdx.x);
    const int dims = static_cast<int>(params.dims);
    const std::size_t nx = params.nx;
    const std::size_t ny = dims >= 2 ? params.ny : 1;
    const std::size_t nz = dims == 3 ? params.nz : 1;

    std::size_t x0 = 0, y0 = 0, z0 = 0, sx = 0, sy = 1, sz = 1;
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
    const std::size_t per_lane = dims == 1 ? kChunk1D : (block_elements + kWarpSize - 1) / kWarpSize;
    const std::size_t first = static_cast<std::size_t>(lane) * per_lane;
    const std::size_t last = min(first + per_lane, block_elements);

    unsigned long long local_sum = 0;
    unsigned long long local_count = 0;
    unsigned long long local_min = ~0ull;
    unsigned long long local_max = 0;
    for (std::size_t local = first; local < last; ++local) {
        const T value = input[flat_index(local, x0, y0, z0, sx, sy, params)];
        const unsigned long long wide = static_cast<unsigned long long>(value);
        local_sum += wide;
        ++local_count;
        local_min = min(local_min, wide);
        local_max = max(local_max, wide);
    }
    sums[lane] = local_sum;
    counts[lane] = local_count;
    mins[lane] = local_min;
    maxs[lane] = local_max;
    __syncthreads();

    if (lane == 0) {
        total_sum = 0;
        total_count = 0;
        total_min = ~0ull;
        total_max = 0;
        for (int i = 0; i < kWarpSize; ++i) {
            total_sum += sums[i];
            total_count += counts[i];
            total_min = min(total_min, mins[i]);
            total_max = max(total_max, maxs[i]);
        }
        use_adm = total_count != 0 && (total_max - total_min) < kThreshold;
        center = use_adm ? static_cast<T>(total_sum / total_count) : static_cast<T>(0);
        centers[block] = center;
        if (use_adm) atomicOr(&flags_words[block >> 5], 1u << (31 - (block & 31)));
        max_signal_bytes = 0;
    }
    __syncthreads();

    std::uint8_t* signal = lane_signals +
        (block * kWarpSize + static_cast<std::size_t>(lane)) * lane_signal_stride;
    for (std::size_t i = 0; i < lane_signal_stride; ++i) signal[i] = 0;

    std::size_t signal_bits = 0;
    for (std::size_t local = first; local < last; ++local) {
        const std::size_t q = flat_index(local, x0, y0, z0, sx, sy, params);
        const T value = input[q];
        if (use_adm == 0) {
            store_le(signal + (local - first) * sizeof(T), value);
            continue;
        }
        const unsigned long long value_wide = static_cast<unsigned long long>(value);
        const unsigned long long center_wide = static_cast<unsigned long long>(center);
        const unsigned long long diff = value_wide >= center_wide
            ? value_wide - center_wide : center_wide - value_wide;
        const std::size_t output_bits = value == center ? 1 : (diff + 125) / 126;
        const unsigned long long residual = diff + 126 -
            static_cast<unsigned long long>(output_bits) * 126;
        codes[q] = value == center ? 1 : static_cast<std::uint8_t>(
            residual * 2 + (value > center ? 0 : 1));
        signal[signal_bits >> 3] |= static_cast<std::uint8_t>(1u << (7 - (signal_bits & 7)));
        signal_bits += output_bits;
    }

    const unsigned int signal_bytes = static_cast<unsigned int>((signal_bits + 7) / 8);
    atomicMax(&max_signal_bytes, signal_bytes);
    __syncthreads();
    if (use_adm != 0 && signal_bits < static_cast<std::size_t>(max_signal_bytes) * 8) {
        signal[signal_bits >> 3] |= static_cast<std::uint8_t>(0xffu >> (signal_bits & 7));
    }
    if (lane == 0) {
        signal_lengths[block] = use_adm != 0
            ? max_signal_bytes : static_cast<std::uint32_t>(per_lane * sizeof(T));
    }
}

__global__ void validate_payload(const std::uint8_t* payload,
                                std::size_t payload_size,
                                std::size_t blocks,
                                std::size_t fixed_bytes,
                                std::uint32_t* error) {
    const std::size_t block = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t available = payload_size - fixed_bytes;
    if (block == 0) {
        if (load_i32_le(payload) != 0) atomicExch(error, 1u);
        const std::int32_t final_i = load_i32_le(
            payload + blocks * sizeof(std::int32_t));
        if (final_i < 0 || static_cast<std::uint64_t>(final_i) * kWarpSize != available) {
            atomicExch(error, 1u);
        }
    }
    if (block >= blocks) return;

    const std::int32_t start_i = load_i32_le(
        payload + block * sizeof(std::int32_t));
    const std::int32_t end_i = load_i32_le(
        payload + (block + 1) * sizeof(std::int32_t));
    if (start_i < 0 || end_i < start_i) {
        atomicExch(error, 1u);
        return;
    }
    const std::uint64_t signal_offset =
        static_cast<std::uint64_t>(start_i) * kWarpSize;
    const std::uint64_t signal_bytes =
        static_cast<std::uint64_t>(end_i - start_i) * kWarpSize;
    if (signal_offset > available || signal_bytes > available - signal_offset) {
        atomicExch(error, 1u);
    }
}

template <typename T>
__global__ void decode(const std::uint8_t* payload, std::size_t payload_size,
                       std::size_t num_elements, mans::MansParams params,
                       T* output, std::uint32_t* error) {
    const std::size_t block = static_cast<std::size_t>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    const std::size_t blocks = params.dims == 1
        ? (num_elements + kBlock1D - 1) / kBlock1D
        : ((params.nx + kTile - 1) / kTile) * ((params.ny + kTile - 1) / kTile) *
          (params.dims == 3 ? (params.nz + kTile - 1) / kTile : 1);
    if (block >= blocks) return;

    const std::size_t offsets_bytes = (blocks + 1) * sizeof(std::int32_t);
    const std::size_t centers_bytes = blocks * sizeof(T);
    const std::size_t flags_bytes = (blocks + 7) / 8;
    const std::size_t fixed_bytes = offsets_bytes + centers_bytes + flags_bytes + num_elements;
    if (payload_size < fixed_bytes) { if (lane == 0) *error = 1; return; }

    const std::uint8_t* flags = payload + offsets_bytes + centers_bytes;
    const std::uint8_t* codes = flags + flags_bytes;
    const std::uint8_t* signals = codes + num_elements;
    const std::int32_t start_i = load_i32_le(payload + block * sizeof(std::int32_t));
    const std::int32_t end_i = load_i32_le(payload + (block + 1) * sizeof(std::int32_t));
    if (start_i < 0 || end_i < start_i) { if (lane == 0) *error = 1; return; }
    const std::size_t lane_length = static_cast<std::size_t>(end_i - start_i);
    const std::size_t signal_offset = static_cast<std::size_t>(start_i) * kWarpSize;
    if (signal_offset > payload_size - fixed_bytes ||
        lane_length > (payload_size - fixed_bytes - signal_offset) / kWarpSize) {
        if (lane == 0) *error = 1;
        return;
    }

    std::size_t x0 = 0, y0 = 0, z0 = 0, sx = 0, sy = 1, sz = 1;
    if (params.dims == 1) {
        x0 = block * kBlock1D;
        sx = x0 < num_elements ? min(static_cast<std::size_t>(kBlock1D), num_elements - x0) : 0;
    } else {
        const std::size_t gx = (params.nx + kTile - 1) / kTile;
        const std::size_t gy = (params.ny + kTile - 1) / kTile;
        const std::size_t bx = block % gx;
        const std::size_t t = block / gx;
        const std::size_t by = t % gy;
        const std::size_t bz = params.dims == 3 ? t / gy : 0;
        x0 = bx * kTile; y0 = by * kTile; z0 = bz * kTile;
        sx = x0 < params.nx ? min(static_cast<std::size_t>(kTile), static_cast<std::size_t>(params.nx) - x0) : 0;
        sy = y0 < params.ny ? min(static_cast<std::size_t>(kTile), static_cast<std::size_t>(params.ny) - y0) : 0;
        sz = params.dims == 3 && z0 < params.nz
            ? min(static_cast<std::size_t>(kTile), static_cast<std::size_t>(params.nz) - z0) : 1;
    }
    const std::size_t block_elements = sx * sy * sz;
    const std::size_t per_lane = params.dims == 1 ? kChunk1D :
        (block_elements + kWarpSize - 1) / kWarpSize;
    const std::size_t first = static_cast<std::size_t>(lane) * per_lane;
    const std::size_t last = first < block_elements
        ? min(first + per_lane, block_elements)
        : first;
    const T center = load_le<T>(payload + offsets_bytes + block * sizeof(T));
    const bool use_adm = (flags[block >> 3] & (1u << (7 - (block & 7)))) != 0;
    const std::uint8_t* lane_signal = signals + signal_offset +
        static_cast<std::size_t>(lane) * lane_length;

    if (!use_adm) {
        if (lane_length < (last - first) * sizeof(T)) { if (lane == 0) *error = 1; return; }
        for (std::size_t local = first; local < last; ++local) {
            output[flat_index(local, x0, y0, z0, sx, sy, params)] =
                load_le<T>(lane_signal + (local - first) * sizeof(T));
        }
        return;
    }

    // A signal is a sequence of value-start bits followed by zero continuation
    // bits. A new one terminates the previous value; the last value terminates
    // at the end of the fixed lane region.
    const std::size_t lane_values = last - first;
    std::size_t value_index = 0;
    std::size_t zero_count = 0;
    bool started = false;
    for (std::size_t byte = 0; byte < lane_length && value_index < lane_values; ++byte) {
        for (int bit = 7; bit >= 0; --bit) {
            const bool one = (lane_signal[byte] & (1u << bit)) != 0;
            if (one) {
                if (started) {
                    const std::size_t local = first + value_index;
                    const std::size_t q = flat_index(local, x0, y0, z0, sx, sy, params);
                    const std::uint8_t code = codes[q];
                    const std::uint64_t base = (code & 1u) ? (code - 1u) / 2u : code / 2u;
                    const std::uint64_t diff = base + zero_count * 126;
                    const std::uint64_t c = static_cast<std::uint64_t>(center);
                    const std::uint64_t maxv = sizeof(T) == 2 ? 0xffffull : 0xffffffffull;
                    const bool negative = (code & 1u) != 0;
                    if ((negative && diff > c) || (!negative && diff > maxv - c)) {
                        if (lane == 0) *error = 1;
                        return;
                    }
                    output[q] = static_cast<T>(negative ? c - diff : c + diff);
                    ++value_index;
                }
                started = true;
                zero_count = 0;
            } else if (started) {
                ++zero_count;
            } else {
                if (lane == 0) *error = 1;
                return;
            }
            if (value_index == lane_values) return;
        }
    }
    if (started && value_index < lane_values) {
        const std::size_t local = first + value_index;
        const std::size_t q = flat_index(local, x0, y0, z0, sx, sy, params);
        const std::uint8_t code = codes[q];
        const std::uint64_t base = (code & 1u) ? (code - 1u) / 2u : code / 2u;
        const std::uint64_t diff = base + zero_count * 126;
        const std::uint64_t c = static_cast<std::uint64_t>(center);
        const std::uint64_t maxv = sizeof(T) == 2 ? 0xffffull : 0xffffffffull;
        const bool negative = (code & 1u) != 0;
        if ((negative && diff > c) || (!negative && diff > maxv - c)) {
            if (lane == 0) *error = 1;
            return;
        }
        output[q] = static_cast<T>(negative ? c - diff : c + diff);
        ++value_index;
    }
    if (value_index != lane_values && lane == 0) *error = 1;
}

} // namespace mans::dcu::adm::kernel
