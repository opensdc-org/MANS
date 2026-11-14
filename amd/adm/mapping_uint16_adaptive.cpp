// Reorganized AMD MANS ADM flow (1D uint16), GPU kernels only.
//
// Compression:
//   1. gsize-based execution choice: fused ADM (`map_values_kernel_decoupled`)
//      for small inputs, unfused ADM (`map_values_kernel_thrust` + thrust scan +
//      `concat`) for large inputs. The crossover is expressed in gsize
//      (gsize = ceil(num_elements / (cmp_tblock_size * cmp_chunk))), i.e. the
//      number of 512-element blocks.
//   2. Per-block ADM-vs-raw decision inside the mapping kernel: if a block's
//      value range (max - min) reaches kThreshold the block is stored verbatim
//      (raw uint16 passthrough) instead of ADM, mirroring the NV encode kernel.
//      The per-block choice is recorded in a flags bitmask.
//   3. The resulting ADM payload is then fed to the ANS stage (see
//      mans/amd/ans/mans_amd_ans.{h,cpp}) which entropy-codes it as plain bytes.
//
// Decompression:
//   Always the fused `decompress` kernel. Blocks flagged as raw are copied
//   through; blocks flagged as ADM are decoded. No unfused decode path.
//
// Kernels are the optimized MI210 (gfx90a) kernels from
// amd/adm/haac_bench/src/ (center-sum/load merge, register-resident signal
// packing/decode, block size 256 for decompression), extended with the min/max
// range scan and the raw-passthrough branch.
//
// Output `.adm` layout (headerless, little-endian):
//   [output_lengths (gsize * int32)]   exclusive prefix sum of per-lane bytes
//   [centers (gsize * uint16)]         block center (0 for raw blocks)
//   [flags (ceil(gsize/8) bytes)]      bit b = 1 if block b is ADM-encoded
//   [codes (num_elements * uint8)]     ADM codes (unused for raw blocks)
//   [signals (concatenated)]           ADM bit-signals / raw uint16 payload

#include <hip/hip_runtime.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

// Shared constants (match amd/adm/haac_bench/src/common.h).
static const int cmp_tblock_size = 32; // mapping threads per block (one warp)
static const int cmp_chunk = 16;       // elements per mapping thread
static const int max_bytes_signal_per_ele_16b = 2;
static const int decmp_chunk = 32;     // one decomp thread handles 32 elements
static const int warp_size = 32;
static const int k = 4;
static const int kn = 16;
static const int decmp_block_size = 256;
static const int concat_block_size = 256;
static const int timing_iters = 10;

// ADM vs raw fallback: a block whose value range reaches this threshold is
// stored verbatim instead of ADM (matches nv/adm/adm_kernel.cuh kThreshold).
static const unsigned long long kThreshold = 3500;
// Raw blocks occupy cmp_chunk * sizeof(uint16_t) = 32 bytes per lane.
static const int raw_per_lane_bytes = cmp_chunk * sizeof(uint16_t);
// Fused vs unfused crossover in gsize units. gsize = num_elements / 512, so
// gsize <= 4096 <=> input <= 4 MiB (fused); larger inputs use unfused (the
// measured MI210 crossover).
static const int kFusedGsizeThreshold = 4096;

// Track the min/max uint16 value over a thread's local_count valid elements,
// loaded as two uint4 (8 packed uint16 each) into b0/b1. Compile-time indexing
// keeps this register-resident (no local-memory spill).
__device__ inline void track_minmax(uint4 b0, uint4 b1, uint32_t count,
                                    uint32_t& mn, uint32_t& mx) {
    const uint32_t words[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
    #pragma unroll
    for (int w = 0; w < 8; ++w) {
        if (2 * w < count) {
            const uint32_t lo = words[w] & 0xFFFFu;
            mn = min(mn, lo);
            mx = max(mx, lo);
        }
        if (2 * w + 1 < count) {
            const uint32_t hi = words[w] >> 16;
            mn = min(mn, hi);
            mx = max(mx, hi);
        }
    }
}

// ---------------------------------------------------------------------------
// Fused compression: single kernel does map + signal encode + decoupled
// look-back prefix sum. Extended with min/max range scan and raw fallback.
// ---------------------------------------------------------------------------
__global__ void map_values_kernel_decoupled(
    const uint16_t* data, uint8_t* code, uint8_t* bit_signal, uint16_t* centers,
    int* signal_length, uint32_t* adm_flags,
    volatile int* const __restrict__ locOffset,
    volatile int* const __restrict__ cmpOffset,
    volatile int* const __restrict__ flag,
    int data_size, int shift)
{
    __shared__ unsigned int excl_sum;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = bid * blockDim.x + tid;
    const int lane = idx & 0x1f;
    const int warp = idx >> 5;
    const int block_num = cmp_chunk >> k;

    int base_block_start_idx;
    int base_thread_start_idx, base_thread_end_idx;
    uint4 tmp_buffer;

    uint8_t local_code[cmp_chunk] = {0};
    uint8_t local_output_idx[cmp_chunk] = {0};

    int diff = 0;
    int output_idx = 0;
    int local_idx = 0;
    int bit_offset = 0;
    uint16_t currValue = 0;

    base_block_start_idx = warp * cmp_chunk * cmp_tblock_size;
    base_thread_start_idx = base_block_start_idx + lane * cmp_chunk;
    base_thread_end_idx = base_thread_start_idx + cmp_chunk;

    uint32_t local_sum = 0;
    uint32_t local_count = 0;
    uint32_t local_min = 0xFFFFu;
    uint32_t local_max = 0u;
    int end = min(base_thread_end_idx, data_size);

    // Load this thread's 16 elements as two vectorized uint4 once and reuse them
    // for the center average, the range scan, and the per-element encoding.
    uint4 buf[2];
    if (base_thread_start_idx < end) {
        buf[0] = reinterpret_cast<const uint4*>(data)[base_thread_start_idx / 8];
        buf[1] = (base_thread_start_idx + 8 < end)
            ? reinterpret_cast<const uint4*>(data)[base_thread_start_idx / 8 + 1]
            : make_uint4(0, 0, 0, 0);
        local_count = end - base_thread_start_idx;
        #pragma unroll
        for (int b = 0; b < 2; ++b) {
            uint32_t t;
            t = buf[b].x; local_sum += (t & 0xFFFF) + (t >> 16);
            t = buf[b].y; local_sum += (t & 0xFFFF) + (t >> 16);
            t = buf[b].z; local_sum += (t & 0xFFFF) + (t >> 16);
            t = buf[b].w; local_sum += (t & 0xFFFF) + (t >> 16);
        }
        track_minmax(buf[0], buf[1], local_count, local_min, local_max);
    } else {
        buf[0] = make_uint4(0, 0, 0, 0);
        buf[1] = make_uint4(0, 0, 0, 0);
    }

    // Warp-shuffle reduce sum, count, min, max.
    #pragma unroll
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down(local_sum, offset);
        local_count += __shfl_down(local_count, offset);
        local_min = min(local_min, __shfl_down(local_min, offset));
        local_max = max(local_max, __shfl_down(local_max, offset));
    }

    // Lane 0 computes the block center and the ADM-vs-raw decision, then
    // broadcasts both so the whole warp agrees (uniform branch below).
    int center = 0;
    center = local_count > 0 ? local_sum / local_count : 0;
    center = __shfl(center, 0);

    int use_adm_v = (local_count > 0 && (local_max - local_min) < kThreshold) ? 1 : 0;
    use_adm_v = __shfl(use_adm_v, 0);
    const bool use_adm = use_adm_v != 0;

    if (use_adm) {
        #pragma unroll
        for (int j = 0; j < block_num; j++)
        {
            base_thread_start_idx = base_block_start_idx + lane * cmp_chunk + j * kn;
            base_thread_end_idx = base_thread_start_idx + kn;
            end = min(base_thread_end_idx, data_size);
            if (base_thread_start_idx > data_size) break;

            #pragma unroll
            for (int b = 0; b < 2; b++)
            {
                tmp_buffer = buf[b];

                currValue = static_cast<uint16_t>(tmp_buffer.x & 0xFFFF);
                diff = (currValue > center) ? currValue - center : center - currValue;
                output_idx = (currValue == center) ? 1
                           : (diff % 126 == 0) ? diff / 126 : diff / 126 + 1;
                local_code[local_idx] = (currValue == center) ? shift
                    : (uint8_t)((currValue > center)
                        ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                        : (diff + 126 - output_idx * 126) * 2 + shift);
                local_output_idx[local_idx] = output_idx;
                local_idx++;
                bit_offset += output_idx;

                currValue = static_cast<uint16_t>((tmp_buffer.x >> 16) & 0xFFFF);
                diff = (currValue > center) ? currValue - center : center - currValue;
                output_idx = (currValue == center) ? 1
                           : (diff % 126 == 0) ? diff / 126 : diff / 126 + 1;
                local_code[local_idx] = (currValue == center) ? shift
                    : (uint8_t)((currValue > center)
                        ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                        : (diff + 126 - output_idx * 126) * 2 + shift);
                local_output_idx[local_idx] = output_idx;
                local_idx++;
                bit_offset += output_idx;

                currValue = static_cast<uint16_t>(tmp_buffer.y & 0xFFFF);
                diff = (currValue > center) ? currValue - center : center - currValue;
                output_idx = (currValue == center) ? 1
                           : (diff % 126 == 0) ? diff / 126 : diff / 126 + 1;
                local_code[local_idx] = (currValue == center) ? shift
                    : (uint8_t)((currValue > center)
                        ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                        : (diff + 126 - output_idx * 126) * 2 + shift);
                local_output_idx[local_idx] = output_idx;
                local_idx++;
                bit_offset += output_idx;

                currValue = static_cast<uint16_t>((tmp_buffer.y >> 16) & 0xFFFF);
                diff = (currValue > center) ? currValue - center : center - currValue;
                output_idx = (currValue == center) ? 1
                           : (diff % 126 == 0) ? diff / 126 : diff / 126 + 1;
                local_code[local_idx] = (currValue == center) ? shift
                    : (uint8_t)((currValue > center)
                        ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                        : (diff + 126 - output_idx * 126) * 2 + shift);
                local_output_idx[local_idx] = output_idx;
                local_idx++;
                bit_offset += output_idx;

                currValue = static_cast<uint16_t>(tmp_buffer.z & 0xFFFF);
                diff = (currValue > center) ? currValue - center : center - currValue;
                output_idx = (currValue == center) ? 1
                           : (diff % 126 == 0) ? diff / 126 : diff / 126 + 1;
                local_code[local_idx] = (currValue == center) ? shift
                    : (uint8_t)((currValue > center)
                        ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                        : (diff + 126 - output_idx * 126) * 2 + shift);
                local_output_idx[local_idx] = output_idx;
                local_idx++;
                bit_offset += output_idx;

                currValue = static_cast<uint16_t>((tmp_buffer.z >> 16) & 0xFFFF);
                diff = (currValue > center) ? currValue - center : center - currValue;
                output_idx = (currValue == center) ? 1
                           : (diff % 126 == 0) ? diff / 126 : diff / 126 + 1;
                local_code[local_idx] = (currValue == center) ? shift
                    : (uint8_t)((currValue > center)
                        ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                        : (diff + 126 - output_idx * 126) * 2 + shift);
                local_output_idx[local_idx] = output_idx;
                local_idx++;
                bit_offset += output_idx;

                currValue = static_cast<uint16_t>(tmp_buffer.w & 0xFFFF);
                diff = (currValue > center) ? currValue - center : center - currValue;
                output_idx = (currValue == center) ? 1
                           : (diff % 126 == 0) ? diff / 126 : diff / 126 + 1;
                local_code[local_idx] = (currValue == center) ? shift
                    : (uint8_t)((currValue > center)
                        ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                        : (diff + 126 - output_idx * 126) * 2 + shift);
                local_output_idx[local_idx] = output_idx;
                local_idx++;
                bit_offset += output_idx;

                currValue = static_cast<uint16_t>((tmp_buffer.w >> 16) & 0xFFFF);
                diff = (currValue > center) ? currValue - center : center - currValue;
                output_idx = (currValue == center) ? 1
                           : (diff % 126 == 0) ? diff / 126 : diff / 126 + 1;
                local_code[local_idx] = (currValue == center) ? shift
                    : (uint8_t)((currValue > center)
                        ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                        : (diff + 126 - output_idx * 126) * 2 + shift);
                local_output_idx[local_idx] = output_idx;
                local_idx++;
                bit_offset += output_idx;
            }
        }
    }

    // Max signal length in the warp (per-lane bytes), only meaningful for ADM.
    uint16_t max_bitstream_length = 0;
    if (use_adm) {
        max_bitstream_length = bit_offset;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            max_bitstream_length = max(max_bitstream_length,
                __shfl_down(max_bitstream_length, offset));
        }
        max_bitstream_length = (max_bitstream_length + 7) / 8;
        max_bitstream_length = __shfl(max_bitstream_length, 0);
    }

    const int per_lane_length = use_adm ? (int)max_bitstream_length : raw_per_lane_bytes;

    if (lane == 31) {
        signal_length[bid] = per_lane_length;
        centers[bid] = use_adm ? (uint16_t)center : (uint16_t)0;
        if (use_adm) {
            atomicOr(&adm_flags[bid >> 5], 1u << (31 - (bid & 31)));
        }
        locOffset[warp + 1] = per_lane_length;
        __threadfence();
        if (warp == 0) {
            flag[0] = 2;
            __threadfence();
            flag[1] = 1;
            __threadfence();
        } else {
            flag[warp + 1] = 1;
            __threadfence();
        }
    }
    __syncthreads();

    // Global-level exclusive prefix sum (decoupled look-back).
    if (warp == 0 && lane == 0) excl_sum = 0;
    __syncthreads();
    if (warp > 0) {
        if (!lane) {
            int lookback = warp;
            int loc_excl_sum = 0;
            while (lookback > 0) {
                int status;
                do {
                    status = flag[lookback];
                    __threadfence();
                } while (status == 0);
                if (status == 2) {
                    loc_excl_sum += cmpOffset[lookback];
                    __threadfence();
                    break;
                }
                if (status == 1) loc_excl_sum += locOffset[lookback];
                lookback--;
                __threadfence();
            }
            excl_sum = loc_excl_sum;
        }
        __syncthreads();
    }
    if (warp > 0) {
        if (!lane) cmpOffset[warp] = excl_sum;
        __threadfence();
        if (!lane) flag[warp] = 2;
        __threadfence();
    }
    __syncthreads();

    if (use_adm) {
        // Pack the variable-length unary signal bits back into bytes using a
        // register accumulator and write straight to global memory.
        int write_pos = excl_sum * cmp_tblock_size + lane * max_bitstream_length;
        uint8_t* bit_out = bit_signal + write_pos;

        uint32_t acc = 0;
        int acc_bits = 0;
        int out_byte = 0;
        #pragma unroll
        for (int i = 0; i < cmp_chunk; ++i) {
            const int oi = local_output_idx[i];
            if (oi > 0) {
                acc = (acc << oi) | (1u << (oi - 1));
                acc_bits += oi;
                while (acc_bits >= 8) {
                    bit_out[out_byte++] = (uint8_t)((acc >> (acc_bits - 8)) & 0xFF);
                    acc_bits -= 8;
                    acc &= (1u << acc_bits) - 1;
                }
            }
        }
        if (acc_bits > 0) {
            bit_out[out_byte] = (uint8_t)(((acc << (8 - acc_bits)) & 0xFF) | (0xFF >> acc_bits));
            for (int i = out_byte + 1; i < max_bitstream_length; ++i) bit_out[i] = 0;
        } else if (out_byte < max_bitstream_length) {
            bit_out[out_byte] = 0xFF;
            for (int i = out_byte + 1; i < max_bitstream_length; ++i) bit_out[i] = 0;
        }

        // d_code
        int4* code_int2 = reinterpret_cast<int4*>(code + base_block_start_idx + lane * cmp_chunk);
        int4* local_code_int2 = reinterpret_cast<int4*>(local_code);
        #pragma unroll
        for (int i = 0; i < cmp_chunk / 16; ++i) {
            code_int2[i] = local_code_int2[i];
        }
    } else {
        // Raw passthrough: store this lane's 16 uint16 verbatim (32 bytes).
        uint4* raw_out = reinterpret_cast<uint4*>(
            bit_signal + excl_sum * cmp_tblock_size + lane * raw_per_lane_bytes);
        raw_out[0] = buf[0];
        raw_out[1] = buf[1];
    }
}

// ---------------------------------------------------------------------------
// Unfused compression: mapping kernel (writes per-thread slots) + thrust scan
// + concat. Extended with min/max range scan and raw fallback.
// ---------------------------------------------------------------------------
__global__ void map_values_kernel_thrust(
    const uint16_t* data, uint8_t* code, uint8_t* bit_signal, uint16_t* centers,
    int* signal_length, uint32_t* adm_flags, int data_size, int shift)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = bid * blockDim.x + tid;
    const int lane = idx & 0x1f;
    const int warp = idx >> 5;
    const int block_num = 1;

    int base_block_start_idx;
    int base_thread_start_idx, base_thread_end_idx;
    uint4 tmp_buffer;

    uint8_t local_code[cmp_chunk] = {0};
    uint8_t local_output_idx[cmp_chunk] = {0};

    int diff = 0;
    int output_idx = 0;
    int local_idx = 0;
    int bit_offset = 0;
    uint16_t currValue = 0;

    base_block_start_idx = warp * cmp_chunk * 32;
    base_thread_start_idx = base_block_start_idx + lane * cmp_chunk;
    base_thread_end_idx = base_thread_start_idx + cmp_chunk;

    uint32_t local_sum = 0;
    uint32_t local_count = 0;
    uint32_t local_min = 0xFFFFu;
    uint32_t local_max = 0u;
    int end = min(base_thread_end_idx, data_size);

    uint4 buf[2];
    if (base_thread_start_idx < end) {
        buf[0] = reinterpret_cast<const uint4*>(data)[base_thread_start_idx / 8];
        buf[1] = (base_thread_start_idx + 8 < end)
            ? reinterpret_cast<const uint4*>(data)[base_thread_start_idx / 8 + 1]
            : make_uint4(0, 0, 0, 0);
        local_count = end - base_thread_start_idx;
        #pragma unroll
        for (int b = 0; b < 2; ++b) {
            uint32_t t;
            t = buf[b].x; local_sum += (t & 0xFFFF) + (t >> 16);
            t = buf[b].y; local_sum += (t & 0xFFFF) + (t >> 16);
            t = buf[b].z; local_sum += (t & 0xFFFF) + (t >> 16);
            t = buf[b].w; local_sum += (t & 0xFFFF) + (t >> 16);
        }
        track_minmax(buf[0], buf[1], local_count, local_min, local_max);
    } else {
        buf[0] = make_uint4(0, 0, 0, 0);
        buf[1] = make_uint4(0, 0, 0, 0);
    }

    #pragma unroll
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down(local_sum, offset);
        local_count += __shfl_down(local_count, offset);
        local_min = min(local_min, __shfl_down(local_min, offset));
        local_max = max(local_max, __shfl_down(local_max, offset));
    }

    int center = 0;
    center = local_count > 0 ? local_sum / local_count : 0;
    center = __shfl(center, 0);

    int use_adm_v = (local_count > 0 && (local_max - local_min) < kThreshold) ? 1 : 0;
    use_adm_v = __shfl(use_adm_v, 0);
    const bool use_adm = use_adm_v != 0;

    if (use_adm) {
        #pragma unroll
        for (int j = 0; j < block_num; j++)
        {
            if (base_thread_start_idx > data_size) break;
            #pragma unroll
            for (int b = 0; b < 2; b++)
            {
                tmp_buffer = buf[b];

                currValue = static_cast<uint16_t>(tmp_buffer.x & 0xFFFF);
                diff = (currValue > center) ? currValue - center : center - currValue;
                output_idx = (currValue == center) ? 1
                           : (diff % 126 == 0) ? diff / 126 : diff / 126 + 1;
                local_code[local_idx] = (currValue == center) ? shift
                    : (uint8_t)((currValue > center)
                        ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                        : (diff + 126 - output_idx * 126) * 2 + shift);
                local_output_idx[local_idx] = output_idx;
                local_idx++;
                bit_offset += output_idx;

                currValue = static_cast<uint16_t>((tmp_buffer.x >> 16) & 0xFFFF);
                diff = (currValue > center) ? currValue - center : center - currValue;
                output_idx = (currValue == center) ? 1
                           : (diff % 126 == 0) ? diff / 126 : diff / 126 + 1;
                local_code[local_idx] = (currValue == center) ? shift
                    : (uint8_t)((currValue > center)
                        ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                        : (diff + 126 - output_idx * 126) * 2 + shift);
                local_output_idx[local_idx] = output_idx;
                local_idx++;
                bit_offset += output_idx;

                currValue = static_cast<uint16_t>(tmp_buffer.y & 0xFFFF);
                diff = (currValue > center) ? currValue - center : center - currValue;
                output_idx = (currValue == center) ? 1
                           : (diff % 126 == 0) ? diff / 126 : diff / 126 + 1;
                local_code[local_idx] = (currValue == center) ? shift
                    : (uint8_t)((currValue > center)
                        ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                        : (diff + 126 - output_idx * 126) * 2 + shift);
                local_output_idx[local_idx] = output_idx;
                local_idx++;
                bit_offset += output_idx;

                currValue = static_cast<uint16_t>((tmp_buffer.y >> 16) & 0xFFFF);
                diff = (currValue > center) ? currValue - center : center - currValue;
                output_idx = (currValue == center) ? 1
                           : (diff % 126 == 0) ? diff / 126 : diff / 126 + 1;
                local_code[local_idx] = (currValue == center) ? shift
                    : (uint8_t)((currValue > center)
                        ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                        : (diff + 126 - output_idx * 126) * 2 + shift);
                local_output_idx[local_idx] = output_idx;
                local_idx++;
                bit_offset += output_idx;

                currValue = static_cast<uint16_t>(tmp_buffer.z & 0xFFFF);
                diff = (currValue > center) ? currValue - center : center - currValue;
                output_idx = (currValue == center) ? 1
                           : (diff % 126 == 0) ? diff / 126 : diff / 126 + 1;
                local_code[local_idx] = (currValue == center) ? shift
                    : (uint8_t)((currValue > center)
                        ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                        : (diff + 126 - output_idx * 126) * 2 + shift);
                local_output_idx[local_idx] = output_idx;
                local_idx++;
                bit_offset += output_idx;

                currValue = static_cast<uint16_t>((tmp_buffer.z >> 16) & 0xFFFF);
                diff = (currValue > center) ? currValue - center : center - currValue;
                output_idx = (currValue == center) ? 1
                           : (diff % 126 == 0) ? diff / 126 : diff / 126 + 1;
                local_code[local_idx] = (currValue == center) ? shift
                    : (uint8_t)((currValue > center)
                        ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                        : (diff + 126 - output_idx * 126) * 2 + shift);
                local_output_idx[local_idx] = output_idx;
                local_idx++;
                bit_offset += output_idx;

                currValue = static_cast<uint16_t>(tmp_buffer.w & 0xFFFF);
                diff = (currValue > center) ? currValue - center : center - currValue;
                output_idx = (currValue == center) ? 1
                           : (diff % 126 == 0) ? diff / 126 : diff / 126 + 1;
                local_code[local_idx] = (currValue == center) ? shift
                    : (uint8_t)((currValue > center)
                        ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                        : (diff + 126 - output_idx * 126) * 2 + shift);
                local_output_idx[local_idx] = output_idx;
                local_idx++;
                bit_offset += output_idx;

                currValue = static_cast<uint16_t>((tmp_buffer.w >> 16) & 0xFFFF);
                diff = (currValue > center) ? currValue - center : center - currValue;
                output_idx = (currValue == center) ? 1
                           : (diff % 126 == 0) ? diff / 126 : diff / 126 + 1;
                local_code[local_idx] = (currValue == center) ? shift
                    : (uint8_t)((currValue > center)
                        ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                        : (diff + 126 - output_idx * 126) * 2 + shift);
                local_output_idx[local_idx] = output_idx;
                local_idx++;
                bit_offset += output_idx;
            }
        }
    }

    uint16_t max_bitstream_length = 0;
    if (use_adm) {
        max_bitstream_length = bit_offset;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            max_bitstream_length = max(max_bitstream_length,
                __shfl_down(max_bitstream_length, offset));
        }
        max_bitstream_length = (max_bitstream_length + 7) / 8;
        max_bitstream_length = __shfl(max_bitstream_length, 0);
    }

    const int per_lane_length = use_adm ? (int)max_bitstream_length : raw_per_lane_bytes;

    if (lane == 0) {
        signal_length[bid] = per_lane_length;
        centers[bid] = use_adm ? (uint16_t)center : (uint16_t)0;
        if (use_adm) {
            atomicOr(&adm_flags[bid >> 5], 1u << (31 - (bid & 31)));
        }
    }

    if (use_adm) {
        int write_pos = idx * cmp_chunk * max_bytes_signal_per_ele_16b;
        uint8_t* bit_out = bit_signal + write_pos;

        uint32_t acc = 0;
        int acc_bits = 0;
        int out_byte = 0;
        #pragma unroll
        for (int i = 0; i < cmp_chunk; ++i) {
            const int oi = local_output_idx[i];
            if (oi > 0) {
                acc = (acc << oi) | (1u << (oi - 1));
                acc_bits += oi;
                while (acc_bits >= 8) {
                    bit_out[out_byte++] = (uint8_t)((acc >> (acc_bits - 8)) & 0xFF);
                    acc_bits -= 8;
                    acc &= (1u << acc_bits) - 1;
                }
            }
        }
        if (acc_bits > 0) {
            bit_out[out_byte] = (uint8_t)(((acc << (8 - acc_bits)) & 0xFF) | (0xFF >> acc_bits));
            for (int i = out_byte + 1; i < max_bitstream_length; ++i) bit_out[i] = 0;
        } else if (out_byte < max_bitstream_length) {
            bit_out[out_byte] = 0xFF;
            for (int i = out_byte + 1; i < max_bitstream_length; ++i) bit_out[i] = 0;
        }

        int4* code_int2 = reinterpret_cast<int4*>(code + base_thread_start_idx);
        int4* local_code_int2 = reinterpret_cast<int4*>(local_code);
        #pragma unroll
        for (int i = 0; i < cmp_chunk / 16; ++i) {
            code_int2[i] = local_code_int2[i];
        }
    } else {
        uint4* raw_out = reinterpret_cast<uint4*>(
            bit_signal + idx * cmp_chunk * max_bytes_signal_per_ele_16b);
        raw_out[0] = buf[0];
        raw_out[1] = buf[1];
    }
}

__global__ void concat(
    const uint8_t* d_bit_signals,
    const int* d_signal_length,
    const int* d_output_lengths,
    uint8_t* d_concatenated_output,
    int gsize,
    int num_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = idx & 0x1f;
    int warp = idx >> 5;

    if (idx * cmp_chunk < num_elements) {
        int length = d_signal_length[warp];
        int bit_signal_start = idx * cmp_chunk * max_bytes_signal_per_ele_16b;
        int concat_start = d_output_lengths[warp] * cmp_tblock_size + lane * length;
        for (int i = 0; i < length; i++) {
            d_concatenated_output[concat_start + i] = d_bit_signals[bit_signal_start + i];
        }
    }
}

// ---------------------------------------------------------------------------
// Fused decompression: single kernel decodes ADM blocks and copies raw blocks.
// ---------------------------------------------------------------------------
__global__ void decompress(
    uint16_t* decmp_data, uint16_t* centers, uint8_t* codes,
    int* d_output_lengths, uint8_t* d_concatenated_signals,
    const uint32_t* adm_flags, int data_size, int last_length, int gsize, int shift)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = bid * blockDim.x + tid;

    int lane = idx & 0x1f;
    int warp = idx >> 5;

    if (idx * decmp_chunk >= data_size) return;

    warp = lane < 16 ? warp * 2 : warp * 2 + 1;
    const int use_adm = (adm_flags[warp >> 5] >> (31 - (warp & 31))) & 1;

    if (!use_adm) {
        // Raw block: copy 32 contiguous uint16 (64 bytes) back verbatim.
        uint16_t* raw_src = reinterpret_cast<uint16_t*>(
            d_concatenated_signals + d_output_lengths[warp] * cmp_tblock_size)
            + (idx & 15) * decmp_chunk;
        int4* src4 = reinterpret_cast<int4*>(raw_src);
        int4* dst4 = reinterpret_cast<int4*>(decmp_data + idx * decmp_chunk);
        #pragma unroll
        for (int i = 0; i < decmp_chunk / 8; ++i) {
            dst4[i] = src4[i];
        }
        return;
    }

    int end = (d_output_lengths[gsize - 1] + last_length) * cmp_tblock_size;
    int length = (warp == (gsize - 1)) ? last_length : d_output_lengths[warp + 1] - d_output_lengths[warp];

    int src_start_idx = d_output_lengths[warp] * cmp_tblock_size;
    src_start_idx = lane < 16 ? src_start_idx + lane * length * 2 : src_start_idx + (lane - 16) * length * 2;
    int dst_start_idx = idx * decmp_chunk;

    uint8_t local_signal[decmp_chunk] = {0};

    // Decode the first 16 elements (indices 0..15) from the first `length` bytes.
    {
        int signal_idx = -1;
        int run = 0;
        for (int offset_byte = 0; offset_byte < length && signal_idx < cmp_chunk; ++offset_byte) {
            const uint8_t bit_buffer = d_concatenated_signals[src_start_idx + offset_byte];
            #pragma unroll
            for (int i = 7; i >= 0 && signal_idx < cmp_chunk; --i) {
                if ((bit_buffer >> i) & 1) {
                    if (signal_idx >= 0 && run != 0) {
                        #pragma unroll
                        for (int j = 0; j < cmp_chunk; ++j) if (j == signal_idx) local_signal[j] = (uint8_t)run;
                    }
                    ++signal_idx;
                    run = 0;
                } else {
                    ++run;
                }
            }
        }
        if (signal_idx >= 0 && run != 0) {
            #pragma unroll
            for (int j = 0; j < cmp_chunk; ++j) if (j == signal_idx) local_signal[j] = (uint8_t)run;
        }
    }

    // Decode the next 16 elements (indices 16..31) from the second `length` bytes.
    {
        int signal_idx = cmp_chunk - 1;
        int run = 0;
        for (int offset_byte = 0; offset_byte < length && signal_idx < decmp_chunk; ++offset_byte) {
            if (src_start_idx + offset_byte + length > end) return;
            const uint8_t bit_buffer = d_concatenated_signals[src_start_idx + offset_byte + length];
            #pragma unroll
            for (int i = 7; i >= 0 && signal_idx < decmp_chunk; --i) {
                if ((bit_buffer >> i) & 1) {
                    if (signal_idx >= cmp_chunk && run != 0) {
                        #pragma unroll
                        for (int j = cmp_chunk; j < decmp_chunk; ++j) if (j == signal_idx) local_signal[j] = (uint8_t)run;
                    }
                    ++signal_idx;
                    run = 0;
                } else {
                    ++run;
                }
            }
        }
        if (signal_idx >= cmp_chunk && run != 0) {
            #pragma unroll
            for (int j = cmp_chunk; j < decmp_chunk; ++j) if (j == signal_idx) local_signal[j] = (uint8_t)run;
        }
    }

    uint8_t local_codes[decmp_chunk];
    int4* local_codes_int2 = reinterpret_cast<int4*>(local_codes);
    int4* codes_int2 = reinterpret_cast<int4*>(codes + dst_start_idx);
    #pragma unroll
    for (int i = 0; i < decmp_chunk / 16; ++i) {
        local_codes_int2[i] = codes_int2[i];
    }

    uint16_t local_result[decmp_chunk];
    int center = centers[warp];

    #pragma unroll
    for (int i = 0; i < decmp_chunk; i++) {
        uint8_t code = local_codes[i];
        uint8_t signal = local_signal[i];
        int diff = (code % 2 == 1) ? ((code - 1) / 2) : (code / 2);
        diff += signal * 126;
        local_result[i] = (code % 2 == 1) ? center - diff : center + diff;
    }

    int4* local_result_int4 = reinterpret_cast<int4*>(local_result);
    int4* decmp_int4 = reinterpret_cast<int4*>(decmp_data + dst_start_idx);
    #pragma unroll
    for (int i = 0; i < decmp_chunk / 8; ++i) {
        decmp_int4[i] = local_result_int4[i];
    }
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------
inline uint16_t* load_file_u16(const char* filename, size_t& size) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return nullptr;
    size = file.tellg();
    file.seekg(0, std::ios::beg);
    size_t num_elements = size / sizeof(uint16_t);
    uint16_t* data = new uint16_t[num_elements];
    file.read(reinterpret_cast<char*>(data), size);
    file.close();
    return data;
}

inline uint8_t* load_file_bytes(const char* filename, size_t& size) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return nullptr;
    size = file.tellg();
    file.seekg(0, std::ios::beg);
    uint8_t* data = new uint8_t[size];
    file.read(reinterpret_cast<char*>(data), size);
    file.close();
    return data;
}

inline void save_file(const char* filename, const uint8_t* data, size_t size) {
    std::ofstream file(filename, std::ios::binary);
    file.write(reinterpret_cast<const char*>(data), size);
    file.close();
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input.u2> <output.adm>" << std::endl;
        return 1;
    }
    const char* input_file = argv[1];
    const char* output_file = argv[2];
    const int shift = 1;

    size_t data_size = 0;
    uint16_t* h_data = load_file_u16(input_file, data_size);
    if (!h_data) {
        std::cerr << "Failed to load input file: " << input_file << std::endl;
        return 1;
    }
    const int num_elements = data_size / sizeof(uint16_t);
    std::cout << "elements num: " << num_elements << std::endl;

    const int bsize = cmp_tblock_size;
    const int gsize = (num_elements + bsize * cmp_chunk - 1) / (bsize * cmp_chunk);
    const size_t flags_words = (gsize + 31) / 32;
    const size_t flags_bytes = (gsize + 7) / 8;
    const bool use_decoupled = gsize <= kFusedGsizeThreshold;

    // Device buffers.
    uint16_t* d_data = nullptr;
    uint16_t* d_centers = nullptr;
    uint8_t* d_codes = nullptr;
    uint8_t* d_bit_signals = nullptr;
    uint8_t* d_concatenated_signals = nullptr;
    int* d_signal_length = nullptr;
    int* d_output_lengths = nullptr;
    int* d_locOffset = nullptr;
    int* d_flag = nullptr;
    uint32_t* d_adm_flags = nullptr;

    hipMalloc((void**)&d_data, data_size);
    hipMalloc((void**)&d_signal_length, gsize * sizeof(int));
    hipMemset(d_signal_length, 0, gsize * sizeof(int));
    hipMalloc((void**)&d_codes, bsize * gsize * cmp_chunk * sizeof(uint8_t));
    hipMalloc((void**)&d_locOffset, (gsize + 1) * sizeof(int));
    hipMalloc((void**)&d_flag, (gsize + 1) * sizeof(int));
    hipMalloc(&d_output_lengths, (gsize + 1) * sizeof(int));
    hipMalloc(&d_centers, gsize * sizeof(uint16_t));
    hipMalloc(&d_adm_flags, flags_words * sizeof(uint32_t));
    hipMalloc(&d_bit_signals, bsize * gsize * cmp_chunk * max_bytes_signal_per_ele_16b * sizeof(uint8_t));
    hipMemset(d_bit_signals, 0, bsize * gsize * cmp_chunk * max_bytes_signal_per_ele_16b * sizeof(uint8_t));
    hipMalloc(&d_concatenated_signals, bsize * gsize * cmp_chunk * max_bytes_signal_per_ele_16b * sizeof(uint8_t));
    hipMemset(d_concatenated_signals, 255, bsize * gsize * cmp_chunk * max_bytes_signal_per_ele_16b * sizeof(uint8_t));

    hipMemcpy(d_data, h_data, data_size, hipMemcpyHostToDevice);

    thrust::device_ptr<int> dev_output_lengths(d_output_lengths);
    thrust::device_ptr<int> dev_signal_length(d_signal_length);

    hipStream_t stream;
    hipStreamCreate(&stream);

    // Warm up the selected path.
    for (int i = 0; i < 5; ++i) {
        hipMemset(d_adm_flags, 0, flags_words * sizeof(uint32_t));
        if (use_decoupled) {
            hipMemset(d_flag, 0, (gsize + 1) * sizeof(int));
            hipMemset(d_locOffset, 0, (gsize + 1) * sizeof(int));
            hipMemset(d_output_lengths, 0, (gsize + 1) * sizeof(int));
            map_values_kernel_decoupled<<<gsize, bsize, 0, stream>>>(
                d_data, d_codes, d_concatenated_signals, d_centers, d_signal_length,
                d_adm_flags, d_locOffset, d_output_lengths, d_flag, num_elements, shift);
        } else {
            map_values_kernel_thrust<<<gsize, bsize, 0, stream>>>(
                d_data, d_codes, d_bit_signals, d_centers, d_signal_length,
                d_adm_flags, num_elements, shift);
            thrust::exclusive_scan(dev_signal_length, dev_signal_length + gsize, dev_output_lengths);
            const int concat_gsize = (num_elements + concat_block_size * cmp_chunk - 1) / (concat_block_size * cmp_chunk);
            concat<<<concat_gsize, concat_block_size, 0, stream>>>(
                d_bit_signals, d_signal_length, d_output_lengths,
                d_concatenated_signals, gsize, num_elements);
        }
        hipStreamSynchronize(stream);
    }

    // Timed compression.
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float cmp_ms = 0;
    for (int i = 0; i < timing_iters; ++i) {
        hipMemset(d_adm_flags, 0, flags_words * sizeof(uint32_t));
        hipEventRecord(start, stream);
        if (use_decoupled) {
            hipMemset(d_flag, 0, (gsize + 1) * sizeof(int));
            hipMemset(d_locOffset, 0, (gsize + 1) * sizeof(int));
            hipMemset(d_output_lengths, 0, (gsize + 1) * sizeof(int));
            map_values_kernel_decoupled<<<gsize, bsize, 0, stream>>>(
                d_data, d_codes, d_concatenated_signals, d_centers, d_signal_length,
                d_adm_flags, d_locOffset, d_output_lengths, d_flag, num_elements, shift);
        } else {
            map_values_kernel_thrust<<<gsize, bsize, 0, stream>>>(
                d_data, d_codes, d_bit_signals, d_centers, d_signal_length,
                d_adm_flags, num_elements, shift);
            thrust::exclusive_scan(dev_signal_length, dev_signal_length + gsize, dev_output_lengths);
            const int concat_gsize = (num_elements + concat_block_size * cmp_chunk - 1) / (concat_block_size * cmp_chunk);
            concat<<<concat_gsize, concat_block_size, 0, stream>>>(
                d_bit_signals, d_signal_length, d_output_lengths,
                d_concatenated_signals, gsize, num_elements);
        }
        hipEventRecord(stop, stream);
        hipEventSynchronize(stop);
        float ms = 0;
        hipEventElapsedTime(&ms, start, stop);
        cmp_ms += ms;
    }
    cmp_ms /= timing_iters;

    // Copy compression results to host.
    std::vector<int> h_signal_length(gsize);
    std::vector<int> h_output_lengths(gsize);
    std::vector<uint32_t> h_adm_flags_words(flags_words);
    hipMemcpy(h_signal_length.data(), d_signal_length, gsize * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(h_output_lengths.data(), d_output_lengths, gsize * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(h_adm_flags_words.data(), d_adm_flags, flags_words * sizeof(uint32_t), hipMemcpyDeviceToHost);

    const size_t bit_signals_size =
        (h_signal_length[gsize - 1] + h_output_lengths[gsize - 1]) * cmp_tblock_size;
    const size_t output_lengths_size = gsize * sizeof(int);
    const size_t centers_size = gsize * sizeof(uint16_t);
    const size_t codes_size = num_elements * sizeof(uint8_t);
    const size_t total_size = output_lengths_size + centers_size + flags_bytes + codes_size + bit_signals_size;

    int adm_blocks = 0;
    for (int b = 0; b < gsize; ++b) {
        if (h_adm_flags_words[b >> 5] & (1u << (31 - (b & 31)))) adm_blocks++;
    }

    std::cout << (use_decoupled ? "=======>Use Fused (decoupled) ADM<=======\n"
                                : "=======>Use Unfused (thrust) ADM<=======\n")
              << "blocks: " << gsize << " (ADM: " << adm_blocks
              << ", raw: " << (gsize - adm_blocks) << ")\n"
              << "Cmp throughput: "
              << (data_size / 1024.0 / 1024 / 1024) / (cmp_ms / 1000.0f) << " GB/s\n";

    // Assemble the headerless .adm buffer on host.
    std::vector<uint8_t> h_centers(gsize * sizeof(uint16_t));
    std::vector<uint8_t> h_codes(num_elements);
    std::vector<uint8_t> h_signals(bit_signals_size);
    hipMemcpy(h_centers.data(), d_centers, centers_size, hipMemcpyDeviceToHost);
    hipMemcpy(h_codes.data(), d_codes, codes_size, hipMemcpyDeviceToHost);
    hipMemcpy(h_signals.data(), d_concatenated_signals, bit_signals_size, hipMemcpyDeviceToHost);

    std::vector<uint8_t> h_flags(flags_bytes, 0);
    for (int b = 0; b < gsize; ++b) {
        if (h_adm_flags_words[b >> 5] & (1u << (31 - (b & 31)))) {
            h_flags[b >> 3] |= (uint8_t)(1u << (7 - (b & 7)));
        }
    }

    std::vector<uint8_t> h_out(total_size);
    std::memcpy(h_out.data(), h_output_lengths.data(), output_lengths_size);
    std::memcpy(h_out.data() + output_lengths_size, h_centers.data(), centers_size);
    std::memcpy(h_out.data() + output_lengths_size + centers_size, h_flags.data(), flags_bytes);
    std::memcpy(h_out.data() + output_lengths_size + centers_size + flags_bytes, h_codes.data(), codes_size);
    std::memcpy(h_out.data() + output_lengths_size + centers_size + flags_bytes + codes_size,
                h_signals.data(), bit_signals_size);
    save_file(output_file, h_out.data(), total_size);

    std::cout << "Original size: " << data_size << " Bytes, Output size: " << total_size
              << " Bytes, CR: " << (data_size * 1.0 / total_size) << " x\n";

    // ---------------- Decompression (always fused) ----------------
    uint16_t* d_decmpdata = nullptr;
    hipMalloc(&d_decmpdata, num_elements * sizeof(uint16_t));

    // Upload parsed sections to device.
    int* d_output_lengths2 = nullptr;
    uint16_t* d_centers2 = nullptr;
    uint8_t* d_codes2 = nullptr;
    uint8_t* d_signals2 = nullptr;
    uint32_t* d_adm_flags2 = nullptr;
    hipMalloc(&d_output_lengths2, output_lengths_size);
    hipMalloc(&d_centers2, centers_size);
    hipMalloc(&d_codes2, codes_size);
    hipMalloc(&d_signals2, bit_signals_size);
    hipMalloc(&d_adm_flags2, flags_words * sizeof(uint32_t));

    hipMemcpy(d_output_lengths2, h_out.data(), output_lengths_size, hipMemcpyHostToDevice);
    hipMemcpy(d_centers2, h_out.data() + output_lengths_size, centers_size, hipMemcpyHostToDevice);
    hipMemcpy(d_codes2, h_out.data() + output_lengths_size + centers_size + flags_bytes,
              codes_size, hipMemcpyHostToDevice);
    hipMemcpy(d_signals2, h_out.data() + output_lengths_size + centers_size + flags_bytes + codes_size,
              bit_signals_size, hipMemcpyHostToDevice);
    // Reconstruct uint32 flags words from the byte-packed section.
    std::vector<uint32_t> h_flags_words2(flags_words, 0);
    for (int b = 0; b < gsize; ++b) {
        const uint8_t byte = h_out[output_lengths_size + centers_size + (b >> 3)];
        if (byte & (1u << (7 - (b & 7)))) {
            h_flags_words2[b >> 5] |= 1u << (31 - (b & 31));
        }
    }
    hipMemcpy(d_adm_flags2, h_flags_words2.data(), flags_words * sizeof(uint32_t), hipMemcpyHostToDevice);

    const int last_length = bit_signals_size / cmp_tblock_size - h_output_lengths[gsize - 1];
    const int decmp_gsize = (num_elements + decmp_block_size * decmp_chunk - 1) / (decmp_block_size * decmp_chunk);

    for (int i = 0; i < 5; ++i) {
        decompress<<<decmp_gsize, decmp_block_size, 0, stream>>>(
            d_decmpdata, d_centers2, d_codes2, d_output_lengths2, d_signals2,
            d_adm_flags2, num_elements, last_length, gsize, shift);
        hipStreamSynchronize(stream);
    }
    hipEventRecord(start, stream);
    decompress<<<decmp_gsize, decmp_block_size, 0, stream>>>(
        d_decmpdata, d_centers2, d_codes2, d_output_lengths2, d_signals2,
        d_adm_flags2, num_elements, last_length, gsize, shift);
    hipEventRecord(stop, stream);
    hipEventSynchronize(stop);
    float decmp_ms = 0;
    hipEventElapsedTime(&decmp_ms, start, stop);

    std::vector<uint16_t> h_decmpdata2(num_elements);
    hipMemcpy(h_decmpdata2.data(), d_decmpdata, num_elements * sizeof(uint16_t), hipMemcpyDeviceToHost);

    bool pass = true;
    for (int i = 0; i < num_elements; ++i) {
        if (h_decmpdata2[i] != h_data[i]) {
            std::cout << "\033[0;31mFail error check!\033[0m\n"
                      << "Error: Data mismatch at index " << i << ", original = "
                      << h_data[i] << ", decompressed = " << h_decmpdata2[i] << std::endl;
            pass = false;
            break;
        }
    }
    std::cout << (pass ? "\033[0;32mPass error check!\033[0m\n"
                       : "\033[0;31mFail error check!\033[0m\n");
    std::cout << "Decmp throughput: "
              << (data_size / 1024.0 / 1024 / 1024) / (decmp_ms / 1000.0f)
              << " GB/s (" << decmp_ms << " ms)" << std::endl;

    // Cleanup.
    hipFree(d_data);
    hipFree(d_centers);
    hipFree(d_codes);
    hipFree(d_bit_signals);
    hipFree(d_concatenated_signals);
    hipFree(d_signal_length);
    hipFree(d_output_lengths);
    hipFree(d_locOffset);
    hipFree(d_flag);
    hipFree(d_adm_flags);
    hipFree(d_decmpdata);
    hipFree(d_output_lengths2);
    hipFree(d_centers2);
    hipFree(d_codes2);
    hipFree(d_signals2);
    hipFree(d_adm_flags2);
    hipEventDestroy(start);
    hipEventDestroy(stop);
    hipStreamDestroy(stream);

    delete[] h_data;
    std::cout << "Memory has been successfully freed." << std::endl;
    return 0;
}
