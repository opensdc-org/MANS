#include "mans_nv_adm.h"

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace mans {
namespace nv {
namespace adm {
namespace {

// Shared constants (mirror cpu/adm/adm.h and amd/adm/haac_bench/src/common.h).
constexpr int cmp_tblock_size = 32; // mapping threads per block (one warp)
constexpr int cmp_chunk = 16;       // elements per mapping thread
// CPU ADM reserves 4 signal bytes per element per lane so a block whose range
// is just below the 3500 threshold (output_idx up to ~28 bits) cannot overflow
// the unfused per-thread scratch slot.
constexpr int max_bytes_signal_per_ele_16b = 4;
constexpr int max_bytes_signal_per_ele_32b = 4;
constexpr int decmp_chunk = 32;     // one decomp thread handles 32 elements
constexpr int warp_size = 32;
constexpr int k = 4;
constexpr int kn = 16;
constexpr int decmp_block_size = 256;
constexpr int concat_block_size = 256;

// ADM vs raw fallback threshold (matches nv/adm/adm_kernel.cuh kThreshold<T>).
// The kernel tests `range < kThreshold`, and ADM must never exceed raw size:
// code 1B/elem + unary signal ceil(diff/126) bits <= 2B (u16) / 4B (u32) means
// diff <= 1008 (u16) / 3024 (u32), so the constant is one past: 1009 / 3025.
template <typename T>
__host__ __device__ constexpr unsigned long long kThresholdFor() {
    return sizeof(T) == 2 ? 1009ull : 3025ull;
}
// Raw blocks occupy cmp_chunk * sizeof(uint16_t) = 32 bytes per lane.
constexpr int raw_per_lane_bytes = cmp_chunk * sizeof(uint16_t);
// Fused vs unfused crossover in gsize units (gsize = num_elements / 512).
constexpr int kFusedGsizeThreshold = 4096;
// ADM code base offset (matches cpu/adm/adm.h encode +1).
constexpr int kShift = 1;
// 2D/3D tile sizes (mirror cpu/adm/adm.h cmp_block_x/y/z).
constexpr int cmp_block_x = 16;
constexpr int cmp_block_y = 16;
constexpr int cmp_block_z = 16;
// Max elements a single lane handles in a 2D/3D block (16*16*16 / 32).
constexpr int kMaxLaneElems =
    (cmp_block_x * cmp_block_y * cmp_block_z + cmp_tblock_size - 1) / cmp_tblock_size;

void check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

// cudaEvent-backed timer used only when a timing out-param is requested. A single
// instance is reused across the sequential launch phases of a compress/decompress
// call; each end() synchronizes so per-kernel times are exact and ordered.
struct GpuTimer {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    GpuTimer() {
        (void)cudaEventCreate(&start);
        (void)cudaEventCreate(&stop);
    }
    ~GpuTimer() {
        if (start) (void)cudaEventDestroy(start);
        if (stop) (void)cudaEventDestroy(stop);
    }
    void begin(cudaStream_t stream) { cudaEventRecord(start, stream); }
    double end(cudaStream_t stream) {
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        return static_cast<double>(ms);
    }
};

class DevBuf {
public:
    DevBuf() = default;
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    ~DevBuf() { reset(); }

    void allocate(std::size_t bytes, const char* what) {
        reset();
        if (bytes != 0) {
            check_cuda(cudaMalloc(&ptr_, bytes), what);
        }
    }

    void reset() {
        if (ptr_ != nullptr) {
            (void)cudaFree(ptr_);
            ptr_ = nullptr;
        }
    }

    void* get() const { return ptr_; }

    template <typename T>
    T* as() const { return static_cast<T*>(ptr_); }

private:
    void* ptr_ = nullptr;
};

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
// 2D/3D block geometry. Replicates cpu/adm/adm.h block_to_coords/block_shape:
// a block is a 16x16 (dims=2) or 16x16x16 (dims=3) tile clamped to the grid.
// ---------------------------------------------------------------------------
struct AdmBlockGeom {
    int x0, y0, z0;   // tile origin in global coordinates
    int sx, sy, sz;   // tile sizes (clamped at the grid boundary)
    int block_elems;  // sx * sy * sz
    int per_lane;     // ceil(block_elems / 32)
};

__device__ __forceinline__ AdmBlockGeom adm_block_geom(int b, int dims, int nx, int ny, int nz) {
    AdmBlockGeom g;
    if (dims == 1) {
        // Contiguous 512-element block (cmp_tblock_size x cmp_chunk), clamped at the tail.
        const int block = cmp_tblock_size * cmp_chunk;
        g.x0 = b * block;
        g.y0 = 0;
        g.z0 = 0;
        g.sx = min(block, nx - g.x0);
        g.sy = 1;
        g.sz = 1;
    } else if (dims == 2) {
        const int grid_x = (nx + cmp_block_x - 1) / cmp_block_x;
        const int bx = b % grid_x;
        const int by = b / grid_x;
        g.x0 = bx * cmp_block_x;
        g.y0 = by * cmp_block_y;
        g.z0 = 0;
        g.sx = min(cmp_block_x, nx - g.x0);
        g.sy = min(cmp_block_y, ny - g.y0);
        g.sz = 1;
    } else { // dims == 3
        const int grid_x = (nx + cmp_block_x - 1) / cmp_block_x;
        const int grid_y = (ny + cmp_block_y - 1) / cmp_block_y;
        const int bx = b % grid_x;
        const int t = b / grid_x;
        const int by = t % grid_y;
        const int bz = t / grid_y;
        g.x0 = bx * cmp_block_x;
        g.y0 = by * cmp_block_y;
        g.z0 = bz * cmp_block_z;
        g.sx = min(cmp_block_x, nx - g.x0);
        g.sy = min(cmp_block_y, ny - g.y0);
        g.sz = min(cmp_block_z, nz - g.z0);
    }
    g.block_elems = g.sx * g.sy * g.sz;
    g.per_lane = (g.block_elems + cmp_tblock_size - 1) / cmp_tblock_size;
    return g;
}

// Flat (row-major) index of the k-th element of a block (k is in tile-row-major
// order: x fastest, then y, then z). Also returns the in-tile (lx, ly, lz).
__device__ __forceinline__ size_t adm_elem_flat(int k, const AdmBlockGeom& g, int nx, int ny,
                                                int& lx, int& ly, int& lz) {
    const int plane = g.sx * g.sy;
    lz = k / plane;
    const int rem = k - lz * plane;
    ly = rem / g.sx;
    lx = rem - ly * g.sx;
    return (size_t)(g.x0 + lx) + (size_t)(g.y0 + ly) * (size_t)nx +
           (size_t)(g.z0 + lz) * (size_t)nx * (size_t)ny;
}

// Per-value-type traits for the generalized (dims 1/2/3) ADM kernels. The ADM
// math is identical for uint16/uint32; only the sum accumulator width differs:
// a full block of uint32 sums beyond uint32 (CPU adm.h uses uint64_t sum too).
template <typename T>
struct AdmTraits;
template <>
struct AdmTraits<std::uint16_t> {
    using Sum = std::uint32_t;
};
template <>
struct AdmTraits<std::uint32_t> {
    using Sum = std::uint64_t;
};

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
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        local_count += __shfl_down_sync(0xffffffff, local_count, offset);
        local_min = min(local_min, __shfl_down_sync(0xffffffff, local_min, offset));
        local_max = max(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }

    // Lane 0 computes the block center and the ADM-vs-raw decision, then
    // broadcasts both so the whole warp agrees (uniform branch below).
    int center = 0;
    center = local_count > 0 ? local_sum / local_count : 0;
    center = __shfl_sync(0xffffffff, center, 0);

    int use_adm_v = (local_count > 0 && (local_max - local_min) < kThresholdFor<std::uint16_t>()) ? 1 : 0;
    use_adm_v = __shfl_sync(0xffffffff, use_adm_v, 0);
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
                output_idx = (currValue == center) ? 1 : (diff + 125) / 126;
                local_code[local_idx] = (currValue == center) ? shift
                    : (uint8_t)((currValue > center)
                        ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                        : (diff + 126 - output_idx * 126) * 2 + shift);
                local_output_idx[local_idx] = output_idx;
                local_idx++;
                bit_offset += output_idx;

                currValue = static_cast<uint16_t>((tmp_buffer.x >> 16) & 0xFFFF);
                diff = (currValue > center) ? currValue - center : center - currValue;
                output_idx = (currValue == center) ? 1 : (diff + 125) / 126;
                local_code[local_idx] = (currValue == center) ? shift
                    : (uint8_t)((currValue > center)
                        ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                        : (diff + 126 - output_idx * 126) * 2 + shift);
                local_output_idx[local_idx] = output_idx;
                local_idx++;
                bit_offset += output_idx;

                currValue = static_cast<uint16_t>(tmp_buffer.y & 0xFFFF);
                diff = (currValue > center) ? currValue - center : center - currValue;
                output_idx = (currValue == center) ? 1 : (diff + 125) / 126;
                local_code[local_idx] = (currValue == center) ? shift
                    : (uint8_t)((currValue > center)
                        ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                        : (diff + 126 - output_idx * 126) * 2 + shift);
                local_output_idx[local_idx] = output_idx;
                local_idx++;
                bit_offset += output_idx;

                currValue = static_cast<uint16_t>((tmp_buffer.y >> 16) & 0xFFFF);
                diff = (currValue > center) ? currValue - center : center - currValue;
                output_idx = (currValue == center) ? 1 : (diff + 125) / 126;
                local_code[local_idx] = (currValue == center) ? shift
                    : (uint8_t)((currValue > center)
                        ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                        : (diff + 126 - output_idx * 126) * 2 + shift);
                local_output_idx[local_idx] = output_idx;
                local_idx++;
                bit_offset += output_idx;

                currValue = static_cast<uint16_t>(tmp_buffer.z & 0xFFFF);
                diff = (currValue > center) ? currValue - center : center - currValue;
                output_idx = (currValue == center) ? 1 : (diff + 125) / 126;
                local_code[local_idx] = (currValue == center) ? shift
                    : (uint8_t)((currValue > center)
                        ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                        : (diff + 126 - output_idx * 126) * 2 + shift);
                local_output_idx[local_idx] = output_idx;
                local_idx++;
                bit_offset += output_idx;

                currValue = static_cast<uint16_t>((tmp_buffer.z >> 16) & 0xFFFF);
                diff = (currValue > center) ? currValue - center : center - currValue;
                output_idx = (currValue == center) ? 1 : (diff + 125) / 126;
                local_code[local_idx] = (currValue == center) ? shift
                    : (uint8_t)((currValue > center)
                        ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                        : (diff + 126 - output_idx * 126) * 2 + shift);
                local_output_idx[local_idx] = output_idx;
                local_idx++;
                bit_offset += output_idx;

                currValue = static_cast<uint16_t>(tmp_buffer.w & 0xFFFF);
                diff = (currValue > center) ? currValue - center : center - currValue;
                output_idx = (currValue == center) ? 1 : (diff + 125) / 126;
                local_code[local_idx] = (currValue == center) ? shift
                    : (uint8_t)((currValue > center)
                        ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                        : (diff + 126 - output_idx * 126) * 2 + shift);
                local_output_idx[local_idx] = output_idx;
                local_idx++;
                bit_offset += output_idx;

                currValue = static_cast<uint16_t>((tmp_buffer.w >> 16) & 0xFFFF);
                diff = (currValue > center) ? currValue - center : center - currValue;
                output_idx = (currValue == center) ? 1 : (diff + 125) / 126;
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
                __shfl_down_sync(0xffffffff, max_bitstream_length, offset));
        }
        max_bitstream_length = (max_bitstream_length + 7) / 8;
        max_bitstream_length = __shfl_sync(0xffffffff, max_bitstream_length, 0);
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

        uint64_t acc = 0;
        int acc_bits = 0;
        int out_byte = 0;
        #pragma unroll
        for (int i = 0; i < cmp_chunk; ++i) {
            const int oi = local_output_idx[i];
            if (oi > 0) {
                acc = (acc << oi) | (1ull << (oi - 1));
                acc_bits += oi;
                while (acc_bits >= 8) {
                    bit_out[out_byte++] = (uint8_t)((acc >> (acc_bits - 8)) & 0xFF);
                    acc_bits -= 8;
                    acc &= (1ull << acc_bits) - 1;
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
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        local_count += __shfl_down_sync(0xffffffff, local_count, offset);
        local_min = min(local_min, __shfl_down_sync(0xffffffff, local_min, offset));
        local_max = max(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }

    int center = 0;
    center = local_count > 0 ? local_sum / local_count : 0;
    center = __shfl_sync(0xffffffff, center, 0);

    int use_adm_v = (local_count > 0 && (local_max - local_min) < kThresholdFor<std::uint16_t>()) ? 1 : 0;
    use_adm_v = __shfl_sync(0xffffffff, use_adm_v, 0);
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
                output_idx = (currValue == center) ? 1 : (diff + 125) / 126;
                local_code[local_idx] = (currValue == center) ? shift
                    : (uint8_t)((currValue > center)
                        ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                        : (diff + 126 - output_idx * 126) * 2 + shift);
                local_output_idx[local_idx] = output_idx;
                local_idx++;
                bit_offset += output_idx;

                currValue = static_cast<uint16_t>((tmp_buffer.x >> 16) & 0xFFFF);
                diff = (currValue > center) ? currValue - center : center - currValue;
                output_idx = (currValue == center) ? 1 : (diff + 125) / 126;
                local_code[local_idx] = (currValue == center) ? shift
                    : (uint8_t)((currValue > center)
                        ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                        : (diff + 126 - output_idx * 126) * 2 + shift);
                local_output_idx[local_idx] = output_idx;
                local_idx++;
                bit_offset += output_idx;

                currValue = static_cast<uint16_t>(tmp_buffer.y & 0xFFFF);
                diff = (currValue > center) ? currValue - center : center - currValue;
                output_idx = (currValue == center) ? 1 : (diff + 125) / 126;
                local_code[local_idx] = (currValue == center) ? shift
                    : (uint8_t)((currValue > center)
                        ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                        : (diff + 126 - output_idx * 126) * 2 + shift);
                local_output_idx[local_idx] = output_idx;
                local_idx++;
                bit_offset += output_idx;

                currValue = static_cast<uint16_t>((tmp_buffer.y >> 16) & 0xFFFF);
                diff = (currValue > center) ? currValue - center : center - currValue;
                output_idx = (currValue == center) ? 1 : (diff + 125) / 126;
                local_code[local_idx] = (currValue == center) ? shift
                    : (uint8_t)((currValue > center)
                        ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                        : (diff + 126 - output_idx * 126) * 2 + shift);
                local_output_idx[local_idx] = output_idx;
                local_idx++;
                bit_offset += output_idx;

                currValue = static_cast<uint16_t>(tmp_buffer.z & 0xFFFF);
                diff = (currValue > center) ? currValue - center : center - currValue;
                output_idx = (currValue == center) ? 1 : (diff + 125) / 126;
                local_code[local_idx] = (currValue == center) ? shift
                    : (uint8_t)((currValue > center)
                        ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                        : (diff + 126 - output_idx * 126) * 2 + shift);
                local_output_idx[local_idx] = output_idx;
                local_idx++;
                bit_offset += output_idx;

                currValue = static_cast<uint16_t>((tmp_buffer.z >> 16) & 0xFFFF);
                diff = (currValue > center) ? currValue - center : center - currValue;
                output_idx = (currValue == center) ? 1 : (diff + 125) / 126;
                local_code[local_idx] = (currValue == center) ? shift
                    : (uint8_t)((currValue > center)
                        ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                        : (diff + 126 - output_idx * 126) * 2 + shift);
                local_output_idx[local_idx] = output_idx;
                local_idx++;
                bit_offset += output_idx;

                currValue = static_cast<uint16_t>(tmp_buffer.w & 0xFFFF);
                diff = (currValue > center) ? currValue - center : center - currValue;
                output_idx = (currValue == center) ? 1 : (diff + 125) / 126;
                local_code[local_idx] = (currValue == center) ? shift
                    : (uint8_t)((currValue > center)
                        ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                        : (diff + 126 - output_idx * 126) * 2 + shift);
                local_output_idx[local_idx] = output_idx;
                local_idx++;
                bit_offset += output_idx;

                currValue = static_cast<uint16_t>((tmp_buffer.w >> 16) & 0xFFFF);
                diff = (currValue > center) ? currValue - center : center - currValue;
                output_idx = (currValue == center) ? 1 : (diff + 125) / 126;
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
                __shfl_down_sync(0xffffffff, max_bitstream_length, offset));
        }
        max_bitstream_length = (max_bitstream_length + 7) / 8;
        max_bitstream_length = __shfl_sync(0xffffffff, max_bitstream_length, 0);
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

        uint64_t acc = 0;
        int acc_bits = 0;
        int out_byte = 0;
        #pragma unroll
        for (int i = 0; i < cmp_chunk; ++i) {
            const int oi = local_output_idx[i];
            if (oi > 0) {
                acc = (acc << oi) | (1ull << (oi - 1));
                acc_bits += oi;
                while (acc_bits >= 8) {
                    bit_out[out_byte++] = (uint8_t)((acc >> (acc_bits - 8)) & 0xFF);
                    acc_bits -= 8;
                    acc &= (1ull << acc_bits) - 1;
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

// ---------------------------------------------------------------------------
// U32 1D vectorized compression (mirrors the uint16 1D kernels, but each thread
// loads four uint4 = 16 uint32 and accumulates in uint64; a boundary group whose
// uint4 would overrun the buffer falls back to scalar loads so the vectorized
// path never reads out of bounds — the uint16 kernels rely on zeroed tail memory
// here, which is not safe for large uint32 centers).
// ---------------------------------------------------------------------------
__global__ void map_values_kernel_decoupled_u32(
    const uint32_t* data, uint8_t* code, uint8_t* bit_signal, uint32_t* centers,
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

    const int base_block_start_idx = warp * cmp_chunk * cmp_tblock_size;
    const int base_thread_start_idx = base_block_start_idx + lane * cmp_chunk;
    const int base_thread_end_idx = base_thread_start_idx + cmp_chunk;
    constexpr int RawBytes = (int)sizeof(uint32_t);

    uint8_t local_code[cmp_chunk] = {0};
    uint8_t local_output_idx[cmp_chunk] = {0};
    uint32_t vals[cmp_chunk] = {0};

    uint64_t local_sum = 0;
    uint32_t local_count = 0;
    uint32_t local_min = 0xFFFFFFFFu;
    uint32_t local_max = 0u;
    const int end = min(base_thread_end_idx, data_size);

    if (base_thread_start_idx < end) {
        local_count = end - base_thread_start_idx;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int off = base_thread_start_idx + i * 4;
            if (off + 4 <= end) {
                const uint4 u = reinterpret_cast<const uint4*>(data)[off / 4];
                vals[i * 4 + 0] = u.x; vals[i * 4 + 1] = u.y;
                vals[i * 4 + 2] = u.z; vals[i * 4 + 3] = u.w;
            } else {
                vals[i * 4 + 0] = (off + 0 < end) ? data[off + 0] : 0u;
                vals[i * 4 + 1] = (off + 1 < end) ? data[off + 1] : 0u;
                vals[i * 4 + 2] = (off + 2 < end) ? data[off + 2] : 0u;
                vals[i * 4 + 3] = (off + 3 < end) ? data[off + 3] : 0u;
            }
        }
        #pragma unroll
        for (int i = 0; i < cmp_chunk; ++i) {
            local_sum += (uint64_t)vals[i];
            if (i < local_count) {
                local_min = min(local_min, vals[i]);
                local_max = max(local_max, vals[i]);
            }
        }
    }

    #pragma unroll
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        local_count += __shfl_down_sync(0xffffffff, local_count, offset);
        local_min = min(local_min, __shfl_down_sync(0xffffffff, local_min, offset));
        local_max = max(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }

    uint32_t center = local_count > 0 ? (uint32_t)(local_sum / local_count) : 0u;
    center = __shfl_sync(0xffffffff, center, 0);
    int use_adm_v = (local_count > 0 && (local_max - local_min) < kThresholdFor<std::uint32_t>()) ? 1 : 0;
    use_adm_v = __shfl_sync(0xffffffff, use_adm_v, 0);
    const bool use_adm = use_adm_v != 0;

    int bit_offset = 0;
    if (use_adm) {
        #pragma unroll
        for (int i = 0; i < cmp_chunk; ++i) {
            if (i < local_count) {
                const uint32_t val = vals[i];
                const uint32_t diff = val > center ? val - center : center - val;
                const int output_idx = (val == center) ? 1
                    : (int)((diff - 1u) / 126u + 1u);
                local_code[i] = (val == center) ? (uint8_t)shift
                    : (uint8_t)((val > center)
                        ? (diff + 126u - (uint32_t)output_idx * 126u) * 2u - 1u + (uint32_t)shift
                        : (diff + 126u - (uint32_t)output_idx * 126u) * 2u + (uint32_t)shift);
                local_output_idx[i] = (uint8_t)output_idx;
                bit_offset += output_idx;
            }
        }
    }

    int max_bitstream_length = 0;
    if (use_adm) {
        max_bitstream_length = bit_offset;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            max_bitstream_length = max(max_bitstream_length,
                __shfl_down_sync(0xffffffff, max_bitstream_length, offset));
        }
        max_bitstream_length = (max_bitstream_length + 7) / 8;
        max_bitstream_length = __shfl_sync(0xffffffff, max_bitstream_length, 0);
    }

    const int per_lane_length = use_adm ? max_bitstream_length : (cmp_chunk * RawBytes);

    if (lane == 31) {
        signal_length[bid] = per_lane_length;
        centers[bid] = use_adm ? center : 0u;
        if (use_adm) atomicOr(&adm_flags[bid >> 5], 1u << (31 - (bid & 31)));
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

    if (warp == 0 && lane == 0) excl_sum = 0;
    __syncthreads();
    if (warp > 0) {
        if (!lane) {
            int lookback = warp;
            int loc_excl_sum = 0;
            while (lookback > 0) {
                int status;
                do { status = flag[lookback]; __threadfence(); } while (status == 0);
                if (status == 2) { loc_excl_sum += cmpOffset[lookback]; __threadfence(); break; }
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
        int write_pos = excl_sum * cmp_tblock_size + lane * max_bitstream_length;
        uint8_t* bit_out = bit_signal + write_pos;
        uint64_t acc = 0;
        int acc_bits = 0;
        int out_byte = 0;
        #pragma unroll
        for (int i = 0; i < cmp_chunk; ++i) {
            const int oi = local_output_idx[i];
            if (oi > 0) {
                acc = (acc << oi) | (1ull << (oi - 1));
                acc_bits += oi;
                while (acc_bits >= 8) {
                    bit_out[out_byte++] = (uint8_t)((acc >> (acc_bits - 8)) & 0xFF);
                    acc_bits -= 8;
                    acc &= (1ull << acc_bits) - 1;
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
        for (int i = 0; i < cmp_chunk / 16; ++i) code_int2[i] = local_code_int2[i];
    } else {
        uint4* raw_out = reinterpret_cast<uint4*>(
            bit_signal + excl_sum * cmp_tblock_size + lane * (cmp_chunk * RawBytes));
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            raw_out[i] = make_uint4(vals[i * 4 + 0], vals[i * 4 + 1],
                                    vals[i * 4 + 2], vals[i * 4 + 3]);
        }
    }
}

// U32 1D unfused compression (writes per-thread scratch slots; thrust scan +
// concat_general follow on the host).
__global__ void map_values_kernel_thrust_u32(
    const uint32_t* data, uint8_t* code, uint8_t* bit_signal, uint32_t* centers,
    int* signal_length, uint32_t* adm_flags, int data_size, int lane_slot_bytes, int shift)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = bid * blockDim.x + tid;
    const int lane = idx & 0x1f;
    const int warp = idx >> 5;

    const int base_block_start_idx = warp * cmp_chunk * cmp_tblock_size;
    const int base_thread_start_idx = base_block_start_idx + lane * cmp_chunk;
    const int base_thread_end_idx = base_thread_start_idx + cmp_chunk;
    constexpr int RawBytes = (int)sizeof(uint32_t);

    uint8_t local_code[cmp_chunk] = {0};
    uint8_t local_output_idx[cmp_chunk] = {0};
    uint32_t vals[cmp_chunk] = {0};

    uint64_t local_sum = 0;
    uint32_t local_count = 0;
    uint32_t local_min = 0xFFFFFFFFu;
    uint32_t local_max = 0u;
    const int end = min(base_thread_end_idx, data_size);

    if (base_thread_start_idx < end) {
        local_count = end - base_thread_start_idx;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int off = base_thread_start_idx + i * 4;
            if (off + 4 <= end) {
                const uint4 u = reinterpret_cast<const uint4*>(data)[off / 4];
                vals[i * 4 + 0] = u.x; vals[i * 4 + 1] = u.y;
                vals[i * 4 + 2] = u.z; vals[i * 4 + 3] = u.w;
            } else {
                vals[i * 4 + 0] = (off + 0 < end) ? data[off + 0] : 0u;
                vals[i * 4 + 1] = (off + 1 < end) ? data[off + 1] : 0u;
                vals[i * 4 + 2] = (off + 2 < end) ? data[off + 2] : 0u;
                vals[i * 4 + 3] = (off + 3 < end) ? data[off + 3] : 0u;
            }
        }
        #pragma unroll
        for (int i = 0; i < cmp_chunk; ++i) {
            local_sum += (uint64_t)vals[i];
            if (i < local_count) {
                local_min = min(local_min, vals[i]);
                local_max = max(local_max, vals[i]);
            }
        }
    }

    #pragma unroll
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        local_count += __shfl_down_sync(0xffffffff, local_count, offset);
        local_min = min(local_min, __shfl_down_sync(0xffffffff, local_min, offset));
        local_max = max(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }

    uint32_t center = local_count > 0 ? (uint32_t)(local_sum / local_count) : 0u;
    center = __shfl_sync(0xffffffff, center, 0);
    int use_adm_v = (local_count > 0 && (local_max - local_min) < kThresholdFor<std::uint32_t>()) ? 1 : 0;
    use_adm_v = __shfl_sync(0xffffffff, use_adm_v, 0);
    const bool use_adm = use_adm_v != 0;

    uint8_t* slot = bit_signal + (size_t)idx * lane_slot_bytes;
    int bit_offset = 0;
    int per_lane_length = 0;

    if (use_adm) {
        #pragma unroll
        for (int i = 0; i < cmp_chunk; ++i) {
            if (i < local_count) {
                const uint32_t val = vals[i];
                const uint32_t diff = val > center ? val - center : center - val;
                const int oi = (val == center) ? 1
                    : (int)((diff - 1u) / 126u + 1u);
                local_code[i] = (val == center) ? (uint8_t)shift
                    : (uint8_t)((val > center)
                        ? (diff + 126u - (uint32_t)oi * 126u) * 2u - 1u + (uint32_t)shift
                        : (diff + 126u - (uint32_t)oi * 126u) * 2u + (uint32_t)shift);
                local_output_idx[i] = (uint8_t)oi;
                bit_offset += oi;
            }
        }
        uint64_t acc = 0;
        int acc_bits = 0;
        int out_byte = 0;
        #pragma unroll
        for (int i = 0; i < cmp_chunk; ++i) {
            const int oi = local_output_idx[i];
            if (oi > 0) {
                acc = (acc << oi) | (1ull << (oi - 1));
                acc_bits += oi;
                while (acc_bits >= 8) {
                    slot[out_byte++] = (uint8_t)((acc >> (acc_bits - 8)) & 0xFF);
                    acc_bits -= 8;
                    acc &= (1ull << acc_bits) - 1;
                }
            }
        }
        int max_bits = bit_offset;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            max_bits = max(max_bits, __shfl_down_sync(0xffffffff, max_bits, offset));
        }
        max_bits = __shfl_sync(0xffffffff, max_bits, 0);
        per_lane_length = (max_bits + 7) / 8;

        if (acc_bits > 0) {
            slot[out_byte] = (uint8_t)(((acc << (8 - acc_bits)) & 0xFF) | (0xFF >> acc_bits));
            for (int i = out_byte + 1; i < per_lane_length; ++i) slot[i] = 0;
        } else if (out_byte < per_lane_length) {
            slot[out_byte] = 0xFF;
            for (int i = out_byte + 1; i < per_lane_length; ++i) slot[i] = 0;
        }

        int4* code_int2 = reinterpret_cast<int4*>(code + base_thread_start_idx);
        int4* local_code_int2 = reinterpret_cast<int4*>(local_code);
        #pragma unroll
        for (int i = 0; i < cmp_chunk / 16; ++i) code_int2[i] = local_code_int2[i];
    } else {
        per_lane_length = cmp_chunk * RawBytes;
        uint4* raw_out = reinterpret_cast<uint4*>(slot);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            raw_out[i] = make_uint4(vals[i * 4 + 0], vals[i * 4 + 1],
                                    vals[i * 4 + 2], vals[i * 4 + 3]);
        }
    }

    if (lane == 0) {
        signal_length[bid] = per_lane_length;
        centers[bid] = use_adm ? center : 0u;
        if (use_adm) atomicOr(&adm_flags[bid >> 5], 1u << (31 - (bid & 31)));
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

// Writes the total exclusive prefix sum into output_lengths[gsize].
__global__ void finalize_lengths(int* output_lengths, const int* signal_length, int gsize) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        output_lengths[gsize] = output_lengths[gsize - 1] + signal_length[gsize - 1];
    }
}

// Converts the word-packed adm_flags (block b at word b>>5, bit 31-(b&31)) to
// the byte-packed payload representation (block b at byte b>>3, bit 7-(b&7)).
// Each thread rewrites one 32-bit word as four big-endian bytes.
__global__ void pack_flags(const uint32_t* words, uint8_t* out_bytes,
                           int flags_words, int flags_bytes) {
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= flags_words) return;
    const uint32_t v = words[w];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int byte_idx = w * 4 + j;
        if (byte_idx < flags_bytes) {
            out_bytes[byte_idx] = (uint8_t)((v >> (24 - 8 * j)) & 0xFF);
        }
    }
}

// ---------------------------------------------------------------------------
// Fused decompression: single kernel decodes ADM blocks and copies raw blocks.
// ---------------------------------------------------------------------------
__global__ void decompress(
    uint16_t* decmp_data, uint16_t* centers, uint8_t* codes,
    int* d_output_lengths, uint8_t* d_concatenated_signals,
    const uint8_t* flags, int data_size, int gsize, int shift)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = bid * blockDim.x + tid;

    int lane = idx & 0x1f;
    int warp = idx >> 5;

    if (idx * decmp_chunk >= data_size) return;

    warp = lane < 16 ? warp * 2 : warp * 2 + 1;
    const int use_adm = (flags[warp >> 3] >> (7 - (warp & 7))) & 1;

    if (!use_adm) {
        // Raw block: copy 32 contiguous uint16 (64 bytes) back verbatim. Byte-wise
        // copy: the assembled payload places the signals section at an arbitrary
        // byte offset, so a vectorized int4 load here would be misaligned on sm_80.
        const uint8_t* raw_src = d_concatenated_signals +
                                 d_output_lengths[warp] * cmp_tblock_size +
                                 (idx & 15) * decmp_chunk * sizeof(uint16_t);
        uint8_t* dst = reinterpret_cast<uint8_t*>(decmp_data + idx * decmp_chunk);
        #pragma unroll
        for (int i = 0; i < decmp_chunk * sizeof(uint16_t); ++i) {
            dst[i] = raw_src[i];
        }
        return;
    }

    int end = d_output_lengths[gsize] * cmp_tblock_size;
    int length = d_output_lengths[warp + 1] - d_output_lengths[warp];

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
    #pragma unroll
    for (int i = 0; i < decmp_chunk; ++i) {
        local_codes[i] = codes[dst_start_idx + i];
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

    #pragma unroll
    for (int i = 0; i < decmp_chunk; ++i) {
        decmp_data[dst_start_idx + i] = local_result[i];
    }
}

// U32 1D vectorized decompression: mirrors the uint16 1D kernel byte-for-byte in
// structure, but elements are uint32 (4 bytes) and the center is a uint32 that may
// exceed INT_MAX, so reconstruction is done in unsigned arithmetic.
__global__ void decompress_u32(
    uint32_t* decmp_data, uint32_t* centers, uint8_t* codes,
    int* d_output_lengths, uint8_t* d_concatenated_signals,
    const uint8_t* flags, int data_size, int gsize, int shift)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = bid * blockDim.x + tid;

    int lane = idx & 0x1f;
    int warp = idx >> 5;

    if (idx * decmp_chunk >= data_size) return;

    warp = lane < 16 ? warp * 2 : warp * 2 + 1;
    const int use_adm = (flags[warp >> 3] >> (7 - (warp & 7))) & 1;

    if (!use_adm) {
        // Raw block: copy 32 contiguous uint32 (128 bytes) back verbatim. Byte-wise
        // copy: the assembled payload places the signals section at an arbitrary
        // byte offset, so a vectorized int4 load here would be misaligned on sm_80.
        const uint8_t* raw_src = d_concatenated_signals +
                                 d_output_lengths[warp] * cmp_tblock_size +
                                 (idx & 15) * decmp_chunk * sizeof(uint32_t);
        uint8_t* dst = reinterpret_cast<uint8_t*>(decmp_data + idx * decmp_chunk);
        #pragma unroll
        for (int i = 0; i < decmp_chunk * sizeof(uint32_t); ++i) {
            dst[i] = raw_src[i];
        }
        return;
    }

    int end = d_output_lengths[gsize] * cmp_tblock_size;
    int length = d_output_lengths[warp + 1] - d_output_lengths[warp];

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
    #pragma unroll
    for (int i = 0; i < decmp_chunk; ++i) {
        local_codes[i] = codes[dst_start_idx + i];
    }

    uint32_t local_result[decmp_chunk];
    uint32_t center = centers[warp];

    #pragma unroll
    for (int i = 0; i < decmp_chunk; i++) {
        uint8_t code = local_codes[i];
        uint8_t signal = local_signal[i];
        int diff = (code % 2 == 1) ? ((code - 1) / 2) : (code / 2);
        diff += signal * 126;
        local_result[i] = (code % 2 == 1)
            ? (uint32_t)(center - (uint32_t)diff)
            : center + (uint32_t)diff;
    }

    #pragma unroll
    for (int i = 0; i < decmp_chunk; ++i) {
        decmp_data[dst_start_idx + i] = local_result[i];
    }
}

// ---------------------------------------------------------------------------
// Generalized (dims=2/3) compression/decompression kernels. The ADM math is
// identical to the 1D path; only the block geometry and element->lane mapping
// differ (a tile instead of a contiguous 512-element chunk), mirroring
// cpu/adm/adm.h block_to_coords/block_shape exactly.
// ---------------------------------------------------------------------------

// Fused generalized compression: one warp per block (dims 1/2/3), decoupled
// look-back prefix sum (same machinery as the 1D fused kernel). The ADM math is
// identical for uint16/uint32; T drives the value/center type and Sum the
// accumulator width (uint32 sums past 2^32 for a full block of uint32).
template <typename T>
__global__ void map_values_kernel_general_decoupled(
    const T* data, uint8_t* code, uint8_t* bit_signal, T* centers,
    int* signal_length, uint32_t* adm_flags,
    volatile int* const __restrict__ locOffset,
    volatile int* const __restrict__ cmpOffset,
    volatile int* const __restrict__ flag,
    int dims, int nx, int ny, int nz, int data_size, int shift)
{
    using SumT = typename AdmTraits<T>::Sum;
    constexpr int RawBytes = (int)sizeof(T);

    __shared__ unsigned int excl_sum;
    const int lane = threadIdx.x;
    const int bid = blockIdx.x;  // blockDim.x == 32, so bid is also the warp index

    const AdmBlockGeom g = adm_block_geom(bid, dims, nx, ny, nz);
    const int k0 = lane * g.per_lane;
    const int k1 = min(k0 + g.per_lane, g.block_elems);

    // Pass 1: block sum/count/min/max for center + range decision.
    SumT local_sum = 0;
    uint32_t local_count = 0, local_min = 0xFFFFFFFFu, local_max = 0u;
    if (k0 < k1) {
        int lx, ly, lz;
        size_t idx = adm_elem_flat(k0, g, nx, ny, lx, ly, lz);
        const size_t step_y = (size_t)(nx - g.sx);
        const size_t step_z = (size_t)nx * (size_t)(ny - g.sy);
        for (int k = k0; k < k1; ++k) {
            const uint32_t v = (uint32_t)data[idx];
            local_sum += (SumT)v; ++local_count;
            local_min = min(local_min, v);
            local_max = max(local_max, v);
            ++lx; ++idx;
            if (lx == g.sx) {
                lx = 0; ++ly; idx += step_y;
                if (ly == g.sy) { ly = 0; ++lz; idx += step_z; }
            }
        }
    }
    #pragma unroll
    for (int off = warp_size / 2; off > 0; off >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, off);
        local_count += __shfl_down_sync(0xffffffff, local_count, off);
        local_min = min(local_min, __shfl_down_sync(0xffffffff, local_min, off));
        local_max = max(local_max, __shfl_down_sync(0xffffffff, local_max, off));
    }
    uint32_t center = local_count > 0 ? (uint32_t)(local_sum / local_count) : 0u;
    center = __shfl_sync(0xffffffff, center, 0);
    int use_adm_v = (local_count > 0 && (local_max - local_min) < kThresholdFor<T>()) ? 1 : 0;
    use_adm_v = __shfl_sync(0xffffffff, use_adm_v, 0);
    const bool use_adm = use_adm_v != 0;

    int per_lane_length = 0;
    int bit_offset = 0;

    if (use_adm) {
        // Pass 2: encode, write codes, accumulate per-lane signal bit length.
        if (k0 < k1) {
            int lx, ly, lz;
            size_t idx = adm_elem_flat(k0, g, nx, ny, lx, ly, lz);
            const size_t step_y = (size_t)(nx - g.sx);
            const size_t step_z = (size_t)nx * (size_t)(ny - g.sy);
            for (int k = k0; k < k1; ++k) {
                const uint32_t uv = (uint32_t)data[idx];
                const uint32_t diff = uv > center ? uv - center : center - uv;
                const int output_idx = (uv == center) ? 1
                    : (int)((diff - 1u) / 126u + 1u);
                code[idx] = (uv == center) ? (uint8_t)shift
                    : (uint8_t)((uv > center)
                        ? (diff + 126u - (uint32_t)output_idx * 126u) * 2u - 1u + (uint32_t)shift
                        : (diff + 126u - (uint32_t)output_idx * 126u) * 2u + (uint32_t)shift);
                bit_offset += output_idx;
                ++lx; ++idx;
                if (lx == g.sx) {
                    lx = 0; ++ly; idx += step_y;
                    if (ly == g.sy) { ly = 0; ++lz; idx += step_z; }
                }
            }
        }
        int max_bits = bit_offset;
        #pragma unroll
        for (int off = warp_size / 2; off > 0; off >>= 1) {
            max_bits = max(max_bits, __shfl_down_sync(0xffffffff, max_bits, off));
        }
        max_bits = __shfl_sync(0xffffffff, max_bits, 0);
        per_lane_length = (max_bits + 7) / 8;
    } else {
        per_lane_length = g.per_lane * RawBytes;
    }

    if (lane == 31) {
        signal_length[bid] = per_lane_length;
        centers[bid] = use_adm ? (T)center : (T)0;
        if (use_adm) atomicOr(&adm_flags[bid >> 5], 1u << (31 - (bid & 31)));
        locOffset[bid + 1] = per_lane_length;
        __threadfence();
        if (bid == 0) {
            flag[0] = 2;
            __threadfence();
            flag[1] = 1;
            __threadfence();
        } else {
            flag[bid + 1] = 1;
            __threadfence();
        }
    }
    __syncthreads();

    if (bid == 0 && lane == 0) excl_sum = 0;
    __syncthreads();
    if (bid > 0) {
        if (lane == 0) {
            int lookback = bid;
            int loc_excl_sum = 0;
            while (lookback > 0) {
                int status;
                do { status = flag[lookback]; __threadfence(); } while (status == 0);
                if (status == 2) { loc_excl_sum += cmpOffset[lookback]; __threadfence(); break; }
                if (status == 1) loc_excl_sum += locOffset[lookback];
                lookback--;
                __threadfence();
            }
            excl_sum = loc_excl_sum;
        }
        __syncthreads();
    }
    if (bid > 0) {
        if (lane == 0) cmpOffset[bid] = excl_sum;
        __threadfence();
        if (lane == 0) flag[bid] = 2;
        __threadfence();
    }
    __syncthreads();

    const int dst_base = (int)excl_sum * cmp_tblock_size + lane * per_lane_length;
    if (use_adm) {
        // Pass 3: pack the variable-length unary signals into the final slot.
        uint64_t acc = 0;
        int acc_bits = 0;
        int out_byte = 0;
        if (k0 < k1) {
            int lx, ly, lz;
            size_t idx = adm_elem_flat(k0, g, nx, ny, lx, ly, lz);
            const size_t step_y = (size_t)(nx - g.sx);
            const size_t step_z = (size_t)nx * (size_t)(ny - g.sy);
            for (int k = k0; k < k1; ++k) {
                const uint32_t uv = (uint32_t)data[idx];
                const uint32_t diff = uv > center ? uv - center : center - uv;
                const int oi = (uv == center) ? 1
                    : (int)((diff - 1u) / 126u + 1u);
                acc = (acc << oi) | (1ull << (oi - 1));
                acc_bits += oi;
                while (acc_bits >= 8) {
                    bit_signal[dst_base + out_byte++] = (uint8_t)((acc >> (acc_bits - 8)) & 0xFF);
                    acc_bits -= 8;
                    acc &= (1ull << acc_bits) - 1;
                }
                ++lx; ++idx;
                if (lx == g.sx) {
                    lx = 0; ++ly; idx += step_y;
                    if (ly == g.sy) { ly = 0; ++lz; idx += step_z; }
                }
            }
        }
        if (acc_bits > 0) {
            bit_signal[dst_base + out_byte] = (uint8_t)(((acc << (8 - acc_bits)) & 0xFF) | (0xFF >> acc_bits));
            for (int i = out_byte + 1; i < per_lane_length; ++i) bit_signal[dst_base + i] = 0;
        } else if (out_byte < per_lane_length) {
            bit_signal[dst_base + out_byte] = 0xFF;
            for (int i = out_byte + 1; i < per_lane_length; ++i) bit_signal[dst_base + i] = 0;
        }
    } else {
        // Raw: zero the slot, then write lane_elems little-endian values.
        for (int i = 0; i < per_lane_length; ++i) bit_signal[dst_base + i] = 0;
        if (k0 < k1) {
            int lx, ly, lz;
            size_t idx = adm_elem_flat(k0, g, nx, ny, lx, ly, lz);
            const size_t step_y = (size_t)(nx - g.sx);
            const size_t step_z = (size_t)nx * (size_t)(ny - g.sy);
            for (int k = k0; k < k1; ++k) {
                const uint32_t v = (uint32_t)data[idx];
                #pragma unroll
                for (int b = 0; b < RawBytes; ++b) {
                    bit_signal[dst_base + (k - k0) * RawBytes + b] = (uint8_t)((v >> (8 * b)) & 0xFFu);
                }
                ++lx; ++idx;
                if (lx == g.sx) {
                    lx = 0; ++ly; idx += step_y;
                    if (ly == g.sy) { ly = 0; ++lz; idx += step_z; }
                }
            }
        }
    }
}

// Unfused generalized compression: writes per-thread scratch slots; the host
// runs thrust::exclusive_scan + concat_general afterwards.
template <typename T>
__global__ void map_values_kernel_general_thrust(
    const T* data, uint8_t* code, uint8_t* bit_signal, T* centers,
    int* signal_length, uint32_t* adm_flags,
    int dims, int nx, int ny, int nz, int data_size, int lane_slot_bytes, int shift)
{
    using SumT = typename AdmTraits<T>::Sum;
    constexpr int RawBytes = (int)sizeof(T);

    const int lane = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = bid * cmp_tblock_size + lane;

    const AdmBlockGeom g = adm_block_geom(bid, dims, nx, ny, nz);
    const int k0 = lane * g.per_lane;
    const int k1 = min(k0 + g.per_lane, g.block_elems);

    SumT local_sum = 0;
    uint32_t local_count = 0, local_min = 0xFFFFFFFFu, local_max = 0u;
    if (k0 < k1) {
        int lx, ly, lz;
        size_t fidx = adm_elem_flat(k0, g, nx, ny, lx, ly, lz);
        const size_t step_y = (size_t)(nx - g.sx);
        const size_t step_z = (size_t)nx * (size_t)(ny - g.sy);
        for (int k = k0; k < k1; ++k) {
            const uint32_t v = (uint32_t)data[fidx];
            local_sum += (SumT)v; ++local_count;
            local_min = min(local_min, v);
            local_max = max(local_max, v);
            ++lx; ++fidx;
            if (lx == g.sx) {
                lx = 0; ++ly; fidx += step_y;
                if (ly == g.sy) { ly = 0; ++lz; fidx += step_z; }
            }
        }
    }
    #pragma unroll
    for (int off = warp_size / 2; off > 0; off >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, off);
        local_count += __shfl_down_sync(0xffffffff, local_count, off);
        local_min = min(local_min, __shfl_down_sync(0xffffffff, local_min, off));
        local_max = max(local_max, __shfl_down_sync(0xffffffff, local_max, off));
    }
    uint32_t center = local_count > 0 ? (uint32_t)(local_sum / local_count) : 0u;
    center = __shfl_sync(0xffffffff, center, 0);
    int use_adm_v = (local_count > 0 && (local_max - local_min) < kThresholdFor<T>()) ? 1 : 0;
    use_adm_v = __shfl_sync(0xffffffff, use_adm_v, 0);
    const bool use_adm = use_adm_v != 0;

    uint8_t* slot = bit_signal + (size_t)idx * lane_slot_bytes;
    int bit_offset = 0;
    int per_lane_length = 0;

    if (use_adm) {
        uint64_t acc = 0;
        int acc_bits = 0;
        int out_byte = 0;
        if (k0 < k1) {
            int lx, ly, lz;
            size_t fidx = adm_elem_flat(k0, g, nx, ny, lx, ly, lz);
            const size_t step_y = (size_t)(nx - g.sx);
            const size_t step_z = (size_t)nx * (size_t)(ny - g.sy);
            for (int k = k0; k < k1; ++k) {
                const uint32_t uv = (uint32_t)data[fidx];
                const uint32_t diff = uv > center ? uv - center : center - uv;
                const int oi = (uv == center) ? 1
                    : (int)((diff - 1u) / 126u + 1u);
                code[fidx] = (uv == center) ? (uint8_t)shift
                    : (uint8_t)((uv > center)
                        ? (diff + 126u - (uint32_t)oi * 126u) * 2u - 1u + (uint32_t)shift
                        : (diff + 126u - (uint32_t)oi * 126u) * 2u + (uint32_t)shift);
                bit_offset += oi;
                acc = (acc << oi) | (1ull << (oi - 1));
                acc_bits += oi;
                while (acc_bits >= 8) {
                    slot[out_byte++] = (uint8_t)((acc >> (acc_bits - 8)) & 0xFF);
                    acc_bits -= 8;
                    acc &= (1ull << acc_bits) - 1;
                }
                ++lx; ++fidx;
                if (lx == g.sx) {
                    lx = 0; ++ly; fidx += step_y;
                    if (ly == g.sy) { ly = 0; ++lz; fidx += step_z; }
                }
            }
        }
        int max_bits = bit_offset;
        #pragma unroll
        for (int off = warp_size / 2; off > 0; off >>= 1) {
            max_bits = max(max_bits, __shfl_down_sync(0xffffffff, max_bits, off));
        }
        max_bits = __shfl_sync(0xffffffff, max_bits, 0);
        per_lane_length = (max_bits + 7) / 8;

        if (acc_bits > 0) {
            slot[out_byte] = (uint8_t)(((acc << (8 - acc_bits)) & 0xFF) | (0xFF >> acc_bits));
            for (int i = out_byte + 1; i < per_lane_length; ++i) slot[i] = 0;
        } else if (out_byte < per_lane_length) {
            slot[out_byte] = 0xFF;
            for (int i = out_byte + 1; i < per_lane_length; ++i) slot[i] = 0;
        }
    } else {
        per_lane_length = g.per_lane * RawBytes;
        for (int i = 0; i < per_lane_length; ++i) slot[i] = 0;
        if (k0 < k1) {
            int lx, ly, lz;
            size_t fidx = adm_elem_flat(k0, g, nx, ny, lx, ly, lz);
            const size_t step_y = (size_t)(nx - g.sx);
            const size_t step_z = (size_t)nx * (size_t)(ny - g.sy);
            for (int k = k0; k < k1; ++k) {
                const uint32_t v = (uint32_t)data[fidx];
                #pragma unroll
                for (int b = 0; b < RawBytes; ++b) {
                    slot[(k - k0) * RawBytes + b] = (uint8_t)((v >> (8 * b)) & 0xFFu);
                }
                ++lx; ++fidx;
                if (lx == g.sx) {
                    lx = 0; ++ly; fidx += step_y;
                    if (ly == g.sy) { ly = 0; ++lz; fidx += step_z; }
                }
            }
        }
    }

    if (lane == 0) {
        signal_length[bid] = per_lane_length;
        centers[bid] = use_adm ? (T)center : (T)0;
        if (use_adm) atomicOr(&adm_flags[bid >> 5], 1u << (31 - (bid & 31)));
    }
}

// Unfused generalized concat: copy each lane's scratch slot into the packed
// signals buffer (lane-by-lane within a block, uniform per-lane length).
__global__ void concat_general(
    const uint8_t* d_bit_signals, const int* d_signal_length,
    const int* d_output_lengths, uint8_t* d_concatenated_output,
    int gsize, int lane_slot_bytes)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= gsize * cmp_tblock_size) return;
    const int lane = idx & 31;
    const int warp = idx >> 5;
    const int length = d_signal_length[warp];
    const int src = idx * lane_slot_bytes;
    const int dst = d_output_lengths[warp] * cmp_tblock_size + lane * length;
    for (int i = 0; i < length; ++i) {
        d_concatenated_output[dst + i] = d_bit_signals[src + i];
    }
}

// Generalized decompression (dims 1/2/3): one warp per block; unary-decode the
// lane's signal then reconstruct values from codes/center/signal, or copy raw.
template <typename T>
__global__ void decompress_general(
    T* decmp_data, const T* centers, const uint8_t* codes,
    const int* d_output_lengths, const uint8_t* flags,
    const uint8_t* d_concatenated_signals,
    int dims, int nx, int ny, int nz, int data_size)
{
    constexpr int RawBytes = (int)sizeof(T);
    const int lane = threadIdx.x;
    const int bid = blockIdx.x;
    const int use_adm = (flags[bid >> 3] >> (7 - (bid & 7))) & 1;

    const AdmBlockGeom g = adm_block_geom(bid, dims, nx, ny, nz);
    const int k0 = lane * g.per_lane;
    const int k1 = min(k0 + g.per_lane, g.block_elems);
    if (k0 >= k1) return;
    const int lane_elems = k1 - k0;

    const int length = d_output_lengths[bid + 1] - d_output_lengths[bid];
    if (length <= 0) return;
    const int src_start = d_output_lengths[bid] * cmp_tblock_size + lane * length;

    int lx, ly, lz;
    size_t idx = adm_elem_flat(k0, g, nx, ny, lx, ly, lz);
    const size_t step_y = (size_t)(nx - g.sx);
    const size_t step_z = (size_t)nx * (size_t)(ny - g.sy);

    if (!use_adm) {
        for (int j = 0; j < lane_elems; ++j) {
            const int off = j * RawBytes;
            uint32_t v = 0;
            #pragma unroll
            for (int b = 0; b < RawBytes; ++b) {
                v |= (uint32_t)d_concatenated_signals[src_start + off + b] << (8 * b);
            }
            decmp_data[idx] = (T)v;
            ++lx; ++idx;
            if (lx == g.sx) {
                lx = 0; ++ly; idx += step_y;
                if (ly == g.sy) { ly = 0; ++lz; idx += step_z; }
            }
        }
        return;
    }

    uint8_t local_signal[kMaxLaneElems] = {0};
    int signal_idx = -1;
    for (int off = 0; off < length && signal_idx < lane_elems; ++off) {
        const uint8_t bit_buffer = d_concatenated_signals[src_start + off];
        #pragma unroll
        for (int i = 7; i >= 0 && signal_idx < lane_elems; --i) {
            if ((bit_buffer >> i) & 1) signal_idx++;
            else local_signal[signal_idx]++;
        }
    }

    const uint32_t center = (uint32_t)centers[bid];
    for (int j = 0; j < lane_elems; ++j) {
        const uint8_t c = codes[idx];
        const uint8_t s = local_signal[j];
        int diff = (c & 1) ? ((c - 1) >> 1) : (c >> 1);
        diff += (int)s * 126;
        decmp_data[idx] = (c & 1) ? (T)(center - (uint32_t)diff)
                                 : (T)(center + (uint32_t)diff);
        ++lx; ++idx;
        if (lx == g.sx) {
            lx = 0; ++ly; idx += step_y;
            if (ly == g.sy) { ly = 0; ++lz; idx += step_z; }
        }
    }
}

inline std::size_t num_elements_from_params(const mans::MansParams& p) {
    std::size_t n = p.nx;
    if (p.dims >= 2) n *= p.ny;
    if (p.dims >= 3) n *= p.nz;
    return n;
}

inline int compute_gsize(std::size_t num_elements, const mans::MansParams& params) {
    if (params.dims <= 1) {
        const std::size_t block = static_cast<std::size_t>(cmp_tblock_size) * cmp_chunk;
        return static_cast<int>((num_elements + block - 1) / block);
    }
    const int grid_x = (int)(((std::size_t)params.nx + cmp_block_x - 1) / cmp_block_x);
    const int grid_y = (int)(((std::size_t)params.ny + cmp_block_y - 1) / cmp_block_y);
    if (params.dims == 2) return grid_x * grid_y;
    const int grid_z = (int)(((std::size_t)params.nz + cmp_block_z - 1) / cmp_block_z);
    return grid_x * grid_y * grid_z;
}

} // namespace

std::size_t get_max_u16_payload_bytes(std::size_t num_elements,
                                      const mans::MansParams& params) {
    if (num_elements == 0) {
        return 0;
    }
    // Mirror cpu/adm/adm_utils.h adm_max_compressed_size<T> exactly.
    const std::size_t gsize = static_cast<std::size_t>(compute_gsize(num_elements, params));
    const std::size_t len1 = (gsize + 1) * sizeof(int);
    const std::size_t len2 = gsize * sizeof(std::uint16_t);
    const std::size_t len3 = (gsize + 7) / 8;
    const std::size_t len4 = num_elements * sizeof(std::uint8_t);
    const std::size_t max_block_elements = params.dims <= 1
        ? static_cast<std::size_t>(cmp_tblock_size) * cmp_chunk
        : params.dims == 2
            ? static_cast<std::size_t>(cmp_block_x) * cmp_block_y
            : static_cast<std::size_t>(cmp_block_x) * cmp_block_y * cmp_block_z;
    const std::size_t max_lane_elements =
        (max_block_elements + cmp_tblock_size - 1) / cmp_tblock_size;
    const std::size_t len5 = gsize * cmp_tblock_size * max_lane_elements *
                             max_bytes_signal_per_ele_16b;
    return len1 + len2 + len3 + len4 + len5;
}

void compress_u16_device(const std::uint16_t* d_input, std::size_t length,
                         const mans::MansParams& params, std::uint8_t* d_out,
                         std::size_t& out_size, cudaStream_t stream,
                         AdmCompressTimings* timings) {
    if (length == 0) {
        out_size = 0;
        return;
    }

    const int dims = params.dims;
    const int nx = static_cast<int>(params.nx);
    const int ny = (dims >= 2) ? static_cast<int>(params.ny) : 1;
    const int nz = (dims >= 3) ? static_cast<int>(params.nz) : 1;
    const int num_elements = static_cast<int>(length);
    const int gsize = compute_gsize(length, params);
    const int flags_words = (gsize + 31) / 32;
    const int flags_bytes = (gsize + 7) / 8;
    const bool use_decoupled = gsize <= kFusedGsizeThreshold;
    GpuTimer timer;
    if (timings) {
        *timings = AdmCompressTimings{};
        timings->fused = use_decoupled;
    }

    const int max_lane_elements = (dims <= 1)
        ? cmp_chunk
        : dims == 2
            ? (cmp_block_x * cmp_block_y + cmp_tblock_size - 1) / cmp_tblock_size
            : (cmp_block_x * cmp_block_y * cmp_block_z + cmp_tblock_size - 1) / cmp_tblock_size;
    const int lane_slot_bytes = max_lane_elements * max_bytes_signal_per_ele_16b;
    const std::size_t codes_bytes = (dims <= 1)
        ? static_cast<std::size_t>(gsize) * cmp_tblock_size * cmp_chunk // rounded up for tail
        : static_cast<std::size_t>(num_elements);
    const std::size_t signals_bytes =
        static_cast<std::size_t>(gsize) * cmp_tblock_size * lane_slot_bytes;

    DevBuf d_centers, d_codes, d_signal_length, d_output_lengths, d_adm_flags;
    DevBuf d_concatenated_signals, d_bit_signals, d_locOffset, d_flag;

    d_centers.allocate(static_cast<std::size_t>(gsize) * sizeof(std::uint16_t),
                       "cudaMalloc adm centers");
    d_codes.allocate(codes_bytes, "cudaMalloc adm codes");
    d_signal_length.allocate(static_cast<std::size_t>(gsize) * sizeof(int),
                             "cudaMalloc adm signal_length");
    d_output_lengths.allocate(static_cast<std::size_t>(gsize + 1) * sizeof(int),
                              "cudaMalloc adm output_lengths");
    d_adm_flags.allocate(static_cast<std::size_t>(flags_words) * sizeof(std::uint32_t),
                         "cudaMalloc adm flags words");
    d_concatenated_signals.allocate(signals_bytes, "cudaMalloc adm concatenated signals");

    check_cuda(cudaMemset(d_adm_flags.get(), 0,
                        static_cast<std::size_t>(flags_words) * sizeof(std::uint32_t)),
              "cudaMemset adm flags words");
    check_cuda(cudaMemset(d_concatenated_signals.get(), 255, signals_bytes),
              "cudaMemset adm concatenated signals");

    if (use_decoupled) {
        d_locOffset.allocate(static_cast<std::size_t>(gsize + 1) * sizeof(int),
                             "cudaMalloc adm locOffset");
        d_flag.allocate(static_cast<std::size_t>(gsize + 1) * sizeof(int),
                        "cudaMalloc adm flag");
        check_cuda(cudaMemset(d_flag.get(), 0, static_cast<std::size_t>(gsize + 1) * sizeof(int)),
                  "cudaMemset adm flag");
        check_cuda(cudaMemset(d_locOffset.get(), 0, static_cast<std::size_t>(gsize + 1) * sizeof(int)),
                  "cudaMemset adm locOffset");
        check_cuda(cudaMemset(d_output_lengths.get(), 0,
                            static_cast<std::size_t>(gsize + 1) * sizeof(int)),
                  "cudaMemset adm output_lengths");
        if (timings) timer.begin(stream);
        if (dims <= 1) {
            map_values_kernel_decoupled<<<gsize, cmp_tblock_size, 0, stream>>>(
                d_input, d_codes.as<std::uint8_t>(), d_concatenated_signals.as<std::uint8_t>(),
                d_centers.as<std::uint16_t>(), d_signal_length.as<int>(), d_adm_flags.as<std::uint32_t>(),
                d_locOffset.as<volatile int>(), d_output_lengths.as<volatile int>(),
                d_flag.as<volatile int>(), num_elements, kShift);
        } else {
            map_values_kernel_general_decoupled<std::uint16_t><<<gsize, cmp_tblock_size, 0, stream>>>(
                d_input, d_codes.as<std::uint8_t>(), d_concatenated_signals.as<std::uint8_t>(),
                d_centers.as<std::uint16_t>(), d_signal_length.as<int>(), d_adm_flags.as<std::uint32_t>(),
                d_locOffset.as<volatile int>(), d_output_lengths.as<volatile int>(),
                d_flag.as<volatile int>(), dims, nx, ny, nz, num_elements, kShift);
        }
        if (timings) timings->map_ms = timer.end(stream);
    } else {
        d_bit_signals.allocate(signals_bytes, "cudaMalloc adm bit_signals scratch");
        check_cuda(cudaMemset(d_bit_signals.get(), 0, signals_bytes),
                  "cudaMemset adm bit_signals scratch");
        if (timings) timer.begin(stream);
        if (dims <= 1) {
            map_values_kernel_thrust<<<gsize, cmp_tblock_size, 0, stream>>>(
                d_input, d_codes.as<std::uint8_t>(), d_bit_signals.as<std::uint8_t>(),
                d_centers.as<std::uint16_t>(), d_signal_length.as<int>(), d_adm_flags.as<std::uint32_t>(),
                num_elements, kShift);
        } else {
            map_values_kernel_general_thrust<std::uint16_t><<<gsize, cmp_tblock_size, 0, stream>>>(
                d_input, d_codes.as<std::uint8_t>(), d_bit_signals.as<std::uint8_t>(),
                d_centers.as<std::uint16_t>(), d_signal_length.as<int>(), d_adm_flags.as<std::uint32_t>(),
                dims, nx, ny, nz, num_elements, lane_slot_bytes, kShift);
        }
        if (timings) timings->map_ms = timer.end(stream);

        thrust::device_ptr<int> dev_signal_length(d_signal_length.as<int>());
        thrust::device_ptr<int> dev_output_lengths(d_output_lengths.as<int>());
        // The AMD production path always runs on the legacy default stream, so the
        // default thrust execution policy is correctly ordered with the kernels above.
        if (timings) timer.begin(stream);
        thrust::exclusive_scan(dev_signal_length, dev_signal_length + gsize, dev_output_lengths);
        if (timings) timings->scan_ms = timer.end(stream);

        if (timings) timer.begin(stream);
        if (dims <= 1) {
            const int concat_gsize =
                (num_elements + concat_block_size * cmp_chunk - 1) / (concat_block_size * cmp_chunk);
            concat<<<concat_gsize, concat_block_size, 0, stream>>>(
                d_bit_signals.as<std::uint8_t>(), d_signal_length.as<int>(), d_output_lengths.as<int>(),
                d_concatenated_signals.as<std::uint8_t>(), gsize, num_elements);
        } else {
            const int concat_gsize = static_cast<int>(
                (static_cast<long long>(gsize) * cmp_tblock_size + concat_block_size - 1) /
                concat_block_size);
            concat_general<<<concat_gsize, concat_block_size, 0, stream>>>(
                d_bit_signals.as<std::uint8_t>(), d_signal_length.as<int>(), d_output_lengths.as<int>(),
                d_concatenated_signals.as<std::uint8_t>(), gsize, lane_slot_bytes);
        }
        if (timings) timings->concat_ms = timer.end(stream);
    }

    // Write output_lengths[gsize] = total per-lane bytes.
    if (timings) timer.begin(stream);
    finalize_lengths<<<1, 1, 0, stream>>>(
        d_output_lengths.as<int>(), d_signal_length.as<int>(), gsize);
    if (timings) timings->finalize_ms = timer.end(stream);

    // Read back the total per-lane byte count to size the signals section.
    int total_lanes = 0;
    check_cuda(cudaMemcpyAsync(&total_lanes, d_output_lengths.as<int>() + gsize,
                             sizeof(int), cudaMemcpyDeviceToHost, stream),
              "cudaMemcpyAsync adm output_lengths total");
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize adm compress");

    const std::size_t signals_used =
        static_cast<std::size_t>(total_lanes) * cmp_tblock_size;
    const std::size_t len1 = static_cast<std::size_t>(gsize + 1) * sizeof(int);
    const std::size_t len2 = static_cast<std::size_t>(gsize) * sizeof(std::uint16_t);
    const std::size_t len3 = static_cast<std::size_t>(flags_bytes);
    const std::size_t len4 = static_cast<std::size_t>(num_elements);

    // Assemble the contiguous headerless ADM payload into d_out.
    check_cuda(cudaMemcpyAsync(d_out, d_output_lengths.get(), len1, cudaMemcpyDeviceToDevice, stream),
              "cudaMemcpyAsync adm output_lengths");
    check_cuda(cudaMemcpyAsync(d_out + len1, d_centers.get(), len2, cudaMemcpyDeviceToDevice, stream),
              "cudaMemcpyAsync adm centers");

    const int pack_blocks = (flags_words + 255) / 256;
    if (timings) timer.begin(stream);
    pack_flags<<<pack_blocks, 256, 0, stream>>>(
        d_adm_flags.as<std::uint32_t>(), d_out + len1 + len2, flags_words, flags_bytes);
    if (timings) timings->pack_flags_ms = timer.end(stream);

    check_cuda(cudaMemcpyAsync(d_out + len1 + len2 + len3, d_codes.get(), len4,
                             cudaMemcpyDeviceToDevice, stream),
              "cudaMemcpyAsync adm codes");
    check_cuda(cudaMemcpyAsync(d_out + len1 + len2 + len3 + len4, d_concatenated_signals.get(),
                             signals_used, cudaMemcpyDeviceToDevice, stream),
              "cudaMemcpyAsync adm signals");

    if (timings) {
        timings->total_ms = timings->map_ms + timings->scan_ms + timings->concat_ms +
                            timings->finalize_ms + timings->pack_flags_ms;
    }
    out_size = len1 + len2 + len3 + len4 + signals_used;
}

void decompress_u16_device(const std::uint8_t* d_input, std::size_t length,
                           const mans::MansParams& params, std::uint16_t* d_out,
                           std::size_t& out_size, cudaStream_t stream,
                           AdmDecompressTimings* timings) {
    const int dims = params.dims;
    const std::size_t num_elements = num_elements_from_params(params);
    if (num_elements == 0) {
        out_size = 0;
        return;
    }

    if (timings) *timings = AdmDecompressTimings{};
    GpuTimer timer;

    const int gsize = compute_gsize(num_elements, params);
    const std::size_t len1 = static_cast<std::size_t>(gsize + 1) * sizeof(int);
    const std::size_t len2 = static_cast<std::size_t>(gsize) * sizeof(std::uint16_t);
    const std::size_t len3 = (static_cast<std::size_t>(gsize) + 7) / 8;
    const std::size_t len4 = num_elements;
    if (length < len1 + len2 + len3 + len4) {
        throw std::runtime_error("mans::nv::adm::decompress_u16_device: payload is truncated.");
    }

    uint8_t* payload = const_cast<std::uint8_t*>(d_input);
    if (timings) timer.begin(stream);
    if (dims <= 1) {
        const int decmp_gsize =
            (static_cast<int>(num_elements) + decmp_block_size * decmp_chunk - 1) /
            (decmp_block_size * decmp_chunk);
        decompress<<<decmp_gsize, decmp_block_size, 0, stream>>>(
            d_out,
            reinterpret_cast<std::uint16_t*>(payload + len1),
            payload + len1 + len2 + len3,
            reinterpret_cast<int*>(payload),
            payload + len1 + len2 + len3 + len4,
            payload + len1 + len2,
            static_cast<int>(num_elements), gsize, kShift);
    } else {
        decompress_general<std::uint16_t><<<gsize, cmp_tblock_size, 0, stream>>>(
            d_out,
            reinterpret_cast<const std::uint16_t*>(payload + len1),
            payload + len1 + len2 + len3,
            reinterpret_cast<const int*>(payload),
            payload + len1 + len2,
            payload + len1 + len2 + len3 + len4,
            dims, static_cast<int>(params.nx), static_cast<int>(params.ny),
            static_cast<int>(params.nz), static_cast<int>(num_elements));
    }
    if (timings) timings->decode_ms = timer.end(stream);

    out_size = num_elements * sizeof(std::uint16_t);
}

std::size_t get_max_u32_payload_bytes(std::size_t num_elements,
                                      const mans::MansParams& params) {
    if (num_elements == 0) {
        return 0;
    }
    // Mirror cpu/adm/adm_utils.h adm_max_compressed_size<std::uint32_t> exactly.
    const std::size_t gsize = static_cast<std::size_t>(compute_gsize(num_elements, params));
    const std::size_t len1 = (gsize + 1) * sizeof(int);
    const std::size_t len2 = gsize * sizeof(std::uint32_t);
    const std::size_t len3 = (gsize + 7) / 8;
    const std::size_t len4 = num_elements * sizeof(std::uint8_t);
    const std::size_t max_block_elements = params.dims <= 1
        ? static_cast<std::size_t>(cmp_tblock_size) * cmp_chunk
        : params.dims == 2
            ? static_cast<std::size_t>(cmp_block_x) * cmp_block_y
            : static_cast<std::size_t>(cmp_block_x) * cmp_block_y * cmp_block_z;
    const std::size_t max_lane_elements =
        (max_block_elements + cmp_tblock_size - 1) / cmp_tblock_size;
    const std::size_t len5 = gsize * cmp_tblock_size * max_lane_elements *
                             max_bytes_signal_per_ele_32b;
    return len1 + len2 + len3 + len4 + len5;
}

void compress_u32_device(const std::uint32_t* d_input, std::size_t length,
                         const mans::MansParams& params, std::uint8_t* d_out,
                         std::size_t& out_size, cudaStream_t stream,
                         AdmCompressTimings* timings) {
    if (length == 0) {
        out_size = 0;
        return;
    }

    const int dims = params.dims;
    const int nx = static_cast<int>(params.nx);
    const int ny = (dims >= 2) ? static_cast<int>(params.ny) : 1;
    const int nz = (dims >= 3) ? static_cast<int>(params.nz) : 1;
    const int num_elements = static_cast<int>(length);
    const int gsize = compute_gsize(length, params);
    const int flags_words = (gsize + 31) / 32;
    const int flags_bytes = (gsize + 7) / 8;
    const bool use_decoupled = gsize <= kFusedGsizeThreshold;
    GpuTimer timer;
    if (timings) {
        *timings = AdmCompressTimings{};
        timings->fused = use_decoupled;
    }

    const int max_lane_elements = (dims <= 1)
        ? cmp_chunk
        : dims == 2
            ? (cmp_block_x * cmp_block_y + cmp_tblock_size - 1) / cmp_tblock_size
            : (cmp_block_x * cmp_block_y * cmp_block_z + cmp_tblock_size - 1) / cmp_tblock_size;
    const int lane_slot_bytes = max_lane_elements * max_bytes_signal_per_ele_32b;
    const std::size_t codes_bytes = (dims <= 1)
        ? static_cast<std::size_t>(gsize) * cmp_tblock_size * cmp_chunk // rounded up for tail
        : static_cast<std::size_t>(num_elements);
    const std::size_t signals_bytes =
        static_cast<std::size_t>(gsize) * cmp_tblock_size * lane_slot_bytes;

    DevBuf d_centers, d_codes, d_signal_length, d_output_lengths, d_adm_flags;
    DevBuf d_concatenated_signals, d_bit_signals, d_locOffset, d_flag;

    d_centers.allocate(static_cast<std::size_t>(gsize) * sizeof(std::uint32_t),
                       "cudaMalloc adm u32 centers");
    d_codes.allocate(codes_bytes, "cudaMalloc adm u32 codes");
    d_signal_length.allocate(static_cast<std::size_t>(gsize) * sizeof(int),
                             "cudaMalloc adm u32 signal_length");
    d_output_lengths.allocate(static_cast<std::size_t>(gsize + 1) * sizeof(int),
                              "cudaMalloc adm u32 output_lengths");
    d_adm_flags.allocate(static_cast<std::size_t>(flags_words) * sizeof(std::uint32_t),
                         "cudaMalloc adm u32 flags words");
    d_concatenated_signals.allocate(signals_bytes, "cudaMalloc adm u32 concatenated signals");

    check_cuda(cudaMemset(d_adm_flags.get(), 0,
                        static_cast<std::size_t>(flags_words) * sizeof(std::uint32_t)),
              "cudaMemset adm u32 flags words");
    check_cuda(cudaMemset(d_concatenated_signals.get(), 255, signals_bytes),
              "cudaMemset adm u32 concatenated signals");

    if (use_decoupled) {
        d_locOffset.allocate(static_cast<std::size_t>(gsize + 1) * sizeof(int),
                             "cudaMalloc adm u32 locOffset");
        d_flag.allocate(static_cast<std::size_t>(gsize + 1) * sizeof(int),
                        "cudaMalloc adm u32 flag");
        check_cuda(cudaMemset(d_flag.get(), 0, static_cast<std::size_t>(gsize + 1) * sizeof(int)),
                  "cudaMemset adm u32 flag");
        check_cuda(cudaMemset(d_locOffset.get(), 0, static_cast<std::size_t>(gsize + 1) * sizeof(int)),
                  "cudaMemset adm u32 locOffset");
        check_cuda(cudaMemset(d_output_lengths.get(), 0,
                            static_cast<std::size_t>(gsize + 1) * sizeof(int)),
                  "cudaMemset adm u32 output_lengths");
        if (timings) timer.begin(stream);
        if (dims <= 1) {
            map_values_kernel_decoupled_u32<<<gsize, cmp_tblock_size, 0, stream>>>(
                d_input, d_codes.as<std::uint8_t>(), d_concatenated_signals.as<std::uint8_t>(),
                d_centers.as<std::uint32_t>(), d_signal_length.as<int>(), d_adm_flags.as<std::uint32_t>(),
                d_locOffset.as<volatile int>(), d_output_lengths.as<volatile int>(),
                d_flag.as<volatile int>(), num_elements, kShift);
        } else {
            map_values_kernel_general_decoupled<std::uint32_t><<<gsize, cmp_tblock_size, 0, stream>>>(
                d_input, d_codes.as<std::uint8_t>(), d_concatenated_signals.as<std::uint8_t>(),
                d_centers.as<std::uint32_t>(), d_signal_length.as<int>(), d_adm_flags.as<std::uint32_t>(),
                d_locOffset.as<volatile int>(), d_output_lengths.as<volatile int>(),
                d_flag.as<volatile int>(), dims, nx, ny, nz, num_elements, kShift);
        }
        if (timings) timings->map_ms = timer.end(stream);
    } else {
        d_bit_signals.allocate(signals_bytes, "cudaMalloc adm u32 bit_signals scratch");
        check_cuda(cudaMemset(d_bit_signals.get(), 0, signals_bytes),
                  "cudaMemset adm u32 bit_signals scratch");
        if (timings) timer.begin(stream);
        if (dims <= 1) {
            map_values_kernel_thrust_u32<<<gsize, cmp_tblock_size, 0, stream>>>(
                d_input, d_codes.as<std::uint8_t>(), d_bit_signals.as<std::uint8_t>(),
                d_centers.as<std::uint32_t>(), d_signal_length.as<int>(), d_adm_flags.as<std::uint32_t>(),
                num_elements, lane_slot_bytes, kShift);
        } else {
            map_values_kernel_general_thrust<std::uint32_t><<<gsize, cmp_tblock_size, 0, stream>>>(
                d_input, d_codes.as<std::uint8_t>(), d_bit_signals.as<std::uint8_t>(),
                d_centers.as<std::uint32_t>(), d_signal_length.as<int>(), d_adm_flags.as<std::uint32_t>(),
                dims, nx, ny, nz, num_elements, lane_slot_bytes, kShift);
        }
        if (timings) timings->map_ms = timer.end(stream);

        thrust::device_ptr<int> dev_signal_length(d_signal_length.as<int>());
        thrust::device_ptr<int> dev_output_lengths(d_output_lengths.as<int>());
        if (timings) timer.begin(stream);
        thrust::exclusive_scan(dev_signal_length, dev_signal_length + gsize, dev_output_lengths);
        if (timings) timings->scan_ms = timer.end(stream);

        const int concat_gsize = static_cast<int>(
            (static_cast<long long>(gsize) * cmp_tblock_size + concat_block_size - 1) /
            concat_block_size);
        if (timings) timer.begin(stream);
        concat_general<<<concat_gsize, concat_block_size, 0, stream>>>(
            d_bit_signals.as<std::uint8_t>(), d_signal_length.as<int>(), d_output_lengths.as<int>(),
            d_concatenated_signals.as<std::uint8_t>(), gsize, lane_slot_bytes);
        if (timings) timings->concat_ms = timer.end(stream);
    }

    if (timings) timer.begin(stream);
    finalize_lengths<<<1, 1, 0, stream>>>(
        d_output_lengths.as<int>(), d_signal_length.as<int>(), gsize);
    if (timings) timings->finalize_ms = timer.end(stream);

    int total_lanes = 0;
    check_cuda(cudaMemcpyAsync(&total_lanes, d_output_lengths.as<int>() + gsize,
                             sizeof(int), cudaMemcpyDeviceToHost, stream),
              "cudaMemcpyAsync adm u32 output_lengths total");
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize adm u32 compress");

    const std::size_t signals_used =
        static_cast<std::size_t>(total_lanes) * cmp_tblock_size;
    const std::size_t len1 = static_cast<std::size_t>(gsize + 1) * sizeof(int);
    const std::size_t len2 = static_cast<std::size_t>(gsize) * sizeof(std::uint32_t);
    const std::size_t len3 = static_cast<std::size_t>(flags_bytes);
    const std::size_t len4 = static_cast<std::size_t>(num_elements);

    check_cuda(cudaMemcpyAsync(d_out, d_output_lengths.get(), len1, cudaMemcpyDeviceToDevice, stream),
              "cudaMemcpyAsync adm u32 output_lengths");
    check_cuda(cudaMemcpyAsync(d_out + len1, d_centers.get(), len2, cudaMemcpyDeviceToDevice, stream),
              "cudaMemcpyAsync adm u32 centers");

    const int pack_blocks = (flags_words + 255) / 256;
    if (timings) timer.begin(stream);
    pack_flags<<<pack_blocks, 256, 0, stream>>>(
        d_adm_flags.as<std::uint32_t>(), d_out + len1 + len2, flags_words, flags_bytes);
    if (timings) timings->pack_flags_ms = timer.end(stream);

    check_cuda(cudaMemcpyAsync(d_out + len1 + len2 + len3, d_codes.get(), len4,
                             cudaMemcpyDeviceToDevice, stream),
              "cudaMemcpyAsync adm u32 codes");
    check_cuda(cudaMemcpyAsync(d_out + len1 + len2 + len3 + len4, d_concatenated_signals.get(),
                             signals_used, cudaMemcpyDeviceToDevice, stream),
              "cudaMemcpyAsync adm u32 signals");

    if (timings) {
        timings->total_ms = timings->map_ms + timings->scan_ms + timings->concat_ms +
                            timings->finalize_ms + timings->pack_flags_ms;
    }
    out_size = len1 + len2 + len3 + len4 + signals_used;
}

void decompress_u32_device(const std::uint8_t* d_input, std::size_t length,
                           const mans::MansParams& params, std::uint32_t* d_out,
                           std::size_t& out_size, cudaStream_t stream,
                           AdmDecompressTimings* timings) {
    const int dims = params.dims;
    const std::size_t num_elements = num_elements_from_params(params);
    if (num_elements == 0) {
        out_size = 0;
        return;
    }

    if (timings) *timings = AdmDecompressTimings{};
    GpuTimer timer;

    const int gsize = compute_gsize(num_elements, params);
    const std::size_t len1 = static_cast<std::size_t>(gsize + 1) * sizeof(int);
    const std::size_t len2 = static_cast<std::size_t>(gsize) * sizeof(std::uint32_t);
    const std::size_t len3 = (static_cast<std::size_t>(gsize) + 7) / 8;
    const std::size_t len4 = num_elements;
    if (length < len1 + len2 + len3 + len4) {
        throw std::runtime_error("mans::nv::adm::decompress_u32_device: payload is truncated.");
    }

    uint8_t* payload = const_cast<std::uint8_t*>(d_input);
    if (timings) timer.begin(stream);
    if (dims <= 1) {
        const int decmp_gsize =
            (static_cast<int>(num_elements) + decmp_block_size * decmp_chunk - 1) /
            (decmp_block_size * decmp_chunk);
        decompress_u32<<<decmp_gsize, decmp_block_size, 0, stream>>>(
            d_out,
            reinterpret_cast<std::uint32_t*>(payload + len1),
            payload + len1 + len2 + len3,
            reinterpret_cast<int*>(payload),
            payload + len1 + len2 + len3 + len4,
            payload + len1 + len2,
            static_cast<int>(num_elements), gsize, kShift);
    } else {
        decompress_general<std::uint32_t><<<gsize, cmp_tblock_size, 0, stream>>>(
            d_out,
            reinterpret_cast<const std::uint32_t*>(payload + len1),
            payload + len1 + len2 + len3,
            reinterpret_cast<const int*>(payload),
            payload + len1 + len2,
            payload + len1 + len2 + len3 + len4,
            dims, static_cast<int>(params.nx), static_cast<int>(params.ny),
            static_cast<int>(params.nz), static_cast<int>(num_elements));
    }
    if (timings) timings->decode_ms = timer.end(stream);

    out_size = num_elements * sizeof(std::uint32_t);
}

} // namespace adm
} // namespace nv
} // namespace mans
