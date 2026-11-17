#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "../../mans_defs.h"

namespace mans {
namespace nv {
namespace adm {

// GPU ADM (uint16/uint32, dims 1/2/3) — production payload format matches cpu/adm/adm.h
// byte-for-byte so CPU and GPU ADM are interchangeable for the same input:
//
//   [output_lengths  (gsize + 1) * int32]  exclusive prefix sums of per-lane
//                                          byte lengths; [gsize] == total lanes
//   [centers         gsize * uint16]        block center (0 for raw blocks)
//   [flags           ceil(gsize/8) bytes]   bit b == 1 iff block b is ADM
//   [codes           num_elements * uint8]  ADM residual codes (raw: unused)
//   [signals         output_lengths[gsize] * 32 bytes]
//                                          ADM bit-signals / raw uint16 LE
//
// Compression picks the fused (decoupled look-back) kernel when
// gsize <= kFusedGsizeThreshold and the unfused (thrust scan + concat) kernel
// otherwise, then decides per block whether to ADM-encode or store raw
// (kThreshold = 3500, mirroring the NV encode kernel). Decompression is always
// the fused kernel.

// GPU kernel-time breakdown (cudaEvent-based). Pass a non-null pointer to one of
// the device entry points below to collect per-kernel times; nullptr keeps the
// production path unchanged. All values are in milliseconds.
struct AdmCompressTimings {
    bool fused = false;         // true = decoupled single-kernel path was taken
    double map_ms = 0.0;        // fused: decoupled kernel; unfused: mapping kernel
    double scan_ms = 0.0;       // unfused only: thrust exclusive_scan
    double concat_ms = 0.0;     // unfused only: concat kernel
    double finalize_ms = 0.0;   // finalize_lengths kernel (both paths)
    double pack_flags_ms = 0.0; // pack_flags kernel (both paths)
    double total_ms = 0.0;      // map + scan + concat + finalize + pack_flags
};

struct AdmDecompressTimings {
    double decode_ms = 0.0;     // fused decode kernel
};

void compress_u16_device(const std::uint16_t* d_input, std::size_t length,
                         const mans::MansParams& params,
                         std::uint8_t* d_out, std::size_t& out_size,
                         cudaStream_t stream,
                         AdmCompressTimings* timings = nullptr);

void decompress_u16_device(const std::uint8_t* d_input, std::size_t length,
                           const mans::MansParams& params,
                           std::uint16_t* d_out, std::size_t& out_size,
                           cudaStream_t stream,
                           AdmDecompressTimings* timings = nullptr);

std::size_t get_max_u16_payload_bytes(std::size_t num_elements,
                                      const mans::MansParams& params);

void compress_u32_device(const std::uint32_t* d_input, std::size_t length,
                         const mans::MansParams& params,
                         std::uint8_t* d_out, std::size_t& out_size,
                         cudaStream_t stream,
                         AdmCompressTimings* timings = nullptr);

void decompress_u32_device(const std::uint8_t* d_input, std::size_t length,
                           const mans::MansParams& params,
                           std::uint32_t* d_out, std::size_t& out_size,
                           cudaStream_t stream,
                           AdmDecompressTimings* timings = nullptr);

std::size_t get_max_u32_payload_bytes(std::size_t num_elements,
                                      const mans::MansParams& params);

} // namespace adm
} // namespace nv
} // namespace mans
