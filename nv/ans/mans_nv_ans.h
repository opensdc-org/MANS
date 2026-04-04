#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "../../mans_defs.h"

namespace mans {
namespace nv {
namespace ans {

void compress_stage_device(
    const std::uint8_t* d_input,
    std::size_t input_size,
    std::uint8_t* d_output,
    std::size_t& output_size,
    std::uint32_t mode,
    const MansParams& params,
    cudaStream_t stream = nullptr
);

void decompress_stage_device(
    const std::uint8_t* d_input,
    std::size_t compressed_size,
    std::uint8_t* d_output,
    std::size_t max_output_size,
    std::size_t& output_size,
    std::uint32_t mode,
    const MansParams& params,
    cudaStream_t stream = nullptr
);

std::size_t get_max_compress_bytes(
    std::size_t input_bytes,
    std::uint32_t mode,
    const MansParams& params
);

} // namespace ans
} // namespace nv
} // namespace mans
