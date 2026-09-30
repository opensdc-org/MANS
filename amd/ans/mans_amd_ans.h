#pragma once

#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime.h>

namespace mans {
namespace amd {
namespace ans {

void compress_stage_device(const std::uint8_t* d_input,
                           std::size_t input_size,
                           std::uint8_t* d_output,
                           std::size_t output_capacity,
                           std::size_t& output_size,
                           hipStream_t stream);

void decompress_stage_device(const std::uint8_t* d_input,
                             std::size_t compressed_size,
                             std::uint8_t* d_output,
                             std::size_t output_capacity,
                             std::size_t& output_size,
                             hipStream_t stream);

std::size_t get_max_compress_bytes(std::size_t input_bytes);
std::size_t get_decompressed_bytes(const void* compressed_data, std::size_t compressed_size);

} // namespace ans
} // namespace amd
} // namespace mans
