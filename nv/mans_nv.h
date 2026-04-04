#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include <cuda_runtime.h>

#include "../mans_defs.h"

namespace mans {
namespace nv {

void compress_internal_device(
    const void* d_input_data,
    size_t length,
    const MansParams& params,
    std::uint8_t* d_out,
    std::size_t& out_size,
    cudaStream_t stream = nullptr
);

void decompress_internal_device(
    const void* d_input_data,
    size_t length,
    const MansParams& params,
    std::uint8_t* d_out,
    std::size_t& out_size,
    cudaStream_t stream = nullptr
);

void compress_internal(
    const void* input_data,
    size_t length,
    const MansParams& params,
    std::uint8_t* out,
    std::size_t& out_size,
    bool save_adm,
    const std::string& dump_path
);

void decompress_internal(
    const void* input_data,
    size_t length,
    const MansParams& params,
    std::uint8_t* out,
    std::size_t& out_size,
    bool save_adm,
    const std::string& dump_path
);

std::size_t get_max_compress_bytes(
    std::size_t num_elements,
    const MansParams& params
);

std::size_t get_exact_decompress_bytes(
    const void* compressed_data,
    std::size_t compressed_len,
    const MansParams& params
);

} // namespace nv
} // namespace mans
