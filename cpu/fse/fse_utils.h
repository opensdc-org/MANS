#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace mans {
namespace cpu {

void fse_compress(
    const std::uint8_t* input_data,
    std::size_t input_len,
    std::uint8_t* output_data,
    std::size_t& output_len,
    double& duration_ms
);

void fse_decompress(
    const std::uint8_t* compressed_data,
    std::size_t compressed_len,
    std::uint8_t* decompressed_data,
    std::size_t& decompressed_len,
    double& duration_ms
);

bool get_fse_compress_and_decompressed_len(
    const std::uint8_t* compressed_data,
    std::size_t compressed_len,
    std::size_t& frame_len,
    std::size_t& decompressed_len,
    std::string* error = nullptr
);

} // namespace cpu
} // namespace mans

