#pragma once

#include <cstddef>
#include <cstdint>

namespace mans::dcu::ans {

std::size_t get_max_compressed_size(std::size_t input_bytes);

void compress_device(const std::uint8_t* d_input, std::size_t input_bytes,
                     std::uint8_t* d_output, std::size_t output_capacity,
                     std::size_t& output_bytes);

void decompress_device(const std::uint8_t* d_input, std::size_t input_bytes,
                       std::uint8_t* d_output, std::size_t output_capacity,
                       std::size_t& output_bytes);

} // namespace mans::dcu::ans
