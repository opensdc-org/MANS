#pragma once
#include <vector>
#include <cstdint>
#include <cstddef>
#include "mans_defs.h"
#include "mans_data_gen.h"

namespace mans {


void compress(
    const void* input_data,
    size_t length,
    const MansParams& params,
    uint8_t* out,
    size_t& out_size
);


void decompress(
    const void* input_data,
    size_t length,
    const MansParams& params,
    uint8_t* out,
    size_t& out_size
);

std::size_t get_mans_max_compress_bytes_p(
    std::size_t num_elements,
    const MansParams& params
);

std::size_t get_mans_exact_decompress_bytes_p(
    const void* compressed_data,
    std::size_t compressed_len,
    const MansParams& params
);

} // namespace mans