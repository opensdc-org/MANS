#pragma once
#include <vector>
#include <cstdint>
#include <cstddef>
#include "mans_defs.h"

namespace mans {


void compress(
    const void* input_data, 
    size_t length, 
    const MansParams& params, 
    uint8_t* out,
    size_t & out_size
);


void decompress(
    const void* input_data, 
    size_t length,
    const MansParams& params, 
    uint8_t* out,
    size_t & out_size
);

} // namespace mans