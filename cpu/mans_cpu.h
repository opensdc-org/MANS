#pragma once
#include <vector>
#include <string>
#include "../mans_defs.h" 
namespace mans {
namespace cpu {


void compress_internal(
    const void* input_data,
    size_t length,
    const MansParams& params,
    std::uint8_t* out,
    std::size_t& out_size,
    bool save_adm,
    const std::string& dump_path,
    bool open_benchmark
);


void decompress_internal(
    const void* input_data,
    size_t length,
    const MansParams& params,
    std::uint8_t* out,
    std::size_t& out_size,
    bool save_adm,
    const std::string& dump_path,
    bool open_benchmark
);

}
}