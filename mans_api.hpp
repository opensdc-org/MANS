#pragma once
#include <vector>
#include <cstdint>
#include <cstddef>
#include <string>
#include "mans_defs.h"
#include "mans_data_gen.h"

namespace mans {

struct MansAutotuneSweepRow {
    std::size_t chunk_elements = 0;
    std::uint32_t dims = 1;
    std::string mode;
    std::uint32_t threads = 0;
    double throughput_mbps = 0.0;
};

struct MansAutotuneBestConfig {
    std::size_t chunk_elements = 0;
    std::uint32_t dims = 1;
    std::uint32_t compress_thread = 0;
    std::uint32_t decompress_thread = 0;
};

struct MansAutotuneOptions {
    std::vector<double> data_size_mb_list = {
        4.0 / 1024.0,
        8.0 / 1024.0,
        16.0 / 1024.0,
        32.0 / 1024.0,
        64.0 / 1024.0,
        128.0 / 1024.0,
        256.0 / 1024.0,
        512.0 / 1024.0,
        1.0,
        4.0,
        16.0,
        256.0
    };
    std::vector<std::uint32_t> dims_list = {1, 2, 3};
    int threads_min = 1;
    int threads_max = 0;
    int stride = 8;
    std::uint32_t iter = 10;
    data_gen::SyntheticConfig synth_cfg{};
    std::vector<MansAutotuneSweepRow> sweep_rows;
    std::vector<MansAutotuneBestConfig> best_configs;
    bool verbose = false;
};

void compress_device(
    const void* input_data,
    size_t length,
    const MansParams& params,
    uint8_t* out,
    size_t& out_size
);

void decompress_device(
    const void* input_data,
    size_t length,
    const MansParams& params,
    uint8_t* out,
    size_t& out_size
);

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

std::size_t get_mans_max_compress_bytes(
    std::size_t num_elements,
    const MansParams& params
);

std::size_t get_mans_exact_decompress_bytes(
    const void* compressed_data,
    std::size_t compressed_len,
    const MansParams& params
);

void autotune(MansAutotuneOptions& options);

} // namespace mans
