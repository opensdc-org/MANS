#include "adm_utils.h"
#include "adm.h" 
#include "../../mans_timing.h"
#include <iostream>
#include <cstring>
#include <chrono>
#include <type_traits>
#include <stdexcept>
#include <cstdio> 

namespace {

template <typename T>
void compute_adm_layout(std::size_t num_elements,
                        const mans::MansParams& params,
                        std::size_t& gsize,
                        std::size_t& len1,
                        std::size_t& len2,
                        std::size_t& len3,
                        std::size_t& len4) {
    const int dims = static_cast<int>(params.dims);
    const int nx = static_cast<int>(params.nx);
    const int ny = static_cast<int>(params.ny);
    const int nz = static_cast<int>(params.nz);
    const int ny_eff = (dims >= 2) ? ny : 1;
    const int nz_eff = (dims == 3) ? nz : 1;

    if (dims <= 1) {
        const std::size_t block =
            static_cast<std::size_t>(adm::cmp_tblock_size) * adm::cmp_chunk;
        gsize = (num_elements + block - 1) / block;
    } else if (dims == 2) {
        const std::size_t grid_x =
            (static_cast<std::size_t>(nx) + adm::cmp_block_x - 1) / adm::cmp_block_x;
        const std::size_t grid_y =
            (static_cast<std::size_t>(ny_eff) + adm::cmp_block_y - 1) / adm::cmp_block_y;
        gsize = grid_x * grid_y;
    } else {
        const std::size_t grid_x =
            (static_cast<std::size_t>(nx) + adm::cmp_block_x - 1) / adm::cmp_block_x;
        const std::size_t grid_y =
            (static_cast<std::size_t>(ny_eff) + adm::cmp_block_y - 1) / adm::cmp_block_y;
        const std::size_t grid_z =
            (static_cast<std::size_t>(nz_eff) + adm::cmp_block_z - 1) / adm::cmp_block_z;
        gsize = grid_x * grid_y * grid_z;
    }

    len1 = (gsize + 1) * sizeof(int);
    len2 = gsize * sizeof(T);
    len3 = (gsize + 7) / 8;
    len4 = num_elements * sizeof(std::uint8_t);
}

} // namespace

bool bytes_equal(
    const std::uint8_t* a, std::size_t a_size,
    const std::uint8_t* b, std::size_t b_size)
{
    if (a_size != b_size) {
        return false;
    }
    if (a_size == 0) {
        return true;
    }
    if (a == nullptr || b == nullptr) {
        return false;
    }
    return std::memcmp(a, b, a_size) == 0;
}

// raw data->adm compressed data
template<typename T>
void adm_compress(
    const T* input_data,                
    std::size_t input_len,
    std::uint8_t* output,              
    std::size_t& output_size,
    const mans::MansParams& params
    )          
{
    int dims = params.dims;
    int nx = params.nx;
    int ny = params.ny;
    int nz = params.nz;
    if (dims < 1 || dims > 3) return;
    if (nx <= 0) return;
    if (dims >= 2 && ny <= 0) return;
    if (dims == 3 && nz <= 0) return;

    std::size_t num_elements = input_len; 
    if (num_elements == 0) {
        output_size = 0;
        return;
    }

    // std::uint64_t gsize = (num_elements
    //     + adm::cmp_tblock_size * adm::cmp_chunk - 1)
    //     / (adm::cmp_tblock_size * adm::cmp_chunk);
    constexpr int blk_x = adm::cmp_block_x;
    constexpr int blk_y = adm::cmp_block_y;
    constexpr int blk_z = adm::cmp_block_z;
    const int ny_eff = (dims >= 2) ? ny : 1;
    const int nz_eff = (dims == 3) ? nz : 1;

    std::uint64_t gsize = 0;
    std::uint64_t grid_x = 0, grid_y = 1, grid_z = 1;

    if (dims == 1) {
        // keep original: 32*16
        int block_elems_max = adm::cmp_tblock_size * adm::cmp_chunk; // 512
        gsize = (int)((num_elements + block_elems_max - 1) / block_elems_max);
        grid_x = gsize; grid_y = 1; grid_z = 1; // logical
    } else if (dims == 2) {
        grid_x = (nx + blk_x - 1) / blk_x;
        grid_y = (ny_eff + blk_y - 1) / blk_y;
        grid_z = 1;
        gsize = grid_x * grid_y;
        // block_elems_max = blk_x * blk_y; // 256
    } else { // dims == 3
        grid_x = (nx + blk_x - 1) / blk_x;
        grid_y = (ny_eff + blk_y - 1) / blk_y;
        grid_z = (nz_eff + blk_z - 1) / blk_z;
        gsize = grid_x * grid_y * grid_z;
        // block_elems_max = blk_x * blk_y * blk_z; // 4096
    }

    const std::size_t len1 = (gsize + 1) * sizeof(int);
    const std::size_t len2 = gsize * sizeof(T);
    const std::size_t len3 = (gsize + 7) / 8;
    const std::size_t len4 = num_elements * sizeof(std::uint8_t);

    if (!output) {
        std::cerr << "adm_compress error: output buffer is null.\n";
        output_size = 0;
        return;
    }

    std::size_t offset = 0;
    int* output_lengths_ptr = reinterpret_cast<int*>(output + offset);
    offset += len1;
    T* centers_ptr = reinterpret_cast<T*>(output + offset);
    offset += len2;
    std::uint8_t* flags_ptr = output + offset;
    offset += len3;
    std::uint8_t* codes_ptr = output + offset;
    offset += len4;
    std::uint8_t* bit_signals_ptr = output + offset;
    std::size_t bit_signals_len = 0;
    std::memset(flags_ptr, 0xFF, len3);

    if constexpr (std::is_same_v<T, std::uint16_t>) {
        MANS_TIMING_START("mans/adm_encode_core");
        adm::compress_uint16(input_data, input_len, output_lengths_ptr, centers_ptr, flags_ptr, codes_ptr,
                             bit_signals_ptr, bit_signals_len, params);
        MANS_TIMING_STOP("mans/adm_encode_core");
    } else if constexpr (std::is_same_v<T, std::uint32_t>) {
        MANS_TIMING_START("mans/adm_encode_core");
        adm::compress_uint32(input_data, input_len, output_lengths_ptr, centers_ptr, flags_ptr, codes_ptr,
                             bit_signals_ptr, bit_signals_len, params);
        MANS_TIMING_STOP("mans/adm_encode_core");
    } else {
        static_assert(std::is_same_v<T, std::uint16_t> || std::is_same_v<T, std::uint32_t>,
                      "adm_compress only supports uint16_t and uint32_t");
    }

    output_size = len1 + len2 + len3 + len4 + bit_signals_len;
}

template<typename T>
void adm_decompress(
    const std::uint8_t* merged,    
    std::size_t merged_size,       
    T* recovered,
    std::size_t num_elements,
    const mans::MansParams& params
)
{
    std::size_t gsize = 0;
    std::size_t len1 = 0;
    std::size_t len2 = 0;
    std::size_t len3 = 0;
    std::size_t len4 = 0;
    compute_adm_layout<T>(num_elements, params, gsize, len1, len2, len3, len4);

    std::size_t offset = 0;
    if (merged_size < offset + len1 + len2 + len3 + len4) {
        throw std::runtime_error("Corrupted file: not enough data.");
    }


    
    // Part 1: output_lengths (int array)
    const int* output_lengths = reinterpret_cast<const int*>(merged + offset);
    offset += len1;

    // Part 2: centers (T array)
    const T* centers = reinterpret_cast<const T*>(merged + offset);
    offset += len2;

    // Part 3: flags (uint8_t array)
    const std::uint8_t* flags = merged + offset;
    offset += len3;

    // Part 4: codes (uint8_t array)
    const std::uint8_t* codes = merged + offset;
    offset += len4;

    // Part 5: bit_signals (uint8_t array)
    const std::uint8_t* bit_signals = merged + offset;
    // offset += len4; 

    if constexpr (std::is_same_v<T, std::uint16_t>) {
        MANS_TIMING_START("mans/adm_decode_core");
        adm::decompress_uint16(
            output_lengths, 
            gsize,
            centers, 
            codes, 
            flags,
            num_elements, 
            bit_signals, 
            recovered,
            params
        );
        MANS_TIMING_STOP("mans/adm_decode_core");
    } else if constexpr (std::is_same_v<T, std::uint32_t>) {
        MANS_TIMING_START("mans/adm_decode_core");
        adm::decompress_uint32(
            output_lengths, 
            gsize,
            centers, 
            codes, 
            flags,
            num_elements, 
            bit_signals, 
            recovered,
            params
        );
        MANS_TIMING_STOP("mans/adm_decode_core");
    } else {
        static_assert(std::is_same_v<T, std::uint16_t> || std::is_same_v<T, std::uint32_t>,
                      "adm_decompress only supports uint16_t and uint32_t");
    }
}
// ===== compress_and_benchmark =====
template<typename T>
void adm_compress_and_benchmark(
    const T* input_data,             
    std::size_t input_len, 
    std::uint8_t* output,
    std::size_t& output_size,
    const mans::MansParams& params)
{
    std::size_t num_elements = input_len; 
    if (num_elements == 0) {
        output_size = 0;
        return;
    }

    const std::size_t max_compressed_size = adm_max_compressed_size<T>(input_len, params);
    std::vector<std::uint8_t> tmp_buf(max_compressed_size);
    std::vector<std::uint8_t> last_tmp_buf(max_compressed_size);

    // warmup
    for (int i = 0; i < 5; ++i) {
        std::size_t tmp_sz = 0;
        adm_compress<T>(input_data, input_len, tmp_buf.data(), tmp_sz,params);
    }

    if constexpr (std::is_same_v<T, std::uint16_t>) {
        std::cout << "\033[0;36m=======> Start ADM Compress (uint16, benchmark) <=======\033[0m\n";
    } else {
        std::cout << "\033[0;36m=======> Start ADM Compress (uint32, benchmark) <=======\033[0m\n";
    }

    int   times   = 10;
    float exe_min = 1e30f;
    bool  all_same = true;
    bool  first_run = true;
    std::size_t last_size = 0;

    for (int i = 0; i < times; ++i) {
        std::size_t current_size = 0;
        
        auto start = std::chrono::high_resolution_clock::now();
        adm_compress<T>(input_data, input_len, tmp_buf.data(), current_size,params);
        auto end   = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double, std::milli> dur = (end - start);
        exe_min = std::min(exe_min, static_cast<float>(dur.count()));

        if (!first_run) {
            if (current_size != last_size ||
                std::memcmp(last_tmp_buf.data(), tmp_buf.data(), current_size) != 0) {
                all_same = false;
            }
        }
        std::memcpy(last_tmp_buf.data(), tmp_buf.data(), current_size);
        last_size = current_size;
        first_run = false;
    }

    if (output != nullptr) {
        std::memcpy(output, last_tmp_buf.data(), last_size);
    }

    output_size = last_size;

    if (all_same) {
        std::cout << "\033[32m[adm_compress] all " << times
                  << " runs produce IDENTICAL bitstreams.\033[0m\n";
    } else {
        std::cout << "\033[31m[adm_compress] WARNING: bitstreams differ between runs.\033[0m\n";
    }

    std::size_t element_bytes = sizeof(T);
    double throughput = num_elements * element_bytes * 1.0 / 1024.0 / 1024.0 / (exe_min / 1000.0);
    std::printf("compress cost %.2f ms, throughput %.2f MB/s\n", exe_min, throughput);

    double cr = 0.0;
    if (last_size > 0) {
        cr = num_elements * element_bytes * 1.0 / last_size;
    }
    std::printf("\033[0;36m=======> Compression Ratio <=======\033[0m\n");
    std::printf("CR : %.2f x\n", cr);
}
// ===== decompress_and_benchmark =====
template<typename T>
void adm_decompress_and_benchmark(
    const std::uint8_t* merged,      
    std::size_t merged_size,         
    T* recovered,
    std::size_t num_elements,
    const mans::MansParams& params)
{
    if constexpr (std::is_same_v<T, std::uint16_t>) {
        std::cout << "\033[0;36m=======> Start ADM Decompress (uint16, benchmark) <=======\033[0m\n";
    } else {
        std::cout << "\033[0;36m=======> Start ADM Decompress (uint32, benchmark) <=======\033[0m\n";
    }

    int   times   = 10;
    float exe_min = 1e30f;

    for (int i = 0; i < times; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        adm_decompress<T>(merged, merged_size, recovered, num_elements, params);
        auto end   = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> dur = (end - start);
        exe_min = std::min(exe_min, static_cast<float>(dur.count()));
    }

    std::size_t element_bytes = sizeof(T);
    double throughput = num_elements * element_bytes * 1.0 / 1024.0 / 1024.0 / (exe_min / 1000.0);
    std::printf("decompress cost %.2f ms, throughput %.2f MB/s\n", exe_min, throughput);
}

// ==========================================================
// Explicit Instantiation
// Must be placed at the end of the .cpp file, otherwise the linker cannot find the symbols
// ==========================================================

template void adm_compress<uint16_t>(const uint16_t*, std::size_t, std::uint8_t*, std::size_t&,
                                     const mans::MansParams& params);
template void adm_compress<uint32_t>(const uint32_t*, std::size_t, std::uint8_t*, std::size_t&,
                                     const mans::MansParams& params);


template void adm_decompress<uint16_t>(const std::uint8_t*, std::size_t, uint16_t*, std::size_t,
                                       const mans::MansParams& params);
template void adm_decompress<uint32_t>(const std::uint8_t*, std::size_t, uint32_t*, std::size_t,
                                       const mans::MansParams& params);

template void adm_compress_and_benchmark<uint16_t>(const uint16_t*, std::size_t, std::uint8_t*, std::size_t&,const mans::MansParams& params);
template void adm_compress_and_benchmark<uint32_t>(const uint32_t*, std::size_t, std::uint8_t*, std::size_t&,const mans::MansParams& params);

template void adm_decompress_and_benchmark<uint16_t>(const std::uint8_t*, std::size_t, uint16_t*, std::size_t,const mans::MansParams& params);
template void adm_decompress_and_benchmark<uint32_t>(const std::uint8_t*, std::size_t, uint32_t*, std::size_t,const mans::MansParams& params);
