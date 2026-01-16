#include "adm_utils.h"
#include "adm.h" 
#include <iostream>
#include <cstring>
#include <chrono>
#include <type_traits>
#include <stdexcept>
#include <cstdio> 

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
    std::size_t num_elements = input_len; 
    if (num_elements == 0) {
        output_size = 0;
        return;
    }

    std::uint64_t gsize = (num_elements
        + adm::cmp_tblock_size * adm::cmp_chunk - 1)
        / (adm::cmp_tblock_size * adm::cmp_chunk);

    std::vector<int>              output_lengths(gsize + 1);
    std::vector<T>                centers(gsize);
    std::vector<std::uint8_t>     codes(num_elements);
    std::vector<std::uint8_t>     bit_signals;

    // call adm compress function
    if constexpr (std::is_same_v<T, std::uint16_t>) {
        adm::compress_uint16(input_data, input_len, output_lengths, centers, codes, bit_signals,params);
    } else if constexpr (std::is_same_v<T, std::uint32_t>) {
        adm::compress_uint32(input_data, input_len, output_lengths, centers, codes, bit_signals,params);
    } else {
        static_assert(std::is_same_v<T, std::uint16_t> || std::is_same_v<T, std::uint32_t>,
                      "adm_compress only supports uint16_t and uint32_t");
    }

    adm::FileHeader header;
    header.num_elements = static_cast<std::uint64_t>(num_elements);
    header.gsize        = gsize;

    std::size_t len1 = output_lengths.size() * sizeof(int);
    std::size_t len2 = centers.size()       * sizeof(T);
    std::size_t len3 = codes.size()         * sizeof(std::uint8_t);
    std::size_t len4 = bit_signals.size();

    header.len1 = len1;
    header.len2 = len2;
    header.len3 = len3;
    header.len4 = len4;

    std::size_t len_header = sizeof(header);
    std::size_t total_size = len_header + len1 + len2 + len3 + len4;

    // 1. set output size
    output_size = total_size;

    if (output) {
        std::size_t offset = 0;
        std::memcpy(output + offset, &header,           len_header); offset += len_header;
        std::memcpy(output + offset, output_lengths.data(), len1);  offset += len1;
        std::memcpy(output + offset, centers.data(),        len2);  offset += len2;
        std::memcpy(output + offset, codes.data(),          len3);  offset += len3;
        std::memcpy(output + offset, bit_signals.data(),    len4);
    }
}

template<typename T>
void adm_decompress(
    const std::uint8_t* merged,    
    std::size_t merged_size,       
    T* recovered,
    std::size_t& num_elements,
    const mans::MansParams& params                   
)
{
    if (merged_size < sizeof(adm::FileHeader)) {
        throw std::runtime_error("File too small or invalid format.");
    }

    adm::FileHeader header;
    std::memcpy(&header, merged, sizeof(header));
    
    std::size_t offset = sizeof(header);


    num_elements = static_cast<std::size_t>(header.num_elements);
    std::size_t len1 = static_cast<std::size_t>(header.len1); // output_lengths (bytes)
    std::size_t len2 = static_cast<std::size_t>(header.len2); // centers (bytes)
    std::size_t len3 = static_cast<std::size_t>(header.len3); // codes (bytes)
    std::size_t len4 = static_cast<std::size_t>(header.len4); // bit_signals (bytes)

    if (merged_size < offset + len1 + len2 + len3 + len4) {
        throw std::runtime_error("Corrupted file: not enough data.");
    }


    
    // Part 1: output_lengths (int array)
    const int* output_lengths = reinterpret_cast<const int*>(merged + offset);
    offset += len1;

    // Part 2: centers (T array)
    const T* centers = reinterpret_cast<const T*>(merged + offset);
    offset += len2;

    // Part 3: codes (uint8_t array)
    const std::uint8_t* codes = merged + offset;
    offset += len3;

    // Part 4: bit_signals (uint8_t array)
    const std::uint8_t* bit_signals = merged + offset;
    // offset += len4; 

    if constexpr (std::is_same_v<T, std::uint16_t>) {
        adm::decompress_uint16(
            output_lengths, 
            len1 / sizeof(int), // gsize
            centers, 
            codes, 
            num_elements, 
            bit_signals, 
            recovered,
            params
        );
    } else if constexpr (std::is_same_v<T, std::uint32_t>) {
        adm::decompress_uint32(
            output_lengths, 
            len1 / sizeof(int), // gsize
            centers, 
            codes, 
            num_elements, 
            bit_signals, 
            recovered,
            params
        );
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

    // 1. First do a dry run (without writing any data) to get the exact compressed size.
    // This allows pre-allocating the exact amount of memory and avoids reallocations
    // during benchmarking from affecting the results.
    std::size_t exact_compressed_size = 0;
    adm_compress<T>(input_data, input_len, nullptr, exact_compressed_size,params);

    std::vector<std::uint8_t> tmp_buf(exact_compressed_size);
    std::vector<std::uint8_t> last_tmp_buf(exact_compressed_size);

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

    for (int i = 0; i < times; ++i) {
        std::size_t current_size = 0;
        
        auto start = std::chrono::high_resolution_clock::now();
        adm_compress<T>(input_data, input_len, tmp_buf.data(), current_size,params);
        auto end   = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double, std::milli> dur = (end - start);
        exe_min = std::min(exe_min, static_cast<float>(dur.count()));

        if (!first_run) {
            if (std::memcmp(last_tmp_buf.data(), tmp_buf.data(), exact_compressed_size) != 0) {
                all_same = false;
            }
        }
        std::memcpy(last_tmp_buf.data(), tmp_buf.data(), exact_compressed_size);
        first_run = false;
    }

    if (output != nullptr) {
        std::memcpy(output, last_tmp_buf.data(), exact_compressed_size);
    }

    output_size = exact_compressed_size;

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
    if (exact_compressed_size > 0) {
        cr = num_elements * element_bytes * 1.0 / exact_compressed_size;
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
    std::size_t &num_elements,
    const mans::MansParams& params)
{
    num_elements = 0;
    if constexpr (std::is_same_v<T, std::uint16_t>) {
        std::cout << "\033[0;36m=======> Start ADM Decompress (uint16, benchmark) <=======\033[0m\n";
    } else {
        std::cout << "\033[0;36m=======> Start ADM Decompress (uint32, benchmark) <=======\033[0m\n";
    }

    int   times   = 10;
    float exe_min = 1e30f;

    for (int i = 0; i < times; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        adm_decompress<T>(merged, merged_size, recovered, num_elements,params);
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

template void adm_compress<uint16_t>(const uint16_t*, std::size_t, std::uint8_t*, std::size_t&,const mans::MansParams& params);
template void adm_compress<uint32_t>(const uint32_t*, std::size_t, std::uint8_t*, std::size_t&,const mans::MansParams& params);

template void adm_decompress<uint16_t>(const std::uint8_t*, std::size_t, uint16_t*, std::size_t&,const mans::MansParams& params);
template void adm_decompress<uint32_t>(const std::uint8_t*, std::size_t, uint32_t*, std::size_t&,const mans::MansParams& params);

template void adm_compress_and_benchmark<uint16_t>(const uint16_t*, std::size_t, std::uint8_t*, std::size_t&,const mans::MansParams& params);
template void adm_compress_and_benchmark<uint32_t>(const uint32_t*, std::size_t, std::uint8_t*, std::size_t&,const mans::MansParams& params);

template void adm_decompress_and_benchmark<uint16_t>(const std::uint8_t*, std::size_t, uint16_t*, std::size_t&,const mans::MansParams& params);
template void adm_decompress_and_benchmark<uint32_t>(const std::uint8_t*, std::size_t, uint32_t*, std::size_t&,const mans::MansParams& params);
