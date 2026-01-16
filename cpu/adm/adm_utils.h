// adm_benchmark.h
#ifndef ADM_UTILS_H
#define ADM_UTILS_H

#include <vector>
#include <cstdint>
#include <cstddef> 
#include "../../mans_defs.h"
template<typename T>
void adm_compress(
    const T* input_data,
    std::size_t input_len,
    std::uint8_t* output,
    std::size_t& output_size,
    const mans::MansParams& params
);

template<typename T>
void adm_decompress(
    const std::uint8_t* merged,
    std::size_t merged_size,
    T* recovered,
    std::size_t& num_elements,
    const mans::MansParams& params
);

template<typename T>
void adm_compress_and_benchmark(
    const T* input_data,
    std::size_t input_len,
    std::uint8_t* output,
    std::size_t& output_size,
    const mans::MansParams& params
);

template<typename T>
void adm_decompress_and_benchmark(
    const std::uint8_t* merged,
    std::size_t merged_size,
    T* recovered,
    std::size_t &num_elements,
    const mans::MansParams& params
);

#endif // ADM_UTILS_H