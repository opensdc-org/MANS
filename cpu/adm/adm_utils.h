// adm_benchmark.h
#ifndef ADM_UTILS_H
#define ADM_UTILS_H

#include <vector>
#include <cstdint>

template<typename T>
void adm_compress(
    const std::vector<T>& input_data,
    std::vector<std::uint8_t>& output
);


template<typename T>
void adm_decompress(
    const std::vector<std::uint8_t>& merged,
    std::vector<T>& recovered
);


template<typename T>
void adm_compress_and_benchmark(
    const std::vector<T>& input_data,
    std::vector<std::uint8_t>& output
);

template<typename T>
void adm_decompress_and_benchmark(
    const std::vector<std::uint8_t>& merged,
    std::vector<T>& recovered
);

#endif // ADM_UTILS_H