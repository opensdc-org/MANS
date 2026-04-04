// adm_benchmark.h
#ifndef ADM_UTILS_H
#define ADM_UTILS_H

#include <vector>
#include <cstdint>
#include <cstddef>
#include <type_traits>
#include "adm.h"
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
    std::size_t num_elements,
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
    std::size_t num_elements,
    const mans::MansParams& params
);

template <typename T>
std::size_t adm_max_compressed_size(std::size_t num_elements,
                                    const mans::MansParams& params) {
    const std::uint32_t dims = params.dims;
    const std::uint32_t nx = params.nx;
    const std::uint32_t ny = params.ny;
    const std::uint32_t nz = params.nz;
    const std::size_t gsize = [&]() -> std::size_t {
        if (dims <= 1) {
            const std::size_t block =
                static_cast<std::size_t>(adm::cmp_tblock_size) * adm::cmp_chunk;
            return (num_elements + block - 1) / block;
        }
        const std::size_t grid_x = (static_cast<std::size_t>(nx) + adm::cmp_block_x - 1) / adm::cmp_block_x;
        const std::size_t grid_y = (static_cast<std::size_t>(ny) + adm::cmp_block_y - 1) / adm::cmp_block_y;
        if (dims == 2) {
            return grid_x * grid_y;
        }
        const std::size_t grid_z = (static_cast<std::size_t>(nz) + adm::cmp_block_z - 1) / adm::cmp_block_z;
        return grid_x * grid_y * grid_z;
    }();
    const std::size_t len1 = (gsize + 1) * sizeof(int);
    const std::size_t len2 = gsize * sizeof(T);
    const std::size_t len3 = (gsize + 7) / 8;
    const std::size_t len4 = num_elements * sizeof(std::uint8_t);

    if constexpr (std::is_same_v<T, std::uint16_t>) {
        const std::size_t max_len5 =
            gsize * adm::cmp_tblock_size * adm::cmp_chunk *
            adm::max_bytes_signal_per_ele_16b;
        return len1 + len2 + len3 + len4 + max_len5;
    } else if constexpr (std::is_same_v<T, std::uint32_t>) {
        const std::size_t max_len5 =
            gsize * adm::cmp_tblock_size * adm::cmp_chunk *
            adm::max_bytes_signal_per_ele_32b;
        return len1 + len2 + len3 + len4 + max_len5;
    } else {
        static_assert(std::is_same_v<T, std::uint16_t> || std::is_same_v<T, std::uint32_t>,
                      "adm_max_compressed_size only supports uint16_t and uint32_t");
        return 0;
    }
}

#endif // ADM_UTILS_H
