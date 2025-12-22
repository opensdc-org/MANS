// pans_utils.h
#ifndef PANS_UTILS_H
#define PANS_UTILS_H

#include <cstdint>
#include <vector>

// tool function：raw_data or adm_compressed_data -> pans_compressed_data
void pans_compress(
    std::vector<uint8_t>& inputData,
    std::vector<uint8_t>& compressedData,
    uint32_t& batchSize,
    uint32_t& compressedSize
);

// tool function：pans_compressed_data -> raw_data or adm_compressed_data
void pans_decompress(
    std::vector<uint8_t>& compressedData,
    std::vector<uint8_t>& decompressedData,
    uint32_t& batchSize,
    uint32_t& compressedSize
);

// benchmark: internally calls pans_compress, precision uses the macro PANS_PRECISION
void pans_compress_and_benchmark(
    std::vector<uint8_t>& inputData,
    std::vector<uint8_t>& compressedData
);

// benchmark: internally calls pans_decompress, precision uses the macro PANS_PRECISION
void pans_decompress_and_benchmark(
    std::vector<uint8_t>& compressedData,
    std::vector<uint8_t>& decompressedData
);

#endif // PANS_UTILS_H