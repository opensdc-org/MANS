// pans_utils.h
#ifndef PANS_UTILS_H
#define PANS_UTILS_H

#include <cstdint>
#include <vector>

// 纯压缩：inputData -> compressedData，并返回 batchSize / compressedSize
void pans_compress(
    std::vector<uint8_t>& inputData,
    std::vector<uint8_t>& compressedData,
    uint32_t& batchSize,
    uint32_t& compressedSize
);

// 纯解压：compressedData -> decompressedData，并返回 batchSize / compressedSize
void pans_decompress(
    std::vector<uint8_t>& compressedData,
    std::vector<uint8_t>& decompressedData,
    uint32_t& batchSize,
    uint32_t& compressedSize
);

// 仅 benchmark：内部调用 pans_compress，precision 使用宏 PANS_PRECISION
void pans_compress_and_benchmark(
    std::vector<uint8_t>& inputData,
    std::vector<uint8_t>& compressedData
);

// 仅 benchmark：内部调用 pans_decompress，precision 使用宏 PANS_PRECISION
void pans_decompress_and_benchmark(
    std::vector<uint8_t>& compressedData,
    std::vector<uint8_t>& decompressedData
);

#endif // PANS_UTILS_H