// pans_utils.h
#ifndef PANS_UTILS_H
#define PANS_UTILS_H

#include <cstdint>
#include <cstddef>
#include <vector>

// tool function：raw_data or adm_compressed_data -> pans_compressed_data
void pans_compress(
    const uint8_t* inputData,
    size_t inputLen,
    uint8_t* outputData,
    size_t &outputLen,
    double &duration
);

// tool function：pans_compressed_data -> raw_data or adm_compressed_data
void pans_decompress(
    const uint8_t* compressedData,
    size_t compressedLen,
    uint8_t* decompressedData,
    size_t &decompressedLen,
    double &duration
);

// benchmark: internally calls pans_compress, precision uses the macro PANS_PRECISION
void pans_compress_and_benchmark(
    const uint8_t* inputData,
    size_t inputLen,
    uint8_t* outputData,
    size_t &outputLen
);

// benchmark: internally calls pans_decompress, precision uses the macro PANS_PRECISION
void pans_decompress_and_benchmark(
    const uint8_t* compressedData,
    size_t compressedLen,
    uint8_t* decompressedData,
    size_t &decompressedLen
);


void get_compress_and_decompressed_len(
    const uint8_t* compressedData,
    size_t &compress_len,
    size_t &decompressed_len
);
#endif // PANS_UTILS_H