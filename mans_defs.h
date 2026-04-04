#pragma once
#include <cstddef>
#include <cstdint>

namespace mans {


struct MansParams {
    uint32_t backend = 0;       // 0: CPU, 1: GPU
    uint32_t dtype = 0;         // 0: U16, 1: U32

    // --- ADM Thread Config ---
    uint32_t adm_compress_thread = 32;    // Shared thread count for all ADM compress stages
    uint32_t adm_decompress_thread = 32;  // Shared thread count for all ADM decompress stages

    // --- Pipeline mode ---
    // 0: p-mode (ADM -> PANS)
    // 1: r-mode (ADM -> FSE)
    uint32_t mode = 1;

    // --- 3D Mapping ---
    uint32_t dims=1;
    uint32_t nx=0;
    uint32_t ny=0;
    uint32_t nz=0;
};
    
static_assert(sizeof(MansParams) % 4 == 0, "MansParams size must be multiple of 4 bytes");

namespace Backend {
    constexpr uint32_t CPU = 0;
    constexpr uint32_t NVIDIA = 1;
}

namespace DataType {
    constexpr uint32_t U16 = 0;
    constexpr uint32_t U32 = 1;
}

namespace Mode {
    constexpr uint32_t P = 0;
    constexpr uint32_t R = 1;
}

struct MansHeader {
    std::uint8_t codec;            // 1 = ADM payload, 2 = RAW payload
    std::uint8_t mode;             // 0 = p-mode, 1 = r-mode
    std::uint8_t dims;             // 1/2/3, effective dims used for this chunk
    std::uint8_t raw_bytes_le[8];  // little-endian raw byte length
    std::uint64_t nx;              // effective nx used for this chunk
    std::uint64_t ny;              // effective ny used for this chunk
    std::uint64_t nz;              // effective nz used for this chunk
};

constexpr std::size_t kMansHeaderBytes = sizeof(MansHeader);

} // namespace mans
