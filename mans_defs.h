#pragma once
#include <cstddef>
#include <cstdint>

namespace mans {


struct MansParams {
    uint32_t backend = 0;       // 0: CPU, 1: GPU
    uint32_t dtype = 0;         // 0: U16, 1: U32
    uint32_t adm_threshold = 4000; // (block max diff > adm_threshold) -> skip adm mode
    uint32_t adm_decide_threads = 16; // ADM decision (decide_use_adm)

    // --- ADM Compression Threads Config ---
    uint32_t adm_center_calc_threads = 32;    // Center calculation
    uint32_t adm_encode_threads = 32;         // Encoding
    uint32_t adm_warp_reduce_threads = 32;    // Warp reduction
    uint32_t adm_fill_tail_threads = 32;      // Fill tail bits
    uint32_t adm_write_back_threads = 32;     // Write back bit signals

    // --- ADM Decompression Threads Config ---
    uint32_t adm_restore_signals_threads = 32; // Restore signals
    uint32_t adm_decode_values_threads = 32;   // Decode values

    // --- Pipeline mode ---
    // 0: p-mode (ADM -> PANS)
    // 1: r-mode (ADM -> FSE)
    uint32_t mode = 1;
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
    std::uint8_t raw_bytes_le[8];  // little-endian raw byte length
};
static_assert(sizeof(MansHeader) == 10, "MansHeader must be 10 bytes");

constexpr std::size_t kMansHeaderBytes = sizeof(MansHeader);

} // namespace mans
