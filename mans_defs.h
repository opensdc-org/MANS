#pragma once
#include <cstdint>

namespace mans {


struct MansParams {
    uint32_t backend;       // 0: CPU, 1: GPU
    uint32_t dtype;         // 0: U16, 1: U32
    uint32_t adm_threshold; // (block max diff > adm_threshold) -> skip adm mode
    uint32_t adm_decide_threads; // ADM decision (decide_use_adm)

    // --- ADM Compression Threads Config ---
    uint32_t adm_center_calc_threads;    // Center calculation
    uint32_t adm_encode_threads;         // Encoding
    uint32_t adm_warp_reduce_threads;    // Warp reduction
    uint32_t adm_fill_tail_threads;      // Fill tail bits
    uint32_t adm_write_back_threads;     // Write back bit signals

    // --- ADM Decompression Threads Config ---
    uint32_t adm_restore_signals_threads; // Restore signals
    uint32_t adm_decode_values_threads;   // Decode values
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

struct MansHeader {
    std::uint8_t codec;  // 1 = ADM, 2 = ANS
};
// Static assertion: ensure the compiler doesn't add padding so it remains 1 byte in size
static_assert(sizeof(MansHeader) == 1, "MansHeader must be 1 byte");

} // namespace mans
