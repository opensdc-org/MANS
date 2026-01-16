#include "mans_api.hpp"
#include <stdexcept>
#include <string>


#ifdef MANS_ENABLE_CPU
    #include "cpu/mans_cpu.h"
#endif

#ifdef MANS_ENABLE_NV
    // #include "nv/mans_nv.h"
#endif

namespace mans {

void compress(const void* input_data, size_t length, const MansParams& params, uint8_t* out,size_t& out_size) {
    // 1. CPU branch
    if (params.backend == Backend::CPU) {
#ifdef MANS_ENABLE_CPU
        mans::cpu::compress_internal(input_data, length, params, out,out_size, false, "", false);
        return;
#else
        throw std::runtime_error("MANS::compress: CPU backend was NOT compiled.");
#endif
    }
    
    // 2. NV branch
    if (params.backend == Backend::NVIDIA) {
#ifdef MANS_ENABLE_NV
        // mans::nv::compress_internal(...);
        throw std::runtime_error("MANS::compress: NVIDIA backend not implemented yet.");
        return;
#else
        throw std::runtime_error("MANS::compress: NVIDIA backend was NOT compiled.");
#endif
    }

    throw std::runtime_error("MANS::compress: Unknown backend type.");
}

void decompress(const void* input_data, size_t length, const MansParams& params, uint8_t* out,size_t& out_size) {
    if (params.backend == Backend::CPU) {
#ifdef MANS_ENABLE_CPU
        mans::cpu::decompress_internal(input_data, length, params, out,out_size, false, "", false);
        return;
#else
        throw std::runtime_error("MANS::decompress: CPU backend was NOT compiled.");
#endif
    }
    
    if (params.backend == Backend::NVIDIA) {
#ifdef MANS_ENABLE_NV
        // mans::nv::decompress_internal(...);
        throw std::runtime_error("MANS::decompress: NVIDIA backend not implemented yet.");
        return;
#else
        throw std::runtime_error("MANS::decompress: NVIDIA backend was NOT compiled.");
#endif
    }

    throw std::runtime_error("MANS::decompress: Unknown backend type.");
}

} // namespace mans