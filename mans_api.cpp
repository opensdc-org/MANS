#include "mans_api.hpp"
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cstring>
#include <limits>
#include <vector>

namespace {

inline std::uint64_t read_le64(const std::uint8_t* p) {
    return static_cast<std::uint64_t>(p[0]) |
           (static_cast<std::uint64_t>(p[1]) << 8) |
           (static_cast<std::uint64_t>(p[2]) << 16) |
           (static_cast<std::uint64_t>(p[3]) << 24) |
           (static_cast<std::uint64_t>(p[4]) << 32) |
           (static_cast<std::uint64_t>(p[5]) << 40) |
           (static_cast<std::uint64_t>(p[6]) << 48) |
           (static_cast<std::uint64_t>(p[7]) << 56);
}

std::size_t parse_raw_bytes_from_header_or_throw(const void* compressed_data,
                                                 std::size_t compressed_len) {
    if (!compressed_data) {
        throw std::runtime_error("MANS::get_mans_exact_decompress_bytes: compressed_data is null.");
    }
    if (compressed_len < mans::kMansHeaderBytes) {
        throw std::runtime_error("MANS::get_mans_exact_decompress_bytes: compressed_len too small.");
    }

    mans::MansHeader header{};
    std::memcpy(&header, compressed_data, sizeof(header));

    if (header.codec != 1 && header.codec != 2) {
        throw std::runtime_error("MANS::get_mans_exact_decompress_bytes: unknown codec.");
    }
    if (header.mode != mans::Mode::P && header.mode != mans::Mode::R) {
        throw std::runtime_error("MANS::get_mans_exact_decompress_bytes: unknown mode in header.");
    }
    const std::uint64_t raw_bytes_u64 = read_le64(header.raw_bytes_le);
    if (raw_bytes_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("MANS::get_mans_exact_decompress_bytes: raw size overflows size_t.");
    }
    return static_cast<std::size_t>(raw_bytes_u64);
}

} // namespace


#ifdef MANS_ENABLE_CPU
    #include "cpu/mans_cpu.h"
    #include "cpu/adm/adm_utils.h"
    #include "cpu/adm/adm.h"
    #include "cpu/pans/CpuANSUtils.h"
    #include "cpu/pans/pans_utils.h"
    #include "cpu/fse/fse_utils.h"
    extern "C" {
        #include "cpu/fse/include/fse.h"
    }
#endif

#ifdef MANS_ENABLE_NV
    // #include "nv/mans_nv.h"
#endif

namespace mans {

void compress(const void* input_data,
              size_t length,
              const MansParams& params,
              uint8_t* out,
              size_t& out_size) {
    // 1. CPU branch
    if (params.backend == Backend::CPU) {
#ifdef MANS_ENABLE_CPU
        mans::cpu::compress_internal(input_data,
                                     length,
                                     params,
                                     out,
                                     out_size,
                                     false,
                                     "");
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

void decompress(const void* input_data,
                size_t length,
                const MansParams& params,
                uint8_t* out,
                size_t& out_size) {
    if (params.backend == Backend::CPU) {
#ifdef MANS_ENABLE_CPU
        mans::cpu::decompress_internal(input_data,
                                       length,
                                       params,
                                       out,
                                       out_size,
                                       false,
                                       "");
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

std::size_t get_mans_max_compress_bytes(std::size_t num_elements, const MansParams& params) {
    if (num_elements == 0) {
        return 0;
    }

    if (params.backend == Backend::CPU) {
#ifdef MANS_ENABLE_CPU
        std::size_t elem_size = 0;
        if (params.dtype == DataType::U16) {
            elem_size = sizeof(std::uint16_t);
        } else if (params.dtype == DataType::U32) {
            elem_size = sizeof(std::uint32_t);
        } else {
            throw std::runtime_error("MANS::get_mans_max_compress_bytes: Unsupported dtype.");
        }

        const std::size_t max_u32 = std::numeric_limits<uint32_t>::max();
        if (num_elements > max_u32 / elem_size) {
            throw std::runtime_error("MANS::get_mans_max_compress_bytes: raw_bytes exceeds 32-bit limit.");
        }
        const std::size_t raw_bytes = num_elements * elem_size;
        if (raw_bytes > max_u32) {
            throw std::runtime_error("MANS::get_mans_max_compress_bytes: raw_bytes exceeds 32-bit limit.");
        }

        std::size_t adm_bytes = 0;
        if (params.dtype == DataType::U16) {
            adm_bytes = adm_max_compressed_size<std::uint16_t>(num_elements);
        } else {
            adm_bytes = adm_max_compressed_size<std::uint32_t>(num_elements);
        }
        if (adm_bytes > max_u32) {
            throw std::runtime_error("MANS::get_mans_max_compress_bytes: adm_bytes exceeds 32-bit limit.");
        }

        const std::size_t pans_raw =
            static_cast<std::size_t>(cpu_ans::getMaxCompressedSize(static_cast<uint32_t>(raw_bytes)));
        const std::size_t pans_adm =
            static_cast<std::size_t>(cpu_ans::getMaxCompressedSize(static_cast<uint32_t>(adm_bytes)));
        const std::size_t fse_raw = FSE_compressBound(raw_bytes);
        const std::size_t fse_adm = FSE_compressBound(adm_bytes);
        if (fse_raw == 0 || fse_adm == 0) {
            throw std::runtime_error("MANS::get_mans_max_compress_bytes: FSE_compressBound overflow.");
        }

        const std::uint32_t mode = (params.mode == Mode::R) ? Mode::R : Mode::P;
        if (mode == Mode::P) {
            return kMansHeaderBytes + std::max(pans_raw, pans_adm);
        }
        if (mode == Mode::R) {
            return kMansHeaderBytes + std::max(fse_raw, fse_adm);
        }
        throw std::runtime_error("MANS::get_mans_max_compress_bytes: Unknown mode.");
#else
        throw std::runtime_error("MANS::get_mans_max_compress_bytes: CPU backend was NOT compiled.");
#endif
    }

    if (params.backend == Backend::NVIDIA) {
#ifdef MANS_ENABLE_NV
        throw std::runtime_error("MANS::get_mans_max_compress_bytes: NVIDIA backend not implemented yet.");
#else
        throw std::runtime_error("MANS::get_mans_max_compress_bytes: NVIDIA backend was NOT compiled.");
#endif
    }

    throw std::runtime_error("MANS::get_mans_max_compress_bytes: Unknown backend type.");
}

std::size_t get_mans_exact_decompress_bytes(const void* compressed_data,
                                            std::size_t compressed_len,
                                            const MansParams& params) {
    if (params.backend == Backend::CPU) {
#ifdef MANS_ENABLE_CPU
        const std::size_t raw_bytes = parse_raw_bytes_from_header_or_throw(compressed_data, compressed_len);
        if (compressed_len <= kMansHeaderBytes) {
            throw std::runtime_error("MANS::get_mans_exact_decompress_bytes: missing payload.");
        }

        std::size_t elem_size = 0;
        if (params.dtype == DataType::U16) {
            elem_size = sizeof(std::uint16_t);
        } else if (params.dtype == DataType::U32) {
            elem_size = sizeof(std::uint32_t);
        } else {
            throw std::runtime_error("MANS::get_mans_exact_decompress_bytes: Unsupported dtype.");
        }

        if (raw_bytes == 0 || raw_bytes % elem_size != 0) {
            throw std::runtime_error("MANS::get_mans_exact_decompress_bytes: invalid raw size in header.");
        }
        return raw_bytes;
#else
        throw std::runtime_error("MANS::get_mans_exact_decompress_bytes: CPU backend was NOT compiled.");
#endif
    }

    if (params.backend == Backend::NVIDIA) {
#ifdef MANS_ENABLE_NV
        throw std::runtime_error("MANS::get_mans_exact_decompress_bytes: NVIDIA backend not implemented yet.");
#else
        throw std::runtime_error("MANS::get_mans_exact_decompress_bytes: NVIDIA backend was NOT compiled.");
#endif
    }

    throw std::runtime_error("MANS::get_mans_exact_decompress_bytes: Unknown backend type.");
}

} // namespace mans
