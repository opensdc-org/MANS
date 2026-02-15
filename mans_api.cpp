#include "mans_api.hpp"
#include <stdexcept>
#include <string>
#include <algorithm>
#include <limits>
#include <vector>


#ifdef MANS_ENABLE_CPU
    #include "cpu/mans_cpu.h"
    #include "cpu/adm/adm_utils.h"
    #include "cpu/adm/adm.h"
    #include "cpu/pans/CpuANSUtils.h"
    #include "cpu/pans/pans_utils.h"
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

std::size_t get_mans_max_compress_bytes_p(std::size_t num_elements, const MansParams& params) {
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
            throw std::runtime_error("MANS::get_mans_max_compress_bytes_p: Unsupported dtype.");
        }

        const std::size_t max_u32 = std::numeric_limits<uint32_t>::max();
        if (num_elements > max_u32 / elem_size) {
            throw std::runtime_error("MANS::get_mans_max_compress_bytes_p: raw_bytes exceeds 32-bit limit.");
        }
        const std::size_t raw_bytes = num_elements * elem_size;
        if (raw_bytes > max_u32) {
            throw std::runtime_error("MANS::get_mans_max_compress_bytes_p: raw_bytes exceeds 32-bit limit.");
        }

        std::size_t adm_bytes = 0;
        if (params.dtype == DataType::U16) {
            adm_bytes = adm_max_compressed_size<std::uint16_t>(num_elements);
        } else {
            adm_bytes = adm_max_compressed_size<std::uint32_t>(num_elements);
        }
        if (adm_bytes > max_u32) {
            throw std::runtime_error("MANS::get_mans_max_compress_bytes_p: adm_bytes exceeds 32-bit limit.");
        }

        const std::size_t pans_raw =
            static_cast<std::size_t>(cpu_ans::getMaxCompressedSize(static_cast<uint32_t>(raw_bytes)));
        const std::size_t pans_adm =
            static_cast<std::size_t>(cpu_ans::getMaxCompressedSize(static_cast<uint32_t>(adm_bytes)));

        return 1 + std::max(pans_raw, pans_adm); // 1 byte for codec header
#else
        throw std::runtime_error("MANS::get_mans_max_compress_bytes_p: CPU backend was NOT compiled.");
#endif
    }

    if (params.backend == Backend::NVIDIA) {
#ifdef MANS_ENABLE_NV
        throw std::runtime_error("MANS::get_mans_max_compress_bytes_p: NVIDIA backend not implemented yet.");
#else
        throw std::runtime_error("MANS::get_mans_max_compress_bytes_p: NVIDIA backend was NOT compiled.");
#endif
    }

    throw std::runtime_error("MANS::get_mans_max_compress_bytes_p: Unknown backend type.");
}

std::size_t get_mans_exact_decompress_bytes_p(const void* compressed_data,
                                              std::size_t compressed_len,
                                              const MansParams& params) {
    if (!compressed_data) {
        throw std::runtime_error("MANS::get_mans_exact_decompress_bytes_p: compressed_data is null.");
    }
    if (compressed_len < 1) {
        throw std::runtime_error("MANS::get_mans_exact_decompress_bytes_p: compressed_len too small.");
    }

    if (params.backend == Backend::CPU) {
#ifdef MANS_ENABLE_CPU
        const auto* bytes = static_cast<const std::uint8_t*>(compressed_data);
        const std::uint8_t codec = bytes[0];
        if (compressed_len <= 1) {
            throw std::runtime_error("MANS::get_mans_exact_decompress_bytes_p: missing PANS payload.");
        }

        const std::uint8_t* pans_ptr = bytes + 1;
        const std::size_t pans_len = compressed_len - 1;

        std::size_t pans_compress_len = 0;
        std::size_t pans_decompress_len = 0;
        get_compress_and_decompressed_len(pans_ptr, pans_compress_len, pans_decompress_len);
        if (pans_decompress_len == 0) {
            throw std::runtime_error("MANS::get_mans_exact_decompress_bytes_p: failed to parse PANS header.");
        }
        if (pans_compress_len > pans_len) {
            throw std::runtime_error("MANS::get_mans_exact_decompress_bytes_p: PANS payload truncated.");
        }

        if (codec == 2) {
            return pans_decompress_len; // direct PANS -> raw bytes
        }
        if (codec != 1) {
            throw std::runtime_error("MANS::get_mans_exact_decompress_bytes_p: unknown codec.");
        }

        std::vector<std::uint8_t> adm_blob(pans_decompress_len);
        std::size_t out_len = 0;
        double dur = 0.0;
        pans_decompress(pans_ptr, pans_len, adm_blob.data(), out_len, dur);
        if (out_len < sizeof(adm::FileHeader)) {
            throw std::runtime_error("MANS::get_mans_exact_decompress_bytes_p: ADM header too small.");
        }

        const auto* hdr = reinterpret_cast<const adm::FileHeader*>(adm_blob.data());
        std::size_t elem_size = 0;
        if (params.dtype == DataType::U16) {
            elem_size = sizeof(std::uint16_t);
        } else if (params.dtype == DataType::U32) {
            elem_size = sizeof(std::uint32_t);
        } else {
            throw std::runtime_error("MANS::get_mans_exact_decompress_bytes_p: Unsupported dtype.");
        }

        const std::size_t max_size = std::numeric_limits<std::size_t>::max();
        if (hdr->num_elements > max_size / elem_size) {
            throw std::runtime_error("MANS::get_mans_exact_decompress_bytes_p: size overflow.");
        }
        return static_cast<std::size_t>(hdr->num_elements) * elem_size;
#else
        throw std::runtime_error("MANS::get_mans_exact_decompress_bytes_p: CPU backend was NOT compiled.");
#endif
    }

    if (params.backend == Backend::NVIDIA) {
#ifdef MANS_ENABLE_NV
        throw std::runtime_error("MANS::get_mans_exact_decompress_bytes_p: NVIDIA backend not implemented yet.");
#else
        throw std::runtime_error("MANS::get_mans_exact_decompress_bytes_p: NVIDIA backend was NOT compiled.");
#endif
    }

    throw std::runtime_error("MANS::get_mans_exact_decompress_bytes_p: Unknown backend type.");
}

} // namespace mans
