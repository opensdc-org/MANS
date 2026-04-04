#include "mans_cpu.h"
#include <iostream>
#include <cstring>
#include <limits>
#include <algorithm>
#include <new>
#include <memory>
#include <cstdlib>

#include "adm/adm_utils.h"
#include "pans/pans_utils.h"
#include "pans/CpuANSUtils.h"
#include "fse/fse_utils.h"
extern "C" {
#include "fse/include/fse.h"
}
#include "../mans_utils.h"
#include "buffer_cache.h"
#include "../mans_timing.h"
#define DEBUG_PRINT(msg) \
    std::cerr << "\033[1;35m[PLUGIN-CORE]\033[0m " << msg << "\n"

namespace mans {
namespace cpu {

static std::uint32_t normalize_mode(std::uint32_t mode) {
    if (mode == Mode::R) {
        return Mode::R;
    }
    return Mode::P;
}

// ==========================================
// 2. Decompress Helper Function
// ==========================================

static bool parse_header(const uint8_t* data,
                         size_t length,
                         MansHeader& out_header,
                         std::size_t& out_raw_bytes) {
    std::string error;
    if (!mans::parse_mans_header(data, length, out_header, out_raw_bytes, &error)) {
        std::cerr << "[Error] " << error << ".\n";
        return false;
    }
    return true;
}

// ==========================================
// 3. Core Compress/Decompress Loginic
// ==========================================

template<typename T>
void do_compress_t(
    const T* data_ptr,
    size_t length,
    const MansParams& params,
    std::uint8_t* final_out,
    std::size_t& final_out_size,
    bool save_adm,
    const std::string& dump_path
) {
    const std::uint32_t mode = normalize_mode(params.mode);
    constexpr std::uint8_t codec_code = 1; // ADM

    final_out_size = 0;
    if (!final_out) {
        return;
    }

    // second-stage input pointer/length (raw bytes or ADM blob)
    const std::uint8_t* stage2_in_ptr = nullptr;
    std::size_t stage2_in_len = 0;


    std::uint8_t* mans_intermediate_buf_local = nullptr;
    std::size_t adm_cap = 0;

    if (length > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        std::cerr << "[Error] Input size overflow in do_compress_t.\n";
        return;
    }
    const std::size_t raw_bytes = length * sizeof(T);

    try {
            adm_cap = adm_max_compressed_size<T>(length, params);
            mans_intermediate_buf_local =
                mans::cpu::BufferCache::instance().get_t<std::uint8_t>(
                    "mans_adm_intermediate", adm_cap);
            if (!mans_intermediate_buf_local) {
                std::cerr << "[Error] Out of memory during alloc_adm_buf.\n";
                return;
            }
            std::size_t adm_size = 0;
            {
                MANS_TIMING_SCOPE("adm_compress");
                adm_compress<T>(data_ptr, length, mans_intermediate_buf_local, adm_size, params);
            }
            if (adm_size > adm_cap) {
                std::cerr << "[Error] adm_buf overflow: adm_size > adm_cap.\n";
                return;
            }

            if (save_adm && !dump_path.empty()) {
                std::vector<std::uint8_t> tmp(
                    mans_intermediate_buf_local,
                    mans_intermediate_buf_local + adm_size);
                mans::save_u8_file(dump_path, tmp);
            }

            stage2_in_ptr = mans_intermediate_buf_local;
            stage2_in_len = adm_size;
        std::size_t stage2_out_len = 0;
        double stage2_dur = 0.0;
        if (mode == Mode::P) {
            MANS_TIMING_SCOPE("ans_compress");
            pans_compress(
                stage2_in_ptr,
                stage2_in_len,
                final_out + kMansHeaderBytes, // reserve header
                stage2_out_len,
                stage2_dur
            );
        } else {
            MANS_TIMING_SCOPE("fse_compress");
            fse_compress(
                stage2_in_ptr,
                stage2_in_len,
                final_out + kMansHeaderBytes, // reserve header
                stage2_out_len,
                stage2_dur
            );
        }
        if (stage2_out_len == 0) {
            return;
        }

        MansHeader header{};
        header.codec = codec_code;
        header.mode = static_cast<std::uint8_t>(mode);
        header.dims = static_cast<std::uint8_t>(params.dims);
        mans::write_le64(header.raw_bytes_le, static_cast<std::uint64_t>(raw_bytes));
        header.nx = static_cast<std::uint64_t>(params.nx);
        header.ny = static_cast<std::uint64_t>(params.ny);
        header.nz = static_cast<std::uint64_t>(params.nz);
        std::memcpy(final_out, &header, sizeof(header));
        final_out_size = kMansHeaderBytes + stage2_out_len;
    }
    catch (const std::bad_alloc&) {
        std::cerr << "[Error] Out of memory during do_compress_t.\n";
        final_out_size = 0;
    }

}

template<typename T>
void do_decompress_t(
    const uint8_t* input_ptr,
    size_t length,
    std::uint8_t* final_out,
    std::size_t& final_out_size,
    const MansParams& params,

    bool save_adm,
    const std::string& dump_path
) {
    const std::size_t out_capacity = final_out_size;
    final_out_size = 0;

    MansHeader header{};
    std::size_t raw_bytes = 0;
    if (!parse_header(input_ptr, length, header, raw_bytes)) {
        return;
    }
    if (!final_out) {
        std::cerr << "[Error] final_out is null.\n";
        return;
    }
    if (length <= kMansHeaderBytes) {
        std::cerr << "[Error] payload is empty.\n";
        return;
    }
    if (raw_bytes == 0 || (raw_bytes % sizeof(T)) != 0) {
        std::cerr << "[Error] Invalid raw size in mans header.\n";
        return;
    }

    MansParams effective_params = params;
    effective_params.dims = static_cast<std::uint32_t>(header.dims);
    effective_params.nx = static_cast<std::uint32_t>(header.nx);
    effective_params.ny = static_cast<std::uint32_t>(header.ny);
    effective_params.nz = static_cast<std::uint32_t>(header.nz);

    const uint8_t* payload_ptr = input_ptr + kMansHeaderBytes;
    size_t payload_len = length - kMansHeaderBytes;

    std::uint8_t* stage2_dec_buf = nullptr;

    try {
        std::size_t stage2_decomp_len = 0;
        const std::uint32_t mode = static_cast<std::uint32_t>(header.mode);
        if (mode == Mode::P) {
            std::size_t pans_comp_len = payload_len;
            get_compress_and_decompressed_len(payload_ptr, pans_comp_len, stage2_decomp_len);
            if (stage2_decomp_len == 0 || pans_comp_len != payload_len) {
                std::cerr << "[Error] Invalid PANS payload.\n";
                return;
            }
            {
                MANS_TIMING_SCOPE("alloc_pans_decomp_buf");
                stage2_dec_buf =
                    mans::cpu::BufferCache::instance().get_t<std::uint8_t>(
                        "mans_pans_decomp", stage2_decomp_len);
            }
            if (!stage2_dec_buf) {
                std::cerr << "[Error] Out of memory.\n";
                return;
            }
            double pans_dur = 0.0;
            {
                MANS_TIMING_SCOPE("ans_decompress");
                pans_decompress(payload_ptr, payload_len, stage2_dec_buf, stage2_decomp_len, pans_dur);
            }
            if (stage2_decomp_len == 0) {
                std::cerr << "[Error] PANS decompress failed.\n";
                return;
            }
        } else {
            std::size_t frame_len = 0;
            std::string parse_error;
            if (!get_fse_compress_and_decompressed_len(payload_ptr, payload_len, frame_len,
                                                       stage2_decomp_len, &parse_error)) {
                std::cerr << "[Error] " << parse_error << "\n";
                return;
            }
            if (frame_len != payload_len) {
                std::cerr << "[Error] Invalid FSE frame length.\n";
                return;
            }
            {
                MANS_TIMING_SCOPE("alloc_fse_decomp_buf");
                stage2_dec_buf =
                    mans::cpu::BufferCache::instance().get_t<std::uint8_t>(
                        "mans_fse_decomp", stage2_decomp_len);
            }
            if (!stage2_dec_buf) {
                std::cerr << "[Error] Out of memory.\n";
                return;
            }
            double fse_dur = 0.0;
            {
                MANS_TIMING_SCOPE("fse_decompress");
                fse_decompress(payload_ptr, payload_len, stage2_dec_buf, stage2_decomp_len, fse_dur);
            }
            if (stage2_decomp_len == 0) {
                std::cerr << "[Error] FSE decompress failed.\n";
                return;
            }
        }

        if (header.codec == 2) {
            // Direct Mode
            if (stage2_decomp_len != raw_bytes) {
                std::cerr << "[Error] Raw size mismatch in direct payload.\n";
                return;
            }
            if (raw_bytes > out_capacity) {
                std::cerr << "[Error] Output buffer too small for direct payload.\n";
                return;
            }
            if (raw_bytes > 0) {
                std::memcpy(final_out, stage2_dec_buf, raw_bytes);
            }
            final_out_size = raw_bytes;
        }
        else if (header.codec == 1) {
            // ADM Mode
            if (save_adm && !dump_path.empty()) {
                std::vector<std::uint8_t> tmp(stage2_dec_buf, stage2_dec_buf + stage2_decomp_len);
                mans::save_u8_file(dump_path, tmp);
            }
            const std::size_t num_elements = raw_bytes / sizeof(T);
            const std::size_t expected_bytes = num_elements * sizeof(T);
            if (expected_bytes > out_capacity) {
                std::cerr << "[Error] Output buffer too small for ADM payload.\n";
                return;
            }
            T* recovered = reinterpret_cast<T*>(final_out);

            {
                MANS_TIMING_SCOPE("adm_decompress");
                adm_decompress<T>(stage2_dec_buf, stage2_decomp_len, recovered,
                                  num_elements, effective_params);
            }
            final_out_size = expected_bytes;
            if (final_out_size != raw_bytes) {
                std::cerr << "[Error] Raw size mismatch after ADM decompression.\n";
                final_out_size = 0;
                return;
            }
        }
        else {
            std::cerr << "[Error] Unknown codec type: " << int(header.codec) << "\n";
        }
    }
    catch (const std::bad_alloc&) {
        std::cerr << "[Error] Out of memory.\n";
        final_out_size = 0;
        return; 
    }
    catch (const std::exception& e) { 
        std::cerr << "[Error] An exception occurred: " << e.what() << "\n";
        final_out_size = 0;
        return;
    }
    catch (...) { 
        std::cerr << "[Error] An unknown exception occurred.\n";
        final_out_size = 0;
        return;
    }
}

// ==========================================
// 5. Exposed implementation interface
// ==========================================

void compress_internal(
    const void* input_data,
    size_t length,
    const MansParams& params,  
    std::uint8_t* out,
    std::size_t& out_size,
    bool save_adm,
    const std::string& dump_path
) {
    if (params.dtype == DataType::U16) {
        do_compress_t(
            static_cast<const uint16_t*>(input_data),
            length,
            params,
            out,
            out_size,
            save_adm,
            dump_path
        );
    } else if (params.dtype == DataType::U32) {
        do_compress_t(
            static_cast<const uint32_t*>(input_data),
            length,
            params,
            out,
            out_size,
            save_adm,
            dump_path
        );
    }
}

void decompress_internal(
    const void* input_data,
    size_t length,
    const MansParams& params,
    std::uint8_t* out,
    std::size_t& out_size,
    bool save_adm,
    const std::string& dump_path
) {
    const uint8_t* ptr = static_cast<const uint8_t*>(input_data);

    if (params.dtype == DataType::U16) {
        do_decompress_t<uint16_t>(
            ptr, length, out, out_size, params, save_adm, dump_path
        );
    } else if (params.dtype == DataType::U32) {
        do_decompress_t<uint32_t>(
            ptr, length, out, out_size, params, save_adm, dump_path
        );
    }
}


std::size_t get_max_compress_bytes(std::size_t num_elements, const MansParams& params) {
    if (num_elements == 0) {
        return 0;
    }

    std::size_t elem_size = 0;
    if (!mans::get_dtype_size(params.dtype, elem_size)) {
        throw std::runtime_error("mans::cpu::get_max_compress_bytes: Unsupported dtype.");
    }

    const std::size_t max_u32 = std::numeric_limits<std::uint32_t>::max();
    if (num_elements > max_u32 / elem_size) {
        throw std::runtime_error("mans::cpu::get_max_compress_bytes: raw_bytes exceeds 32-bit limit.");
    }
    const std::size_t raw_bytes = num_elements * elem_size;
    if (raw_bytes > max_u32) {
        throw std::runtime_error("mans::cpu::get_max_compress_bytes: raw_bytes exceeds 32-bit limit.");
    }

    std::size_t adm_bytes = 0;
    if (params.dtype == DataType::U16) {
        adm_bytes = adm_max_compressed_size<std::uint16_t>(num_elements, params);
    } else {
        adm_bytes = adm_max_compressed_size<std::uint32_t>(num_elements, params);
    }
    if (adm_bytes > max_u32) {
        throw std::runtime_error("mans::cpu::get_max_compress_bytes: adm_bytes exceeds 32-bit limit.");
    }

    const std::size_t pans_raw =
        static_cast<std::size_t>(cpu_ans::getMaxCompressedSize(static_cast<std::uint32_t>(raw_bytes)));
    const std::size_t pans_adm =
        static_cast<std::size_t>(cpu_ans::getMaxCompressedSize(static_cast<std::uint32_t>(adm_bytes)));
    const std::size_t fse_raw = FSE_compressBound(raw_bytes);
    const std::size_t fse_adm = FSE_compressBound(adm_bytes);
    if (fse_raw == 0 || fse_adm == 0) {
        throw std::runtime_error("mans::cpu::get_max_compress_bytes: FSE_compressBound overflow.");
    }

    const std::uint32_t mode = normalize_mode(params.mode);
    if (mode == Mode::P) {
        return kMansHeaderBytes + std::max(pans_raw, pans_adm);
    }
    if (mode == Mode::R) {
        return kMansHeaderBytes + std::max(fse_raw, fse_adm);
    }
    throw std::runtime_error("mans::cpu::get_max_compress_bytes: Unknown mode.");
}

std::size_t get_exact_decompress_bytes(const void* compressed_data,
                                       std::size_t compressed_len,
                                       const MansParams& params) {
    if (compressed_len <= kMansHeaderBytes) {
        throw std::runtime_error("mans::cpu::get_exact_decompress_bytes: missing payload.");
    }

    std::size_t raw_bytes = 0;
    std::string parse_error;
    if (!mans::parse_mans_raw_bytes(compressed_data, compressed_len, raw_bytes, &parse_error)) {
        throw std::runtime_error("mans::cpu::get_exact_decompress_bytes: " + parse_error + ".");
    }

    std::size_t elem_size = 0;
    if (!mans::get_dtype_size(params.dtype, elem_size)) {
        throw std::runtime_error("mans::cpu::get_exact_decompress_bytes: Unsupported dtype.");
    }
    if (raw_bytes % elem_size != 0) {
        throw std::runtime_error("mans::cpu::get_exact_decompress_bytes: invalid raw size in header.");
    }
    return raw_bytes;
}

} // namespace cpu
} // namespace mans
