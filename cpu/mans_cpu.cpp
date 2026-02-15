#include "mans_cpu.h"
#include <iostream>
#include <cstring>
#include <limits>
#include <algorithm>
#include <new>
#include <memory>
#include <cstdlib>
#include <omp.h>

#include "adm/adm_utils.h"
#include "pans/pans_utils.h"
#include "file_utils.h"
#include "buffer_cache.h"
#include "../mans_timing.h"
#define DEBUG_PRINT(msg) \
    std::cerr << "\033[1;35m[PLUGIN-CORE]\033[0m " << msg << "\n"

namespace mans {
namespace cpu {

// ==========================================
// 1.  Compress Helper Function
// ==========================================

template<typename T>
static bool decide_use_adm(const T* data, size_t size, uint32_t threshold, uint32_t threads) {
    const std::size_t block_size = 512;
    std::uint64_t max_block_diff = 0;
    const std::size_t blocks = (size + block_size - 1) / block_size;
    const int num_threads = threads == 0 ? 16 : static_cast<int>(threads);

    #pragma omp parallel for num_threads(num_threads) reduction(max:max_block_diff)
    for (std::size_t b = 0; b < blocks; ++b) {
        std::size_t i = b * block_size;
        std::size_t end = std::min(i + block_size, size);
        T bmin = std::numeric_limits<T>::max();
        T bmax = std::numeric_limits<T>::min();

        for (std::size_t j = i; j < end; ++j) {
            T v = data[j];
            if (v < bmin) bmin = v;
            if (v > bmax) bmax = v;
        }

        std::uint64_t diff = static_cast<std::uint64_t>(bmax) - static_cast<std::uint64_t>(bmin);
        if (diff > max_block_diff) {
            max_block_diff = diff;
        }
    }
    return (max_block_diff <= threshold);
}

static void prepend_header(
    const std::uint8_t* payload,
    std::size_t payload_size,
    std::uint8_t* final_payload,
    std::size_t& final_size,
    std::uint8_t codec)
{
    final_size = 0;

    if (!final_payload) {
        return;
    }

    final_payload[0] = codec;

    if (payload_size > 0) {
        std::memcpy(final_payload + 1, payload, payload_size);
    }

    final_size = 1 + payload_size;
}

// ==========================================
// 2. Decompress Helper Function
// ==========================================

static bool parse_header(
    const uint8_t* data, 
    size_t length, 
    uint8_t& codec)
{
    if (length < sizeof(MansHeader)) {
        std::cerr << "[Error] File too small, invalid mans format.\n";
        return false;
    }
    // read first byte as codec
    codec = data[0]; 
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
    uint32_t threshold = params.adm_threshold;
    if (threshold == 0) threshold = 4000;

    bool use_adm = false;
    {
        MANS_TIMING_SCOPE("decide_adm");
        use_adm = decide_use_adm(data_ptr, length, threshold, params.adm_decide_threads);
    }
    std::uint8_t codec_code = 0;

    final_out_size = 0;
    if (!final_out) {
        return;
    }

    // PANS input pointer/length (raw bytes or adm-compressed bytes)
    const std::uint8_t* pans_in_ptr = nullptr;
    std::size_t pans_in_len = 0;


    std::uint8_t* mans_intermediate_buf_local = nullptr;
    std::size_t adm_cap = 0;

    // simple conservative bounds
    auto raw_bytes = length * sizeof(T);

    try {

        if (use_adm) {
            codec_code = 1; // ADM
            adm_cap = adm_max_compressed_size<T>(length);
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
                std::vector<std::uint8_t> tmp(mans_intermediate_buf_local, mans_intermediate_buf_local + adm_size);
                save_u8_file(dump_path, tmp);
            }

            pans_in_ptr = mans_intermediate_buf_local;
            pans_in_len = adm_size;
        } else {
            codec_code = 2; // Direct
            pans_in_ptr = reinterpret_cast<const std::uint8_t*>(data_ptr);
            pans_in_len = raw_bytes;
        }

        std::size_t pans_out_len = 0;
        double pans_dur = 0.0;

        {
            MANS_TIMING_SCOPE("ans_compress");
            pans_compress(
                pans_in_ptr,
                pans_in_len,
                final_out + 1, //reserve 1 byte for codec
                pans_out_len,
                pans_dur
            );
        }

        final_out[0] = codec_code;
        final_out_size=1+pans_out_len;//include codec byte
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
    final_out_size = 0;

    uint8_t codec = 0;
    if (!parse_header(input_ptr, length, codec)) {
        return;
    }


    const uint8_t* payload_ptr = input_ptr + 1;
    size_t payload_len = length - 1;

    std::uint8_t* mans_intermediate_buf_local = nullptr;

    try {
        std::size_t pans_decomp_len = 0;
        get_compress_and_decompressed_len(payload_ptr,payload_len,pans_decomp_len);
        {
            MANS_TIMING_SCOPE("alloc_pans_decomp_buf");
            mans_intermediate_buf_local =
                mans::cpu::BufferCache::instance().get_t<std::uint8_t>(
                    "mans_pans_decomp", pans_decomp_len);
        }
        if (!mans_intermediate_buf_local) {
            std::cerr << "[Error] Out of memory.\n";
            final_out_size = 0;
            return;
        }
        double pans_dur = 0.0;

        {
            MANS_TIMING_SCOPE("ans_decompress");
            pans_decompress(payload_ptr, payload_len, mans_intermediate_buf_local, pans_decomp_len, pans_dur);
        }

        
        if (codec == 2) {
            // Direct Mode
            if (pans_decomp_len > 0) {
                std::memcpy(final_out, mans_intermediate_buf_local, pans_decomp_len);
            }
            final_out_size = pans_decomp_len;
        }
        else if (codec == 1) {
            // ADM Mode
            if (save_adm && !dump_path.empty()) {
                std::vector<std::uint8_t> tmp(mans_intermediate_buf_local, mans_intermediate_buf_local + pans_decomp_len);
                save_u8_file(dump_path, tmp);
            }
            T* recovered = reinterpret_cast<T*>(final_out);

  
            std::size_t num_elements = 0;

            {
                MANS_TIMING_SCOPE("adm_decompress");
                adm_decompress<T>(mans_intermediate_buf_local, pans_decomp_len, recovered,
                                  num_elements, params);
            }

            final_out_size = num_elements * sizeof(T);
        }
        else {
            std::cerr << "[Error] Unknown codec type: " << int(codec) << "\n";
        }
        if (!final_out) {
            std::cerr << "[Error] final_out is null in ADM mode, cannot decompress.\n";
            return;
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

} // namespace cpu
} // namespace mans
