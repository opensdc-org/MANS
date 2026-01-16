#include "mans_cpu.h"
#include <iostream>
#include <cstring>
#include <limits>
#include <algorithm>
#include <new>

#include "adm/adm_utils.h"
#include "pans/pans_utils.h"
#include "file_utils.h"
#define DEBUG_PRINT(msg) \
    std::cerr << "\033[1;35m[PLUGIN-CORE]\033[0m " << msg << "\n"

namespace mans {
namespace cpu {

// ==========================================
// 1.  Compress Helper Function
// ==========================================

template<typename T>
static bool decide_use_adm(const T* data, size_t size, uint32_t threshold) {
    const std::size_t block_size = 512;
    std::uint64_t max_block_diff = 0;

    for (std::size_t i = 0; i < size; i += block_size) {
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
        if (max_block_diff > threshold) return false;
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
    const std::string& dump_path,
    bool open_benchmark
) {
    uint32_t threshold = params.adm_threshold;
    if (threshold == 0) threshold = 4000;

    bool use_adm = decide_use_adm(data_ptr, length, threshold);
    std::uint8_t codec_code = 0;

    final_out_size = 0;
    if (!final_out) {
        return;
    }

    // PANS input pointer/length (raw bytes or adm-compressed bytes)
    const std::uint8_t* pans_in_ptr = nullptr;
    std::size_t pans_in_len = 0;


    std::uint8_t* adm_buf = nullptr;
    std::size_t adm_cap = 0;
    std::uint8_t* pans_out_buf = nullptr;

    // simple conservative bounds
    auto raw_bytes = length * sizeof(T);

    try {

        if (use_adm) {
            codec_code = 1; // ADM
            adm_cap = raw_bytes * 4 + 256;
            adm_buf = new std::uint8_t[adm_cap];

            std::size_t adm_size = 0;

            if (open_benchmark) {

                adm_compress_and_benchmark<T>(data_ptr, length, adm_buf, adm_size,params);
            } else {
                adm_compress<T>(data_ptr, length, adm_buf, adm_size,params);
            }

            if (adm_size > adm_cap) {
                std::cerr << "[Error] adm_buf overflow: adm_size > adm_cap.\n";
                return;
            }

            if (save_adm && !dump_path.empty()) {
                std::vector<std::uint8_t> tmp(adm_buf, adm_buf + adm_size);
                save_u8_file(dump_path, tmp);
            }

            pans_in_ptr = adm_buf;
            pans_in_len = adm_size;
        } else {
            codec_code = 2; // Direct
            pans_in_ptr = reinterpret_cast<const std::uint8_t*>(data_ptr);
            pans_in_len = raw_bytes;
        }

        std::size_t pans_out_len = 0;
        double pans_dur = 0.0;

        if (open_benchmark) {
            pans_compress_and_benchmark(
                pans_in_ptr,
                pans_in_len,
                final_out + 1, //reserve 1 byte for codec
                pans_out_len
            );
        } else {
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

    delete[] adm_buf;
    delete[] pans_out_buf;
}

template<typename T>
void do_decompress_t(
    const uint8_t* input_ptr,
    size_t length,
    std::uint8_t* final_out,
    std::size_t& final_out_size,
    const MansParams& params,

    bool save_adm,
    const std::string& dump_path,
    bool open_benchmark
) {
    final_out_size = 0;

    uint8_t codec = 0;
    if (!parse_header(input_ptr, length, codec)) {
        return;
    }


    const uint8_t* payload_ptr = input_ptr + 1;
    size_t payload_len = length - 1;

    std::uint8_t* pans_decomp_buf = nullptr;

    try {
        std::size_t pans_decomp_len = 0;
        get_compress_and_decompressed_len(payload_ptr,payload_len,pans_decomp_len);
        pans_decomp_buf = new std::uint8_t[pans_decomp_len];
        double pans_dur = 0.0;

        if (open_benchmark) {
            pans_decompress_and_benchmark(payload_ptr, payload_len, pans_decomp_buf, pans_decomp_len);
        } else {
            pans_decompress(payload_ptr, payload_len, pans_decomp_buf, pans_decomp_len, pans_dur);
        }

        
        if (codec == 2) {
            // Direct Mode
            if (pans_decomp_len > 0) {
                std::memcpy(final_out, pans_decomp_buf, pans_decomp_len);
            }
            final_out_size = pans_decomp_len;
        }
        else if (codec == 1) {
            // ADM Mode
            if (save_adm && !dump_path.empty()) {
                std::vector<std::uint8_t> tmp(pans_decomp_buf, pans_decomp_buf + pans_decomp_len);
                save_u8_file(dump_path, tmp);
            }
            T* recovered = reinterpret_cast<T*>(final_out);

  
            std::size_t num_elements = 0;

            if (open_benchmark) {
                adm_decompress_and_benchmark<T>(pans_decomp_buf, pans_decomp_len, recovered, num_elements,params);
            } else {
                adm_decompress<T>(pans_decomp_buf, pans_decomp_len, recovered, num_elements,params);
            }

            final_out_size = num_elements * sizeof(T);
        }
        else {
            std::cerr << "[Error] Unknown codec type: " << int(codec) << "\n";
        }
        if (!final_out) {
            std::cerr << "[Error] final_out is null in ADM mode, cannot decompress.\n";
            delete[] pans_decomp_buf;
            return;
        }
    }
    catch (const std::bad_alloc&) {
        std::cerr << "[Error] Out of memory.\n";
        final_out_size = 0;
        delete[] pans_decomp_buf;
        return; 
    }
    catch (const std::exception& e) { 
        std::cerr << "[Error] An exception occurred: " << e.what() << "\n";
        final_out_size = 0;
        delete[] pans_decomp_buf; 
        return;
    }
    catch (...) { 
        std::cerr << "[Error] An unknown exception occurred.\n";
        final_out_size = 0;
        delete[] pans_decomp_buf;
        return;
    }

    delete[] pans_decomp_buf;
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
    const std::string& dump_path,
    bool open_benchmark
) {
    if (params.dtype == DataType::U16) {
        do_compress_t(
            static_cast<const uint16_t*>(input_data),
            length,
            params,
            out,
            out_size,
            save_adm,
            dump_path,
            open_benchmark
        );
    } else if (params.dtype == DataType::U32) {
        do_compress_t(
            static_cast<const uint32_t*>(input_data),
            length,
            params,
            out,
            out_size,
            save_adm,
            dump_path,
            open_benchmark
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
    const std::string& dump_path,
    bool open_benchmark
) {
    const uint8_t* ptr = static_cast<const uint8_t*>(input_data);

    if (params.dtype == DataType::U16) {
        do_decompress_t<uint16_t>(
            ptr, length, out, out_size,params,save_adm, dump_path, open_benchmark
        );
    } else if (params.dtype == DataType::U32) {
        do_decompress_t<uint32_t>(
            ptr, length, out, out_size,params, save_adm, dump_path, open_benchmark
        );
    }
}

} // namespace cpu
} // namespace mans