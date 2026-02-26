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
#include "fse/fse_utils.h"
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

static std::uint32_t normalize_mode(std::uint32_t mode) {
    if (mode == Mode::R) {
        return Mode::R;
    }
    return Mode::P;
}

// ==========================================
// 2. Decompress Helper Function
// ==========================================

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

inline void write_le64(std::uint8_t* p, std::uint64_t v) {
    p[0] = static_cast<std::uint8_t>(v & 0xFFu);
    p[1] = static_cast<std::uint8_t>((v >> 8) & 0xFFu);
    p[2] = static_cast<std::uint8_t>((v >> 16) & 0xFFu);
    p[3] = static_cast<std::uint8_t>((v >> 24) & 0xFFu);
    p[4] = static_cast<std::uint8_t>((v >> 32) & 0xFFu);
    p[5] = static_cast<std::uint8_t>((v >> 40) & 0xFFu);
    p[6] = static_cast<std::uint8_t>((v >> 48) & 0xFFu);
    p[7] = static_cast<std::uint8_t>((v >> 56) & 0xFFu);
}

static bool parse_header(const uint8_t* data,
                         size_t length,
                         MansHeader& out_header,
                         std::size_t& out_raw_bytes) {
    if (length < kMansHeaderBytes) {
        std::cerr << "[Error] File too small, invalid mans format.\n";
        return false;
    }

    MansHeader header{};
    std::memcpy(&header, data, sizeof(header));
    out_header = header;

    if (header.codec != 1 && header.codec != 2) {
        std::cerr << "[Error] Unknown codec type: " << int(header.codec) << "\n";
        return false;
    }
    if (header.mode != Mode::P && header.mode != Mode::R) {
        std::cerr << "[Error] Unknown mode in header: " << int(header.mode) << "\n";
        return false;
    }
    const std::uint64_t raw_bytes_u64 = read_le64(header.raw_bytes_le);
    if (raw_bytes_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        std::cerr << "[Error] raw size overflows size_t.\n";
        return false;
    }
    out_raw_bytes = static_cast<std::size_t>(raw_bytes_u64);
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
    const std::uint32_t mode = normalize_mode(params.mode);

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

            stage2_in_ptr = mans_intermediate_buf_local;
            stage2_in_len = adm_size;
        } else {
            codec_code = 2; // Direct
            stage2_in_ptr = reinterpret_cast<const std::uint8_t*>(data_ptr);
            stage2_in_len = raw_bytes;
        }

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
        write_le64(header.raw_bytes_le, static_cast<std::uint64_t>(raw_bytes));
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
                save_u8_file(dump_path, tmp);
            }
            if (stage2_decomp_len < sizeof(adm::FileHeader)) {
                std::cerr << "[Error] ADM payload is too small.\n";
                return;
            }
            const auto* hdr = reinterpret_cast<const adm::FileHeader*>(stage2_dec_buf);
            if (hdr->num_elements >
                std::numeric_limits<std::size_t>::max() / sizeof(T)) {
                std::cerr << "[Error] ADM output size overflow.\n";
                return;
            }
            const std::size_t expected_bytes =
                static_cast<std::size_t>(hdr->num_elements) * sizeof(T);
            if (expected_bytes != raw_bytes) {
                std::cerr << "[Error] Raw size mismatch between mans header and ADM header.\n";
                return;
            }
            if (expected_bytes > out_capacity) {
                std::cerr << "[Error] Output buffer too small for ADM payload.\n";
                return;
            }
            T* recovered = reinterpret_cast<T*>(final_out);
            std::size_t num_elements = 0;

            {
                MANS_TIMING_SCOPE("adm_decompress");
                adm_decompress<T>(stage2_dec_buf, stage2_decomp_len, recovered,
                                  num_elements, params);
            }
            final_out_size = num_elements * sizeof(T);
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

} // namespace cpu
} // namespace mans
