#include "fse_utils.h"

#include "../buffer_cache.h"

#include <chrono>
#include <cstring>
#include <iostream>
#include <limits>

extern "C" {
#include "include/fse.h"
}

namespace mans {
namespace cpu {
namespace {

constexpr std::size_t kFrameHeaderSize = 20;  // magic(4) + blockSize(4) + rawSize(8) + blockCount(4)
constexpr std::size_t kBlockHeaderSize = 9;   // mode(1) + rawSize(4) + storedSize(4)
constexpr std::uint32_t kFrameBlockSize = 32U * 1024U;

inline std::uint32_t read_le32(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0]) |
           (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) |
           (static_cast<std::uint32_t>(p[3]) << 24);
}

inline std::uint64_t read_le64(const std::uint8_t* p) {
    return static_cast<std::uint64_t>(read_le32(p)) |
           (static_cast<std::uint64_t>(read_le32(p + 4)) << 32);
}

bool parse_fse_frame(const std::uint8_t* compressed_data,
                     std::size_t compressed_len,
                     std::size_t& frame_len,
                     std::size_t& decompressed_len,
                     std::string* error) {
    frame_len = 0;
    decompressed_len = 0;

    auto set_error = [&](const char* msg) {
        if (error) {
            *error = msg;
        }
    };

    if (!compressed_data) {
        set_error("compressed_data is null");
        return false;
    }
    if (compressed_len < kFrameHeaderSize) {
        set_error("compressed_len too small for FSE frame header");
        return false;
    }
    if (compressed_data[0] != 'M' || compressed_data[1] != 'F' ||
        compressed_data[2] != 'S' || compressed_data[3] != 'E') {
        set_error("invalid FSE magic");
        return false;
    }

    const std::uint32_t block_size = read_le32(compressed_data + 4);
    if (block_size != kFrameBlockSize) {
        set_error("unexpected FSE block size");
        return false;
    }

    const std::uint64_t raw_size_u64 = read_le64(compressed_data + 8);
    if (raw_size_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        set_error("raw size overflows size_t");
        return false;
    }
    const std::size_t raw_size = static_cast<std::size_t>(raw_size_u64);
    const std::uint32_t block_count = read_le32(compressed_data + 16);

    std::size_t offset = kFrameHeaderSize;
    std::size_t raw_acc = 0;
    for (std::uint32_t i = 0; i < block_count; ++i) {
        if (offset > compressed_len || compressed_len - offset < kBlockHeaderSize) {
            set_error("truncated FSE block header");
            return false;
        }

        const std::uint8_t mode = compressed_data[offset];
        const std::uint32_t block_raw = read_le32(compressed_data + offset + 1);
        const std::uint32_t block_stored = read_le32(compressed_data + offset + 5);
        offset += kBlockHeaderSize;

        if (mode > 2) {
            set_error("invalid FSE block mode");
            return false;
        }
        if (block_stored > compressed_len - offset) {
            set_error("truncated FSE block payload");
            return false;
        }
        if (raw_acc > std::numeric_limits<std::size_t>::max() - static_cast<std::size_t>(block_raw)) {
            set_error("raw size accumulation overflow");
            return false;
        }

        raw_acc += static_cast<std::size_t>(block_raw);
        offset += static_cast<std::size_t>(block_stored);
    }

    if (raw_acc != raw_size) {
        set_error("raw size mismatch in FSE frame");
        return false;
    }
    if (offset != compressed_len) {
        set_error("extra bytes after FSE frame");
        return false;
    }

    frame_len = offset;
    decompressed_len = raw_size;
    return true;
}

} // namespace

void fse_compress(const std::uint8_t* input_data,
                  std::size_t input_len,
                  std::uint8_t* output_data,
                  std::size_t& output_len,
                  double& duration_ms) {
    output_len = 0;
    duration_ms = 0.0;

    if (!input_data || input_len == 0) {
        std::cerr << "Error: fse_compress input is empty.\n";
        return;
    }

    const std::size_t bound = FSE_compressBound(input_len);
    if (bound == 0) {
        std::cerr << "Error: FSE_compressBound overflow.\n";
        return;
    }

    auto* scratch = mans::cpu::BufferCache::instance().get_t<std::uint8_t>("fse_enc", bound);
    if (!scratch) {
        std::cerr << "Error: fse_compress allocation failed.\n";
        return;
    }

    const auto start = std::chrono::high_resolution_clock::now();
    const std::size_t csize = FSE_compress(scratch, bound, input_data, input_len);
    const auto end = std::chrono::high_resolution_clock::now();
    duration_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e3;

    if (FSE_isError(csize)) {
        std::cerr << "Error: FSE_compress failed: " << FSE_getErrorName(csize) << "\n";
        return;
    }

    if (output_data && csize > 0) {
        std::memcpy(output_data, scratch, csize);
    }
    output_len = csize;
}

void fse_decompress(const std::uint8_t* compressed_data,
                    std::size_t compressed_len,
                    std::uint8_t* decompressed_data,
                    std::size_t& decompressed_len,
                    double& duration_ms) {
    duration_ms = 0.0;

    std::size_t frame_len = 0;
    std::size_t raw_len = 0;
    std::string parse_error;
    if (!get_fse_compress_and_decompressed_len(compressed_data, compressed_len, frame_len,
                                               raw_len, &parse_error)) {
        std::cerr << "Error: " << parse_error << "\n";
        decompressed_len = 0;
        return;
    }
    if (frame_len != compressed_len) {
        std::cerr << "Error: unexpected trailing bytes in FSE payload.\n";
        decompressed_len = 0;
        return;
    }

    if (!decompressed_data) {
        decompressed_data = mans::cpu::BufferCache::instance().get_t<std::uint8_t>("fse_dec", raw_len);
        if (!decompressed_data) {
            std::cerr << "Error: fse_decompress allocation failed.\n";
            decompressed_len = 0;
            return;
        }
    }

    if (decompressed_len > 0 && decompressed_len < raw_len) {
        std::cerr << "Error: fse_decompress output buffer is too small.\n";
        decompressed_len = 0;
        return;
    }

    const auto start = std::chrono::high_resolution_clock::now();
    const std::size_t dsize = FSE_decompress(decompressed_data, raw_len, compressed_data, compressed_len);
    const auto end = std::chrono::high_resolution_clock::now();
    duration_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e3;

    if (FSE_isError(dsize)) {
        std::cerr << "Error: FSE_decompress failed: " << FSE_getErrorName(dsize) << "\n";
        decompressed_len = 0;
        return;
    }
    if (dsize != raw_len) {
        std::cerr << "Error: FSE_decompress output size mismatch.\n";
        decompressed_len = 0;
        return;
    }

    decompressed_len = dsize;
}

bool get_fse_compress_and_decompressed_len(const std::uint8_t* compressed_data,
                                           std::size_t compressed_len,
                                           std::size_t& frame_len,
                                           std::size_t& decompressed_len,
                                           std::string* error) {
    return parse_fse_frame(compressed_data, compressed_len, frame_len, decompressed_len, error);
}

} // namespace cpu
} // namespace mans
