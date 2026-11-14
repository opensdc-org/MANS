#include "mans_amd_ans.h"
#include "utils/AnsConfig.h"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace pans_hip {
void ansEncodeBatch(int precision, uint8_t* in, uint32_t inSize, uint8_t* out,
                    uint32_t* outSize, hipStream_t stream);
void ansDecodeBatch(int precision, uint8_t* in, uint8_t* out, hipStream_t stream);
} // namespace pans_hip

namespace mans {
namespace amd {
namespace ans {
namespace {

constexpr int kPrecision = 10;
constexpr std::size_t kAlignment = 4;
constexpr std::size_t kBlockSize = 8192;
constexpr std::uint32_t kMagic = 0xd00d;
constexpr std::uint32_t kVersion = 1;

struct AmdAnsHeader {
    std::uint32_t magic_and_version;
    std::uint32_t num_blocks;
    std::uint32_t total_uncompressed_words;
    std::uint32_t total_compressed_words;
    std::uint32_t options;
    std::uint32_t checksum;
    std::uint32_t unused0;
    std::uint32_t unused1;

    std::size_t compressed_overhead() const {
        return compressed_overhead_for(num_blocks);
    }

    static std::size_t compressed_overhead_for(std::size_t blocks) {
        const std::size_t aligned_blocks = (blocks + 1) & ~std::size_t(1);
        return sizeof(AmdAnsHeader) + 256 * sizeof(std::uint16_t) +
               blocks * pans_hip::kWarpSize * sizeof(std::uint32_t) +
               aligned_blocks * sizeof(std::uint32_t) * 2;
    }

    std::size_t total_compressed_size() const {
        return compressed_overhead() + static_cast<std::size_t>(total_compressed_words) * 2;
    }

    std::uint32_t prob_bits() const { return options & 0xfU; }
};
static_assert(sizeof(AmdAnsHeader) == 32, "AMD ANS header layout changed");

void check_hip(hipError_t status, const char* what) {
    if (status != hipSuccess) {
        throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(status));
    }
}

void check_size(std::size_t size, const char* what) {
    if (size > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::runtime_error(std::string(what) + " exceeds uint32_t ANS limit.");
    }
}

AmdAnsHeader read_header(const std::uint8_t* bytes) {
    AmdAnsHeader header{};
    std::memcpy(&header, bytes, sizeof(header));
    return header;
}

void validate_header(const AmdAnsHeader& header, std::size_t compressed_size) {
    if ((header.magic_and_version >> 16) != kMagic ||
        (header.magic_and_version & 0xffffU) != kVersion ||
        header.prob_bits() != kPrecision || header.num_blocks == 0 ||
        header.total_uncompressed_words == 0 ||
        header.total_compressed_size() != compressed_size) {
        throw std::runtime_error("mans::amd::ans: invalid compressed header or payload length.");
    }
}

} // namespace

void compress_stage_device(const std::uint8_t* d_input,
                           std::size_t input_size,
                           std::uint8_t* d_output,
                           std::size_t output_capacity,
                           std::size_t& output_size,
                           hipStream_t stream) {
    output_size = 0;
    if (!d_input && input_size != 0) {
        throw std::runtime_error("mans::amd::ans: null input device pointer.");
    }
    if (!d_output && output_capacity != 0) {
        throw std::runtime_error("mans::amd::ans: null output device pointer.");
    }
    if (input_size == 0) {
        return;
    }
    check_size(input_size, "AMD ANS input");
    const std::size_t bound = get_max_compress_bytes(input_size);
    if (output_capacity < bound) {
        throw std::runtime_error("mans::amd::ans: output buffer is smaller than the ANS bound.");
    }

    // The encoded size is recovered from the coalesced header already written to
    // d_output, so we need a single D2H copy (instead of a separate size buffer
    // plus a header copy), cutting one host round-trip on the small-file path.
    pans_hip::ansEncodeBatch(kPrecision, const_cast<std::uint8_t*>(d_input),
                             static_cast<std::uint32_t>(input_size), d_output,
                             nullptr, stream);
    check_hip(hipGetLastError(), "AMD ANS encode launch");
    check_hip(hipStreamSynchronize(stream), "AMD ANS encode synchronize");

    std::uint8_t header_bytes[sizeof(AmdAnsHeader)] = {};
    check_hip(hipMemcpy(header_bytes, d_output, sizeof(header_bytes), hipMemcpyDeviceToHost),
              "AMD ANS header copy");
    const AmdAnsHeader header = read_header(header_bytes);
    const std::size_t encoded_size = header.total_compressed_size();
    if (encoded_size > output_capacity || encoded_size < sizeof(AmdAnsHeader)) {
        throw std::runtime_error("mans::amd::ans: invalid encoded size.");
    }
    validate_header(header, encoded_size);
    output_size = encoded_size;
}

void decompress_stage_device(const std::uint8_t* d_input,
                             std::size_t compressed_size,
                             std::uint8_t* d_output,
                             std::size_t output_capacity,
                             std::size_t& output_size,
                             hipStream_t stream) {
    output_size = 0;
    if (!d_input || !d_output) {
        throw std::runtime_error("mans::amd::ans: null device pointer.");
    }
    if (compressed_size < sizeof(AmdAnsHeader)) {
        throw std::runtime_error("mans::amd::ans: compressed input is too small.");
    }
    check_size(compressed_size, "AMD ANS compressed input");

    std::uint8_t header_bytes[sizeof(AmdAnsHeader)] = {};
    check_hip(hipMemcpy(header_bytes, d_input, sizeof(header_bytes), hipMemcpyDeviceToHost),
              "AMD ANS header copy");
    const AmdAnsHeader header = read_header(header_bytes);
    validate_header(header, compressed_size);
    const std::size_t decoded_size = header.total_uncompressed_words;
    if (decoded_size > output_capacity) {
        throw std::runtime_error("mans::amd::ans: output buffer is too small.");
    }

    pans_hip::ansDecodeBatch(kPrecision, const_cast<std::uint8_t*>(d_input), d_output, stream);
    check_hip(hipGetLastError(), "AMD ANS decode launch");
    check_hip(hipStreamSynchronize(stream), "AMD ANS decode synchronize");
    output_size = decoded_size;
}

std::size_t get_max_compress_bytes(std::size_t input_bytes) {
    if (input_bytes == 0) {
        return 0;
    }
    check_size(input_bytes, "AMD ANS input");
    const std::size_t blocks = (input_bytes + kBlockSize - 1) / kBlockSize;
    const std::size_t overhead = AmdAnsHeader::compressed_overhead_for(blocks);
    const std::size_t block_bound = kBlockSize + kBlockSize / 4;
    const std::size_t bound = overhead + block_bound * blocks;
    if (bound > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("mans::amd::ans: compressed bound exceeds uint32_t.");
    }
    return (bound + kAlignment - 1) / kAlignment * kAlignment;
}

std::size_t get_decompressed_bytes(const void* compressed_data, std::size_t compressed_size) {
    if (!compressed_data || compressed_size < sizeof(AmdAnsHeader)) {
        throw std::runtime_error("mans::amd::ans: compressed input is too small.");
    }
    const AmdAnsHeader header = read_header(static_cast<const std::uint8_t*>(compressed_data));
    validate_header(header, compressed_size);
    return header.total_uncompressed_words;
}

} // namespace ans
} // namespace amd
} // namespace mans
