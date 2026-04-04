#include "mans_nv_ans.h"

#include <limits>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#include "ans/BatchPrefixSum.h"
#include "ans/GpuANSCodec.h"

namespace multibyte_ans {
void ansEncodeBatch(
    uint32_t maxNumcompressedWords,
    uint32_t maxNumCompressedBlocks,
    uint4* table_dev,
    uint32_t* tempHistogram_dev,
    uint32_t uncoalescedBlockStride,
    uint8_t* compressedBlocks_dev,
    uint32_t* compressedWords_dev,
    uint32_t* compressedWordsPrefix_dev,
    uint32_t sizeRequired,
    uint8_t* tempPrefixSum_dev,
    int precision,
    uint8_t* in,
    uint32_t inSize,
    uint8_t* out,
    uint32_t* outSize,
    cudaStream_t stream);

void ansDecodeBatch(
    int precision,
    uint8_t* in,
    uint8_t* out,
    cudaStream_t stream);
} // namespace multibyte_ans

namespace mans {
namespace nv {
namespace ans {

namespace {

void check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

void require_p_mode(std::uint32_t mode) {
    if (mode != Mode::P) {
        throw std::runtime_error("mans::nv::ans: NVIDIA backend currently supports only Mode::P.");
    }
}

} // namespace

void compress_stage_device(const std::uint8_t* d_input,
                           std::size_t input_size,
                           std::uint8_t* d_output,
                           std::size_t& output_size,
                           std::uint32_t mode,
                           const MansParams& params,
                           cudaStream_t stream) {
    (void)params;
    require_p_mode(mode);

    if (!d_input && input_size != 0) {
        throw std::runtime_error("mans::nv::ans::compress_stage_device: d_input is null.");
    }
    if (!d_output && input_size != 0) {
        throw std::runtime_error("mans::nv::ans::compress_stage_device: d_output is null.");
    }
    if (input_size == 0) {
        output_size = 0;
        return;
    }
    if (input_size > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::runtime_error("mans::nv::ans::compress_stage_device: input too large.");
    }

    const std::uint32_t in_size_u32 = static_cast<std::uint32_t>(input_size);
    const std::uint32_t max_num_compressed_blocks =
        (in_size_u32 + multibyte_ans::kDefaultBlockSize - 1) / multibyte_ans::kDefaultBlockSize;
    const std::uint32_t uncoalesced_block_stride =
        multibyte_ans::getMaxBlockSizeUnCoalesced(multibyte_ans::kDefaultBlockSize);
    const std::uint32_t prefix_temp_size =
        multibyte_ans::getBatchExclusivePrefixSumTempSize(max_num_compressed_blocks);

    uint4* table_dev = nullptr;
    std::uint32_t* temp_histogram_dev = nullptr;
    std::uint8_t* compressed_blocks_dev = nullptr;
    std::uint32_t* compressed_words_dev = nullptr;
    std::uint32_t* compressed_words_prefix_dev = nullptr;
    std::uint8_t* temp_prefix_sum_dev = nullptr;
    std::uint32_t* out_size_dev = nullptr;

    check_cuda(cudaMalloc(&table_dev, sizeof(uint4) * multibyte_ans::kNumSymbols), "cudaMalloc table_dev");
    check_cuda(cudaMalloc(&temp_histogram_dev, sizeof(std::uint32_t) * multibyte_ans::kNumSymbols),
               "cudaMalloc temp_histogram_dev");
    check_cuda(cudaMalloc(&compressed_blocks_dev,
                          sizeof(std::uint8_t) * max_num_compressed_blocks * uncoalesced_block_stride),
               "cudaMalloc compressed_blocks_dev");
    check_cuda(cudaMalloc(&compressed_words_dev, sizeof(std::uint32_t) * max_num_compressed_blocks),
               "cudaMalloc compressed_words_dev");
    check_cuda(cudaMalloc(&compressed_words_prefix_dev, sizeof(std::uint32_t) * max_num_compressed_blocks),
               "cudaMalloc compressed_words_prefix_dev");
    check_cuda(cudaMalloc(&temp_prefix_sum_dev, sizeof(std::uint8_t) * prefix_temp_size),
               "cudaMalloc temp_prefix_sum_dev");
    check_cuda(cudaMalloc(&out_size_dev, sizeof(std::uint32_t)), "cudaMalloc out_size_dev");

    multibyte_ans::ansEncodeBatch(
        in_size_u32,
        max_num_compressed_blocks,
        table_dev,
        temp_histogram_dev,
        uncoalesced_block_stride,
        compressed_blocks_dev,
        compressed_words_dev,
        compressed_words_prefix_dev,
        prefix_temp_size,
        temp_prefix_sum_dev,
        multibyte_ans::kANSDefaultProbBits,
        const_cast<std::uint8_t*>(d_input),
        in_size_u32,
        d_output,
        out_size_dev,
        stream);
    check_cuda(cudaGetLastError(), "ansEncodeBatch launch");
    check_cuda(cudaStreamSynchronize(stream), "ansEncodeBatch sync");

    std::uint32_t out_size_u32 = 0;
    check_cuda(cudaMemcpy(&out_size_u32, out_size_dev, sizeof(out_size_u32), cudaMemcpyDeviceToHost),
               "cudaMemcpy out_size D2H");
    output_size = static_cast<std::size_t>(out_size_u32);

    cudaFree(table_dev);
    cudaFree(temp_histogram_dev);
    cudaFree(compressed_blocks_dev);
    cudaFree(compressed_words_dev);
    cudaFree(compressed_words_prefix_dev);
    cudaFree(temp_prefix_sum_dev);
    cudaFree(out_size_dev);
}

void decompress_stage_device(const std::uint8_t* d_input,
                             std::size_t compressed_size,
                             std::uint8_t* d_output,
                             std::size_t max_output_size,
                             std::size_t& output_size,
                             std::uint32_t mode,
                             const MansParams& params,
                             cudaStream_t stream) {
    (void)params;
    require_p_mode(mode);

    if (!d_input && compressed_size != 0) {
        throw std::runtime_error("mans::nv::ans::decompress_stage_device: d_input is null.");
    }
    if (!d_output && max_output_size != 0) {
        throw std::runtime_error("mans::nv::ans::decompress_stage_device: d_output is null.");
    }
    if (compressed_size == 0) {
        output_size = 0;
        return;
    }
    if (compressed_size < sizeof(multibyte_ans::ANSCoalescedHeader)) {
        throw std::runtime_error("mans::nv::ans::decompress_stage_device: compressed input too small.");
    }

    multibyte_ans::ANSCoalescedHeader header{};
    check_cuda(cudaMemcpy(&header, d_input, sizeof(header), cudaMemcpyDeviceToHost),
               "cudaMemcpy ans header D2H");
    const std::size_t exact_size = static_cast<std::size_t>(header.getTotalUncompressedWords());
    if (exact_size > max_output_size) {
        throw std::runtime_error("mans::nv::ans::decompress_stage_device: output buffer too small.");
    }

    multibyte_ans::ansDecodeBatch(
        multibyte_ans::kANSDefaultProbBits,
        const_cast<std::uint8_t*>(d_input),
        d_output,
        stream);
    check_cuda(cudaGetLastError(), "ansDecodeBatch launch");
    check_cuda(cudaStreamSynchronize(stream), "ansDecodeBatch sync");
    output_size = exact_size;
}

std::size_t get_max_compress_bytes(std::size_t input_bytes,
                                   std::uint32_t mode,
                                   const MansParams& params) {
    (void)params;
    require_p_mode(mode);
    if (input_bytes > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::runtime_error("mans::nv::ans::get_max_compress_bytes: input too large.");
    }
    return static_cast<std::size_t>(multibyte_ans::getMaxCompressedSize(static_cast<std::uint32_t>(input_bytes)));
}

} // namespace ans
} // namespace nv
} // namespace mans
