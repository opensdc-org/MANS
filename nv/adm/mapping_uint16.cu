#include "mapping_uint16.h"

#include <cuda_runtime.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "adm_kernel.cuh"
#include "adm_reference.h"

namespace mans::nv::adm {
namespace {

void check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
void decode_reference(const std::uint8_t* d_input,
                      std::size_t input_size,
                      T* d_output,
                      std::size_t num_elements,
                      const mans::MansParams& params,
                      cudaStream_t stream) {
    if (!d_input || !d_output) {
        throw std::runtime_error("CUDA ADM input/output is null");
    }
    std::vector<std::uint8_t> encoded(input_size);
    check_cuda(cudaMemcpyAsync(encoded.data(), d_input, input_size,
                               cudaMemcpyDeviceToHost, stream),
               "cudaMemcpy ADM payload D2H");
    check_cuda(cudaStreamSynchronize(stream), "ADM payload D2H sync");

    std::vector<T> output(num_elements);
    reference::decode(encoded.data(), encoded.size(), output.data(), num_elements, params);
    check_cuda(cudaMemcpyAsync(d_output, output.data(), num_elements * sizeof(T),
                               cudaMemcpyHostToDevice, stream),
               "cudaMemcpy ADM decoded H2D");
    check_cuda(cudaStreamSynchronize(stream), "ADM decoded H2D sync");
}

template <typename T>
void compress_kernel(const T* d_input,
                     std::size_t num_elements,
                     const mans::MansParams& params,
                     std::uint8_t* d_output,
                     std::size_t& output_size,
                     cudaStream_t stream) {
    if (!d_input || !d_output) {
        throw std::runtime_error("CUDA ADM input/output is null");
    }
    const std::size_t blocks = reference::block_count(num_elements, params);
    const std::size_t flags_bytes = reference::flags_size(blocks);
    const std::size_t flags_words = (flags_bytes + sizeof(std::uint32_t) - 1) / sizeof(std::uint32_t);
    const std::size_t lane_elements = params.dims == 1
        ? reference::kChunk1D
        : reference::ceil_div(reference::kTileX * reference::kTileY * reference::kTileZ,
                               reference::kWarpSize);
    const std::size_t lane_stride = lane_elements * reference::max_adm_signal_bytes_per_element();

    std::uint32_t* d_flags = nullptr;
    T* d_centers = nullptr;
    std::uint32_t* d_signal_lengths = nullptr;
    std::uint8_t* d_codes = nullptr;
    std::uint8_t* d_lane_signals = nullptr;
    check_cuda(cudaMalloc(&d_flags, flags_words * sizeof(std::uint32_t)), "cudaMalloc ADM flags");
    check_cuda(cudaMalloc(&d_centers, blocks * sizeof(T)), "cudaMalloc ADM centers");
    check_cuda(cudaMalloc(&d_signal_lengths, blocks * sizeof(std::uint32_t)), "cudaMalloc ADM lengths");
    check_cuda(cudaMalloc(&d_codes, num_elements), "cudaMalloc ADM codes");
    check_cuda(cudaMalloc(&d_lane_signals, blocks * reference::kWarpSize * lane_stride),
               "cudaMalloc ADM signals");
    check_cuda(cudaMemsetAsync(d_flags, 0, flags_words * sizeof(std::uint32_t), stream),
               "cudaMemset ADM flags");

    kernel::encode<T><<<static_cast<unsigned int>(blocks), reference::kWarpSize, 0, stream>>>(
        d_input, num_elements, params, d_codes, d_flags, d_centers,
        d_signal_lengths, d_lane_signals, lane_stride);
    check_cuda(cudaGetLastError(), "ADM encode kernel launch");
    check_cuda(cudaStreamSynchronize(stream), "ADM encode kernel sync");

    std::vector<std::uint32_t> signal_lengths(blocks);
    std::vector<std::uint32_t> flags(flags_words);
    std::vector<T> centers(blocks);
    check_cuda(cudaMemcpy(signal_lengths.data(), d_signal_lengths,
                          blocks * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
               "cudaMemcpy ADM lengths D2H");
    check_cuda(cudaMemcpy(flags.data(), d_flags, flags_words * sizeof(std::uint32_t),
                          cudaMemcpyDeviceToHost), "cudaMemcpy ADM flags D2H");
    check_cuda(cudaMemcpy(centers.data(), d_centers, blocks * sizeof(T), cudaMemcpyDeviceToHost),
               "cudaMemcpy ADM centers D2H");

    std::vector<std::int32_t> offsets(blocks + 1, 0);
    for (std::size_t i = 0; i < blocks; ++i) {
        if (signal_lengths[i] > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
            throw std::runtime_error("ADM signal length exceeds 32-bit range");
        }
        offsets[i + 1] = offsets[i] + static_cast<std::int32_t>(signal_lengths[i]);
    }
    const std::size_t offsets_bytes = (blocks + 1) * sizeof(std::int32_t);
    const std::size_t centers_bytes = blocks * sizeof(T);
    const std::size_t codes_bytes = num_elements;
    const std::size_t signal_bytes = static_cast<std::size_t>(offsets.back()) * reference::kWarpSize;
    output_size = offsets_bytes + centers_bytes + flags_bytes + codes_bytes + signal_bytes;

    std::vector<std::uint8_t> host_output(output_size, 0);
    std::memcpy(host_output.data(), offsets.data(), offsets_bytes);
    for (std::size_t i = 0; i < blocks; ++i) {
        reference::write_raw(host_output.data() + offsets_bytes + i * sizeof(T), centers[i]);
    }
    auto* host_flags = host_output.data() + offsets_bytes + centers_bytes;
    std::memset(host_flags, 0, flags_bytes);
    for (std::size_t block = 0; block < blocks; ++block) {
        if (flags[block >> 5] & (1u << (31 - (block & 31)))) {
            host_flags[block >> 3] |= static_cast<std::uint8_t>(1u << (7 - (block & 7)));
        }
    }
    check_cuda(cudaMemcpy(host_output.data() + offsets_bytes + centers_bytes + flags_bytes,
                          d_codes, codes_bytes, cudaMemcpyDeviceToHost),
               "cudaMemcpy ADM codes D2H");
    if (signal_bytes != 0) {
        std::vector<std::uint8_t> lane_signals(blocks * reference::kWarpSize * lane_stride);
        check_cuda(cudaMemcpy(lane_signals.data(), d_lane_signals, lane_signals.size(),
                              cudaMemcpyDeviceToHost), "cudaMemcpy ADM signals D2H");
        auto* dst = host_output.data() + offsets_bytes + centers_bytes + flags_bytes + codes_bytes;
        for (std::size_t block = 0; block < blocks; ++block) {
            const std::size_t length = signal_lengths[block];
            for (std::size_t lane = 0; lane < reference::kWarpSize; ++lane) {
                std::memcpy(dst + offsets[block] * reference::kWarpSize + lane * length,
                            lane_signals.data() + (block * reference::kWarpSize + lane) * lane_stride,
                            length);
            }
        }
    }
    check_cuda(cudaMemcpyAsync(d_output, host_output.data(), output_size,
                               cudaMemcpyHostToDevice, stream), "cudaMemcpy ADM packed H2D");
    check_cuda(cudaStreamSynchronize(stream), "ADM packed H2D sync");

    cudaFree(d_flags);
    cudaFree(d_centers);
    cudaFree(d_signal_lengths);
    cudaFree(d_codes);
    cudaFree(d_lane_signals);
}

} // namespace

void compress_u16_device(const std::uint16_t* d_input,
                         std::size_t num_elements,
                         const mans::MansParams& params,
                         std::uint8_t* d_output,
                         std::size_t& output_size,
                         cudaStream_t stream) {
    output_size = 0;
    if (num_elements == 0) {
        return;
    }
    compress_kernel(d_input, num_elements, params, d_output, output_size, stream);
}

void decompress_u16_device(const std::uint8_t* d_input,
                           std::size_t input_size,
                           std::uint16_t* d_output,
                           std::size_t num_elements,
                           const mans::MansParams& params,
                           cudaStream_t stream) {
    if (num_elements == 0) {
        return;
    }
    decode_reference(d_input, input_size, d_output, num_elements, params, stream);
}

std::size_t get_max_u16_payload_bytes(std::size_t num_elements,
                                      const mans::MansParams& params) {
    return reference::max_payload_bytes<std::uint16_t>(num_elements, params);
}

} // namespace mans::nv::adm
