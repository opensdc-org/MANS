#include "mapping_uint16.h"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "adm_kernel.cuh"
#include "adm_reference.h"

namespace mans::dcu::adm {
namespace {

void check_hip(hipError_t status, const char* what) {
    if (status != hipSuccess) {
        throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(status));
    }
}

class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { reset(); }
    void allocate(std::size_t bytes, const char* what) {
        reset();
        if (bytes != 0) check_hip(hipMalloc(&ptr_, bytes), what);
    }
    void reset() {
        if (ptr_) (void)hipFree(ptr_);
        ptr_ = nullptr;
    }
    std::uint8_t* get() const { return ptr_; }
private:
    std::uint8_t* ptr_ = nullptr;
};

template <typename T>
void compress_kernel(const T* d_input, std::size_t elements,
                     const mans::MansParams& params, std::uint8_t* d_output,
                     std::size_t& output_size, hipStream_t stream) {
    if (!d_input || !d_output || elements == 0) throw std::runtime_error("DCU ADM input/output is null");
    const std::size_t blocks = reference::block_count(elements, params);
    const std::size_t flags_bytes = reference::flags_size(blocks);
    const std::size_t flags_words = (flags_bytes + sizeof(std::uint32_t) - 1) / sizeof(std::uint32_t);
    const std::size_t max_block = params.dims == 1 ? reference::kBlock1D :
        params.dims == 2 ? reference::kTileX * reference::kTileY : reference::kTileX * reference::kTileY * reference::kTileZ;
    const std::size_t lane_stride = reference::ceil_div(max_block, reference::kWarpSize) *
        reference::max_signal_bytes_per_element();

    DeviceBuffer flags, centers, lengths, codes, signals;
    flags.allocate(flags_words * sizeof(std::uint32_t), "hipMalloc DCU ADM flags");
    centers.allocate(blocks * sizeof(T), "hipMalloc DCU ADM centers");
    lengths.allocate(blocks * sizeof(std::uint32_t), "hipMalloc DCU ADM lengths");
    codes.allocate(elements, "hipMalloc DCU ADM codes");
    signals.allocate(blocks * reference::kWarpSize * lane_stride, "hipMalloc DCU ADM signals");
    check_hip(hipMemsetAsync(flags.get(), 0, flags_words * sizeof(std::uint32_t), stream), "DCU ADM flags memset");
    kernel::encode<T><<<static_cast<unsigned int>(blocks), reference::kWarpSize, 0, stream>>>(
        d_input, elements, params, codes.get(), reinterpret_cast<std::uint32_t*>(flags.get()),
        reinterpret_cast<T*>(centers.get()), reinterpret_cast<std::uint32_t*>(lengths.get()),
        signals.get(), lane_stride);
    check_hip(hipGetLastError(), "DCU ADM encode launch");
    check_hip(hipStreamSynchronize(stream), "DCU ADM encode synchronize");

    std::vector<std::uint32_t> host_lengths(blocks), host_flags(flags_words);
    std::vector<T> host_centers(blocks);
    check_hip(hipMemcpy(host_lengths.data(), lengths.get(), blocks * sizeof(std::uint32_t), hipMemcpyDeviceToHost), "DCU ADM lengths D2H");
    check_hip(hipMemcpy(host_flags.data(), flags.get(), flags_words * sizeof(std::uint32_t), hipMemcpyDeviceToHost), "DCU ADM flags D2H");
    check_hip(hipMemcpy(host_centers.data(), centers.get(), blocks * sizeof(T), hipMemcpyDeviceToHost), "DCU ADM centers D2H");

    std::vector<std::int32_t> offsets(blocks + 1, 0);
    for (std::size_t i = 0; i < blocks; ++i) {
        if (host_lengths[i] > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
            static_cast<std::size_t>(offsets[i]) + host_lengths[i] > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
            throw std::runtime_error("DCU ADM signal offsets exceed int32");
        }
        offsets[i + 1] = offsets[i] + static_cast<std::int32_t>(host_lengths[i]);
    }
    const std::size_t offsets_bytes = (blocks + 1) * sizeof(std::int32_t);
    const std::size_t centers_bytes = blocks * sizeof(T);
    const std::size_t signal_bytes = static_cast<std::size_t>(offsets.back()) * reference::kWarpSize;
    output_size = offsets_bytes + centers_bytes + flags_bytes + elements + signal_bytes;
    std::vector<std::uint8_t> packed(output_size, 0);
    for (std::size_t i = 0; i <= blocks; ++i) reference::write_le32(packed.data() + i * 4, static_cast<std::uint32_t>(offsets[i]));
    for (std::size_t i = 0; i < blocks; ++i) reference::write_raw(packed.data() + offsets_bytes + i * sizeof(T), host_centers[i]);
    auto* packed_flags = packed.data() + offsets_bytes + centers_bytes;
    for (std::size_t block = 0; block < blocks; ++block) {
        if (host_flags[block >> 5] & (1u << (31 - (block & 31)))) packed_flags[block >> 3] |= static_cast<std::uint8_t>(1u << (7 - (block & 7)));
    }
    auto* packed_codes = packed_flags + flags_bytes;
    check_hip(hipMemcpy(packed_codes, codes.get(), elements, hipMemcpyDeviceToHost), "DCU ADM codes D2H");
    if (signal_bytes != 0) {
        std::vector<std::uint8_t> scratch(blocks * reference::kWarpSize * lane_stride);
        check_hip(hipMemcpy(scratch.data(), signals.get(), scratch.size(), hipMemcpyDeviceToHost), "DCU ADM signals D2H");
        auto* dst = packed_codes + elements;
        for (std::size_t block = 0; block < blocks; ++block) {
            for (std::size_t lane = 0; lane < reference::kWarpSize; ++lane) {
                std::memcpy(dst + static_cast<std::size_t>(offsets[block]) * reference::kWarpSize + lane * host_lengths[block],
                            scratch.data() + (block * reference::kWarpSize + lane) * lane_stride,
                            host_lengths[block]);
            }
        }
    }
    check_hip(hipMemcpyAsync(d_output, packed.data(), packed.size(), hipMemcpyHostToDevice, stream), "DCU ADM packed H2D");
    check_hip(hipStreamSynchronize(stream), "DCU ADM packed synchronize");
}

template <typename T>
void decompress_kernel(const std::uint8_t* d_input, std::size_t input_size,
                       T* d_output, std::size_t elements,
                       const mans::MansParams& params, hipStream_t stream) {
    if (!d_input || !d_output || elements == 0) throw std::runtime_error("DCU ADM input/output is null");
    const std::size_t blocks = reference::block_count(elements, params);
    const std::size_t fixed = (blocks + 1) * sizeof(std::int32_t) + blocks * sizeof(T) +
                              reference::flags_size(blocks) + elements;
    if (input_size < fixed) throw std::runtime_error("DCU ADM payload is truncated");

    DeviceBuffer error;
    error.allocate(sizeof(std::uint32_t), "hipMalloc DCU ADM error");
    check_hip(hipMemsetAsync(error.get(), 0, sizeof(std::uint32_t), stream), "DCU ADM error memset");
    constexpr unsigned int kValidationThreads = 256;
    const unsigned int validation_grid = static_cast<unsigned int>(
        (blocks + kValidationThreads - 1) / kValidationThreads);
    if (validation_grid == 0) throw std::runtime_error("DCU ADM has no blocks");
    kernel::validate_payload<<<validation_grid, kValidationThreads, 0, stream>>>(
        d_input, input_size, blocks, fixed,
        reinterpret_cast<std::uint32_t*>(error.get()));
    check_hip(hipGetLastError(), "DCU ADM payload validation launch");
    check_hip(hipStreamSynchronize(stream), "DCU ADM payload validation synchronize");
    std::uint32_t host_error = 0;
    check_hip(hipMemcpy(&host_error, error.get(), sizeof(host_error), hipMemcpyDeviceToHost), "DCU ADM validation result");
    if (host_error != 0) throw std::runtime_error("DCU ADM payload offsets are invalid");

    kernel::decode<T><<<static_cast<unsigned int>(blocks), reference::kWarpSize, 0, stream>>>(
        d_input, input_size, elements, params, d_output,
        reinterpret_cast<std::uint32_t*>(error.get()));
    check_hip(hipGetLastError(), "DCU ADM decode launch");
    check_hip(hipStreamSynchronize(stream), "DCU ADM decode synchronize");
    check_hip(hipMemcpy(&host_error, error.get(), sizeof(host_error), hipMemcpyDeviceToHost), "DCU ADM decode result");
    if (host_error != 0) throw std::runtime_error("DCU ADM payload decode failed");
}

} // namespace

void compress_u16_device(const std::uint16_t* d_input, std::size_t n, const mans::MansParams& p,
                         std::uint8_t* d_output, std::size_t& size, hipStream_t stream) {
    compress_kernel(d_input, n, p, d_output, size, stream);
}
void decompress_u16_device(const std::uint8_t* d_input, std::size_t size, std::uint16_t* d_output,
                           std::size_t n, const mans::MansParams& p, hipStream_t stream) {
    decompress_kernel(d_input, size, d_output, n, p, stream);
}
std::size_t get_max_u16_payload_bytes(std::size_t n, const mans::MansParams& p) {
    return reference::max_payload_bytes<std::uint16_t>(n, p);
}

void compress_u32_device(const std::uint32_t* d_input, std::size_t n, const mans::MansParams& p,
                         std::uint8_t* d_output, std::size_t& size, hipStream_t stream) {
    compress_kernel(d_input, n, p, d_output, size, stream);
}
void decompress_u32_device(const std::uint8_t* d_input, std::size_t size, std::uint32_t* d_output,
                           std::size_t n, const mans::MansParams& p, hipStream_t stream) {
    decompress_kernel(d_input, size, d_output, n, p, stream);
}
std::size_t get_max_u32_payload_bytes(std::size_t n, const mans::MansParams& p) {
    return reference::max_payload_bytes<std::uint32_t>(n, p);
}

} // namespace mans::dcu::adm
