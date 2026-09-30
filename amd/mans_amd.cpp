#include "mans_amd.h"

#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include "../mans_utils.h"
#include "../cpu/adm/adm_utils.h"
#include "ans/mans_amd_ans.h"

namespace mans {
namespace amd {
namespace {

constexpr std::uint8_t kAmdCodec = Codec::AMD_ANS;

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
        if (bytes != 0) {
            check_hip(hipMalloc(&ptr_, bytes), what);
        }
    }

    void reset() {
        if (ptr_ != nullptr) {
            (void)hipFree(ptr_);
            ptr_ = nullptr;
        }
    }

    std::uint8_t* get() const { return ptr_; }

private:
    std::uint8_t* ptr_ = nullptr;
};

std::uint32_t require_p_mode(const MansParams& params) {
    if (params.mode != Mode::P) {
        throw std::runtime_error("mans::amd: AMD backend currently supports only Mode::P.");
    }
    return Mode::P;
}

std::size_t checked_raw_bytes(std::size_t length, std::uint32_t dtype) {
    std::size_t elem_size = 0;
    if (!mans::get_dtype_size(dtype, elem_size)) {
        throw std::runtime_error("mans::amd: unsupported dtype.");
    }
    if (length > std::numeric_limits<std::size_t>::max() / elem_size) {
        throw std::runtime_error("mans::amd: raw size overflows size_t.");
    }
    const std::size_t raw_bytes = length * elem_size;
    if (raw_bytes > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("mans::amd: raw size exceeds AMD ANS uint32 limit.");
    }
    return raw_bytes;
}

void validate_compress_geometry(std::size_t length, const MansParams& params) {
    if (params.dims < 1 || params.dims > 3 || params.nx == 0 ||
        (params.dims >= 2 && params.ny == 0) ||
        (params.dims == 3 && params.nz == 0)) {
        throw std::runtime_error("mans::amd: invalid compression geometry.");
    }
    std::size_t product = static_cast<std::size_t>(params.nx);
    if (params.dims >= 2) {
        if (product > std::numeric_limits<std::size_t>::max() / params.ny) {
            throw std::runtime_error("mans::amd: geometry overflows size_t.");
        }
        product *= params.ny;
    }
    if (params.dims == 3) {
        if (product > std::numeric_limits<std::size_t>::max() / params.nz) {
            throw std::runtime_error("mans::amd: geometry overflows size_t.");
        }
        product *= params.nz;
    }
    if (product != length) {
        throw std::runtime_error("mans::amd: geometry does not match element count.");
    }
}

void write_header(std::uint8_t* out, std::size_t raw_bytes, const MansParams& params) {
    mans::write_mans_header(out, raw_bytes, kAmdCodec, Mode::P,
                            static_cast<std::uint8_t>(params.dims),
                            params.nx, params.ny, params.nz);
}

template <typename T>
std::size_t max_adm_bytes(std::size_t elements, const MansParams& params) {
    const std::size_t result = adm_max_compressed_size<T>(elements, params);
    if (result > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("mans::amd: ADM payload exceeds uint32 limit.");
    }
    return result;
}

template <typename T>
std::vector<std::uint8_t> make_adm_payload(const T* input, std::size_t elements,
                                           const MansParams& params) {
    std::vector<std::uint8_t> payload(max_adm_bytes<T>(elements, params));
    std::size_t payload_size = 0;
    adm_compress(input, elements, payload.data(), payload_size, params);
    if (payload_size == 0 || payload_size > payload.size()) {
        throw std::runtime_error("mans::amd: CPU ADM compression failed.");
    }
    payload.resize(payload_size);
    return payload;
}

template <typename T>
void restore_adm_payload(const std::uint8_t* payload, std::size_t payload_size,
                         T* output, std::size_t elements, const MansParams& params) {
    adm_decompress(payload, payload_size, output, elements, params);
}

template <typename T>
void compress_device_t(const T* input, std::size_t length, const MansParams& params,
                       std::uint8_t* d_out, std::size_t output_capacity,
                       std::size_t& out_size, hipStream_t stream) {
    const std::size_t raw_bytes = checked_raw_bytes(length, params.dtype);
    validate_compress_geometry(length, params);
    std::vector<T> host_input(length);
    check_hip(hipMemcpy(host_input.data(), input, raw_bytes, hipMemcpyDeviceToHost),
              "AMD raw input D2H for CPU ADM");
    const std::vector<std::uint8_t> adm_payload = make_adm_payload(host_input.data(), length, params);
    const std::size_t entropy_capacity = ans::get_max_compress_bytes(adm_payload.size());
    if (kMansHeaderBytes + entropy_capacity > output_capacity) {
        throw std::runtime_error("mans::amd: output buffer is too small.");
    }

    DeviceBuffer d_adm;
    DeviceBuffer d_entropy;
    d_adm.allocate(adm_payload.size(), "hipMalloc AMD ADM staging");
    d_entropy.allocate(entropy_capacity, "hipMalloc AMD ANS staging");
    check_hip(hipMemcpyAsync(d_adm.get(), adm_payload.data(), adm_payload.size(),
                            hipMemcpyHostToDevice, stream),
              "AMD ADM payload H2D");

    std::size_t entropy_size = 0;
    ans::compress_stage_device(d_adm.get(), adm_payload.size(), d_entropy.get(),
                               entropy_capacity, entropy_size, stream);
    std::uint8_t header[kMansHeaderBytes] = {};
    write_header(header, raw_bytes, params);
    check_hip(hipMemcpyAsync(d_out, header, kMansHeaderBytes,
                            hipMemcpyHostToDevice, stream),
              "AMD MANS header H2D");
    check_hip(hipMemcpyAsync(d_out + kMansHeaderBytes, d_entropy.get(), entropy_size,
                            hipMemcpyDeviceToDevice, stream),
              "AMD ANS payload D2D");
    check_hip(hipStreamSynchronize(stream), "AMD MANS compression synchronize");
    out_size = kMansHeaderBytes + entropy_size;
}

template <typename T>
void decompress_device_t(const std::uint8_t* d_input, std::size_t length,
                         const MansParams& params, T* d_output,
                         std::size_t output_capacity, std::size_t& out_size,
                         hipStream_t stream) {
    if (length <= kMansHeaderBytes) {
        throw std::runtime_error("mans::amd: compressed stream is too small.");
    }
    std::uint8_t header_bytes[kMansHeaderBytes] = {};
    check_hip(hipMemcpy(header_bytes, d_input, kMansHeaderBytes,
                        hipMemcpyDeviceToHost),
              "AMD MANS header D2H");
    MansHeader header{};
    std::size_t raw_bytes = 0;
    std::string parse_error;
    if (!mans::parse_mans_header(header_bytes, kMansHeaderBytes, header, raw_bytes, &parse_error)) {
        throw std::runtime_error("mans::amd: " + parse_error + ".");
    }
    if (header.codec != kAmdCodec || header.mode != Mode::P) {
        throw std::runtime_error("mans::amd: stream is not an AMD P-mode stream.");
    }
    if (!mans::validate_mans_geometry(header, params.dtype, raw_bytes, &parse_error)) {
        throw std::runtime_error("mans::amd: " + parse_error + ".");
    }
    if (raw_bytes > output_capacity) {
        throw std::runtime_error("mans::amd::decompress_internal_device: output buffer is too small.");
    }
    MansParams effective = params;
    effective.dims = header.dims;
    effective.nx = static_cast<std::uint32_t>(header.nx);
    effective.ny = static_cast<std::uint32_t>(header.ny);
    effective.nz = static_cast<std::uint32_t>(header.nz);
    const std::size_t elements = raw_bytes / sizeof(T);
    const std::size_t payload_size = length - kMansHeaderBytes;
    const std::size_t adm_capacity = max_adm_bytes<T>(elements, effective);

    DeviceBuffer d_adm;
    d_adm.allocate(adm_capacity, "hipMalloc AMD ADM decode staging");
    std::size_t adm_size = 0;
    ans::decompress_stage_device(d_input + kMansHeaderBytes, payload_size,
                                 d_adm.get(), adm_capacity, adm_size, stream);
    std::vector<std::uint8_t> host_adm(adm_size);
    check_hip(hipMemcpy(host_adm.data(), d_adm.get(), adm_size,
                        hipMemcpyDeviceToHost),
              "AMD ADM payload D2H");
    std::vector<T> recovered(elements);
    restore_adm_payload(host_adm.data(), host_adm.size(), recovered.data(), elements, effective);
    check_hip(hipMemcpyAsync(reinterpret_cast<std::uint8_t*>(d_output), recovered.data(), raw_bytes,
                            hipMemcpyHostToDevice, stream),
              "AMD raw output H2D");
    check_hip(hipStreamSynchronize(stream), "AMD MANS decompression synchronize");
    out_size = raw_bytes;
}

} // namespace

void compress_internal_device(const void* d_input_data, std::size_t length,
                              const MansParams& params, std::uint8_t* d_out,
                              std::size_t& out_size) {
    hipStream_t stream = nullptr;
    const std::size_t output_capacity = out_size;
    out_size = 0;
    if (!d_input_data || !d_out) {
        throw std::runtime_error("mans::amd::compress_internal_device: null device pointer.");
    }
    require_p_mode(params);
    const std::size_t required_capacity = get_max_compress_bytes(length, params);
    if (output_capacity < required_capacity) {
        throw std::runtime_error("mans::amd::compress_internal_device: output buffer is too small.");
    }
    checked_raw_bytes(length, params.dtype);
    switch (params.dtype) {
        case DataType::U16:
            compress_device_t(static_cast<const std::uint16_t*>(d_input_data), length,
                              params, d_out, output_capacity, out_size, stream);
            return;
        case DataType::U32:
            compress_device_t(static_cast<const std::uint32_t*>(d_input_data), length,
                              params, d_out, output_capacity, out_size, stream);
            return;
        default:
            throw std::runtime_error("mans::amd::compress_internal_device: unsupported dtype.");
    }
}

void decompress_internal_device(const void* d_input_data, std::size_t length,
                                const MansParams& params, std::uint8_t* d_out,
                                std::size_t& out_size) {
    hipStream_t stream = nullptr;
    const std::size_t output_capacity = out_size;
    out_size = 0;
    if (!d_input_data || !d_out) {
        throw std::runtime_error("mans::amd::decompress_internal_device: null device pointer.");
    }
    require_p_mode(params);
    switch (params.dtype) {
        case DataType::U16:
            decompress_device_t(static_cast<const std::uint8_t*>(d_input_data), length,
                                params, reinterpret_cast<std::uint16_t*>(d_out),
                                output_capacity, out_size, stream);
            return;
        case DataType::U32:
            decompress_device_t(static_cast<const std::uint8_t*>(d_input_data), length,
                                params, reinterpret_cast<std::uint32_t*>(d_out),
                                output_capacity, out_size, stream);
            return;
        default:
            throw std::runtime_error("mans::amd::decompress_internal_device: unsupported dtype.");
    }
}

void compress_internal(const void* input_data, std::size_t length, const MansParams& params,
                       std::uint8_t* out, std::size_t& out_size,
                       bool save_adm, const std::string& dump_path) {
    (void)save_adm;
    (void)dump_path;
    const std::size_t output_capacity = out_size;
    out_size = 0;
    if (!input_data || !out) {
        throw std::runtime_error("mans::amd::compress_internal: null host pointer.");
    }
    const std::size_t required_capacity = get_max_compress_bytes(length, params);
    if (output_capacity < required_capacity) {
        throw std::runtime_error("mans::amd::compress_internal: output buffer is too small.");
    }
    const std::size_t raw_bytes = checked_raw_bytes(length, params.dtype);
    DeviceBuffer d_input;
    DeviceBuffer d_output;
    d_input.allocate(raw_bytes, "hipMalloc AMD raw input");
    d_output.allocate(get_max_compress_bytes(length, params), "hipMalloc AMD compressed output");
    check_hip(hipMemcpy(d_input.get(), input_data, raw_bytes, hipMemcpyHostToDevice),
              "AMD raw input H2D");
    const std::size_t device_capacity = get_max_compress_bytes(length, params);
    std::size_t device_size = device_capacity;
    compress_internal_device(d_input.get(), length, params, d_output.get(), device_size);
    check_hip(hipMemcpy(out, d_output.get(), device_size, hipMemcpyDeviceToHost),
              "AMD compressed output D2H");
    out_size = device_size;
}

void decompress_internal(const void* input_data, std::size_t length, const MansParams& params,
                         std::uint8_t* out, std::size_t& out_size,
                         bool save_adm, const std::string& dump_path) {
    (void)save_adm;
    (void)dump_path;
    const std::size_t output_capacity = out_size;
    out_size = 0;
    if (!input_data || !out) {
        throw std::runtime_error("mans::amd::decompress_internal: null host pointer.");
    }
    const std::size_t raw_bytes = get_exact_decompress_bytes(input_data, length, params);
    if (output_capacity < raw_bytes) {
        throw std::runtime_error("mans::amd::decompress_internal: output buffer is too small.");
    }
    DeviceBuffer d_input;
    DeviceBuffer d_output;
    d_input.allocate(length, "hipMalloc AMD compressed input");
    d_output.allocate(raw_bytes, "hipMalloc AMD raw output");
    check_hip(hipMemcpy(d_input.get(), input_data, length, hipMemcpyHostToDevice),
              "AMD compressed input H2D");
    std::size_t device_size = raw_bytes;
    decompress_internal_device(d_input.get(), length, params, d_output.get(), device_size);
    check_hip(hipMemcpy(out, d_output.get(), device_size, hipMemcpyDeviceToHost),
              "AMD raw output D2H");
    out_size = device_size;
}

std::size_t get_max_compress_bytes(std::size_t num_elements, const MansParams& params) {
    if (num_elements == 0) {
        return 0;
    }
    require_p_mode(params);
    checked_raw_bytes(num_elements, params.dtype);
    validate_compress_geometry(num_elements, params);
    std::size_t adm_bytes = 0;
    if (params.dtype == DataType::U16) {
        adm_bytes = max_adm_bytes<std::uint16_t>(num_elements, params);
    } else if (params.dtype == DataType::U32) {
        adm_bytes = max_adm_bytes<std::uint32_t>(num_elements, params);
    } else {
        throw std::runtime_error("mans::amd::get_max_compress_bytes: unsupported dtype.");
    }
    return kMansHeaderBytes + ans::get_max_compress_bytes(adm_bytes);
}

std::size_t get_exact_decompress_bytes(const void* compressed_data,
                                       std::size_t compressed_len,
                                       const MansParams& params) {
    if (compressed_len <= kMansHeaderBytes) {
        throw std::runtime_error("mans::amd::get_exact_decompress_bytes: missing payload.");
    }
    MansHeader header{};
    std::size_t raw_bytes = 0;
    std::string parse_error;
    if (!mans::parse_mans_header(compressed_data, compressed_len, header, raw_bytes, &parse_error)) {
        throw std::runtime_error("mans::amd::get_exact_decompress_bytes: " + parse_error + ".");
    }
    if (header.codec != kAmdCodec || header.mode != Mode::P) {
        throw std::runtime_error("mans::amd::get_exact_decompress_bytes: not an AMD P-mode stream.");
    }
    if (!mans::validate_mans_geometry(header, params.dtype, raw_bytes, &parse_error)) {
        throw std::runtime_error("mans::amd::get_exact_decompress_bytes: " + parse_error + ".");
    }
    return raw_bytes;
}

} // namespace amd
} // namespace mans
