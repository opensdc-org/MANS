#include "mans_dcu.h"

#include <hip/hip_runtime.h>

#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "../mans_utils.h"
#include "ans/dcu_ans.h"
#include "adm/adm_reference.h"
#include "adm/mapping_uint16.h"
#include "adm/mapping_uint32.h"

namespace mans::dcu {
namespace {

constexpr std::uint8_t kDcuCodec = Codec::ADM;

void check_hip(hipError_t status, const char* what) {
    if (status != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(status));
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
    void reset() { if (ptr_) (void)hipFree(ptr_); ptr_ = nullptr; }
    std::uint8_t* get() const { return ptr_; }
private:
    std::uint8_t* ptr_ = nullptr;
};

std::size_t raw_bytes_for(std::size_t length, std::uint32_t dtype) {
    std::size_t element_size = 0;
    if (!mans::get_dtype_size(dtype, element_size)) throw std::runtime_error("mans::dcu: unsupported dtype");
    if (length > std::numeric_limits<std::size_t>::max() / element_size) throw std::runtime_error("mans::dcu: raw size overflow");
    const std::size_t bytes = length * element_size;
    if (bytes > std::numeric_limits<std::uint32_t>::max()) throw std::runtime_error("mans::dcu: raw size exceeds PANS uint32 limit");
    return bytes;
}

void validate_geometry(std::size_t length, const MansParams& p) {
    if (p.dims < 1 || p.dims > 3 || p.nx == 0 || (p.dims >= 2 && p.ny == 0) || (p.dims == 3 && p.nz == 0)) {
        throw std::runtime_error("mans::dcu: invalid geometry");
    }
    std::size_t product = p.nx;
    if (p.dims >= 2) product *= p.ny;
    if (p.dims == 3) product *= p.nz;
    if (product != length) throw std::runtime_error("mans::dcu: geometry does not match element count");
}

void require_p_mode(const MansParams& p) {
    if (p.mode != Mode::P) throw std::runtime_error("mans::dcu: only P mode is supported");
}

template <typename T>
std::size_t max_adm_bytes(std::size_t n, const MansParams& p) {
    return adm::reference::max_payload_bytes<T>(n, p);
}

template <typename T>
void compress_adm(const T* d_input, std::size_t n, const MansParams& p,
                  std::uint8_t* d_output, std::size_t& output_size) {
    if constexpr (std::is_same_v<T, std::uint16_t>) {
        adm::compress_u16_device(d_input, n, p, d_output, output_size);
    } else {
        adm::compress_u32_device(d_input, n, p, d_output, output_size);
    }
}

template <typename T>
void decompress_adm(const std::uint8_t* d_input, std::size_t input_size, T* d_output,
                    std::size_t n, const MansParams& p) {
    if constexpr (std::is_same_v<T, std::uint16_t>) {
        adm::decompress_u16_device(d_input, input_size, d_output, n, p);
    } else {
        adm::decompress_u32_device(d_input, input_size, d_output, n, p);
    }
}

template <typename T>
void compress_device_t(const T* d_input, std::size_t n, const MansParams& p,
                       std::uint8_t* d_out, std::size_t capacity, std::size_t& out_size) {
    validate_geometry(n, p);
    const std::size_t raw_bytes = raw_bytes_for(n, p.dtype);
    const std::size_t adm_capacity = max_adm_bytes<T>(n, p);
    DeviceBuffer d_adm;
    d_adm.allocate(adm_capacity, "hipMalloc DCU ADM payload");
    std::size_t adm_size = 0;
    compress_adm(d_input, n, p, d_adm.get(), adm_size);
    if (adm_size == 0 || adm_size > adm_capacity) throw std::runtime_error("mans::dcu: ADM encode failed");

    const std::size_t entropy_capacity = ans::get_max_compressed_size(adm_size);
    if (entropy_capacity == 0 || kMansHeaderBytes > capacity ||
        entropy_capacity > capacity - kMansHeaderBytes) {
        throw std::runtime_error("mans::dcu: output buffer is too small");
    }
    std::size_t entropy_size = entropy_capacity;
    ans::compress_device(d_adm.get(), adm_size, d_out + kMansHeaderBytes,
                         capacity - kMansHeaderBytes, entropy_size);
    std::uint8_t header[kMansHeaderBytes] = {};
    mans::write_mans_header(header, raw_bytes, kDcuCodec, Mode::P,
                            static_cast<std::uint8_t>(p.dims), p.nx, p.ny, p.nz);
    check_hip(hipMemcpy(d_out, header, kMansHeaderBytes, hipMemcpyHostToDevice), "DCU MANS header H2D");
    out_size = kMansHeaderBytes + entropy_size;
}

template <typename T>
void decompress_device_t(const std::uint8_t* d_input, std::size_t length, const MansParams& p,
                         T* d_output, std::size_t capacity, std::size_t& out_size) {
    if (length <= kMansHeaderBytes) throw std::runtime_error("mans::dcu: compressed stream is too small");
    std::uint8_t header_bytes[kMansHeaderBytes] = {};
    check_hip(hipMemcpy(header_bytes, d_input, kMansHeaderBytes, hipMemcpyDeviceToHost), "DCU MANS header D2H");
    MansHeader header{};
    std::size_t raw_bytes = 0;
    std::string error;
    if (!mans::parse_mans_header(header_bytes, sizeof(header_bytes), header, raw_bytes, &error)) throw std::runtime_error("mans::dcu: " + error);
    if (header.codec != kDcuCodec || header.mode != Mode::P) throw std::runtime_error("mans::dcu: stream is not codec=1 P-mode");
    if (!mans::validate_mans_geometry(header, p.dtype, raw_bytes, &error)) throw std::runtime_error("mans::dcu: " + error);
    if (raw_bytes > capacity) throw std::runtime_error("mans::dcu: output buffer is too small");

    MansParams effective = p;
    effective.dims = header.dims;
    effective.nx = static_cast<std::uint32_t>(header.nx);
    effective.ny = static_cast<std::uint32_t>(header.ny);
    effective.nz = static_cast<std::uint32_t>(header.nz);
    const std::size_t adm_capacity = max_adm_bytes<T>(raw_bytes / sizeof(T), effective);
    DeviceBuffer d_adm;
    d_adm.allocate(adm_capacity, "hipMalloc DCU decoded ADM payload");

    const std::size_t entropy_size = length - kMansHeaderBytes;
    if (entropy_size < 12) throw std::runtime_error("mans::dcu: ANS payload is too small");
    std::uint32_t adm_bytes = 0;
    check_hip(hipMemcpy(&adm_bytes, d_input + kMansHeaderBytes + 8,
                       sizeof(adm_bytes), hipMemcpyDeviceToHost), "DCU ANS size D2H");
    const std::size_t adm_size = adm_bytes;
    if (adm_size == 0 || adm_size > adm_capacity) throw std::runtime_error("mans::dcu: invalid ANS decoded size");
    std::size_t decoded_size = adm_size;
    ans::decompress_device(d_input + kMansHeaderBytes, entropy_size,
                           d_adm.get(), adm_capacity, decoded_size);
    if (decoded_size != adm_size) throw std::runtime_error("mans::dcu: ANS decode size mismatch");
    decompress_adm(d_adm.get(), adm_size, d_output, raw_bytes / sizeof(T), effective);
    out_size = raw_bytes;
}

} // namespace

void compress_internal_device(const void* d_input_data, std::size_t length, const MansParams& p,
                              std::uint8_t* d_out, std::size_t& out_size) {
    const std::size_t capacity = out_size;
    out_size = 0;
    if (!d_input_data || !d_out) throw std::runtime_error("mans::dcu::compress_internal_device: null device pointer");
    require_p_mode(p);
    if (capacity < get_max_compress_bytes(length, p)) throw std::runtime_error("mans::dcu: output buffer is too small");
    if (p.dtype == DataType::U16) compress_device_t(static_cast<const std::uint16_t*>(d_input_data), length, p, d_out, capacity, out_size);
    else if (p.dtype == DataType::U32) compress_device_t(static_cast<const std::uint32_t*>(d_input_data), length, p, d_out, capacity, out_size);
    else throw std::runtime_error("mans::dcu: unsupported dtype");
}

void decompress_internal_device(const void* d_input_data, std::size_t length, const MansParams& p,
                                std::uint8_t* d_out, std::size_t& out_size) {
    const std::size_t capacity = out_size;
    out_size = 0;
    if (!d_input_data || !d_out) throw std::runtime_error("mans::dcu::decompress_internal_device: null device pointer");
    require_p_mode(p);
    if (p.dtype == DataType::U16) decompress_device_t(static_cast<const std::uint8_t*>(d_input_data), length, p, reinterpret_cast<std::uint16_t*>(d_out), capacity, out_size);
    else if (p.dtype == DataType::U32) decompress_device_t(static_cast<const std::uint8_t*>(d_input_data), length, p, reinterpret_cast<std::uint32_t*>(d_out), capacity, out_size);
    else throw std::runtime_error("mans::dcu: unsupported dtype");
}

void compress_internal(const void* input_data, std::size_t length, const MansParams& p,
                       std::uint8_t* out, std::size_t& out_size, bool save_adm, const std::string& dump_path) {
    (void)save_adm; (void)dump_path;
    const std::size_t capacity = out_size;
    out_size = 0;
    const std::size_t raw_bytes = raw_bytes_for(length, p.dtype);
    DeviceBuffer d_input, d_output;
    d_input.allocate(raw_bytes, "hipMalloc DCU raw input");
    d_output.allocate(get_max_compress_bytes(length, p), "hipMalloc DCU compressed output");
    check_hip(hipMemcpy(d_input.get(), input_data, raw_bytes, hipMemcpyHostToDevice), "DCU raw input H2D");
    std::size_t device_size = d_output.get() ? get_max_compress_bytes(length, p) : 0;
    compress_internal_device(d_input.get(), length, p, d_output.get(), device_size);
    if (device_size > capacity) throw std::runtime_error("mans::dcu: output buffer is too small");
    check_hip(hipMemcpy(out, d_output.get(), device_size, hipMemcpyDeviceToHost), "DCU compressed output D2H");
    out_size = device_size;
}

void decompress_internal(const void* input_data, std::size_t length, const MansParams& p,
                         std::uint8_t* out, std::size_t& out_size, bool save_adm, const std::string& dump_path) {
    (void)save_adm; (void)dump_path;
    const std::size_t capacity = out_size;
    out_size = 0;
    const std::size_t raw_bytes = get_exact_decompress_bytes(input_data, length, p);
    DeviceBuffer d_input, d_output;
    d_input.allocate(length, "hipMalloc DCU compressed input");
    d_output.allocate(raw_bytes, "hipMalloc DCU raw output");
    check_hip(hipMemcpy(d_input.get(), input_data, length, hipMemcpyHostToDevice), "DCU compressed input H2D");
    std::size_t device_size = raw_bytes;
    decompress_internal_device(d_input.get(), length, p, d_output.get(), device_size);
    if (device_size > capacity) throw std::runtime_error("mans::dcu: output buffer is too small");
    check_hip(hipMemcpy(out, d_output.get(), device_size, hipMemcpyDeviceToHost), "DCU raw output D2H");
    out_size = device_size;
}

std::size_t get_max_compress_bytes(std::size_t n, const MansParams& p) {
    if (n == 0) return 0;
    require_p_mode(p);
    validate_geometry(n, p);
    raw_bytes_for(n, p.dtype);
    const std::size_t adm_bytes = p.dtype == DataType::U16 ? max_adm_bytes<std::uint16_t>(n, p) : max_adm_bytes<std::uint32_t>(n, p);
    const std::size_t ans_bytes = ans::get_max_compressed_size(adm_bytes);
    if (ans_bytes == 0 || ans_bytes > std::numeric_limits<std::size_t>::max() - kMansHeaderBytes) {
        throw std::runtime_error("mans::dcu: compressed-size bound overflow");
    }
    return kMansHeaderBytes + ans_bytes;
}

std::size_t get_exact_decompress_bytes(const void* data, std::size_t length, const MansParams& p) {
    if (length <= kMansHeaderBytes) throw std::runtime_error("mans::dcu: compressed stream is too small");
    MansHeader header{};
    std::size_t raw_bytes = 0;
    std::string error;
    if (!mans::parse_mans_header(data, length, header, raw_bytes, &error)) throw std::runtime_error("mans::dcu: " + error);
    if (header.codec != kDcuCodec || header.mode != Mode::P) throw std::runtime_error("mans::dcu: not a codec=1 P-mode stream");
    if (!mans::validate_mans_geometry(header, p.dtype, raw_bytes, &error)) throw std::runtime_error("mans::dcu: " + error);
    return raw_bytes;
}

} // namespace mans::dcu
