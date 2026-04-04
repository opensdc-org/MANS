#include "mans_nv.h"

#include <cstring>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#include "../mans_utils.h"
#include "adm/mapping_uint16.h"
#include "adm/mapping_uint32.h"
#include "ans/mans_nv_ans.h"

namespace mans {
namespace nv {

namespace {

void check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

std::uint32_t normalize_mode(std::uint32_t mode) {
    return (mode == Mode::R) ? Mode::R : Mode::P;
}

void write_header(std::uint8_t* out,
                  std::size_t raw_bytes,
                  const MansParams& params,
                  std::uint32_t mode) {
    MansHeader header{};
    header.codec = 1;
    header.mode = static_cast<std::uint8_t>(mode);
    header.dims = static_cast<std::uint8_t>(params.dims);
    mans::write_le64(header.raw_bytes_le, static_cast<std::uint64_t>(raw_bytes));
    header.nx = static_cast<std::uint64_t>(params.nx);
    header.ny = static_cast<std::uint64_t>(params.ny);
    header.nz = static_cast<std::uint64_t>(params.nz);
    std::memcpy(out, &header, sizeof(header));
}

void compress_adm_payload_device(const void* d_input_data,
                                 std::size_t num_elements,
                                 const MansParams& params,
                                 std::uint8_t* d_output,
                                 std::size_t& output_size,
                                 cudaStream_t stream) {
    if (num_elements == 0) {
        output_size = 0;
        return;
    }

    switch (params.dtype) {
        case DataType::U16:
            mans::nv::adm::compress_u16_device(
                static_cast<const std::uint16_t*>(d_input_data), num_elements, params, d_output, output_size, stream);
            return;
        case DataType::U32:
            mans::nv::adm::compress_u32_device(
                static_cast<const std::uint32_t*>(d_input_data), num_elements, params, d_output, output_size, stream);
            return;
        default:
            throw std::runtime_error("mans::nv::compress_internal_device: Unsupported dtype.");
    }
}

void decompress_adm_payload_device(const std::uint8_t* d_input,
                                   std::size_t input_size,
                                   const MansParams& params,
                                   std::uint8_t* d_output,
                                   std::size_t raw_bytes,
                                   cudaStream_t stream) {
    std::size_t elem_size = 0;
    if (!mans::get_dtype_size(params.dtype, elem_size) || elem_size == 0) {
        throw std::runtime_error("mans::nv::decompress_internal_device: Unsupported dtype.");
    }
    if (raw_bytes % elem_size != 0) {
        throw std::runtime_error("mans::nv::decompress_internal_device: raw size is not aligned to dtype.");
    }

    const std::size_t num_elements = raw_bytes / elem_size;
    if (num_elements == 0) {
        return;
    }

    switch (params.dtype) {
        case DataType::U16:
            mans::nv::adm::decompress_u16_device(
                d_input, input_size, reinterpret_cast<std::uint16_t*>(d_output), num_elements, params, stream);
            return;
        case DataType::U32:
            mans::nv::adm::decompress_u32_device(
                d_input, input_size, reinterpret_cast<std::uint32_t*>(d_output), num_elements, params, stream);
            return;
        default:
            throw std::runtime_error("mans::nv::decompress_internal_device: Unsupported dtype.");
    }
}

std::size_t get_max_adm_payload_bytes(std::size_t num_elements,
                                      const MansParams& params) {
    switch (params.dtype) {
        case DataType::U16:
            return mans::nv::adm::get_max_u16_payload_bytes(num_elements, params);
        case DataType::U32:
            return mans::nv::adm::get_max_u32_payload_bytes(num_elements, params);
        default:
            throw std::runtime_error("mans::nv::get_max_compress_bytes: Unsupported dtype.");
    }
}

} // namespace

void compress_internal_device(const void* d_input_data,
                              size_t length,
                              const MansParams& params,
                              std::uint8_t* d_out,
                              std::size_t& out_size,
                              cudaStream_t stream) {
    out_size = 0;
    if (!d_input_data || !d_out) {
        throw std::runtime_error("mans::nv::compress_internal_device: input/output buffer is null.");
    }

    std::size_t elem_size = 0;
    if (!mans::get_dtype_size(params.dtype, elem_size)) {
        throw std::runtime_error("mans::nv::compress_internal_device: Unsupported dtype.");
    }

    const std::uint32_t mode = normalize_mode(params.mode);
    const std::size_t raw_bytes = length * elem_size;
    const std::size_t max_adm_bytes = get_max_adm_payload_bytes(length, params);
    const std::size_t max_entropy_bytes =
        mans::nv::ans::get_max_compress_bytes(max_adm_bytes, mode, params);

    std::uint8_t* d_adm_payload = nullptr;
    std::uint8_t* d_entropy_payload = nullptr;
    std::size_t adm_size = 0;
    std::size_t entropy_size = 0;

    std::uint8_t header_bytes[kMansHeaderBytes] = {};
    write_header(header_bytes, raw_bytes, params, mode);

    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_adm_payload), max_adm_bytes), "cudaMalloc d_adm_payload");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_entropy_payload), max_entropy_bytes), "cudaMalloc d_entropy_payload");

    compress_adm_payload_device(d_input_data, length, params, d_adm_payload, adm_size, stream);
    mans::nv::ans::compress_stage_device(
        d_adm_payload, adm_size, d_entropy_payload, entropy_size, mode, params, stream);

    check_cuda(cudaMemcpyAsync(d_out,
                               header_bytes,
                               kMansHeaderBytes,
                               cudaMemcpyHostToDevice,
                               stream),
               "cudaMemcpyAsync H2D MANS header");
    if (entropy_size != 0) {
        check_cuda(cudaMemcpyAsync(d_out + kMansHeaderBytes,
                                   d_entropy_payload,
                                   entropy_size,
                                   cudaMemcpyDeviceToDevice,
                                   stream),
                   "cudaMemcpyAsync D2D entropy payload");
    }
    check_cuda(cudaStreamSynchronize(stream), "compress_internal_device sync");
    out_size = kMansHeaderBytes + entropy_size;

    cudaFree(d_adm_payload);
    cudaFree(d_entropy_payload);
}

void decompress_internal_device(const void* d_input_data,
                                size_t length,
                                const MansParams& params,
                                std::uint8_t* d_out,
                                std::size_t& out_size,
                                cudaStream_t stream) {
    out_size = 0;
    if (!d_input_data || !d_out) {
        throw std::runtime_error("mans::nv::decompress_internal_device: input/output buffer is null.");
    }

    if (length <= kMansHeaderBytes) {
        throw std::runtime_error("mans::nv::decompress_internal_device: missing payload.");
    }

    MansHeader header{};
    check_cuda(cudaMemcpy(&header, d_input_data, sizeof(header), cudaMemcpyDeviceToHost),
               "cudaMemcpy D2H MANS header");

    std::size_t raw_bytes = 0;
    std::string parse_error;
    if (!mans::parse_mans_raw_bytes(&header, sizeof(header), raw_bytes, &parse_error)) {
        throw std::runtime_error("mans::nv::decompress_internal_device: " + parse_error + ".");
    }

    MansParams effective_params = params;
    effective_params.dims = static_cast<std::uint32_t>(header.dims);
    effective_params.nx = static_cast<std::uint32_t>(header.nx);
    effective_params.ny = static_cast<std::uint32_t>(header.ny);
    effective_params.nz = static_cast<std::uint32_t>(header.nz);

    std::size_t elem_size = 0;
    if (!mans::get_dtype_size(effective_params.dtype, elem_size) || elem_size == 0) {
        throw std::runtime_error("mans::nv::decompress_internal_device: Unsupported dtype.");
    }
    const std::size_t num_elements = raw_bytes / elem_size;
    const std::size_t max_adm_bytes = get_max_adm_payload_bytes(num_elements, effective_params);

    const auto* d_payload = static_cast<const std::uint8_t*>(d_input_data) + kMansHeaderBytes;
    const std::size_t payload_len = length - kMansHeaderBytes;

    std::uint8_t* d_adm_payload = nullptr;
    std::size_t adm_size = 0;

    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_adm_payload), max_adm_bytes), "cudaMalloc d_adm_payload");

    mans::nv::ans::decompress_stage_device(
        d_payload,
        payload_len,
        d_adm_payload,
        max_adm_bytes,
        adm_size,
        static_cast<std::uint32_t>(header.mode),
        effective_params,
        stream);
    decompress_adm_payload_device(d_adm_payload, adm_size, effective_params, d_out, raw_bytes, stream);
    check_cuda(cudaStreamSynchronize(stream), "decompress_internal_device sync");
    out_size = raw_bytes;

    cudaFree(d_adm_payload);
}

void compress_internal(const void* input_data,
                      size_t length,
                      const MansParams& params,
                      std::uint8_t* out,
                      std::size_t& out_size,
                      bool save_adm,
                      const std::string& dump_path) {
    (void)save_adm;
    (void)dump_path;

    out_size = 0;
    if (!input_data || !out) {
        throw std::runtime_error("mans::nv::compress_internal: input/output buffer is null.");
    }

    std::size_t elem_size = 0;
    if (!mans::get_dtype_size(params.dtype, elem_size)) {
        throw std::runtime_error("mans::nv::compress_internal: Unsupported dtype.");
    }

    const std::size_t input_bytes = length * elem_size;
    const std::size_t max_output_bytes = get_max_compress_bytes(length, params);

    std::uint8_t* d_input = nullptr;
    std::uint8_t* d_output = nullptr;

    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_input), input_bytes), "cudaMalloc d_input");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_output), max_output_bytes), "cudaMalloc d_output");
    if (input_bytes != 0) {
        check_cuda(cudaMemcpy(d_input, input_data, input_bytes, cudaMemcpyHostToDevice),
                   "cudaMemcpy H2D raw input");
    }

    compress_internal_device(d_input, length, params, d_output, out_size);
    if (out_size != 0) {
        check_cuda(cudaMemcpy(out, d_output, out_size, cudaMemcpyDeviceToHost),
                   "cudaMemcpy D2H compressed output");
    }

    cudaFree(d_input);
    cudaFree(d_output);
}

void decompress_internal(const void* input_data,
                        size_t length,
                        const MansParams& params,
                        std::uint8_t* out,
                        std::size_t& out_size,
                        bool save_adm,
                        const std::string& dump_path) {
    (void)save_adm;
    (void)dump_path;

    if (!input_data || !out) {
        throw std::runtime_error("mans::nv::decompress_internal: input/output buffer is null.");
    }

    const std::size_t raw_bytes = get_exact_decompress_bytes(input_data, length, params);
    std::uint8_t* d_input = nullptr;
    std::uint8_t* d_output = nullptr;

    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_input), length), "cudaMalloc d_input");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_output), raw_bytes), "cudaMalloc d_output");
    check_cuda(cudaMemcpy(d_input, input_data, length, cudaMemcpyHostToDevice),
               "cudaMemcpy H2D compressed input");

    decompress_internal_device(d_input, length, params, d_output, out_size);
    if (out_size != 0) {
        check_cuda(cudaMemcpy(out, d_output, out_size, cudaMemcpyDeviceToHost),
                   "cudaMemcpy D2H raw output");
    }

    cudaFree(d_input);
    cudaFree(d_output);
}

std::size_t get_max_compress_bytes(std::size_t num_elements,
                                   const MansParams& params) {
    if (num_elements == 0) {
        return 0;
    }

    const std::size_t adm_bytes = get_max_adm_payload_bytes(num_elements, params);
    const std::size_t entropy_bytes =
        mans::nv::ans::get_max_compress_bytes(adm_bytes, normalize_mode(params.mode), params);
    return kMansHeaderBytes + entropy_bytes;
}

std::size_t get_exact_decompress_bytes(const void* compressed_data,
                                       std::size_t compressed_len,
                                       const MansParams& params) {
    (void)params;

    if (compressed_len <= kMansHeaderBytes) {
        throw std::runtime_error("mans::nv::get_exact_decompress_bytes: missing payload.");
    }

    std::size_t raw_bytes = 0;
    std::string parse_error;
    if (!mans::parse_mans_raw_bytes(compressed_data, compressed_len, raw_bytes, &parse_error)) {
        throw std::runtime_error("mans::nv::get_exact_decompress_bytes: " + parse_error + ".");
    }
    return raw_bytes;
}

} // namespace nv
} // namespace mans
