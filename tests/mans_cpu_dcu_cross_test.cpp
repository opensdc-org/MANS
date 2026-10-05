#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../mans_api.hpp"
#include "../mans_utils.h"
#include "../cpu/pans/CpuANSUtils.h"
#include "../cpu/pans/pans_utils.h"
#include "../dcu/adm/adm_reference.h"
#include "../dcu/adm/mapping_uint16.h"

namespace {

void check_hip(hipError_t status, const char* what) {
    if (status != hipSuccess) {
        throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(status));
    }
}

struct Shape {
    std::uint32_t dims;
    std::uint32_t nx;
    std::uint32_t ny;
    std::uint32_t nz;

    std::size_t elements() const {
        return static_cast<std::size_t>(nx) * (dims >= 2 ? ny : 1) * (dims == 3 ? nz : 1);
    }
};

template <typename T>
std::vector<T> make_data(const Shape& shape, int pattern) {
    std::vector<T> data(shape.elements());
    for (std::size_t i = 0; i < data.size(); ++i) {
        if (pattern == 0) {
            data[i] = static_cast<T>(1000 + (i % 17));
        } else if (pattern == 1) {
            data[i] = static_cast<T>(i < data.size() / 2 ? 100 : 3599);
        } else if (pattern == 2) {
            data[i] = static_cast<T>(i < data.size() / 2 ? 100 : 3600);
        } else {
            data[i] = static_cast<T>((i * 7919u + (i / 13u) * 31u) &
                                     (sizeof(T) == sizeof(std::uint16_t) ? 0xffffu : 0xffffffffu));
        }
    }
    return data;
}

template <typename T>
mans::MansParams make_params(std::uint32_t backend, const Shape& shape) {
    mans::MansParams params{};
    params.backend = backend;
    params.dtype = sizeof(T) == sizeof(std::uint16_t) ? mans::DataType::U16 : mans::DataType::U32;
    params.mode = mans::Mode::P;
    params.dims = shape.dims;
    params.nx = shape.nx;
    params.ny = shape.ny;
    params.nz = shape.nz;
    return params;
}

std::size_t max_capacity(std::size_t elements,
                         const mans::MansParams& first,
                         const mans::MansParams& second) {
    return std::max(mans::get_mans_max_compress_bytes(elements, first),
                    mans::get_mans_max_compress_bytes(elements, second));
}

template <typename T>
void check_header(const std::vector<std::uint8_t>& stream,
                 std::size_t stream_size,
                 const mans::MansParams& params) {
    mans::MansHeader header{};
    std::size_t raw_bytes = 0;
    std::string error;
    if (!mans::parse_mans_header(stream.data(), stream_size, header, raw_bytes, &error)) {
        throw std::runtime_error("header parse failed: " + error);
    }
    if (header.codec != mans::Codec::ADM || header.mode != mans::Mode::P ||
        header.dims != params.dims || raw_bytes == 0) {
        throw std::runtime_error("stream header is not codec=1 P-mode");
    }
}

template <typename T>
void run_case(const Shape& shape, int pattern) {
    std::cerr << "case dtype=" << (sizeof(T) == 2 ? "u16" : "u32")
              << " dims=" << shape.dims << " shape=" << shape.nx << "x"
              << shape.ny << "x" << shape.nz << " pattern=" << pattern << "\n";
    const auto input = make_data<T>(shape, pattern);
    const auto cpu = make_params<T>(mans::Backend::CPU, shape);
    const auto dcu = make_params<T>(mans::Backend::DCU, shape);
    const std::size_t raw_bytes = input.size() * sizeof(T);
    const std::size_t capacity = max_capacity(input.size(), cpu, dcu);

    std::vector<std::uint8_t> cpu_stream(capacity);
    std::size_t cpu_stream_size = cpu_stream.size();
    mans::compress(input.data(), input.size(), cpu, cpu_stream.data(), cpu_stream_size);
    if (cpu_stream_size == 0) throw std::runtime_error("CPU compression returned an empty stream");
    check_header<T>(cpu_stream, cpu_stream_size, cpu);

    std::vector<std::uint8_t> cpu_self_output(raw_bytes);
    std::size_t cpu_self_output_size = cpu_self_output.size();
    mans::decompress(cpu_stream.data(), cpu_stream_size, cpu,
                     cpu_self_output.data(), cpu_self_output_size);
    if (cpu_self_output_size != raw_bytes ||
        std::memcmp(cpu_self_output.data(), input.data(), raw_bytes) != 0) {
        throw std::runtime_error("CPU self-decode mismatch before DCU decode");
    }

    std::vector<std::uint8_t> dcu_host_output(raw_bytes);
    std::size_t dcu_host_output_size = dcu_host_output.size();
    mans::decompress(cpu_stream.data(), cpu_stream_size, dcu,
                     dcu_host_output.data(), dcu_host_output_size);
    if (dcu_host_output_size != raw_bytes ||
        std::memcmp(dcu_host_output.data(), input.data(), raw_bytes) != 0) {
        const T* restored = reinterpret_cast<const T*>(dcu_host_output.data());
        std::size_t mismatch = 0;
        while (mismatch < input.size() && restored[mismatch] == input[mismatch]) ++mismatch;
        std::cerr << "mismatch index=" << mismatch << " expected="
                  << static_cast<unsigned long long>(input[mismatch]) << " actual="
                  << static_cast<unsigned long long>(restored[mismatch]) << "\\n";
        if (shape.dims == 2 && shape.nx == 17 && shape.ny == 31 && pattern == 1) {
            cpu_ans::ANSCoalescedHeader ans_header{};
            std::memcpy(&ans_header, cpu_stream.data() + mans::kMansHeaderBytes,
                        sizeof(ans_header));
            const std::size_t adm_size = ans_header.getTotalUncompressedWords();
            std::vector<std::uint8_t> adm(adm_size);
            std::size_t decoded = adm_size;
            double duration = 0.0;
            pans_decompress(cpu_stream.data() + mans::kMansHeaderBytes,
                            cpu_stream_size - mans::kMansHeaderBytes,
                            adm.data(), decoded, duration);
            const std::size_t blocks = 4;
            const std::size_t offsets_bytes = (blocks + 1) * 4;
            const std::size_t centers_bytes = blocks * sizeof(T);
            const std::size_t flags_bytes = 1;
            std::cerr << "adm_size=" << decoded << " offsets=";
            for (std::size_t i = 0; i <= blocks; ++i) {
                std::int32_t value = 0;
                std::memcpy(&value, adm.data() + i * 4, 4);
                std::cerr << value << (i == blocks ? "" : ",");
            }
            std::cerr << " centers=";
            for (std::size_t i = 0; i < blocks; ++i) {
                T center{};
                std::memcpy(&center, adm.data() + offsets_bytes + i * sizeof(T), sizeof(T));
                std::cerr << static_cast<unsigned long long>(center) << (i + 1 == blocks ? "" : ",");
            }
            const auto* flags = adm.data() + offsets_bytes + centers_bytes;
            std::cerr << " flags=0x" << std::hex << static_cast<unsigned int>(flags[0])
                      << std::dec << " code=" << static_cast<unsigned int>(adm[offsets_bytes + centers_bytes + flags_bytes + mismatch]) << "\\n";
        }
        throw std::runtime_error("CPU compress -> DCU host decompress mismatch");
    }

    std::uint8_t* d_cpu_stream = nullptr;
    std::uint8_t* d_device_output = nullptr;
    check_hip(hipMalloc(&d_cpu_stream, cpu_stream_size), "hipMalloc CPU stream");
    check_hip(hipMalloc(&d_device_output, raw_bytes), "hipMalloc DCU output");
    check_hip(hipMemcpy(d_cpu_stream, cpu_stream.data(), cpu_stream_size,
                       hipMemcpyHostToDevice), "CPU stream H2D");
    std::size_t device_output_size = raw_bytes;
    mans::decompress_device(d_cpu_stream, cpu_stream_size, dcu,
                            d_device_output, device_output_size);
    std::vector<T> device_restored(input.size());
    check_hip(hipMemcpy(device_restored.data(), d_device_output, raw_bytes,
                       hipMemcpyDeviceToHost), "DCU output D2H");
    if (device_output_size != raw_bytes ||
        std::memcmp(device_restored.data(), input.data(), raw_bytes) != 0) {
        throw std::runtime_error("CPU compress -> DCU device decompress mismatch");
    }

    std::vector<std::uint8_t> dcu_stream(capacity);
    std::size_t dcu_stream_size = dcu_stream.size();
    mans::compress(input.data(), input.size(), dcu, dcu_stream.data(), dcu_stream_size);
    if (dcu_stream_size == 0) throw std::runtime_error("DCU compression returned an empty stream");
    check_header<T>(dcu_stream, dcu_stream_size, dcu);

    std::vector<std::uint8_t> cpu_output(raw_bytes);
    std::size_t cpu_output_size = cpu_output.size();
    mans::decompress(dcu_stream.data(), dcu_stream_size, cpu,
                     cpu_output.data(), cpu_output_size);
    if (cpu_output_size != raw_bytes ||
        std::memcmp(cpu_output.data(), input.data(), raw_bytes) != 0) {
        throw std::runtime_error("DCU compress -> CPU decompress mismatch");
    }

    std::uint8_t* d_input = nullptr;
    std::uint8_t* d_dcu_stream = nullptr;
    check_hip(hipMalloc(&d_input, raw_bytes), "hipMalloc DCU input");
    check_hip(hipMalloc(&d_dcu_stream, capacity), "hipMalloc DCU stream");
    check_hip(hipMemcpy(d_input, input.data(), raw_bytes, hipMemcpyHostToDevice),
              "DCU input H2D");
    std::size_t dcu_device_stream_size = capacity;
    mans::compress_device(d_input, input.size(), dcu, d_dcu_stream,
                          dcu_device_stream_size);
    std::vector<std::uint8_t> dcu_device_stream(dcu_device_stream_size);
    check_hip(hipMemcpy(dcu_device_stream.data(), d_dcu_stream,
                       dcu_device_stream_size, hipMemcpyDeviceToHost),
              "DCU stream D2H");
    check_header<T>(dcu_device_stream, dcu_device_stream_size, dcu);
    std::fill(cpu_output.begin(), cpu_output.end(), 0);
    cpu_output_size = cpu_output.size();
    mans::decompress(dcu_device_stream.data(), dcu_device_stream_size, cpu,
                     cpu_output.data(), cpu_output_size);
    if (cpu_output_size != raw_bytes ||
        std::memcmp(cpu_output.data(), input.data(), raw_bytes) != 0) {
        throw std::runtime_error("DCU device compress -> CPU decompress mismatch");
    }

    hipFree(d_cpu_stream);
    hipFree(d_device_output);
    hipFree(d_input);
    hipFree(d_dcu_stream);
}

} // namespace

int main() {
    int device_count = 0;
    try {
        check_hip(hipGetDeviceCount(&device_count), "hipGetDeviceCount");
        if (device_count == 0) {
            std::cout << "No DCU device available; skipping CPU/DCU cross test.\n";
            return 0;
        }

        for (int pattern = 0; pattern < 4; ++pattern) {
            run_case<std::uint16_t>({1, 513, 1, 1}, pattern);
            run_case<std::uint16_t>({2, 17, 31, 1}, pattern);
            run_case<std::uint16_t>({3, 5, 17, 19}, pattern);
            run_case<std::uint32_t>({1, 513, 1, 1}, pattern);
            run_case<std::uint32_t>({2, 17, 31, 1}, pattern);
            run_case<std::uint32_t>({3, 5, 17, 19}, pattern);
        }
        for (std::uint32_t length : {1u, 511u, 512u, 4097u}) {
            run_case<std::uint16_t>({1, length, 1, 1}, 3);
            run_case<std::uint32_t>({1, length, 1, 1}, 3);
        }
        std::cout << "CPU/DCU P-mode cross-backend tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "CPU/DCU cross test failed: " << error.what() << "\n";
        return 1;
    }
}
