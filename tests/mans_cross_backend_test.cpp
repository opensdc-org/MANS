#include "../mans_api.hpp"
#include "../mans_utils.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Shape {
    std::uint32_t dims;
    std::uint32_t nx;
    std::uint32_t ny;
    std::uint32_t nz;
};

template <typename T>
void check_equal(const std::vector<T>& expected,
                 const std::vector<std::uint8_t>& actual,
                 const std::string& name) {
    if (actual.size() != expected.size() * sizeof(T) ||
        std::memcmp(actual.data(), expected.data(), actual.size()) != 0) {
        throw std::runtime_error(name + ": decompressed data mismatch");
    }
}

template <typename T>
std::vector<T> make_data(const Shape& shape, int variant) {
    const std::size_t count = static_cast<std::size_t>(shape.nx) *
                              (shape.dims >= 2 ? shape.ny : 1) *
                              (shape.dims == 3 ? shape.nz : 1);
    std::vector<T> data(count);
    std::mt19937 rng(0x4d414e53u + static_cast<unsigned>(variant));
    for (std::size_t i = 0; i < count; ++i) {
        if (variant == 0) {
            data[i] = static_cast<T>(17);
        } else if (variant == 1) {
            data[i] = static_cast<T>((i * 13u + 7u) % 3000u);
        } else if (variant == 2) {
            data[i] = static_cast<T>((i & 1u) ? 0 : (sizeof(T) == 2 ? 65535u : 0xffffffffu));
        } else {
            data[i] = static_cast<T>(rng());
        }
    }
    return data;
}

template <typename T>
void run_case(const Shape& shape, int variant) {
    const auto input = make_data<T>(shape, variant);
    mans::MansParams cpu{};
    cpu.backend = mans::Backend::CPU;
    cpu.dtype = sizeof(T) == 2 ? mans::DataType::U16 : mans::DataType::U32;
    cpu.mode = mans::Mode::P;
    cpu.dims = shape.dims;
    cpu.nx = shape.nx;
    cpu.ny = shape.ny;
    cpu.nz = shape.nz;

    mans::MansParams nv = cpu;
    nv.backend = mans::Backend::NVIDIA;

    const std::size_t cpu_cap = mans::get_mans_max_compress_bytes(input.size(), cpu);
    const std::size_t nv_cap = mans::get_mans_max_compress_bytes(input.size(), nv);
    std::vector<std::uint8_t> cpu_stream(cpu_cap);
    std::vector<std::uint8_t> nv_stream(nv_cap);
    std::size_t cpu_size = cpu_stream.size();
    std::size_t nv_size = nv_stream.size();
    mans::compress(input.data(), input.size(), cpu, cpu_stream.data(), cpu_size);
    mans::compress(input.data(), input.size(), nv, nv_stream.data(), nv_size);
    if (cpu_size == 0 || nv_size == 0) {
        throw std::runtime_error("compression failed");
    }
    cpu_stream.resize(cpu_size);
    nv_stream.resize(nv_size);

    std::size_t raw_bytes = 0;
    mans::MansHeader header{};
    std::string error;
    if (!mans::parse_mans_header(cpu_stream.data(), cpu_stream.size(), header, raw_bytes, &error) ||
        header.mode != mans::Mode::P || header.codec != 1 || raw_bytes != input.size() * sizeof(T)) {
        throw std::runtime_error("invalid CPU stream header: " + error);
    }
    if (!mans::parse_mans_header(nv_stream.data(), nv_stream.size(), header, raw_bytes, &error) ||
        header.mode != mans::Mode::P || header.codec != 1 || raw_bytes != input.size() * sizeof(T)) {
        throw std::runtime_error("invalid CUDA stream header: " + error);
    }

    std::vector<std::uint8_t> recovered(input.size() * sizeof(T));
    std::size_t recovered_size = recovered.size();
    mans::MansParams nv_self_decode = nv;
    mans::decompress(nv_stream.data(), nv_stream.size(), nv_self_decode,
                     recovered.data(), recovered_size);
    recovered.assign(input.size() * sizeof(T), 0);
    recovered_size = recovered.size();
    mans::MansParams cpu_decode = cpu;
    mans::decompress(nv_stream.data(), nv_stream.size(), cpu_decode,
                     recovered.data(), recovered_size);
    if (recovered_size != recovered.size()) {
        throw std::runtime_error("CPU decode size mismatch");
    }
    check_equal(input, recovered, "CUDA -> CPU");

    recovered.assign(input.size() * sizeof(T), 0);
    recovered_size = recovered.size();
    mans::MansParams nv_decode = nv;
    mans::decompress(cpu_stream.data(), cpu_stream.size(), nv_decode,
                     recovered.data(), recovered_size);
    if (recovered_size != recovered.size()) {
        throw std::runtime_error("CUDA decode size mismatch");
    }
    check_equal(input, recovered, "CPU -> CUDA");
}

void run_all() {
    const std::vector<Shape> shapes = {
        {1, 1, 0, 0}, {1, 511, 0, 0}, {1, 512, 0, 0}, {1, 513, 0, 0},
        {1, 4097, 0, 0}, {2, 17, 17, 0}, {2, 31, 17, 0},
        {3, 17, 17, 17}, {3, 31, 16, 33}};
    for (const Shape& shape : shapes) {
        for (int variant = 0; variant < 4; ++variant) {
            run_case<std::uint16_t>(shape, variant);
            run_case<std::uint32_t>(shape, variant);
        }
    }
}

} // namespace

int main() {
    try {
        int device_count = 0;
        const cudaError_t status = cudaGetDeviceCount(&device_count);
        if (status != cudaSuccess || device_count == 0) {
            std::cout << "CUDA device unavailable; cross-backend test skipped.\n";
            return 0;
        }
        run_all();
        std::cout << "cross-backend P-mode tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "cross-backend test failed: " << error.what() << "\n";
        return 1;
    }
}
