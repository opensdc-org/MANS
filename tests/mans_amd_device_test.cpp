#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../mans_api.hpp"

namespace {
void check(hipError_t status, const char* what) {
    if (status != hipSuccess) {
        throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(status));
    }
}
}

int main() {
    int device_count = 0;
    check(hipGetDeviceCount(&device_count), "hipGetDeviceCount");
    if (device_count == 0) {
        std::cout << "No AMD device available; skipping AMD device API test.\n";
        return 0;
    }

    try {
        constexpr std::size_t elements = 513;
        std::vector<std::uint16_t> input(elements);
        for (std::size_t i = 0; i < elements; ++i) {
            input[i] = static_cast<std::uint16_t>((i * 19) & 0x0fffU);
        }
        mans::MansParams params{};
#ifdef MANS_ENABLE_DCU
        params.backend = mans::Backend::DCU;
#else
        params.backend = mans::Backend::AMD;
#endif
        params.dtype = mans::DataType::U16;
        params.mode = mans::Mode::P;
        params.dims = 1;
        params.nx = static_cast<std::uint32_t>(elements);

        std::vector<std::uint8_t> host_compressed(
            mans::get_mans_max_compress_bytes(elements, params));
        std::size_t host_compressed_size = host_compressed.size();
        std::uint8_t* d_input = nullptr;
        std::uint8_t* d_compressed = nullptr;
        std::uint8_t* d_output = nullptr;
        check(hipMalloc(&d_input, input.size() * sizeof(input[0])), "hipMalloc input");
        check(hipMalloc(&d_compressed, host_compressed.size()), "hipMalloc compressed");
        check(hipMalloc(&d_output, input.size() * sizeof(input[0])), "hipMalloc output");
        check(hipMemcpy(d_input, input.data(), input.size() * sizeof(input[0]),
                        hipMemcpyHostToDevice), "input H2D");

        mans::compress_device(d_input, elements, params, d_compressed, host_compressed_size);
        check(hipMemcpy(host_compressed.data(), d_compressed, host_compressed_size,
                        hipMemcpyDeviceToHost), "compressed D2H");

        std::size_t output_size = input.size() * sizeof(input[0]);
        mans::decompress_device(d_compressed, host_compressed_size, params, d_output, output_size);
        std::vector<std::uint16_t> restored(elements);
        check(hipMemcpy(restored.data(), d_output, output_size, hipMemcpyDeviceToHost),
              "output D2H");
        if (output_size != input.size() * sizeof(input[0]) ||
            std::memcmp(input.data(), restored.data(), output_size) != 0) {
            throw std::runtime_error("AMD device round-trip mismatch");
        }

        hipFree(d_input);
        hipFree(d_compressed);
        hipFree(d_output);
        std::cout << "AMD backend device round-trip test passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "AMD device test failed: " << error.what() << "\n";
        return 1;
    }
}
