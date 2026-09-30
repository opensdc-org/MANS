#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "../mans_api.hpp"
#include "../mans_utils.h"

namespace {

template <typename T>
void run_case(std::uint32_t dims, std::uint32_t nx, std::uint32_t ny,
              std::uint32_t nz, std::size_t elements) {
    std::vector<T> input(elements);
    for (std::size_t i = 0; i < elements; ++i) {
        input[i] = static_cast<T>((i * 17 + (i / 7) * 3) & 0x0fffU);
    }
    mans::MansParams params{};
#ifdef MANS_ENABLE_DCU
    params.backend = mans::Backend::DCU;
#else
    params.backend = mans::Backend::AMD;
#endif
    params.dtype = sizeof(T) == sizeof(std::uint16_t) ? mans::DataType::U16 : mans::DataType::U32;
    params.mode = mans::Mode::P;
    params.dims = dims;
    params.nx = nx;
    params.ny = ny;
    params.nz = nz;

    const std::size_t capacity = mans::get_mans_max_compress_bytes(elements, params);
    std::vector<std::uint8_t> compressed(capacity);
    std::size_t compressed_size = compressed.size();
    mans::compress(input.data(), input.size(), params, compressed.data(), compressed_size);
    if (compressed_size == 0 || compressed_size > capacity) {
        throw std::runtime_error("AMD compression returned an invalid size");
    }

    const std::size_t raw_bytes = mans::get_mans_exact_decompress_bytes(
        compressed.data(), compressed_size, params);
    std::vector<std::uint8_t> restored(raw_bytes);
    std::size_t restored_size = restored.size();
    mans::decompress(compressed.data(), compressed_size, params,
                     restored.data(), restored_size);
    if (restored_size != input.size() * sizeof(T) ||
        std::memcmp(restored.data(), input.data(), restored_size) != 0) {
        throw std::runtime_error("AMD host round-trip mismatch");
    }
}

} // namespace

int main() {
    try {
        run_case<std::uint16_t>(1, 513, 0, 0, 513);
        run_case<std::uint16_t>(2, 17, 31, 0, 17 * 31);
        run_case<std::uint32_t>(3, 5, 7, 9, 5 * 7 * 9);
        std::cout << "AMD backend host round-trip tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "AMD backend test failed: " << error.what() << "\n";
        return 1;
    }
}
