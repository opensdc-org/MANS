#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "../mans_api.hpp"
#include "../mans_utils.h"

namespace {

void usage(const char* program) {
    std::cerr << "Usage: " << program
              << " <-u2|-u4> <input_file> <output_file> [--mode p] [--dims x [y z]]\n";
}

template <typename T>
bool run(const std::string& input_file, const std::string& output_file,
         mans::MansParams params, const std::vector<std::uint32_t>& requested_dims) {
    std::vector<T> input;
    if (!mans::load_typed_file<T>(input_file, input) || input.empty()) {
        std::cerr << "Failed to load a non-empty input file: " << input_file << "\n";
        return false;
    }

    std::vector<std::uint32_t> dims = requested_dims;
    if (dims.empty()) {
        if (input.size() > std::numeric_limits<std::uint32_t>::max()) {
            std::cerr << "Input is too large for 1D geometry.\n";
            return false;
        }
        dims.push_back(static_cast<std::uint32_t>(input.size()));
    }
    std::size_t dim_elements = 0;
    if (!mans::dims_product(dims, dim_elements) || dim_elements != input.size()) {
        std::cerr << "--dims element count does not match the input file.\n";
        return false;
    }

    params.dims = static_cast<std::uint32_t>(dims.size());
    params.nx = dims[0];
    params.ny = dims.size() > 1 ? dims[1] : 0;
    params.nz = dims.size() > 2 ? dims[2] : 0;

    const std::size_t capacity = mans::get_mans_max_compress_bytes(input.size(), params);
    std::vector<std::uint8_t> compressed(capacity);
    std::size_t compressed_size = compressed.size();
    mans::compress(input.data(), input.size(), params, compressed.data(), compressed_size);
    if (compressed_size == 0 || compressed_size > compressed.size()) {
        std::cerr << "AMD compression failed.\n";
        return false;
    }
    compressed.resize(compressed_size);
    if (!mans::save_u8_file(output_file, compressed)) {
        std::cerr << "Failed to write output file: " << output_file << "\n";
        return false;
    }
    std::cout << "Compressed " << input_file << " -> " << output_file
              << " (" << static_cast<double>(input.size() * sizeof(T)) / compressed_size
              << "x)\n";
    return true;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        usage(argv[0]);
        return 1;
    }
    const std::string dtype = argv[1];
    const bool u16 = dtype == "-u2" || dtype == "u2";
    const bool u32 = dtype == "-u4" || dtype == "u4";
    if (!u16 && !u32) {
        usage(argv[0]);
        return 1;
    }

    mans::MansParams params{};
#ifdef MANS_ENABLE_DCU
    params.backend = mans::Backend::DCU;
#else
    params.backend = mans::Backend::AMD;
#endif
    params.dtype = u16 ? mans::DataType::U16 : mans::DataType::U32;
    params.mode = mans::Mode::P;
    std::vector<std::uint32_t> dims;
    for (int i = 4; i < argc; ++i) {
        if (std::strcmp(argv[i], "--mode") == 0) {
            if (++i >= argc || !mans::parse_mode(argv[i], params.mode) || params.mode != mans::Mode::P) {
                std::cerr << "AMD backend supports only --mode p.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "--dims") == 0) {
            int count = 0;
            while (i + 1 < argc && std::strncmp(argv[i + 1], "--", 2) != 0) {
                std::uint32_t dim = 0;
                if (!mans::parse_positive_u32(argv[++i], dim) || dims.size() == 3) {
                    std::cerr << "Use --dims x [y z].\n";
                    return 1;
                }
                dims.push_back(dim);
                ++count;
            }
            if (count == 0) {
                std::cerr << "Use --dims x [y z].\n";
                return 1;
            }
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    return u16 ? (run<std::uint16_t>(argv[2], argv[3], params, dims) ? 0 : 1)
               : (run<std::uint32_t>(argv[2], argv[3], params, dims) ? 0 : 1);
}
