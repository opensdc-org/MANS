// compiler: g++ -std=c++17 -mavx512f -fopenmp -march=native -O3 compress.cpp -o compress
// exec: OMP_NUM_THREADS=4 ./compress -u2 input.u2 output.bin --dims 32768
//       OMP_NUM_THREADS=4 ./compress -u4 input.u4 output.bin --dims 256 256

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "adm_utils.h"
#include "../../mans_defs.h"
#include "../../mans_utils.h"

namespace {

void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " <-u2|-u4> <input_file> <output_bin_file> --dims x [y z]\n";
}

bool parse_positive_u32(const char* text, std::uint32_t& out) {
    if (!text || *text == '\0') {
        return false;
    }
    char* end = nullptr;
    errno = 0;
    unsigned long long value = std::strtoull(text, &end, 10);
    if (errno != 0 || end == nullptr || *end != '\0' || value == 0 ||
        value > static_cast<unsigned long long>(std::numeric_limits<std::uint32_t>::max())) {
        return false;
    }
    out = static_cast<std::uint32_t>(value);
    return true;
}

bool dims_product(const std::vector<std::uint32_t>& dims, std::size_t& out) {
    out = 1;
    for (std::uint32_t dim : dims) {
        if (dim == 0) {
            return false;
        }
        if (out > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(dim)) {
            return false;
        }
        out *= static_cast<std::size_t>(dim);
    }
    return true;
}

template <typename T>
bool run_compress(const std::string& input_file,
                  const std::string& output_file,
                  const std::vector<std::uint32_t>& dims,
                  mans::MansParams params) {
    std::vector<T> input_data;
    if constexpr (std::is_same_v<T, std::uint16_t>) {
        if (!mans::load_u16_file(input_file, input_data)) {
            std::cerr << "Failed to load input file: " << input_file << "\n";
            return false;
        }
    } else {
        if (!mans::load_u32_file(input_file, input_data)) {
            std::cerr << "Failed to load input file: " << input_file << "\n";
            return false;
        }
    }

    std::size_t total_elements = 0;
    if (!dims_product(dims, total_elements)) {
        std::cerr << "Invalid --dims values.\n";
        return false;
    }
    if (total_elements != input_data.size()) {
        std::cerr << "--dims element count mismatch: dims_elems=" << total_elements
                  << " file_elems=" << input_data.size() << "\n";
        return false;
    }

    params.dims = static_cast<std::uint32_t>(dims.size());
    params.nx = dims[0];
    params.ny = dims.size() >= 2 ? dims[1] : 0;
    params.nz = dims.size() >= 3 ? dims[2] : 0;

    std::size_t max_buffer_size = adm_max_compressed_size<T>(input_data.size(), params);
    max_buffer_size += 1024;

    std::vector<std::uint8_t> output(max_buffer_size);
    std::size_t compressed_size = 0;
    adm_compress_and_benchmark<T>(
        input_data.data(),
        input_data.size(),
        output.data(),
        compressed_size,
        params
    );

    output.resize(compressed_size);
    if (!mans::save_u8_file(output_file, output)) {
        std::cerr << "Failed to write output file: " << output_file << "\n";
        return false;
    }

    std::cout << "ADM finished! Write to " << output_file << " (Size: " << compressed_size << " bytes)\n";
    return true;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 6) {
        print_usage(argv[0]);
        return 1;
    }

    const std::string dtype_flag = argv[1];
    const bool is_u2 = (dtype_flag == "-u2" || dtype_flag == "u2");
    const bool is_u4 = (dtype_flag == "-u4" || dtype_flag == "u4");
    if (!is_u2 && !is_u4) {
        print_usage(argv[0]);
        return 1;
    }

    const std::string input_file = argv[2];
    const std::string output_file = argv[3];

    std::vector<std::uint32_t> dims;
    for (int i = 4; i < argc; ++i) {
        if (std::strcmp(argv[i], "--dims") == 0) {
            int consumed = 0;
            while (i + 1 < argc && std::strncmp(argv[i + 1], "--", 2) != 0) {
                std::uint32_t dim = 0;
                if (!parse_positive_u32(argv[i + 1], dim)) {
                    std::cerr << "Invalid dim value: " << argv[i + 1] << "\n";
                    return 1;
                }
                dims.push_back(dim);
                ++i;
                ++consumed;
            }
            if (consumed == 0 || dims.size() > 3) {
                std::cerr << "Use --dims x [y z].\n";
                return 1;
            }
        } else {
            std::cerr << "Unknown argument: " << argv[i] << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    if (dims.empty()) {
        std::cerr << "--dims is required.\n";
        return 1;
    }

    mans::MansParams params{};
    params.dtype = is_u2 ? mans::DataType::U16 : mans::DataType::U32;

    if (is_u2) {
        return run_compress<std::uint16_t>(input_file, output_file, dims, params) ? 0 : 1;
    }
    return run_compress<std::uint32_t>(input_file, output_file, dims, params) ? 0 : 1;
}
