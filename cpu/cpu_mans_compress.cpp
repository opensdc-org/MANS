#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "mans_cpu.h"
#include "../mans_utils.h"
#include "adm/adm_utils.h"
#include "fse/include/fse.h"

namespace {

void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " <-u2|-u4> <input_file> <output_file> [--mode p|r] [--dims x [y z]]\n";
}

template <typename T>
bool run_compress(const std::string& input_file,
                  const std::string& output_file,
                  mans::MansParams params,
                  const std::vector<std::uint32_t>& cli_dims) {
    std::vector<T> input;
    if (!mans::load_typed_file<T>(input_file, input)) {
        std::cerr << "Failed to load input file: " << input_file << "\n";
        return false;
    }
    if (input.empty()) {
        std::cerr << "Input file is empty: " << input_file << "\n";
        return false;
    }

    std::vector<std::uint32_t> dims = cli_dims;
    if (dims.empty()) {
        if (input.size() > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
            std::cerr << "Input is too large for default 1D dims.\n";
            return false;
        }
        dims.push_back(static_cast<std::uint32_t>(input.size()));
    }

    std::size_t total_elements = 0;
    if (!mans::dims_product(dims, total_elements)) {
        std::cerr << "Invalid --dims values.\n";
        return false;
    }
    if (total_elements != input.size()) {
        std::cerr << "--dims element count mismatch: dims_elems=" << total_elements
                  << " file_elems=" << input.size() << "\n";
        return false;
    }

    params.dims = static_cast<std::uint32_t>(dims.size());
    params.nx = dims[0];
    params.ny = dims.size() >= 2 ? dims[1] : 0;
    params.nz = dims.size() >= 3 ? dims[2] : 0;

    const std::size_t adm_cap = adm_max_compressed_size<T>(input.size(), params);
    std::size_t stage2_cap = adm_cap * 2 + 4096;
    if (params.mode == mans::Mode::R) {
        const std::size_t fse_bound = FSE_compressBound(adm_cap);
        if (fse_bound > 0) {
            stage2_cap = fse_bound;
        }
    } else {
        stage2_cap = adm_cap + adm_cap / 2 + 4096;
    }
    if (stage2_cap < adm_cap) {
        std::cerr << "Compressed-size estimate overflow.\n";
        return false;
    }
    const std::size_t output_cap = mans::kMansHeaderBytes + stage2_cap;
    if (output_cap < stage2_cap) {
        std::cerr << "Compressed-size estimate overflow.\n";
        return false;
    }

    std::vector<std::uint8_t> output(output_cap);
    std::size_t output_size = output.size();
    mans::cpu::compress_internal(input.data(), input.size(), params,
                                 output.data(), output_size,
                                 false, std::string());
    if (output_size == 0) {
        std::cerr << "Compression failed.\n";
        return false;
    }

    output.resize(output_size);
    if (!mans::save_u8_file(output_file, output)) {
        std::cerr << "Failed to write output file: " << output_file << "\n";
        return false;
    }

    const double ratio = static_cast<double>(input.size() * sizeof(T)) /
                         static_cast<double>(output_size);
    std::cout << "Compressed " << input_file << " -> " << output_file
              << " (" << ratio << "x)\n";
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
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

    mans::MansParams params{};
    params.backend = mans::Backend::CPU;
    params.dtype = is_u2 ? mans::DataType::U16 : mans::DataType::U32;
    params.mode = mans::Mode::R;

    std::vector<std::uint32_t> dims;
    for (int i = 4; i < argc; ++i) {
        if (std::strcmp(argv[i], "--mode") == 0) {
            if (i + 1 >= argc || !mans::parse_mode(argv[i + 1], params.mode)) {
                std::cerr << "Invalid --mode value. Use p or r.\n";
                return 1;
            }
            ++i;
        } else if (std::strcmp(argv[i], "--dims") == 0) {
            int consumed = 0;
            while (i + 1 < argc && std::strncmp(argv[i + 1], "--", 2) != 0) {
                std::uint32_t dim = 0;
                if (!mans::parse_positive_u32(argv[i + 1], dim)) {
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

    if (is_u2) {
        return run_compress<std::uint16_t>(input_file, output_file, params, dims) ? 0 : 1;
    }
    return run_compress<std::uint32_t>(input_file, output_file, params, dims) ? 0 : 1;
}
