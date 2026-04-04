#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>

#include "mapping_uint16.h"
#include "mapping_uint32.h"
#include "../../mans_defs.h"
#include "../../mans_utils.h"

namespace {

void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " <-u2|-u4> <input_file> <output_bin_file> --dims x [y z]\n";
}

void check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
bool run_compress(const std::string& input_file,
                  const std::string& output_file,
                  const std::vector<std::uint32_t>& dims) {
    std::vector<T> input_data;
    if (!mans::load_typed_file(input_file, input_data)) {
        std::cerr << "Failed to load input file: " << input_file << "\n";
        return false;
    }

    std::size_t total_elements = 0;
    if (!mans::dims_product(dims, total_elements)) {
        std::cerr << "Invalid --dims values.\n";
        return false;
    }
    if (total_elements != input_data.size()) {
        std::cerr << "--dims element count mismatch: dims_elems=" << total_elements
                  << " file_elems=" << input_data.size() << "\n";
        return false;
    }

    try {
        mans::MansParams params{};
        params.backend = mans::Backend::NVIDIA;
        params.dtype = std::is_same_v<T, std::uint16_t> ? mans::DataType::U16 : mans::DataType::U32;
        params.dims = static_cast<std::uint32_t>(dims.size());
        params.nx = dims.size() >= 1 ? dims[0] : 0;
        params.ny = dims.size() >= 2 ? dims[1] : 0;
        params.nz = dims.size() >= 3 ? dims[2] : 0;

        T* d_input = nullptr;
        std::uint8_t* d_output = nullptr;
        const std::size_t input_bytes = input_data.size() * sizeof(T);
        const std::size_t max_output_size = std::is_same_v<T, std::uint16_t>
            ? mans::nv::adm::get_max_u16_payload_bytes(input_data.size(), params)
            : mans::nv::adm::get_max_u32_payload_bytes(input_data.size(), params);
        std::size_t output_size = 0;

        if (input_bytes != 0) {
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_input), input_bytes), "cudaMalloc d_input");
            check_cuda(cudaMemcpy(d_input, input_data.data(), input_bytes, cudaMemcpyHostToDevice),
                       "cudaMemcpy H2D input");
        }
        if (max_output_size != 0) {
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_output), max_output_size), "cudaMalloc d_output");
        }

        if constexpr (std::is_same_v<T, std::uint16_t>) {
            mans::nv::adm::compress_u16_device(d_input, input_data.size(), params, d_output, output_size);
        } else {
            mans::nv::adm::compress_u32_device(d_input, input_data.size(), params, d_output, output_size);
        }

        std::vector<std::uint8_t> output(output_size);
        if (output_size != 0) {
            check_cuda(cudaMemcpy(output.data(), d_output, output_size, cudaMemcpyDeviceToHost),
                       "cudaMemcpy D2H output");
        }

        cudaFree(d_input);
        cudaFree(d_output);

        if (!mans::save_u8_file(output_file, output)) {
            std::cerr << "Failed to write output file: " << output_file << "\n";
            return false;
        }

        std::cout << "NV ADM finished! Write to " << output_file
                  << " (Size: " << output.size() << " bytes)\n";
        return true;
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return false;
    }
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

    if (dims.empty()) {
        std::cerr << "--dims is required.\n";
        return 1;
    }

    if (is_u2) {
        return run_compress<std::uint16_t>(input_file, output_file, dims) ? 0 : 1;
    }
    return run_compress<std::uint32_t>(input_file, output_file, dims) ? 0 : 1;
}
