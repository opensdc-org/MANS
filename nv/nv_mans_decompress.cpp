#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "../mans_api.hpp"
#include "../mans_utils.h"

namespace {

void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " <-u2|-u4> <input_file> <output_file>\n";
}

bool parse_raw_bytes(const std::vector<std::uint8_t>& input, std::size_t& raw_bytes) {
    std::string error;
    if (!mans::parse_mans_raw_bytes(input.data(), input.size(), raw_bytes, &error)) {
        std::cerr << error << ".\n";
        return false;
    }
    return true;
}

template <typename T>
bool run_decompress(const std::string& input_file,
                    const std::string& output_file,
                    mans::MansParams params) {
    std::vector<std::uint8_t> input;
    if (!mans::load_u8_file(input_file, input)) {
        std::cerr << "Failed to load input file: " << input_file << "\n";
        return false;
    }

    std::size_t raw_bytes = 0;
    if (!parse_raw_bytes(input, raw_bytes)) {
        return false;
    }
    if ((raw_bytes % sizeof(T)) != 0) {
        std::cerr << "Raw byte size is incompatible with the requested dtype.\n";
        return false;
    }

    std::vector<std::uint8_t> output(raw_bytes);
    std::size_t output_size = output.size();
    mans::decompress(input.data(), input.size(), params, output.data(), output_size);
    if (output_size == 0) {
        std::cerr << "Decompression failed.\n";
        return false;
    }
    if (output_size != raw_bytes) {
        std::cerr << "Unexpected decompressed size: " << output_size
                  << " expected " << raw_bytes << "\n";
        return false;
    }
    if (!mans::save_typed_bytes_file<T>(output_file, output)) {
        std::cerr << "Failed to write output file: " << output_file << "\n";
        return false;
    }

    std::cout << "Decompressed " << input_file << " -> " << output_file
              << " (" << output_size << " bytes)\n";
    return true;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
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

    mans::MansParams params{};
    params.backend = mans::Backend::NVIDIA;
    params.dtype = is_u2 ? mans::DataType::U16 : mans::DataType::U32;

    const std::string input_file = argv[2];
    const std::string output_file = argv[3];

    if (is_u2) {
        return run_decompress<std::uint16_t>(input_file, output_file, params) ? 0 : 1;
    }
    return run_decompress<std::uint32_t>(input_file, output_file, params) ? 0 : 1;
}
