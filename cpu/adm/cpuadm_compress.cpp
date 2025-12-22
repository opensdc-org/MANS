// compiler： g++ -std=c++17 -mavx512f -fopenmp -march=native -O3 compress.cpp -o compress
// exec: OMP_NUM_THREADS=4 ./compress u2 input.u2 output.bin
//       OMP_NUM_THREADS=4 ./compress u4 input.u2 output.bin

#include <iostream>
#include <vector>
#include <cstdint>
#include <string>

#include "adm_utils.h" 
#include "../file_utils.h"

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Use: " << argv[0]
                  << " <-u2|-u4> <input_file> <output_bin_file>\n";
        return 1;
    }

    std::string input_type  = argv[1];
    std::string input_file  = argv[2];
    std::string output_file = argv[3];

    bool is_u2 = (input_type == "-u2" || input_type == "u2");
    bool is_u4 = (input_type == "-u4" || input_type == "u4");

    if (!is_u2 && !is_u4) {
        std::cerr << "Unknown data type flag: " << input_type
                  << "\nUse: -u2 or -u4\n";
        return 1;
    }

    std::vector<std::uint8_t> output;

    if (is_u2) {
        std::vector<std::uint16_t> input_data;
        if (!load_u16_file(input_file, input_data)) {
            std::cerr << "Failed to load input file: " << input_file << "\n";
            return 1;
        }
        adm_compress_and_benchmark(input_data, output);
    } else {
        std::vector<std::uint32_t> input_data;
        if (!load_u32_file(input_file, input_data)) {
            std::cerr << "Failed to load input file: " << input_file << "\n";
            return 1;
        }
        adm_compress_and_benchmark(input_data, output);
    }

    if (!save_u8_file(output_file, output)) {
        std::cerr << "Failed to write output file: " << output_file << "\n";
        return 1;
    }

    std::cout << "ADM finished! Write to " << output_file << "\n";

    return 0;
}