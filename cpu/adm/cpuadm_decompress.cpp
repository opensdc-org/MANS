// compiler: g++ -std=c++17 -mavx512f -fopenmp -march=native -O3 decompress.cpp -o decompress
// exec   : OMP_NUM_THREADS=4 ./decompress u2 input.bin output.u2
//          OMP_NUM_THREADS=4 ./decompress u4 input.bin output.u4

// cpuadm_decompress.cpp
#include <iostream>
#include <vector>
#include <cstdint>
#include <string>
#include <cstring> 
#include "adm.h" 
#include "adm_utils.h" 
#include "../file_utils.h"

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Use: " << argv[0]
                  << " <-u2|-u4> <input_bin_file> <output_u2/u4_file>\n";
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

    std::vector<std::uint8_t> merged;
    if (!load_u8_file(input_file, merged)) {
        std::cerr << "Failed to load input file: " << input_file << "\n";
        return 1;
    }

    if (merged.size() < sizeof(adm::FileHeader)) {
        std::cerr << "Error: File too small or invalid format.\n";
        return 1;
    }
    mans::MansParams params{};
    adm::FileHeader header;
    std::memcpy(&header, merged.data(), sizeof(header));
    std::size_t num_elements = static_cast<std::size_t>(header.num_elements);
    try {
        if (is_u2) {
            std::vector<std::uint16_t> recovered(num_elements);
            
            
            adm_decompress_and_benchmark<std::uint16_t>(
                merged.data(), 
                merged.size(), 
                recovered.data(),
                num_elements,
                params
            );
            
            if (!save_u16_file(output_file, recovered)) {
                std::cerr << "Failed to write output file: " << output_file << "\n";
                return 1;
            }
        } else {
            std::vector<std::uint32_t> recovered(num_elements);
            adm_decompress_and_benchmark<std::uint32_t>(
                merged.data(), 
                merged.size(), 
                recovered.data(),
                num_elements,
                params
            );
            
            if (!save_u32_file(output_file, recovered)) {
                std::cerr << "Failed to write output file: " << output_file << "\n";
                return 1;
            }
        }

        std::cout << "Decompress finished! Write to " << output_file << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
