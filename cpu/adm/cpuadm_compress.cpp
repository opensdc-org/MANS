// compiler： g++ -std=c++17 -mavx512f -fopenmp -march=native -O3 compress.cpp -o compress
// exec: OMP_NUM_THREADS=4 ./compress u2 input.u2 output.bin
//       OMP_NUM_THREADS=4 ./compress u4 input.u2 output.bin

#include <iostream>
#include <vector>
#include <cstdint>
#include <string>

#include "adm_utils.h" 
#include "../file_utils.h"
#include "../../mans_defs.h"
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
    mans::MansParams params;
    params.dtype = is_u2 ? mans::DataType::U16 : mans::DataType::U32;
    params.backend = mans::Backend::CPU;
    params.adm_threshold = 4000; // default threshold
    // Default thread settings for ADM compression/decompression
    params.adm_center_calc_threads      = 32;
    params.adm_encode_threads           = 32;
    params.adm_warp_reduce_threads      = 32;
    params.adm_fill_tail_threads        = 16;
    params.adm_write_back_threads       = 16;
    params.adm_restore_signals_threads  = 32;
    params.adm_decode_values_threads    = 16;
    std::vector<std::uint8_t> output;
    std::size_t compressed_size = 0;

    if (is_u2) {
        std::vector<std::uint16_t> input_data;
        if (!load_u16_file(input_file, input_data)) {
            std::cerr << "Failed to load input file: " << input_file << "\n";
            return 1;
        }


        std::size_t max_buffer_size = input_data.size() * sizeof(std::uint16_t) * 3 / 2;
        // Add header size just in case the data is extremely small
        max_buffer_size += 1024; 
        
        output.resize(max_buffer_size);

        // Run the benchmark: write results into `output`, and store the actual size in `compressed_size`
        adm_compress_and_benchmark<std::uint16_t>(
            input_data.data(), 
            input_data.size(), 
            output.data(), 
            compressed_size,
            params
        );

    } else {
        std::vector<std::uint32_t> input_data;
        if (!load_u32_file(input_file, input_data)) {
            std::cerr << "Failed to load input file: " << input_file << "\n";
            return 1;
        }

        std::size_t max_buffer_size = input_data.size() * sizeof(std::uint32_t) * 2;
        max_buffer_size += 1024;

        output.resize(max_buffer_size);

        adm_compress_and_benchmark<std::uint32_t>(
            input_data.data(), 
            input_data.size(), 
            output.data(), 
            compressed_size,
            params
        );
    }
    output.resize(compressed_size);

    if (!save_u8_file(output_file, output)) {
        std::cerr << "Failed to write output file: " << output_file << "\n";
        return 1;
    }

    std::cout << "ADM finished! Write to " << output_file << " (Size: " << compressed_size << " bytes)\n";

    return 0;
}