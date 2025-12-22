// compiler： g++ -std=c++17 -O3 cpu_mans_decompress.cpp -o cpu_mans_decompress
// exec   :  OMP_NUM_THREADS=4 ./cpu_mans_decompress u2 input.bin output.u2
//           OMP_NUM_THREADS=4 ./cpu_mans_decompress u4 input.bin output.u4

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <cstdlib>  // remove

#include "file_utils.h"
#include "adm/adm_utils.h"
#include "pans/pans_utils.h"

struct MansHeader {
    std::uint8_t codec;  // 1 = ADM, 2 = ANS
};
static_assert(sizeof(MansHeader) == 1, "MansHeader must be 1 byte");


inline bool strip_header(
    const std::vector<std::uint8_t>& all,
    std::vector<std::uint8_t>& payload,
    std::uint8_t& codec)
{
    if (all.size() < sizeof(MansHeader)) {
        std::cerr << "File too small, invalid mans format.\n";
        return false;
    }
    codec = all[0];
    payload.assign(all.begin() + 1, all.end());
    return true;
}


int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Use: " << argv[0]
                  << " <u2|u4> <input_bin_file> <output_u2/u4_file> <save_adm>\n";
        return 1;
    }

    std::string dtype       = argv[1];  // "u2" / "u4"
    std::string input_file  = argv[2];
    std::string output_file = argv[3];
    std::string save_adm_flag = argv[4];
    bool save_adm = (save_adm_flag == "1");
    bool is_u2 = (dtype == "-u2" || dtype == "u2");
    bool is_u4 = (dtype == "-u4" || dtype == "u4");

    if (!is_u2 && !is_u4) {
        std::cerr << "Unknown data type flag: " << dtype
                  << "\nUse: u2 or u4 (or -u2/-u4)\n";
        return 1;
    }
    std::uint8_t codec;
    std::string tmp_pans_out;
    
    std::vector<std::uint8_t> input_data_with_header;
    std::vector<std::uint8_t> input_data;
    std::vector<std::uint8_t> pans_data;
    if(!load_u8_file(input_file, input_data_with_header)) {
        std::cerr << "Failed to load input file: " << input_file << "\n";
        return 1;
    }

    if (!strip_header(input_data_with_header, input_data, codec)) {
        return 1;
    }
    
    if (codec == 1) {
        tmp_pans_out = output_file + ".adm";
    } else if (codec == 2) {
        tmp_pans_out = output_file;
    } else {
        std::cerr << "Unknown codec type in mans header: " << int(codec) << "\n";
        return 1;
    }
    // PANS decompress
    pans_decompress_and_benchmark(
        input_data,
        pans_data
    );
    if(codec==2){ // If ADM is not used, directly output the result
        if(!save_u8_file(output_file, pans_data)) {
            std::cerr << "Failed to write PANS output file: " << output_file << "\n";
            return 1;
        }
    }
    else{
        if(save_adm){
            if (!save_u8_file(tmp_pans_out, pans_data)) {
                std::cerr << "Failed to write ADM tmp file: " << tmp_pans_out << "\n";
                return 1;
            }
        }
        // ADM decompress
        if (codec == 1) {
            if (is_u2) {
                std::vector<std::uint16_t> recovered;
                adm_decompress_and_benchmark(pans_data, recovered);
                if (!save_u16_file(output_file, recovered)) {
                    std::cerr << "Failed to write output file: " << output_file << "\n";
                    return 1;
                }
            } else {
                std::vector<std::uint32_t> recovered;
                adm_decompress_and_benchmark(pans_data, recovered);
                if (!save_u32_file(output_file, recovered)) {
                    std::cerr << "Failed to write output file: " << output_file << "\n";
                    return 1;
                }
            }
        }
    }

    

    std::cout << "mans decompress finished! Output: " << output_file << "\n";
    return 0;
}
