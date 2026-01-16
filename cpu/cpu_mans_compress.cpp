// compiler: g++ -std=c++17 -O3 cpu_mans_compress.cpp mans_cpu.cpp -o cpu_mans_compress -fopenmp
// exec    : OMP_NUM_THREADS=4 ./cpu_mans_compress u2 input.u2 output.bin 1
//           OMP_NUM_THREADS=4 ./cpu_mans_compress u4 input.u4 output.bin 0

#include <iostream>
#include <string>
#include <vector>

#include "../mans_defs.h" 
#include "mans_cpu.h"
#include "file_utils.h"

int main(int argc, char** argv) {

    if (argc < 5) {
        std::cerr << "Use: " << argv[0] 
                  << " <u2|u4> <input_file> <output_bin_file> <save_adm(0|1)> [threshold=4000]\n";
        return 1;
    }

    std::string dtype_str   = argv[1];
    std::string input_file  = argv[2];
    std::string output_file = argv[3];
    std::string save_flag   = argv[4];
    
    // 1. save intermediate ADM compressed data or not
    bool save_adm = (save_flag == "1");
    uint32_t threshold = 4000; 
    if (argc >= 6) {
        threshold = std::stoul(argv[5]);
    }

    // 2. build MansParams
    mans::MansParams params{};
    params.backend = mans::Backend::CPU;
    params.adm_threshold = threshold;

    if (dtype_str == "u2" || dtype_str == "-u2") {
        params.dtype = mans::DataType::U16;
    } else if (dtype_str == "u4" || dtype_str == "-u4") {
        params.dtype = mans::DataType::U32;
    } else {
        std::cerr << "Unknown data type flag: " << dtype_str << "\nUse: u2 or u4\n";
        return 1;
    }

    // 3. load data
    std::vector<uint8_t> compressed_data;
    
    if (params.dtype == mans::DataType::U16) {
        std::vector<uint16_t> host_data;
        if (!load_u16_file(input_file, host_data)) {
            std::cerr << "Failed to load input file: " << input_file << "\n";
            return 1;
        }
        if (host_data.empty()) {
            std::cerr << "Input file is empty.\n";
            return 1;
        }

        std::cout << "Compressing U16 (size=" << host_data.size() << ")...\n";

        // === 预分配输出缓冲区 ===
        size_t input_bytes = host_data.size() * sizeof(uint16_t);
        size_t max_out_size = input_bytes*2 + 4096; 
        compressed_data.resize(max_out_size);
        
        // 此变量传入时表示 Buffer 容量，返回时表示实际写入大小
        size_t final_size = max_out_size;

        // core compress set debug parameters:save_adm, dump_path, open_benchmark=true
        mans::cpu::compress_internal(
            host_data.data(), 
            host_data.size(), 
            params, 
            compressed_data.data(),
            final_size,            
            save_adm, 
            output_file + ".adm",   // debug path
            true                    // open_benchmark
        );

    
        if (final_size > max_out_size) {
            
             std::cerr << "Warning: Output truncated or buffer overflow logic triggered.\n";
        }
        compressed_data.resize(final_size);

    } else { // U32
        std::vector<uint32_t> host_data;
        if (!load_u32_file(input_file, host_data)) {
            std::cerr << "Failed to load input file: " << input_file << "\n";
            return 1;
        }
        if (host_data.empty()) {
            std::cerr << "Input file is empty.\n";
            return 1;
        }

        std::cout << "Compressing U32 (size=" << host_data.size() << ")...\n";

        
        size_t input_bytes = host_data.size() * sizeof(uint32_t);
        size_t max_out_size = input_bytes*2 + 4096;
        compressed_data.resize(max_out_size);

        size_t final_size = max_out_size;
        
        mans::cpu::compress_internal(
            host_data.data(), 
            host_data.size(), 
            params, 
            compressed_data.data(), 
            final_size,
            save_adm, 
            output_file + ".adm", 
            true
        );

        compressed_data.resize(final_size);
    }

    if (!save_u8_file(output_file, compressed_data)) {
        std::cerr << "Failed to write Final output: " << output_file << "\n";
        return 1;
    }

    std::cout << "Mans compress finished! Written to " << output_file 
              << " (Size: " << compressed_data.size() << ")\n";
    
    return 0;
}