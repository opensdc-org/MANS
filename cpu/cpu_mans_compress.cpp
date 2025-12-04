// compiler： g++ -std=c++17 -O3 cpu_mans_compress.cpp -o cpu_mans_compress
// exec   :  OMP_NUM_THREADS=4 ./cpu_mans_compress u2 input.u2 output.bin
//           OMP_NUM_THREADS=4 ./cpu_mans_compress u4 input.u4 output.bin

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <limits>
#include <cstdlib>  // system, remove

const int threshold= 4000;

struct MansHeader {
    std::uint8_t codec;  // 1 = ADM, 2 = ANS
};
static_assert(sizeof(MansHeader) == 1, "MansHeader must be 1 byte");


bool load_u16_file(const std::string& filename, std::vector<std::uint16_t>& data) {
    std::ifstream in(filename, std::ios::binary | std::ios::ate);
    if (!in.is_open()) return false;
    std::streamsize size = in.tellg();
    if (size < 0) return false;
    in.seekg(0, std::ios::beg);
    data.resize(static_cast<std::size_t>(size) / sizeof(std::uint16_t));
    return static_cast<bool>(in.read(reinterpret_cast<char*>(data.data()), size));
}

bool load_u32_file(const std::string& filename, std::vector<std::uint32_t>& data) {
    std::ifstream in(filename, std::ios::binary | std::ios::ate);
    if (!in.is_open()) return false;
    std::streamsize size = in.tellg();
    if (size < 0) return false;
    in.seekg(0, std::ios::beg);
    data.resize(static_cast<std::size_t>(size) / sizeof(std::uint32_t));
    return static_cast<bool>(in.read(reinterpret_cast<char*>(data.data()), size));
}

bool load_u8_file(const std::string& filename, std::vector<std::uint8_t>& data) {
    std::ifstream in(filename, std::ios::binary | std::ios::ate);
    if (!in.is_open()) return false;
    std::streamsize size = in.tellg();
    if (size < 0) return false;
    in.seekg(0, std::ios::beg);
    data.resize(static_cast<std::size_t>(size));
    return static_cast<bool>(in.read(reinterpret_cast<char*>(data.data()), size));
}

bool save_u8_file(const std::string& filename, const std::vector<std::uint8_t>& data) {
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) return false;
    out.write(reinterpret_cast<const char*>(data.data()),
              static_cast<std::streamsize>(data.size()));
    return static_cast<bool>(out);
}


int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Use: " << argv[0]
                  << " <u2|u4> <input_file> <output_bin_file>\n";
        return 1;
    }

    std::string dtype       = argv[1];   // "u2" or "u4"
    std::string input_file  = argv[2];
    std::string output_file = argv[3];

    bool is_u2 = (dtype == "-u2" || dtype == "u2");
    bool is_u4 = (dtype == "-u4" || dtype == "u4");

    if (!is_u2 && !is_u4) {
        std::cerr << "Unknown data type flag: " << dtype
                  << "\nUse: u2 or u4 (or -u2/-u4)\n";
        return 1;
    }

        // 1. 读取数据 & 按 512 块计算 max-min
    bool use_adm = false;
    const std::size_t block_size = 512;

    if (is_u2) {
        std::vector<std::uint16_t> data;
        if (!load_u16_file(input_file, data)) {
            std::cerr << "Failed to load input file: " << input_file << "\n";
            return 1;
        }
        if (data.empty()) {
            std::cerr << "Input file is empty.\n";
            return 1;
        }

        std::uint64_t max_block_diff = 0;

        for (std::size_t i = 0; i < data.size(); i += block_size) {
            std::size_t end = std::min(i + block_size, data.size());

            std::uint16_t bmin = std::numeric_limits<std::uint16_t>::max();
            std::uint16_t bmax = std::numeric_limits<std::uint16_t>::min();

            for (std::size_t j = i; j < end; ++j) {
                std::uint16_t v = data[j];
                if (v < bmin) bmin = v;
                if (v > bmax) bmax = v;
            }

            std::uint32_t diff =
                static_cast<std::uint32_t>(bmax) - static_cast<std::uint32_t>(bmin);
            if (diff > max_block_diff) {
                max_block_diff = diff;
            }
        }

        std::cout << "[mans] u2 block range (block_size=512): max_diff="
                  << max_block_diff << "\n";

        use_adm = (max_block_diff <= threshold);

    } else { // is_u4

        std::vector<std::uint32_t> data;
        if (!load_u32_file(input_file, data)) {
            std::cerr << "Failed to load input file: " << input_file << "\n";
            return 1;
        }
        if (data.empty()) {
            std::cerr << "Input file is empty.\n";
            return 1;
        }

        std::uint64_t max_block_diff = 0;

        for (std::size_t i = 0; i < data.size(); i += block_size) {
            std::size_t end = std::min(i + block_size, data.size());

            std::uint32_t bmin = std::numeric_limits<std::uint32_t>::max();
            std::uint32_t bmax = std::numeric_limits<std::uint32_t>::min();

            for (std::size_t j = i; j < end; ++j) {
                std::uint32_t v = data[j];
                if (v < bmin) bmin = v;
                if (v > bmax) bmax = v;
            }

            std::uint64_t diff =
                static_cast<std::uint64_t>(bmax) - static_cast<std::uint64_t>(bmin);
            if (diff > max_block_diff) {
                max_block_diff = diff;
            }
        }

        std::cout << "[mans] u4 block range (block_size=512): max_diff="
                  << max_block_diff << "\n";

        use_adm = (max_block_diff <= threshold);
    }

    std::string tmp_out = output_file + ".adm"; 

    std::string cmd;

    MansHeader mh{};
    if (use_adm) {
        mh.codec = 1; // ADM
        cmd = "./build/bin/cpu/adm/compress " + dtype + " " + input_file + " " + tmp_out;
    } 
    else{
        mh.codec = 2; // without ADM
    }
    
    int ret = std::system(cmd.c_str());
    if (ret != 0) {
        std::cerr << "Failed to run command: " << cmd << ", ret=" << ret << "\n";
        return 1;
    }

    if (use_adm) {
        cmd = "./build/bin/cpu/pans/cpuans_compress " + tmp_out + " " + output_file;
    } 
    else{
        cmd = "./build/bin/cpu/pans/cpuans_compress " + input_file + " " + output_file;
    }

    ret = std::system(cmd.c_str());
    if (ret != 0) {
        std::cerr << "Failed to run command: " << cmd << ", ret=" << ret << "\n";
        return 1;
    }

    std::vector<std::uint8_t> payload;
    if (!load_u8_file(output_file, payload)) {
        std::cerr << "Failed to load tmp output file: " << output_file << "\n";
        return 1;
    }

    std::vector<std::uint8_t> final_data;
    final_data.reserve(1 + payload.size());
    final_data.push_back(mh.codec);
    final_data.insert(final_data.end(), payload.begin(), payload.end());

    if (!save_u8_file(output_file, final_data)) {
        std::cerr << "Failed to write output file: " << output_file << "\n";
        return 1;
    }

    std::cout << "mans compress finished! Write to " << output_file
              << " (codec=" << int(mh.codec) << ")\n";
    return 0;
}
