// compiler： g++ -std=c++17 -O3 cpu_mans_decompress.cpp -o cpu_mans_decompress
// exec   :  OMP_NUM_THREADS=4 ./cpu_mans_decompress u2 input.bin output.u2
//           OMP_NUM_THREADS=4 ./cpu_mans_decompress u4 input.bin output.u4

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <cstdlib>  // system, remove

struct MansHeader {
    std::uint8_t codec;  // 1 = ADM, 2 = ANS
};
static_assert(sizeof(MansHeader) == 1, "MansHeader must be 1 byte");

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
                  << " <u2|u4> <input_bin_file> <output_u2/u4_file>\n";
        return 1;
    }

    std::string dtype       = argv[1];  // "u2" / "u4"
    std::string input_file  = argv[2];
    std::string output_file = argv[3];

    bool is_u2 = (dtype == "-u2" || dtype == "u2");
    bool is_u4 = (dtype == "-u4" || dtype == "u4");

    if (!is_u2 && !is_u4) {
        std::cerr << "Unknown data type flag: " << dtype
                  << "\nUse: u2 or u4 (or -u2/-u4)\n";
        return 1;
    }

    std::vector<std::uint8_t> all;
    if (!load_u8_file(input_file, all)) {
        std::cerr << "Failed to load input file: " << input_file << "\n";
        return 1;
    }
    if (all.size() < sizeof(MansHeader)) {
        std::cerr << "File too small, invalid mans format.\n";
        return 1;
    }

    // 2. weather use adm
    MansHeader mh;
    mh.codec = all[0];

    std::vector<std::uint8_t> payload(all.begin() + 1, all.end());
    std::string tmp_in = input_file + ".tmp_payload";

    if (!save_u8_file(tmp_in, payload)) {
        std::cerr << "Failed to write tmp payload file: " << tmp_in << "\n";
        return 1;
    }

    std::string tmp_adm = output_file + ".adm";
    std::string cmd;
    if(mh.codec == 1)
    {
        cmd = "./build/bin/cpu/pans/cpuans_decompress " + tmp_in + " " + tmp_adm;
    }
    else if(mh.codec == 2) {
        // adm/decompress: ./decompress u2/u4 input.bin output.u2/u4
        cmd = "./build/bin/cpu/pans/cpuans_decompress " + tmp_in + " " + output_file;
    }
    else {
        std::cerr << "Unknown codec type in mans header: " << int(mh.codec) << "\n";
        return 1;
    }

    int ret = std::system(cmd.c_str());
    if (ret != 0) {
        std::cerr << "Failed to run command: " << cmd << ", ret=" << ret << "\n";
        return 1;
    }

    if(mh.codec == 1)
    {
        cmd = "./build/bin/cpu/adm/decompress " + dtype + " " + tmp_adm + " " + output_file;
    }

    ret = std::system(cmd.c_str());
    if (ret != 0) {
        std::cerr << "Failed to run command: " << cmd << ", ret=" << ret << "\n";
        return 1;
    }

    std::remove(tmp_in.c_str());

    std::cout << "mans decompress finished! Output: " << output_file << "\n";
    return 0;
}
