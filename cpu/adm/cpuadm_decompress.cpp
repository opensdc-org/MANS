// compiler: g++ -std=c++17 -mavx512f -fopenmp -march=native -O3 decompress.cpp -o decompress
// exec   : OMP_NUM_THREADS=4 ./decompress u2 input.bin output.u2
//          OMP_NUM_THREADS=4 ./decompress u4 input.bin output.u4

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <chrono>

#include "adm.h"

bool load_u8_file(const std::string& filename, std::vector<std::uint8_t>& data) {
    std::ifstream in(filename, std::ios::binary | std::ios::ate);
    if (!in.is_open()) return false;
    std::streamsize size = in.tellg();
    if (size < 0) return false;
    in.seekg(0, std::ios::beg);
    data.resize(static_cast<std::size_t>(size));
    return static_cast<bool>(in.read(reinterpret_cast<char*>(data.data()), size));
}

bool save_u16_file(const std::string& filename, const std::vector<std::uint16_t>& data) {
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) return false;
    out.write(reinterpret_cast<const char*>(data.data()),
              static_cast<std::streamsize>(data.size() * sizeof(std::uint16_t)));
    return static_cast<bool>(out);
}

bool save_u32_file(const std::string& filename, const std::vector<std::uint32_t>& data) {
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) return false;
    out.write(reinterpret_cast<const char*>(data.data()),
              static_cast<std::streamsize>(data.size() * sizeof(std::uint32_t)));
    return static_cast<bool>(out);
}

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
        std::cerr << "File too small or invalid format.\n";
        return 1;
    }

    adm::FileHeader header;
    std::size_t offset = 0;
    std::memcpy(&header, merged.data(), sizeof(header));
    offset += sizeof(header);

    std::size_t num_elements = static_cast<std::size_t>(header.num_elements);
    std::size_t gsize        = static_cast<std::size_t>(header.gsize);
    std::size_t len1         = static_cast<std::size_t>(header.len1);
    std::size_t len2         = static_cast<std::size_t>(header.len2);
    std::size_t len3         = static_cast<std::size_t>(header.len3);
    std::size_t len4         = static_cast<std::size_t>(header.len4);

    if (merged.size() < offset + len1 + len2 + len3 + len4) {
        std::cerr << "Corrupted file: not enough data.\n";
        return 1;
    }

    // ====================== uint16 ======================
    if (is_u2) {
        std::vector<int>            output_lengths(len1 / sizeof(int));
        std::vector<std::uint16_t>  centers(len2 / sizeof(std::uint16_t));
        std::vector<std::uint8_t>   codes(len3 / sizeof(std::uint8_t));
        std::vector<std::uint8_t>   bit_signals(len4);

        std::size_t off = offset;
        std::memcpy(output_lengths.data(), merged.data() + off, len1); off += len1;
        std::memcpy(centers.data(),        merged.data() + off, len2); off += len2;
        std::memcpy(codes.data(),          merged.data() + off, len3); off += len3;
        std::memcpy(bit_signals.data(),    merged.data() + off, len4);

        std::cout << "\033[0;36m=======> Start ADM Decompress (uint16) <=======\033[0m\n";

        int   times   = 10;
        float exe_min = 1e30f;
        std::vector<std::uint16_t> recovered(num_elements);

        for (int i = 0; i < times; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            adm::decompress_uint16(output_lengths, centers, codes, bit_signals, recovered);
            auto end   = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> dur = (end - start);
            exe_min = std::min(exe_min, static_cast<float>(dur.count()));
        }

        double throughput = num_elements * 2.0 / 1024.0 / 1024.0 / (exe_min / 1000.0);
        std::printf("decompress cost %.2f ms, throughput %.2f MB/s\n",
                    exe_min, throughput);

        if (!save_u16_file(output_file, recovered)) {
            std::cerr << "Failed to write output file: " << output_file << "\n";
            return 1;
        }

        std::cout << "Decompress finished! Write to " << output_file << "\n";
        return 0;
    }

    // ====================== uint32 ======================
    if (is_u4) {
        std::vector<int>            output_lengths(len1 / sizeof(int));
        std::vector<std::uint32_t>  centers(len2 / sizeof(std::uint32_t));
        std::vector<std::uint8_t>   codes(len3 / sizeof(std::uint8_t));
        std::vector<std::uint8_t>   bit_signals(len4);

        std::size_t off = offset;
        std::memcpy(output_lengths.data(), merged.data() + off, len1); off += len1;
        std::memcpy(centers.data(),        merged.data() + off, len2); off += len2;
        std::memcpy(codes.data(),          merged.data() + off, len3); off += len3;
        std::memcpy(bit_signals.data(),    merged.data() + off, len4);

        std::cout << "\033[0;36m=======> Start ADM Decompress (uint32) <=======\033[0m\n";

        int   times   = 10;
        float exe_min = 1e30f;
        std::vector<std::uint32_t> recovered(num_elements);

        for (int i = 0; i < times; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            adm::decompress_uint32(output_lengths, centers, codes, bit_signals, recovered);
            auto end   = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> dur = (end - start);
            exe_min = std::min(exe_min, static_cast<float>(dur.count()));
        }

        double throughput = num_elements * 4.0 / 1024.0 / 1024.0 / (exe_min / 1000.0);
        std::printf("decompress cost %.2f ms, throughput %.2f MB/s\n",
                    exe_min, throughput);

        if (!save_u32_file(output_file, recovered)) {
            std::cerr << "Failed to write output file: " << output_file << "\n";
            return 1;
        }

        std::cout << "Decompress finished! Write to " << output_file << "\n";
        return 0;
    }

    return 0;
}
