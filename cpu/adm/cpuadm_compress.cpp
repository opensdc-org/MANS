// compiler： g++ -std=c++17 -mavx512f -fopenmp -march=native -O3 compress.cpp -o compress
// exec: OMP_NUM_THREADS=4 ./compress u2 input.u2 output.bin
//       OMP_NUM_THREADS=4 ./compress u4 input.u2 output.bin

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <chrono>

#include "adm.h"

const int cmp_tblock_size = 32;
const int cmp_chunk = 16;
const int decmp_chunk = 32;
const int max_bytes_signal_per_ele_16b = 2;
const int warp_size = 32;

bool load_u16_file(const std::string& filename, std::vector<std::uint16_t>& data) {
    std::ifstream in(filename, std::ios::binary | std::ios::ate);
    if (!in.is_open()) return false;
    std::streamsize size = in.tellg();
    if (size < 0) return false;
    in.seekg(0, std::ios::beg);
    data.resize(static_cast<std::size_t>(size) / sizeof(std::uint16_t));
    return static_cast<bool>(in.read(
        reinterpret_cast<char*>(data.data()), size));
}

bool load_u32_file(const std::string& filename, std::vector<std::uint32_t>& data) {
    std::ifstream in(filename, std::ios::binary | std::ios::ate);
    if (!in.is_open()) return false;
    std::streamsize size = in.tellg();
    if (size < 0) return false;
    in.seekg(0, std::ios::beg);
    data.resize(static_cast<std::size_t>(size) / sizeof(std::uint32_t));
    return static_cast<bool>(in.read(
        reinterpret_cast<char*>(data.data()), size));
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

    std::vector<std::uint8_t> merged;
    double cr = 0.0;

    // ====================== uint16 流程 ======================
    if (is_u2) {

        // load data
        std::vector<std::uint16_t> input_data;
        if (!load_u16_file(input_file, input_data)) {
            std::cerr << "Failed to load input file: " << input_file << "\n";
            return 1;
        }

        std::size_t num_elements = input_data.size();
        std::uint64_t gsize = (num_elements
            + adm::cmp_tblock_size * adm::cmp_chunk - 1)
            / (adm::cmp_tblock_size * adm::cmp_chunk);

        std::vector<int>            output_lengths(gsize + 1);
        std::vector<std::uint16_t>  centers(gsize);
        std::vector<std::uint8_t>   codes(num_elements);
        std::vector<std::uint8_t>   bit_signals;
        // int last_length = 0;

        // warmup
        for (int i = 0; i < 5; ++i) {
            adm::compress_uint16(input_data, output_lengths, centers, codes, bit_signals);
        }

        // 正式计时
        std::cout << "\033[0;36m=======> Start ADM Compress <=======\033[0m\n";
        int   times   = 10;
        float exe_min = 1e30f;

        for (int i = 0; i < times; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            adm::compress_uint16(input_data, output_lengths, centers, codes, bit_signals);
            auto end   = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> dur = (end - start);
            exe_min = std::min(exe_min, static_cast<float>(dur.count()));
        }

        double throughput = num_elements * 2.0 / 1024.0 / 1024.0 / (exe_min / 1000.0);
        std::printf("compress cost %.2f ms, throughput %.2f MB/s\n",
                    exe_min, throughput);

        // ----------- save data -----------
        // file format:
        // [FileHeader][output_lengths(int[gsize+1])]
        // [centers(uint16[gsize])][codes(uint8[num_elements])]
        // [bit_signals(byte[...])]
        adm::FileHeader header;
        header.num_elements = static_cast<std::uint64_t>(num_elements);
        header.gsize = gsize;

        std::size_t len1 = output_lengths.size() * sizeof(int);
        std::size_t len2 = centers.size()       * sizeof(std::uint16_t);
        std::size_t len3 = codes.size()         * sizeof(std::uint8_t);
        std::size_t len4 = bit_signals.size();

        header.len1 = len1;
        header.len2 = len2;
        header.len3 = len3;
        header.len4 = len4;

        std::size_t len_header = sizeof(header);

        std::size_t total_size = len_header + len1 + len2 + len3 + len4;
        // std::vector<std::uint8_t> merged(total_size);

        std::size_t offset = 0;
        merged.resize(total_size);
        std::memcpy(merged.data() + offset, &header, len_header); offset += len_header;
        std::memcpy(merged.data() + offset, output_lengths.data(), len1); offset += len1;
        std::memcpy(merged.data() + offset, centers.data(),        len2); offset += len2;
        std::memcpy(merged.data() + offset, codes.data(),          len3); offset += len3;
        std::memcpy(merged.data() + offset, bit_signals.data(),    len4);

        cr = num_elements * sizeof(std::uint16_t) * 1.0 / total_size;
    }

    // ====================== uint32 流程 ======================
    if (is_u4) {
        std::vector<std::uint32_t> input_data;
        if (!load_u32_file(input_file, input_data)) {
            std::cerr << "Failed to load input file: " << input_file << "\n";
            return 1;
        }

        std::size_t num_elements = input_data.size();
        std::uint64_t gsize = (num_elements
            + adm::cmp_tblock_size * adm::cmp_chunk - 1)
            / (adm::cmp_tblock_size * adm::cmp_chunk);

        std::vector<int>              output_lengths(gsize + 1);
        std::vector<std::uint32_t>    centers(gsize);
        std::vector<std::uint8_t>     codes(num_elements);
        std::vector<std::uint8_t>     bit_signals;

        // 预热几次
        for (int i = 0; i < 5; ++i) { 
            adm::compress_uint32(input_data, output_lengths, centers, codes, bit_signals);
        }

        // 正式计时
        std::cout << "\033[0;36m=======> Start ADM Compress (uint32) <=======\033[0m\n";
        int   times   = 10;
        float exe_min = 1e30f;

        for (int i = 0; i < times; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            adm::compress_uint32(input_data, output_lengths, centers, codes, bit_signals);
            auto end   = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> dur = (end - start);
            exe_min = std::min(exe_min, static_cast<float>(dur.count()));
        }

        double throughput = num_elements * 4.0 / 1024.0 / 1024.0 / (exe_min / 1000.0);
        std::printf("compress cost %.2f ms, throughput %.2f MB/s\n",
                    exe_min, throughput);

        // ----------- save data -----------
        // file format:
        // [FileHeader][output_lengths(int[gsize+1])]
        // [centers(uint16[gsize])][codes(uint8[num_elements])]
        // [bit_signals(byte[...])]
        adm::FileHeader header;
        header.num_elements = static_cast<std::uint64_t>(num_elements);
        header.gsize = gsize;

        // 填写 header & 合并数据
        std::size_t len1 = output_lengths.size() * sizeof(int);
        std::size_t len2 = centers.size()       * sizeof(std::uint32_t);
        std::size_t len3 = codes.size()         * sizeof(std::uint8_t);
        std::size_t len4 = bit_signals.size();

        header.num_elements = static_cast<std::uint64_t>(num_elements);
        header.gsize        = gsize;
        header.len1         = len1;
        header.len2         = len2;
        header.len3         = len3;
        header.len4         = len4;

        std::size_t len_header = sizeof(header);
        std::size_t total_size = len_header + len1 + len2 + len3 + len4;

        std::size_t offset = 0;
        merged.resize(total_size);
        std::memcpy(merged.data() + offset, &header, len_header);         offset += len_header;
        std::memcpy(merged.data() + offset, output_lengths.data(), len1); offset += len1;
        std::memcpy(merged.data() + offset, centers.data(),        len2); offset += len2;
        std::memcpy(merged.data() + offset, codes.data(),          len3); offset += len3;
        std::memcpy(merged.data() + offset, bit_signals.data(),    len4);

        cr = num_elements * sizeof(std::uint32_t) * 1.0 / total_size;
    }

    if (!save_u8_file(output_file, merged)) {
        std::cerr << "Failed to write output file: " << output_file << "\n";
        return 1;
    }

    std::cout << "ADM finished! Write to " << output_file << "\n";

    // 压缩率
    std::printf("\033[0;36m=======> Compression Ratio <=======\033[0m\n");
    std::printf("CR : %.2f x\n", cr);

    return 0;
}
