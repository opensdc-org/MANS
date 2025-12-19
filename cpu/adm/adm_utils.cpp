#include "adm_utils.h"
#include "adm.h" // 包含核心算法和 FileHeader 定义
#include <iostream>
#include <cstring>
#include <chrono>
#include <type_traits>
#include <stdexcept>
#include <cstdio> 

// 纯压缩：input_data -> merged output
template<typename T>
void adm_compress(
    const std::vector<T>& input_data,
    std::vector<std::uint8_t>& output)
{
    std::size_t num_elements = input_data.size();
    if (num_elements == 0) {
        output.clear();
        return;
    }

    std::uint64_t gsize = (num_elements
        + adm::cmp_tblock_size * adm::cmp_chunk - 1)
        / (adm::cmp_tblock_size * adm::cmp_chunk);

    std::vector<int>              output_lengths(gsize + 1);
    std::vector<T>                centers(gsize);
    std::vector<std::uint8_t>     codes(num_elements);
    std::vector<std::uint8_t>     bit_signals;

    // 实际压缩
    if constexpr (std::is_same_v<T, std::uint16_t>) {
        adm::compress_uint16(input_data, output_lengths, centers, codes, bit_signals);
    } else if constexpr (std::is_same_v<T, std::uint32_t>) {
        adm::compress_uint32(input_data, output_lengths, centers, codes, bit_signals);
    } else {
        static_assert(std::is_same_v<T, std::uint16_t> || std::is_same_v<T, std::uint32_t>,
                      "adm_compress only supports uint16_t and uint32_t");
    }

    // 打包数据到一个连续 buffer
    adm::FileHeader header;
    header.num_elements = static_cast<std::uint64_t>(num_elements);
    header.gsize        = gsize;

    std::size_t len1 = output_lengths.size() * sizeof(int);
    std::size_t len2 = centers.size()       * sizeof(T);
    std::size_t len3 = codes.size()         * sizeof(std::uint8_t);
    std::size_t len4 = bit_signals.size();

    header.len1 = len1;
    header.len2 = len2;
    header.len3 = len3;
    header.len4 = len4;

    std::size_t len_header = sizeof(header);
    std::size_t total_size = len_header + len1 + len2 + len3 + len4;

    output.resize(total_size);
    std::size_t offset = 0;
    std::memcpy(output.data() + offset, &header,           len_header); offset += len_header;
    std::memcpy(output.data() + offset, output_lengths.data(), len1);  offset += len1;
    std::memcpy(output.data() + offset, centers.data(),        len2);  offset += len2;
    std::memcpy(output.data() + offset, codes.data(),          len3);  offset += len3;
    std::memcpy(output.data() + offset, bit_signals.data(),    len4);
}

// 解压：merged -> recovered
template<typename T>
void adm_decompress(
    const std::vector<std::uint8_t>& merged,
    std::vector<T>& recovered)
{
    if (merged.size() < sizeof(adm::FileHeader)) {
        throw std::runtime_error("File too small or invalid format.");
    }

    adm::FileHeader header;
    std::size_t offset = 0;
    std::memcpy(&header, merged.data(), sizeof(header));
    offset += sizeof(header);

    std::size_t num_elements = static_cast<std::size_t>(header.num_elements);
    std::size_t len1 = static_cast<std::size_t>(header.len1);
    std::size_t len2 = static_cast<std::size_t>(header.len2);
    std::size_t len3 = static_cast<std::size_t>(header.len3);
    std::size_t len4 = static_cast<std::size_t>(header.len4);

    if (merged.size() < offset + len1 + len2 + len3 + len4) {
        throw std::runtime_error("Corrupted file: not enough data.");
    }

    std::vector<int>            output_lengths(len1 / sizeof(int));
    std::vector<T>              centers(len2 / sizeof(T));
    std::vector<std::uint8_t>   codes(len3);
    std::vector<std::uint8_t>   bit_signals(len4);

    std::size_t off = offset;
    std::memcpy(output_lengths.data(), merged.data() + off, len1); off += len1;
    std::memcpy(centers.data(),        merged.data() + off, len2); off += len2;
    std::memcpy(codes.data(),          merged.data() + off, len3); off += len3;
    std::memcpy(bit_signals.data(),    merged.data() + off, len4);

    recovered.resize(num_elements);

    if constexpr (std::is_same_v<T, std::uint16_t>) {
        adm::decompress_uint16(output_lengths, centers, codes, bit_signals, recovered);
    } else if constexpr (std::is_same_v<T, std::uint32_t>) {
        adm::decompress_uint32(output_lengths, centers, codes, bit_signals, recovered);
    } else {
        static_assert(std::is_same_v<T, std::uint16_t> || std::is_same_v<T, std::uint32_t>,
                      "adm_decompress only supports uint16_t and uint32_t");
    }
}

// ===== compress_and_benchmark =====
template<typename T>
void adm_compress_and_benchmark(
    const std::vector<T>& input_data,
    std::vector<std::uint8_t>& output)
{
    std::size_t num_elements = input_data.size();
    if (num_elements == 0) {
        output.clear();
        return;
    }

    // warmup
    for (int i = 0; i < 5; ++i) {
        std::vector<std::uint8_t> tmp;
        adm_compress<T>(input_data, tmp);
    }

    if constexpr (std::is_same_v<T, std::uint16_t>) {
        std::cout << "\033[0;36m=======> Start ADM Compress (uint16, benchmark) <=======\033[0m\n";
    } else {
        std::cout << "\033[0;36m=======> Start ADM Compress (uint32, benchmark) <=======\033[0m\n";
    }

    int   times   = 10;
    float exe_min = 1e30f;

    for (int i = 0; i < times; ++i) {
        std::vector<std::uint8_t> tmp;
        auto start = std::chrono::high_resolution_clock::now();
        adm_compress<T>(input_data, tmp);
        auto end   = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> dur = (end - start);
        exe_min = std::min(exe_min, static_cast<float>(dur.count()));

        // 最后一次迭代把结果返回给调用者
        if (i == times - 1) {
            output.swap(tmp);
        }
    }

    std::size_t element_bytes = sizeof(T);
    double throughput = num_elements * element_bytes * 1.0 / 1024.0 / 1024.0 / (exe_min / 1000.0);
    std::printf("compress cost %.2f ms, throughput %.2f MB/s\n", exe_min, throughput);

    double cr = 0.0;
    if (!output.empty()) {
        std::size_t total_size = output.size();
        cr = num_elements * element_bytes * 1.0 / total_size;
    }
    std::printf("\033[0;36m=======> Compression Ratio <=======\033[0m\n");
    std::printf("CR : %.2f x\n", cr);
}

// ===== decompress_and_benchmark =====
template<typename T>
void adm_decompress_and_benchmark(
    const std::vector<std::uint8_t>& merged,
    std::vector<T>& recovered)
{
    if (merged.size() < sizeof(adm::FileHeader)) {
        throw std::runtime_error("File too small or invalid format.");
    }

    // 先从 header 里读出元素个数，用于 throughput 统计
    adm::FileHeader header;
    std::memcpy(&header, merged.data(), sizeof(header));
    std::size_t num_elements = static_cast<std::size_t>(header.num_elements);

    if constexpr (std::is_same_v<T, std::uint16_t>) {
        std::cout << "\033[0;36m=======> Start ADM Decompress (uint16, benchmark) <=======\033[0m\n";
    } else {
        std::cout << "\033[0;36m=======> Start ADM Decompress (uint32, benchmark) <=======\033[0m\n";
    }

    int   times   = 10;
    float exe_min = 1e30f;

    for (int i = 0; i < times; ++i) {
        std::vector<T> tmp;
        auto start = std::chrono::high_resolution_clock::now();
        adm_decompress<T>(merged, tmp);
        auto end   = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> dur = (end - start);
        exe_min = std::min(exe_min, static_cast<float>(dur.count()));

        if (i == times - 1) {
            recovered.swap(tmp);
        }
    }

    std::size_t element_bytes = sizeof(T);
    double throughput = num_elements * element_bytes * 1.0 / 1024.0 / 1024.0 / (exe_min / 1000.0);
    std::printf("decompress cost %.2f ms, throughput %.2f MB/s\n", exe_min, throughput);
}



// ==========================================================
// 显式实例化 (Explicit Instantiation)
// 必须放在 .cpp 文件末尾，否则链接器找不到符号
// ==========================================================
template void adm_compress<uint16_t>(const std::vector<uint16_t>&, std::vector<uint8_t>&);
template void adm_compress<uint32_t>(const std::vector<uint32_t>&, std::vector<uint8_t>&);

template void adm_decompress<uint16_t>(const std::vector<uint8_t>&, std::vector<uint16_t>&);
template void adm_decompress<uint32_t>(const std::vector<uint8_t>&, std::vector<uint32_t>&);

template void adm_compress_and_benchmark<uint16_t>(const std::vector<uint16_t>&, std::vector<uint8_t>&);
template void adm_compress_and_benchmark<uint32_t>(const std::vector<uint32_t>&, std::vector<uint8_t>&);

template void adm_decompress_and_benchmark<uint16_t>(const std::vector<uint8_t>&, std::vector<uint16_t>&);
template void adm_decompress_and_benchmark<uint32_t>(const std::vector<uint8_t>&, std::vector<uint32_t>&);