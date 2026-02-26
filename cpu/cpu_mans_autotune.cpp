#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <omp.h>

#include "../mans_data_gen.h"
#include "../mans_defs.h"
#include "mans_cpu.h"

namespace {

struct ChunkInfo {
    std::size_t offset = 0;
    std::size_t len = 0;
};

struct ThroughputResult {
    double comp_mbps = 0.0;
    double decomp_mbps = 0.0;
    bool ok = true;
    std::string error;
};

struct BestEntry {
    double throughput = -1.0;
    int threads = 0;
};

std::vector<ChunkInfo> build_chunks(std::size_t total_elements, std::size_t chunk_elements) {
    std::vector<ChunkInfo> chunks;
    if (chunk_elements == 0 || total_elements == 0) {
        return chunks;
    }
    std::size_t offset = 0;
    while (offset < total_elements) {
        std::size_t len = std::min(chunk_elements, total_elements - offset);
        chunks.push_back(ChunkInfo{offset, len});
        offset += len;
    }
    return chunks;
}

std::size_t max_chunk_len(const std::vector<ChunkInfo>& chunks) {
    std::size_t max_len = 0;
    for (const auto& chunk : chunks) {
        max_len = std::max(max_len, chunk.len);
    }
    return max_len;
}

template<typename T>
static bool decide_use_adm(const T* data, std::size_t size, std::uint32_t threshold, std::uint32_t threads) {
    const std::size_t block_size = 512;
    std::uint64_t max_block_diff = 0;
    const std::size_t blocks = (size + block_size - 1) / block_size;
    const int num_threads = threads == 0 ? 16 : static_cast<int>(threads);

    #pragma omp parallel for num_threads(num_threads) reduction(max:max_block_diff)
    for (std::size_t b = 0; b < blocks; ++b) {
        std::size_t i = b * block_size;
        std::size_t end = std::min(i + block_size, size);
        T bmin = std::numeric_limits<T>::max();
        T bmax = std::numeric_limits<T>::min();

        for (std::size_t j = i; j < end; ++j) {
            T v = data[j];
            if (v < bmin) bmin = v;
            if (v > bmax) bmax = v;
        }

        std::uint64_t diff = static_cast<std::uint64_t>(bmax) - static_cast<std::uint64_t>(bmin);
        if (diff > max_block_diff) {
            max_block_diff = diff;
        }
    }
    return (max_block_diff <= threshold);
}

std::vector<int> build_thread_list(int threads_min, int threads_max, int stride) {
    std::vector<int> threads;
    if (stride <= 0 || threads_min <= 0 || threads_max <= 0) {
        return threads;
    }
    if (threads_min > threads_max) {
        return threads;
    }

    if (threads_min == 1) {
        threads.push_back(1);
        for (int mult = 1;; ++mult) {
            long long value = static_cast<long long>(stride) * static_cast<long long>(mult);
            if (value > threads_max) {
                break;
            }
            if (value == 1) {
                continue;
            }
            threads.push_back(static_cast<int>(value));
        }
        return threads;
    }

    for (int value = threads_min; value <= threads_max; value += stride) {
        threads.push_back(value);
    }
    return threads;
}

constexpr std::uint32_t kAdmThreshold = 4000;

void print_progress_line(const std::string& line, bool newline) {
    std::cout << "\r\033[K" << line;
    if (newline) {
        std::cout << "\n";
    }
    std::cout << std::flush;
}

bool open_csv_new(const std::string& path, std::ofstream& out) {
    out.open(path, std::ios::trunc);
    if (!out) {
        return false;
    }
    out << "chunk_elements,mode,threads,throughput_mbps\n";
    out.flush();
    return true;
}

void append_csv_row(std::ofstream& out, std::size_t chunk_elements, const std::string& mode,
                    int threads, double throughput_mbps) {
    out << chunk_elements << "," << mode << "," << threads << ","
        << std::fixed << std::setprecision(2) << throughput_mbps << "\n";
    out.flush();
}

mans::MansParams default_params() {
    mans::MansParams params{};
    params.backend = mans::Backend::CPU;
    params.dtype = mans::DataType::U16;
    params.mode = mans::Mode::P;
    params.adm_threshold = kAdmThreshold;
    params.adm_decide_threads = 32;
    params.adm_center_calc_threads = 32;
    params.adm_encode_threads = 32;
    params.adm_warp_reduce_threads = 32;
    params.adm_fill_tail_threads = 32;
    params.adm_write_back_threads = 32;
    params.adm_restore_signals_threads = 32;
    params.adm_decode_values_threads = 32;
    return params;
}

ThroughputResult run_compress_decompress(const std::uint16_t* data,
                                         const std::vector<ChunkInfo>& chunks,
                                         std::size_t total_elements,
                                         const mans::MansParams& params) {
    ThroughputResult result;
    if (chunks.empty()) {
        result.ok = false;
        result.error = "No chunks generated.";
        return result;
    }

    const std::size_t max_len = max_chunk_len(chunks);
    const std::size_t max_chunk_bytes = max_len * sizeof(std::uint16_t);
    const std::size_t max_out_bytes = max_chunk_bytes * 2 + 4096;
    std::vector<std::uint8_t> comp_buf(max_out_bytes);
    std::vector<std::uint8_t> decomp_buf(max_chunk_bytes);

    double comp_ms = 0.0;
    double decomp_ms = 0.0;

    for (const auto& chunk : chunks) {
        std::size_t out_size = comp_buf.size();
        auto comp_start = std::chrono::high_resolution_clock::now();
        mans::cpu::compress_internal(
            data + chunk.offset,
            chunk.len,
            params,
            comp_buf.data(),
            out_size,
            false,
            ""
        );
        auto comp_end = std::chrono::high_resolution_clock::now();
        comp_ms += std::chrono::duration<double, std::milli>(comp_end - comp_start).count();

        if (out_size == 0) {
            result.ok = false;
            result.error = "Compression failed (out_size=0).";
            return result;
        }

        std::size_t out_bytes = max_chunk_bytes;
        auto decomp_start = std::chrono::high_resolution_clock::now();
        mans::cpu::decompress_internal(
            comp_buf.data(),
            out_size,
            params,
            decomp_buf.data(),
            out_bytes,
            false,
            ""
        );
        auto decomp_end = std::chrono::high_resolution_clock::now();
        decomp_ms += std::chrono::duration<double, std::milli>(decomp_end - decomp_start).count();

        const std::size_t expected_bytes = chunk.len * sizeof(std::uint16_t);
        if (out_bytes != expected_bytes) {
            result.ok = false;
            result.error = "Decompressed size mismatch.";
            return result;
        }
        if (std::memcmp(decomp_buf.data(), data + chunk.offset, expected_bytes) != 0) {
            result.ok = false;
            result.error = "Decompressed data mismatch.";
            return result;
        }
    }

    const double total_bytes = static_cast<double>(total_elements * sizeof(std::uint16_t));
    result.comp_mbps = comp_ms > 0.0 ? (total_bytes / 1e6) / (comp_ms / 1e3) : 0.0;
    result.decomp_mbps = decomp_ms > 0.0 ? (total_bytes / 1e6) / (decomp_ms / 1e3) : 0.0;
    return result;
}

double run_decide_throughput(const std::uint16_t* data,
                             const std::vector<ChunkInfo>& chunks,
                             std::size_t total_elements,
                             int threads) {
    if (chunks.empty()) {
        return 0.0;
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (const auto& chunk : chunks) {
        (void)decide_use_adm<std::uint16_t>(
            data + chunk.offset,
            chunk.len,
            kAdmThreshold,
            static_cast<std::uint32_t>(threads)
        );
    }
    auto end = std::chrono::high_resolution_clock::now();

    const double ms = std::chrono::duration<double, std::milli>(end - start).count();
    const double total_bytes = static_cast<double>(total_elements * sizeof(std::uint16_t));
    return ms > 0.0 ? (total_bytes / 1e6) / (ms / 1e3) : 0.0;
}

int sweep_decide_threads(const std::uint16_t* data,
                         const std::vector<ChunkInfo>& chunks,
                         std::size_t total_elements,
                         std::size_t chunk_elements,
                         int threads_min,
                         int threads_max,
                         int stride,
                         std::ofstream* csv) {
    std::vector<int> thread_list = build_thread_list(threads_min, threads_max, stride);
    if (thread_list.empty()) {
        std::cerr << "[decide] Invalid thread list.\n";
        return threads_min;
    }

    int best_threads = thread_list.front();
    double best_thr = -1.0;

    for (int threads : thread_list) {
        {
            std::ostringstream line;
            line << "  [adm_decide] threads=" << threads << " ... running";
            print_progress_line(line.str(), false);
        }
        double throughput = run_decide_throughput(
            data, chunks, total_elements, threads);
        std::ostringstream line;
        line << "  [adm_decide] threads=" << threads << " ... "
             << std::fixed << std::setprecision(2) << throughput << " MB/s";
        print_progress_line(line.str(), true);

        if (csv && *csv) {
            append_csv_row(*csv, chunk_elements, "adm_decide", threads, throughput);
        }
        if (throughput > best_thr) {
            best_thr = throughput;
            best_threads = threads;
        }
    }

    std::cout << "\n  [adm_decide] best threads=" << best_threads
              << " throughput=" << std::fixed << std::setprecision(2) << best_thr
              << " MB/s\n";
    return best_threads;
}

void sweep_codec_threads(const std::uint16_t* data,
                         const std::vector<ChunkInfo>& chunks,
                         std::size_t total_elements,
                         std::size_t chunk_elements,
                         int decide_threads,
                         int threads_min,
                         int threads_max,
                         int stride,
                         std::ofstream* csv) {
    std::vector<int> thread_list = build_thread_list(threads_min, threads_max, stride);
    if (thread_list.empty()) {
        std::cerr << "[codec] Invalid thread list.\n";
        return;
    }

    for (int threads : thread_list) {
        mans::MansParams params = default_params();
        params.adm_decide_threads = static_cast<std::uint32_t>(decide_threads);
        params.adm_center_calc_threads = static_cast<std::uint32_t>(threads);
        params.adm_encode_threads = static_cast<std::uint32_t>(threads);
        params.adm_warp_reduce_threads = static_cast<std::uint32_t>(threads);
        params.adm_fill_tail_threads = static_cast<std::uint32_t>(threads);
        params.adm_write_back_threads = static_cast<std::uint32_t>(threads);
        params.adm_restore_signals_threads = static_cast<std::uint32_t>(threads);
        params.adm_decode_values_threads = static_cast<std::uint32_t>(threads);

        {
            std::ostringstream line;
            line << "  [codec] threads=" << threads << " ... running";
            print_progress_line(line.str(), false);
        }
        ThroughputResult result = run_compress_decompress(data, chunks, total_elements, params);
        if (!result.ok) {
            std::cerr << "\n  [codec] failed: " << result.error << "\n";
            continue;
        }
        std::ostringstream line;
        line << "  [codec] threads=" << threads << " ... "
             << "comp=" << std::fixed << std::setprecision(2) << result.comp_mbps
             << " MB/s, decomp=" << result.decomp_mbps << " MB/s";
        print_progress_line(line.str(), true);

        if (csv && *csv) {
            append_csv_row(*csv, chunk_elements, "compress", threads, result.comp_mbps);
            append_csv_row(*csv, chunk_elements, "decompress", threads, result.decomp_mbps);
        }
    }
    std::cout << "\n";
}

bool write_best_threads_csv(const std::string& input_csv,
                            const std::string& output_csv) {
    std::ifstream in(input_csv);
    if (!in.is_open()) {
        std::cerr << "Failed to open input CSV: " << input_csv << "\n";
        return false;
    }

    std::map<std::size_t, std::map<std::string, BestEntry>> best;
    std::string line;
    bool first = true;
    while (std::getline(in, line)) {
        if (first) {
            first = false;
            continue;
        }
        if (line.empty()) {
            continue;
        }
        std::stringstream ss(line);
        std::string chunk_str;
        std::string mode;
        std::string threads_str;
        std::string thr_str;
        if (!std::getline(ss, chunk_str, ',')) {
            continue;
        }
        if (!std::getline(ss, mode, ',')) {
            continue;
        }
        if (!std::getline(ss, threads_str, ',')) {
            continue;
        }
        if (!std::getline(ss, thr_str, ',')) {
            continue;
        }

        std::size_t chunk_elements = 0;
        int threads = 0;
        double throughput = 0.0;
        try {
            chunk_elements = static_cast<std::size_t>(std::stoull(chunk_str));
            threads = std::stoi(threads_str);
            throughput = std::stod(thr_str);
        } catch (const std::exception&) {
            continue;
        }
        auto& entry = best[chunk_elements][mode];
        if (throughput > entry.throughput) {
            entry.throughput = throughput;
            entry.threads = threads;
        }
    }

    std::ofstream out(output_csv);
    if (!out.is_open()) {
        std::cerr << "Failed to open output CSV: " << output_csv << "\n";
        return false;
    }

    out << "chunk_elements,adm_decide_threads,compress_threads,decompress_threads\n";
    for (const auto& chunk_pair : best) {
        std::size_t chunk_elements = chunk_pair.first;
        const auto& modes = chunk_pair.second;
        int decide_threads = 0;
        int compress_threads = 0;
        int decompress_threads = 0;

        auto it_decide = modes.find("adm_decide");
        if (it_decide != modes.end()) {
            decide_threads = it_decide->second.threads;
        }
        auto it_comp = modes.find("compress");
        if (it_comp != modes.end()) {
            compress_threads = it_comp->second.threads;
        }
        auto it_decomp = modes.find("decompress");
        if (it_decomp != modes.end()) {
            decompress_threads = it_decomp->second.threads;
        }

        out << chunk_elements << "," << decide_threads << ","
            << compress_threads << "," << decompress_threads << "\n";
    }

    return true;
}

int default_max_threads() {
    int nproc = omp_get_num_procs();
    if (nproc > 0) {
        return nproc;
    }
    unsigned int hw = std::thread::hardware_concurrency();
    if (hw > 0) {
        return static_cast<int>(hw);
    }
    return 1;
}

void print_usage(const char* argv0) {
    std::cout << "Usage: " << argv0
              << " [--chunk-mb-min 0.00390625] [--chunk-mb-max 4]"
              << " [--threads-min 1] [--threads-max NPROC] [--stride 8]"
              << " [--csv thread_sweep.csv] [--out best_threads.csv]\n";
}

} // namespace

int main(int argc, char** argv) {
    double chunk_mb_min = 4.0 / 1024.0; // 4KB
    double chunk_mb_max = 4.0;          // 4MB
    int threads_min = 1;
    int threads_max = default_max_threads();
    int stride = 8;
    std::string csv_path = "thread_sweep.csv";
    std::string out_path = "best_threads.csv";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--chunk-mb-min" && i + 1 < argc) {
            chunk_mb_min = std::stod(argv[++i]);
        } else if (arg == "--chunk-mb-max" && i + 1 < argc) {
            chunk_mb_max = std::stod(argv[++i]);
        } else if (arg == "--threads-min" && i + 1 < argc) {
            threads_min = std::stoi(argv[++i]);
        } else if (arg == "--threads-max" && i + 1 < argc) {
            threads_max = std::stoi(argv[++i]);
        } else if (arg == "--stride" && i + 1 < argc) {
            stride = std::stoi(argv[++i]);
        } else if (arg == "--csv" && i + 1 < argc) {
            csv_path = argv[++i];
        } else if (arg == "--out" && i + 1 < argc) {
            out_path = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    if (chunk_mb_min <= 0.0 || chunk_mb_max <= 0.0 || chunk_mb_min > chunk_mb_max) {
        std::cerr << "Invalid chunk MB range.\n";
        return 1;
    }
    if (threads_min <= 0 || threads_max <= 0 || threads_min > threads_max) {
        std::cerr << "Invalid thread range.\n";
        return 1;
    }
    if (stride <= 0) {
        std::cerr << "Invalid stride.\n";
        return 1;
    }

    std::cout << "Data: 256MB u16 synthetic\n";
    std::cout << "Chunk MB: " << chunk_mb_min << " -> " << chunk_mb_max << " (x2)\n";
    std::cout << "Threads: " << threads_min << " -> " << threads_max
              << " stride=" << stride << "\n";
    std::cout << "CSV: " << csv_path << "\n";
    std::cout << "Best CSV: " << out_path << "\n\n";

    mans::data_gen::SyntheticConfig synth_cfg;
    synth_cfg.size_per_rank_mb = 256.0;
    const std::size_t total_elements = mans::data_gen::aligned_total_elements(
        synth_cfg.size_per_rank_mb, sizeof(std::uint16_t), synth_cfg.block_size);

    std::vector<std::uint16_t> data = mans::data_gen::generate_synthetic_slice<std::uint16_t>(
        kAdmThreshold, synth_cfg, total_elements, 0, total_elements);

    if (data.empty()) {
        std::cerr << "Failed to generate synthetic data.\n";
        return 1;
    }

    std::ofstream csv;
    if (!open_csv_new(csv_path, csv)) {
        std::cerr << "Failed to open CSV: " << csv_path << "\n";
        return 1;
    }

    std::size_t min_bytes = static_cast<std::size_t>(chunk_mb_min * 1024.0 * 1024.0);
    std::size_t max_bytes = static_cast<std::size_t>(chunk_mb_max * 1024.0 * 1024.0);
    if (min_bytes < sizeof(std::uint16_t)) {
        min_bytes = sizeof(std::uint16_t);
    }
    if (max_bytes < min_bytes) {
        max_bytes = min_bytes;
    }

    for (std::size_t chunk_bytes = min_bytes; chunk_bytes <= max_bytes; chunk_bytes *= 2) {
        std::size_t chunk_elements = chunk_bytes / sizeof(std::uint16_t);
        if (chunk_elements == 0) {
            chunk_elements = 1;
        }
        if (chunk_elements > total_elements) {
            chunk_elements = total_elements;
        }

        std::cout << "Chunk elements: " << chunk_elements
                  << " (" << (chunk_elements * sizeof(std::uint16_t)) << " bytes)\n";

        std::vector<ChunkInfo> chunks = build_chunks(total_elements, chunk_elements);
        if (chunks.empty()) {
            std::cerr << "Failed to build chunks.\n";
            return 1;
        }

        int best_decide_threads = sweep_decide_threads(
            data.data(), chunks, total_elements, chunk_elements,
            threads_min, threads_max, stride, &csv);

        sweep_codec_threads(data.data(), chunks, total_elements, chunk_elements,
                            best_decide_threads,
                            threads_min, threads_max, stride, &csv);

        std::cout << "\n";
        if (chunk_bytes > max_bytes / 2) {
            break;
        }
    }

    if (!write_best_threads_csv(csv_path, out_path)) {
        return 1;
    }

    std::cout << "Done. Best threads saved to " << out_path << "\n";
    return 0;
}
