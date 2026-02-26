#include "mans_cpu.h"
#include "file_utils.h"
#include "../mans_api.hpp"
#include "../mans_defs.h"
#include "../mans_timing.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace {

struct ChunkInfo {
    std::size_t offset = 0;
    std::size_t len = 0;
};

struct RunStats {
    double comp_ms = 0.0;
    double decomp_ms = 0.0;
    std::size_t comp_bytes = 0;
    bool ok = true;
    std::string error;
};

std::vector<double> parse_chunks(const std::string& arg) {
    std::vector<double> chunks;
    std::stringstream ss(arg);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) {
            continue;
        }
        chunks.push_back(std::stod(item));
    }
    return chunks;
}

bool parse_threads(const std::string& arg, std::vector<int>& threads, std::string& error) {
    threads.clear();
    std::stringstream ss(arg);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) {
            error = "Empty thread entry in --threads list.";
            return false;
        }
        std::size_t idx = 0;
        int value = 0;
        try {
            value = std::stoi(item, &idx);
        } catch (const std::exception&) {
            error = "Invalid thread value in --threads list: " + item;
            return false;
        }
        if (idx != item.size()) {
            error = "Invalid thread value in --threads list: " + item;
            return false;
        }
        if (value <= 0) {
            error = "Thread values must be positive integers.";
            return false;
        }
        threads.push_back(value);
    }
    return true;
}

void apply_thread_overrides(mans::MansParams& params, const std::vector<int>& threads) {
    params.adm_decide_threads = threads[0];
    params.adm_center_calc_threads = threads[1];
    params.adm_encode_threads = threads[2];
    params.adm_warp_reduce_threads = threads[3];
    params.adm_fill_tail_threads = threads[4];
    params.adm_write_back_threads = threads[5];
    params.adm_restore_signals_threads = threads[6];
    params.adm_decode_values_threads = threads[7];
}

void apply_auto_thread_config(mans::MansParams& params, const mans::cpu::CsvThreadConfig& cfg) {
    params.adm_decide_threads = cfg.adm_decide_threads;
    params.adm_center_calc_threads = cfg.compress_threads;
    params.adm_encode_threads = cfg.compress_threads;
    params.adm_warp_reduce_threads = cfg.compress_threads;
    params.adm_fill_tail_threads = cfg.compress_threads;
    params.adm_write_back_threads = cfg.compress_threads;
    params.adm_restore_signals_threads = cfg.decompress_threads;
    params.adm_decode_values_threads = cfg.decompress_threads;
}

std::string format_chunk_label(std::size_t bytes) {
    const std::size_t kib = 1024;
    const std::size_t mib = 1024 * 1024;
    std::ostringstream out;
    if (bytes % mib == 0) {
        out << (bytes / mib) << "M";
        return out.str();
    }
    if (bytes % kib == 0) {
        out << (bytes / kib) << "K";
        return out.str();
    }
    out << bytes << "B";
    return out.str();
}

template <typename T>
std::vector<ChunkInfo> build_chunks(std::size_t total_elements, std::size_t chunk_elements) {
    std::vector<ChunkInfo> chunks;
    std::size_t offset = 0;
    while (offset < total_elements) {
        std::size_t len = std::min(chunk_elements, total_elements - offset);
        chunks.push_back(ChunkInfo{offset, len});
        offset += len;
    }
    return chunks;
}

template <typename T>
std::size_t max_mans_chunk_compressed_size(std::size_t chunk_elements, std::uint32_t mode) {
    mans::MansParams bound_params{};
    bound_params.backend = mans::Backend::CPU;
    bound_params.mode = (mode == mans::Mode::R) ? mans::Mode::R : mans::Mode::P;
    if constexpr (std::is_same_v<T, std::uint16_t>) {
        bound_params.dtype = mans::DataType::U16;
    } else if constexpr (std::is_same_v<T, std::uint32_t>) {
        bound_params.dtype = mans::DataType::U32;
    } else {
        return 0;
    }

    try {
        return mans::get_mans_max_compress_bytes(chunk_elements, bound_params);
    } catch (const std::exception&) {
        return 0;
    }
}

template <typename T>
RunStats run_once(const std::vector<T>& input,
                  const std::vector<ChunkInfo>& chunks,
                  const std::vector<std::size_t>& chunk_caps,
                  std::vector<std::uint8_t>& compressed_chunk,
                  std::vector<std::uint8_t>& recovered,
                  const mans::MansParams& params) {
    RunStats stats;

    for (std::size_t i = 0; i < chunks.size(); ++i) {
        const auto& chunk = chunks[i];
        if (chunk_caps[i] > compressed_chunk.size()) {
            stats.ok = false;
            stats.error = "Internal compressed buffer too small at chunk index " +
                          std::to_string(i);
            return stats;
        }

        std::size_t compressed_size = chunk_caps[i];
        {
            MANS_TIMING_SCOPE("mans/compress_total");
            auto comp_start = std::chrono::high_resolution_clock::now();
            mans::cpu::compress_internal(
                input.data() + chunk.offset,
                chunk.len,
                params,
                compressed_chunk.data(),
                compressed_size,
                false,
                "");
            auto comp_end = std::chrono::high_resolution_clock::now();
            stats.comp_ms +=
                std::chrono::duration<double, std::milli>(comp_end - comp_start).count();
        }

        if (compressed_size == 0 || compressed_size > chunk_caps[i]) {
            stats.ok = false;
            stats.error = "Compression failed at chunk index " + std::to_string(i);
            return stats;
        }
        stats.comp_bytes += compressed_size;

        const std::size_t expected_bytes = chunk.len * sizeof(T);
        std::size_t recovered_size = expected_bytes;
        {
            MANS_TIMING_SCOPE("mans/decompress_total");
            auto decomp_start = std::chrono::high_resolution_clock::now();
            mans::cpu::decompress_internal(
                compressed_chunk.data(),
                compressed_size,
                params,
                recovered.data(),
                recovered_size,
                false,
                "");
            auto decomp_end = std::chrono::high_resolution_clock::now();
            stats.decomp_ms +=
                std::chrono::duration<double, std::milli>(decomp_end - decomp_start).count();
        }

        if (recovered_size != expected_bytes ||
            std::memcmp(recovered.data(), input.data() + chunk.offset, expected_bytes) != 0) {
            stats.ok = false;
            stats.error = "Decompression mismatch at chunk index " + std::to_string(i);
            return stats;
        }
    }

    return stats;
}

template <typename T>
int run_bench_for_type(const std::vector<T>& input,
                       const std::vector<double>& chunks_mb,
                       mans::MansParams& params,
                       std::ofstream* csv,
                       const std::string& timing_csv_base,
                       bool timing_per_chunk,
                       const std::vector<mans::cpu::CsvThreadConfig>* auto_threads) {
    if (input.empty()) {
        std::cerr << "Input file is empty.\n";
        return 1;
    }

    const std::size_t total_elements = input.size();
    const std::size_t total_bytes = total_elements * sizeof(T);

    for (double chunk_mb : chunks_mb) {
        if (chunk_mb < 0.0) {
            std::cerr << "Skipping invalid chunk size: " << chunk_mb << "\n";
            continue;
        }

        std::size_t chunk_elements = 0;
        if (chunk_mb == 0.0) {
            chunk_elements = total_elements;
        } else {
            chunk_elements = static_cast<std::size_t>(
                (chunk_mb * 1024.0 * 1024.0) / sizeof(T));
            if (chunk_elements == 0) {
                chunk_elements = 1;
            }
        }

        if (auto_threads && !auto_threads->empty()) {
            mans::cpu::CsvThreadConfig chosen{};
            if (mans::cpu::find_nearest_threads(*auto_threads, chunk_elements, chosen)) {
                apply_auto_thread_config(params, chosen);
            } else {
                std::cerr << "No matching thread config found for chunk elements: "
                          << chunk_elements << "\n";
            }
        }

        std::size_t chunk_bytes = chunk_elements * sizeof(T);
        std::string label = format_chunk_label(chunk_bytes);

        std::string timing_path;
        if (!timing_csv_base.empty()) {
            if (timing_per_chunk) {
                timing_path = timing_csv_base + "." + label + ".timing.csv";
            } else {
                timing_path = timing_csv_base;
            }
        }

        std::vector<ChunkInfo> chunks = build_chunks<T>(total_elements, chunk_elements);
        if (chunks.empty()) {
            std::cerr << "No chunks generated for chunk size: " << chunk_mb << " MB\n";
            continue;
        }

        std::vector<std::size_t> chunk_caps(chunks.size(), 0);
        std::size_t max_comp_bytes = 0;
        std::size_t max_chunk_len = 0;
        for (std::size_t i = 0; i < chunks.size(); ++i) {
            const std::size_t cap = max_mans_chunk_compressed_size<T>(chunks[i].len, params.mode);
            if (cap == 0) {
                std::cerr << "Chunk too large for 32-bit PANS input bound, chunk elements: "
                          << chunks[i].len << "\n";
                return 1;
            }
            chunk_caps[i] = cap;
            max_comp_bytes = std::max(max_comp_bytes, cap);
            max_chunk_len = std::max(max_chunk_len, chunks[i].len);
        }

        std::vector<std::uint8_t> compressed_chunk(max_comp_bytes);
        std::vector<std::uint8_t> recovered(max_chunk_len * sizeof(T));

        constexpr int kIters = 11;
        double total_comp_ms = 0.0;
        double total_decomp_ms = 0.0;
        double total_comp_bytes = 0.0;

        MANS_TIMING_RESET();
        for (int iter = 0; iter < kIters; ++iter) {
            MANS_TIMING_RUN_SCOPE();
            RunStats stats = run_once<T>(input, chunks, chunk_caps,
                                         compressed_chunk, recovered, params);
            if (!stats.ok) {
                std::cerr << stats.error << " for chunk " << label << "\n";
                return 1;
            }
            if (iter == 0) {
                continue;
            }
            total_comp_ms += stats.comp_ms;
            total_decomp_ms += stats.decomp_ms;
            total_comp_bytes += static_cast<double>(stats.comp_bytes);
        }

        const double denom = static_cast<double>(kIters - 1);
        const double avg_comp_ms = total_comp_ms / denom;
        const double avg_decomp_ms = total_decomp_ms / denom;
        const double avg_comp_bytes = total_comp_bytes / denom;

        const double ratio = 100.0 * avg_comp_bytes / static_cast<double>(total_bytes);
        const double comp_mbps =
            (static_cast<double>(total_bytes) / 1e6) / (avg_comp_ms / 1e3);
        const double decomp_mbps =
            (static_cast<double>(total_bytes) / 1e6) / (avg_decomp_ms / 1e3);

        std::cout << std::left << std::setw(8) << label
                  << " | " << std::setw(8) << std::fixed << std::setprecision(2)
                  << ratio << "%"
                  << " | " << std::setw(13) << std::fixed << std::setprecision(1)
                  << comp_mbps
                  << " | " << std::setw(13) << std::fixed << std::setprecision(1)
                  << decomp_mbps
                  << "\n";

        if (!timing_path.empty()) {
            MANS_TIMING_DUMP(timing_path);
        }

        if (csv && *csv) {
            (*csv) << label << ","
                   << chunk_bytes << ","
                   << std::fixed << std::setprecision(2) << ratio << ","
                   << std::fixed << std::setprecision(1) << comp_mbps << ","
                   << std::fixed << std::setprecision(1) << decomp_mbps << "\n";
        }
    }

    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <-u2|-u4> <input.bin> [--chunks 0,0.125,0.25,0.5,1,2,8,256]"
                  << " [--mode p|r]"
                  << " [--threshold 4000] [--threads 16,32,32,32,32,32,16,16]"
                  << " [--csv out.csv]"
                  << "\n";
        return 1;
    }

    std::string input_type = argv[1];
    std::string input_path = argv[2];
    std::string chunks_arg = "0.125,0.25,0.5,1,2,8,256";
    std::string csv_path;
    std::string threads_arg;
    bool has_threads = false;
    std::uint32_t mode = mans::Mode::P;
    uint32_t threshold = 4000;

    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--chunks" && i + 1 < argc) {
            chunks_arg = argv[++i];
        } else if (arg == "--mode" && i + 1 < argc) {
            std::string mode_arg = argv[++i];
            if (mode_arg == "p" || mode_arg == "P") {
                mode = mans::Mode::P;
            } else if (mode_arg == "r" || mode_arg == "R") {
                mode = mans::Mode::R;
            } else {
                std::cerr << "Unknown mode: " << mode_arg << " (use p or r)\n";
                return 1;
            }
        } else if (arg == "--threads" && i + 1 < argc) {
            threads_arg = argv[++i];
            has_threads = true;
        } else if (arg == "--csv" && i + 1 < argc) {
            csv_path = argv[++i];
        } else if (arg == "--threshold" && i + 1 < argc) {
            threshold = static_cast<uint32_t>(std::stoul(argv[++i]));
        }
    }

    std::cout << "Command-line arguments:\n";
    std::cout << "  Input type: " << input_type << "\n";
    std::cout << "  Input file: " << input_path << "\n";
    std::cout << "  Chunks (MB): " << chunks_arg << "\n";
    std::cout << "  Mode: " << (mode == mans::Mode::R ? "r" : "p") << "\n";
    std::cout << "  Threshold: " << threshold << "\n";
    if (has_threads) {
        std::cout << "  Threads: " << threads_arg << "\n";
    }
    if (!csv_path.empty()) {
        std::cout << "  Timing CSV: " << csv_path << "\n";
    } else {
        std::cout << "  Timing CSV: mans_timing.csv\n";
    }
    std::cout << "\n";

    const bool is_u2 = (input_type == "-u2" || input_type == "u2");
    const bool is_u4 = (input_type == "-u4" || input_type == "u4");
    if (!is_u2 && !is_u4) {
        std::cerr << "Unknown data type flag: " << input_type
                  << "\nUse: -u2 or -u4\n";
        return 1;
    }

    std::vector<double> chunks_mb = parse_chunks(chunks_arg);
    if (chunks_mb.empty()) {
        std::cerr << "No chunk sizes parsed from --chunks.\n";
        return 1;
    }

    std::vector<int> thread_list;
    if (has_threads) {
        std::string error;
        if (!parse_threads(threads_arg, thread_list, error)) {
            std::cerr << error << "\n";
            return 1;
        }
        if (thread_list.size() != 8) {
            std::cerr << "--threads expects 8 values: decide,center,encode,warp_reduce,"
                      << "fill_tail,write_back,restore_signals,decode_values\n";
            return 1;
        }
    }

    std::cout << "Chunks (MB): " << chunks_arg << "\n\n";

    std::ofstream csv;
    std::string timing_csv_base = "mans_timing.csv";
    bool timing_per_chunk = false;
    if (!csv_path.empty()) {
        csv.open(csv_path);
        if (!csv) {
            std::cerr << "Failed to open CSV output: " << csv_path << "\n";
            return 1;
        }
        csv << "chunk_label,chunk_bytes,ratio_pct,comp_mbps,decomp_mbps\n";
        timing_csv_base = csv_path;
        timing_per_chunk = true;
    }

    std::cout << std::left << std::setw(8) << "Chunk"
              << " | " << std::setw(9) << "Ratio"
              << " | " << std::setw(13) << "Comp MB/s"
              << " | " << std::setw(13) << "Decomp MB/s"
              << "\n";
    std::cout << std::string(52, '-') << "\n";

    if (is_u2) {
        std::vector<std::uint16_t> input;
        if (!load_u16_file(input_path, input)) {
            std::cerr << "Failed to load input file: " << input_path << "\n";
            return 1;
        }

        mans::MansParams params{};
        params.backend = mans::Backend::CPU;
        params.dtype = mans::DataType::U16;
        params.mode = mode;
        params.adm_threshold = threshold;
        if (has_threads) {
            apply_thread_overrides(params, thread_list);
        }
        const char* csv_env = std::getenv("MANS_THREAD_CSV");
        std::string auto_thread_csv =
            (csv_env && csv_env[0] != '\0') ? csv_env : "best_threads.csv";
        std::vector<mans::cpu::CsvThreadConfig> auto_thread_configs;
        bool use_auto_threads = false;
        if (!has_threads) {
            std::string error;
            if (!mans::cpu::load_thread_csv(auto_thread_csv, auto_thread_configs, error)) {
                std::cerr << "Auto thread config disabled: " << error << "\n";
            } else {
                use_auto_threads = true;
            }
        }
        return run_bench_for_type<std::uint16_t>(input, chunks_mb, params, &csv,
                                                 timing_csv_base, timing_per_chunk,
                                                 use_auto_threads ? &auto_thread_configs
                                                                  : nullptr);
    } else {
        std::vector<std::uint32_t> input;
        if (!load_u32_file(input_path, input)) {
            std::cerr << "Failed to load input file: " << input_path << "\n";
            return 1;
        }

        mans::MansParams params{};
        params.backend = mans::Backend::CPU;
        params.dtype = mans::DataType::U32;
        params.mode = mode;
        params.adm_threshold = threshold;
        if (has_threads) {
            apply_thread_overrides(params, thread_list);
        }
        const char* csv_env = std::getenv("MANS_THREAD_CSV");
        std::string auto_thread_csv =
            (csv_env && csv_env[0] != '\0') ? csv_env : "best_threads.csv";
        std::vector<mans::cpu::CsvThreadConfig> auto_thread_configs;
        bool use_auto_threads = false;
        if (!has_threads) {
            std::string error;
            if (!mans::cpu::load_thread_csv(auto_thread_csv, auto_thread_configs, error)) {
                std::cerr << "Auto thread config disabled: " << error << "\n";
            } else {
                use_auto_threads = true;
            }
        }
        return run_bench_for_type<std::uint32_t>(input, chunks_mb, params, &csv,
                                                 timing_csv_base, timing_per_chunk,
                                                 use_auto_threads ? &auto_thread_configs
                                                                  : nullptr);
    }
}
