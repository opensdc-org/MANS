#include "mans_cpu.h"
#include "file_utils.h"
#include "../mans_api.hpp"
#include "../mans_defs.h"
#include "../mans_timing.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace {

struct RunStats {
    double comp_ms = 0.0;
    double decomp_ms = 0.0;
    double comp_should_use_adm_ms = 0.0;
    double comp_adm_core_ms = 0.0;
    double comp_entropy_core_ms = 0.0;
    double decomp_entropy_core_ms = 0.0;
    double decomp_adm_core_ms = 0.0;
    std::size_t comp_bytes = 0;
    bool ok = true;
    std::string error;
};

double last_run_sum_ms(std::initializer_list<const char*> labels) {
#ifdef ENABLE_TIMING
    return mans::TimingCollector::instance().last_run_sum_ms(labels);
#else
    (void)labels;
    return 0.0;
#endif
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

bool parse_positive_u32(const std::string& s, std::uint32_t& out, std::string& error) {
    std::size_t idx = 0;
    unsigned long long v = 0;
    try {
        v = std::stoull(s, &idx);
    } catch (const std::exception&) {
        error = "Invalid positive integer: " + s;
        return false;
    }
    if (idx != s.size() || v == 0 || v > std::numeric_limits<std::uint32_t>::max()) {
        error = "Invalid positive integer: " + s;
        return false;
    }
    out = static_cast<std::uint32_t>(v);
    return true;
}

bool parse_dims_from_argv(int argc, char** argv, int& i,
                          std::vector<std::uint32_t>& dims,
                          std::string& error) {
    if (i + 1 >= argc) {
        error = "--dims requires dimension count.";
        return false;
    }

    std::uint32_t dims_count = 0;
    if (!parse_positive_u32(argv[++i], dims_count, error)) {
        error = "Invalid --dims dimension count: " + std::string(argv[i]);
        return false;
    }
    if (dims_count < 1 || dims_count > 3) {
        error = "--dims dimension count must be 1, 2, or 3.";
        return false;
    }
    if (i + static_cast<int>(dims_count) >= argc) {
        error = "--dims missing dimension values.";
        return false;
    }

    dims.clear();
    dims.reserve(dims_count);
    for (std::uint32_t d = 0; d < dims_count; ++d) {
        std::uint32_t val = 0;
        if (!parse_positive_u32(argv[++i], val, error)) {
            error = "Invalid --dims value: " + std::string(argv[i]);
            return false;
        }
        dims.push_back(val);
    }
    return true;
}

bool validate_dims_product(const std::vector<std::uint32_t>& dims,
                           std::size_t input_elements,
                           std::string& error) {
    if (dims.empty()) {
        error = "--dims is required.";
        return false;
    }

    std::size_t expected = 1;
    for (const std::uint32_t d : dims) {
        if (d == 0) {
            error = "All --dims values must be positive.";
            return false;
        }
        if (expected > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(d)) {
            error = "Dimension product overflows size_t.";
            return false;
        }
        expected *= static_cast<std::size_t>(d);
    }

    if (expected != input_elements) {
        error = "Input element count (" + std::to_string(input_elements) +
                ") does not match dims product (" + std::to_string(expected) + ").";
        return false;
    }
    return true;
}

void apply_dims_override(mans::MansParams& params, const std::vector<std::uint32_t>& dims) {
    if (dims.empty()) {
        return;
    }
    params.dims = static_cast<std::uint32_t>(dims.size());
    params.nx = dims[0];
    params.ny = (dims.size() >= 2) ? dims[1] : 0;
    params.nz = (dims.size() >= 3) ? dims[2] : 0;
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

template <typename T>
std::size_t max_mans_input_compressed_size(std::size_t num_elements, std::uint32_t mode) {
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
        return mans::get_mans_max_compress_bytes(num_elements, bound_params);
    } catch (const std::exception&) {
        return 0;
    }
}

template <typename T>
RunStats run_once(const std::vector<T>& input,
                  std::vector<std::uint8_t>& compressed_input,
                  std::vector<std::uint8_t>& recovered,
                  const mans::MansParams& params) {
    RunStats stats{};

    std::size_t compressed_size = compressed_input.size();
    mans::cpu::compress_internal(
        input.data(),
        input.size(),
        params,
        compressed_input.data(),
        compressed_size,
        false,
        "");

    if (compressed_size == 0 || compressed_size > compressed_input.size()) {
        stats.ok = false;
        stats.error = "Compression failed.";
        return stats;
    }
    stats.comp_bytes = compressed_size;

    const std::size_t expected_bytes = input.size() * sizeof(T);
    std::size_t recovered_size = expected_bytes;
    mans::cpu::decompress_internal(
        compressed_input.data(),
        compressed_size,
        params,
        recovered.data(),
        recovered_size,
        false,
        "");

    if (recovered_size != expected_bytes ||
        std::memcmp(recovered.data(), input.data(), expected_bytes) != 0) {
        stats.ok = false;
        stats.error = "Decompression mismatch.";
        return stats;
    }

    return stats;
}

template <typename T>
int run_bench_for_type(const std::vector<T>& input,
                       mans::MansParams& params,
                       std::ofstream* csv,
                       const std::string& timing_csv_path,
                       const std::vector<mans::cpu::CsvThreadConfig>* auto_threads,
                       std::uint32_t warmup_iters,
                       std::uint32_t bench_iters) {
    if (input.empty()) {
        std::cerr << "Input file is empty.\n";
        return 1;
    }

    const std::size_t total_elements = input.size();
    const std::size_t total_bytes = total_elements * sizeof(T);

    if (auto_threads && !auto_threads->empty()) {
        mans::cpu::CsvThreadConfig chosen{};
        if (mans::cpu::find_nearest_threads(
                *auto_threads,
                total_elements,
                static_cast<uint32_t>(params.dims),
                chosen)) {
            apply_auto_thread_config(params, chosen);
        } else {
            std::cerr << "No matching thread config found for input elements: "
                      << total_elements << ", dims: " << params.dims << "\n";
        }
    }

    const std::size_t max_comp_bytes = max_mans_input_compressed_size<T>(total_elements, params.mode);
    if (max_comp_bytes == 0) {
        std::cerr << "Input too large for 32-bit PANS input bound, elements: "
                  << total_elements << "\n";
        return 1;
    }

    std::vector<std::uint8_t> compressed_input(max_comp_bytes);
    std::vector<std::uint8_t> recovered(total_bytes);

    const std::uint64_t total_runs =
        static_cast<std::uint64_t>(warmup_iters) + static_cast<std::uint64_t>(bench_iters);
    double total_comp_ms = 0.0;
    double total_decomp_ms = 0.0;
    double total_comp_bytes = 0.0;

    MANS_TIMING_RESET();
    for (std::uint64_t iter = 0; iter < total_runs; ++iter) {
        RunStats stats{};
        {
            MANS_TIMING_RUN_SCOPE();
            stats = run_once<T>(input, compressed_input, recovered, params);
        }
        if (!stats.ok) {
            std::cerr << stats.error << "\n";
            return 1;
        }

        stats.comp_should_use_adm_ms = last_run_sum_ms({"mans/should_use_adm"});
        stats.comp_adm_core_ms = last_run_sum_ms({"mans/adm_encode_core"});
        stats.comp_entropy_core_ms = last_run_sum_ms({"mans/entropy_encode_core"});
        stats.decomp_entropy_core_ms = last_run_sum_ms({"mans/entropy_decode_core"});
        stats.decomp_adm_core_ms = last_run_sum_ms({"mans/adm_decode_core"});
        stats.comp_ms = stats.comp_should_use_adm_ms +
                        stats.comp_adm_core_ms +
                        stats.comp_entropy_core_ms;
        stats.decomp_ms = stats.decomp_entropy_core_ms +
                          stats.decomp_adm_core_ms;

        if (iter < warmup_iters) {
            continue;
        }
        total_comp_ms += stats.comp_ms;
        total_decomp_ms += stats.decomp_ms;
        total_comp_bytes += static_cast<double>(stats.comp_bytes);
    }

    const double denom = static_cast<double>(bench_iters);
    const double avg_comp_ms = total_comp_ms / denom;
    const double avg_decomp_ms = total_decomp_ms / denom;
    const double avg_comp_bytes = total_comp_bytes / denom;

    // const double ratio = 100.0 * avg_comp_bytes / static_cast<double>(total_bytes);
    const double ratio = static_cast<double>(total_bytes) / avg_comp_bytes;
    const double comp_mbps =
        (static_cast<double>(total_bytes) / 1e6) / (avg_comp_ms / 1e3);
    const double decomp_mbps =
        (static_cast<double>(total_bytes) / 1e6) / (avg_decomp_ms / 1e3);

    std::cout << std::left << std::setw(8) << "whole"
              << " | " << std::setw(8) << std::fixed << std::setprecision(8)
              << ratio 
              << " | " << std::setw(13) << std::fixed << std::setprecision(1)
              << comp_mbps
              << " | " << std::setw(13) << std::fixed << std::setprecision(1)
              << decomp_mbps
              << "\n";

    if (!timing_csv_path.empty()) {
        MANS_TIMING_DUMP(timing_csv_path);
    }

    if (csv && *csv) {
        (*csv) << "whole,"
               << total_bytes << ","
               << std::fixed << std::setprecision(2) << ratio << ","
               << std::fixed << std::setprecision(1) << comp_mbps << ","
               << std::fixed << std::setprecision(1) << decomp_mbps << "\n";
    }

    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <-u2|-u4> <input.bin>"
                  << " [--mode p|r]"
                  << " [--threshold 4000] [--threads 16,32,32,32,32,32,16,16]"
                  << " [--warmup 5] [--runs 10]"
                  << " --dims D d1 [d2] [d3]"
                  << " [--csv out.csv]"
                  << "\n";
        return 1;
    }

    std::string input_type = argv[1];
    std::string input_path = argv[2];
    std::string csv_path;
    std::string threads_arg;
    std::vector<std::uint32_t> dims_override;
    bool has_threads = false;
    std::uint32_t mode = mans::Mode::P;
    uint32_t threshold = 4000;
    std::uint32_t warmup_iters = 5;
    std::uint32_t bench_iters = 10;

    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--chunks" && i + 1 < argc) {
            std::cerr << "--chunks is no longer supported. Whole-input mode only.\n";
            return 1;
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
        } else if (arg == "--warmup" && i + 1 < argc) {
            std::string error;
            if (!parse_positive_u32(argv[++i], warmup_iters, error)) {
                std::cerr << "Invalid --warmup value: " << argv[i] << "\n";
                return 1;
            }
        } else if ((arg == "--runs" || arg == "--iters") && i + 1 < argc) {
            std::string error;
            if (!parse_positive_u32(argv[++i], bench_iters, error)) {
                std::cerr << "Invalid --runs/--iters value: " << argv[i] << "\n";
                return 1;
            }
        } else if (arg == "--dims") {
            std::string error;
            if (!parse_dims_from_argv(argc, argv, i, dims_override, error)) {
                std::cerr << error << "\n";
                return 1;
            }
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return 1;
        }
    }
    if (dims_override.empty()) {
        std::cerr << "--dims is required.\n";
        return 1;
    }

    std::cout << "Command-line arguments:\n";
    std::cout << "  Input type: " << input_type << "\n";
    std::cout << "  Input file: " << input_path << "\n";
    std::cout << "  Mode: " << (mode == mans::Mode::R ? "r" : "p") << "\n";
    std::cout << "  Threshold: " << threshold << "\n";
    std::cout << "  Warmup iterations: " << warmup_iters << "\n";
    std::cout << "  Benchmark iterations: " << bench_iters << "\n";
    std::cout << "  Dims: " << dims_override.size();
    for (const std::uint32_t d : dims_override) {
        std::cout << " " << d;
    }
    std::cout << "\n";
    if (has_threads) {
        std::cout << "  Threads: " << threads_arg << "\n";
    }
    if (!csv_path.empty()) {
        std::cout << "  Timing CSV: " << csv_path << ".timing.csv\n";
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

    std::ofstream csv;
    std::string timing_csv_path = "mans_timing.csv";
    if (!csv_path.empty()) {
        csv.open(csv_path);
        if (!csv) {
            std::cerr << "Failed to open CSV output: " << csv_path << "\n";
            return 1;
        }
        csv << "input_label,input_bytes,ratio_pct,comp_mbps,decomp_mbps\n";
        timing_csv_path = csv_path + ".timing.csv";
    }

    std::cout << std::left << std::setw(8) << "Input"
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
        std::string dims_error;
        if (!validate_dims_product(dims_override, input.size(), dims_error)) {
            std::cerr << dims_error << "\n";
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
        apply_dims_override(params, dims_override);
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
        return run_bench_for_type<std::uint16_t>(input, params, &csv,
                                                 timing_csv_path,
                                                 use_auto_threads ? &auto_thread_configs
                                                                  : nullptr,
                                                 warmup_iters,
                                                 bench_iters);
    } else {
        std::vector<std::uint32_t> input;
        if (!load_u32_file(input_path, input)) {
            std::cerr << "Failed to load input file: " << input_path << "\n";
            return 1;
        }
        std::string dims_error;
        if (!validate_dims_product(dims_override, input.size(), dims_error)) {
            std::cerr << dims_error << "\n";
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
        apply_dims_override(params, dims_override);
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
        return run_bench_for_type<std::uint32_t>(input, params, &csv,
                                                 timing_csv_path,
                                                 use_auto_threads ? &auto_thread_configs
                                                                  : nullptr,
                                                 warmup_iters,
                                                 bench_iters);
    }
}
