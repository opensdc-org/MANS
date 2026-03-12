#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <omp.h>

#include "../mans_api.hpp"
#include "../mans_data_gen.h"
#include "../mans_defs.h"
#include "mans_cpu.h"

namespace {

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

constexpr std::uint32_t kAdmThreshold = 4000;

void print_progress_line(const std::string& line, bool newline) {
    std::cout << "\r\033[K" << line;
    if (newline) {
        std::cout << "\n";
    }
    std::cout << std::flush;
}

std::vector<int> build_thread_list(int threads_min, int threads_max, int stride) {
    std::vector<int> threads;
    if (stride <= 0 || threads_min <= 0 || threads_max <= 0 || threads_min > threads_max) {
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

bool parse_positive_double_list(const std::string& s,
                                std::vector<double>& out,
                                std::string& error) {
    out.clear();
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) {
            error = "Empty value in --data-size-mb-list.";
            return false;
        }
        std::size_t idx = 0;
        double value = 0.0;
        try {
            value = std::stod(item, &idx);
        } catch (const std::exception&) {
            error = "Invalid value in --data-size-mb-list: " + item;
            return false;
        }
        if (idx != item.size() || value <= 0.0) {
            error = "Invalid value in --data-size-mb-list: " + item;
            return false;
        }
        out.push_back(value);
    }
    if (out.empty()) {
        error = "--data-size-mb-list is empty.";
        return false;
    }
    return true;
}

bool open_csv_new(const std::string& path, std::ofstream& out) {
    out.open(path, std::ios::trunc);
    if (!out) {
        return false;
    }
    out << "chunk_elements,mode,threads,throughput_mbps,dims\n";
    out.flush();
    return true;
}

void append_csv_row(std::ofstream& out,
                    std::size_t chunk_elements,
                    const std::string& mode,
                    int threads,
                    double throughput_mbps,
                    int dims) {
    out << chunk_elements << "," << mode << "," << threads << ","
        << std::fixed << std::setprecision(2) << throughput_mbps << ","
        << dims << "\n";
    out.flush();
}

std::size_t block_elements_for_dims(std::uint32_t dims) {
    if (dims == 1) {
        return 512;
    }
    if (dims == 2) {
        return 256;
    }
    if (dims == 3) {
        return 4096;
    }
    return 512;
}

bool decide_use_adm(const std::uint16_t* data,
                    std::size_t size,
                    std::uint32_t threshold,
                    std::uint32_t threads,
                    const mans::MansParams& params) {
    if (!data || size == 0) {
        return false;
    }

    const std::uint32_t dims = params.dims;
    const int num_threads = threads == 0 ? 16 : static_cast<int>(threads);
    std::uint64_t max_block_diff = 0;

    if (dims <= 1) {
        const std::size_t block_size = block_elements_for_dims(1);
        const std::size_t blocks = (size + block_size - 1) / block_size;
        #pragma omp parallel for num_threads(num_threads) reduction(max:max_block_diff)
        for (long long b = 0; b < static_cast<long long>(blocks); ++b) {
            const std::size_t start = static_cast<std::size_t>(b) * block_size;
            const std::size_t end = std::min(start + block_size, size);
            std::uint16_t bmin = std::numeric_limits<std::uint16_t>::max();
            std::uint16_t bmax = std::numeric_limits<std::uint16_t>::min();
            for (std::size_t i = start; i < end; ++i) {
                const std::uint16_t v = data[i];
                if (v < bmin) bmin = v;
                if (v > bmax) bmax = v;
            }
            const std::uint64_t diff = static_cast<std::uint64_t>(bmax) - static_cast<std::uint64_t>(bmin);
            if (diff > max_block_diff) {
                max_block_diff = diff;
            }
        }
        return max_block_diff <= threshold;
    }

    const std::size_t nx = static_cast<std::size_t>(params.nx);
    const std::size_t ny = (dims >= 2) ? static_cast<std::size_t>(params.ny) : 1;
    const std::size_t nz = (dims == 3) ? static_cast<std::size_t>(params.nz) : 1;
    if (nx == 0 || ny == 0 || nz == 0) {
        return false;
    }

    constexpr std::size_t blk_x = 16;
    constexpr std::size_t blk_y = 16;
    constexpr std::size_t blk_z = 16;

    const std::size_t grid_x = (nx + blk_x - 1) / blk_x;
    const std::size_t grid_y = (ny + blk_y - 1) / blk_y;
    const std::size_t grid_z = (dims == 3) ? ((nz + blk_z - 1) / blk_z) : 1;
    const std::size_t gsize = grid_x * grid_y * grid_z;

    #pragma omp parallel for num_threads(num_threads) reduction(max:max_block_diff)
    for (long long b = 0; b < static_cast<long long>(gsize); ++b) {
        const std::size_t bid = static_cast<std::size_t>(b);
        const std::size_t bx = bid % grid_x;
        const std::size_t t = bid / grid_x;
        const std::size_t by = t % grid_y;
        const std::size_t bz = t / grid_y;

        const std::size_t x0 = bx * blk_x;
        const std::size_t x1 = std::min(x0 + blk_x, nx);
        const std::size_t y0 = by * blk_y;
        const std::size_t y1 = std::min(y0 + blk_y, ny);
        const std::size_t z0 = (dims == 3) ? (bz * blk_z) : 0;
        const std::size_t z1 = (dims == 3) ? std::min(z0 + blk_z, nz) : 1;

        bool has_any = false;
        std::uint16_t bmin = std::numeric_limits<std::uint16_t>::max();
        std::uint16_t bmax = std::numeric_limits<std::uint16_t>::min();

        for (std::size_t z = z0; z < z1; ++z) {
            for (std::size_t y = y0; y < y1; ++y) {
                const std::size_t base = x0 + y * nx + z * nx * ny;
                for (std::size_t x = x0; x < x1; ++x) {
                    const std::size_t idx = base + (x - x0);
                    if (idx >= size) {
                        continue;
                    }
                    const std::uint16_t v = data[idx];
                    if (v < bmin) bmin = v;
                    if (v > bmax) bmax = v;
                    has_any = true;
                }
            }
        }

        if (has_any) {
            const std::uint64_t diff = static_cast<std::uint64_t>(bmax) - static_cast<std::uint64_t>(bmin);
            if (diff > max_block_diff) {
                max_block_diff = diff;
            }
        }
    }

    return max_block_diff <= threshold;
}

template <typename T>
std::size_t max_mans_input_compressed_size(std::size_t num_elements, std::uint32_t mode) {
    mans::MansParams bound_params{};
    bound_params.backend = mans::Backend::CPU;
    bound_params.mode = mode;
    if constexpr (std::is_same_v<T, std::uint16_t>) {
        bound_params.dtype = mans::DataType::U16;
    } else {
        bound_params.dtype = mans::DataType::U32;
    }

    try {
        return mans::get_mans_max_compress_bytes(num_elements, bound_params);
    } catch (const std::exception&) {
        return 0;
    }
}

mans::MansParams default_params(const mans::data_gen::GeneratedDims& shape) {
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

    params.dims = shape.dims;
    params.nx = shape.nx;
    params.ny = shape.ny;
    params.nz = shape.nz;
    return params;
}

ThroughputResult run_compress_decompress(const std::uint16_t* data,
                                         std::size_t total_elements,
                                         const mans::MansParams& params) {
    ThroughputResult result;
    if (total_elements == 0) {
        result.ok = false;
        result.error = "Input is empty.";
        return result;
    }

    const std::size_t max_chunk_bytes = total_elements * sizeof(std::uint16_t);
    const std::size_t max_out_bytes =
        max_mans_input_compressed_size<std::uint16_t>(total_elements, params.mode);
    if (max_out_bytes == 0) {
        result.ok = false;
        result.error = "Input is too large for compressed-size bound.";
        return result;
    }
    std::vector<std::uint8_t> comp_buf(max_out_bytes);
    std::vector<std::uint8_t> decomp_buf(max_chunk_bytes);

    std::size_t out_size = comp_buf.size();
    const auto comp_start = std::chrono::high_resolution_clock::now();
    mans::cpu::compress_internal(
        data,
        total_elements,
        params,
        comp_buf.data(),
        out_size,
        false,
        ""
    );
    const auto comp_end = std::chrono::high_resolution_clock::now();
    const double comp_ms =
        std::chrono::duration<double, std::milli>(comp_end - comp_start).count();

    if (out_size == 0) {
        result.ok = false;
        result.error = "Compression failed (out_size=0).";
        return result;
    }

    std::size_t out_bytes = max_chunk_bytes;
    const auto decomp_start = std::chrono::high_resolution_clock::now();
    mans::cpu::decompress_internal(
        comp_buf.data(),
        out_size,
        params,
        decomp_buf.data(),
        out_bytes,
        false,
        ""
    );
    const auto decomp_end = std::chrono::high_resolution_clock::now();
    const double decomp_ms =
        std::chrono::duration<double, std::milli>(decomp_end - decomp_start).count();

    const std::size_t expected_bytes = total_elements * sizeof(std::uint16_t);
    if (out_bytes != expected_bytes) {
        result.ok = false;
        result.error = "Decompressed size mismatch.";
        return result;
    }
    if (std::memcmp(decomp_buf.data(), data, expected_bytes) != 0) {
        result.ok = false;
        result.error = "Decompressed data mismatch.";
        return result;
    }

    const double total_bytes = static_cast<double>(total_elements * sizeof(std::uint16_t));
    result.comp_mbps = comp_ms > 0.0 ? (total_bytes / 1e6) / (comp_ms / 1e3) : 0.0;
    result.decomp_mbps = decomp_ms > 0.0 ? (total_bytes / 1e6) / (decomp_ms / 1e3) : 0.0;
    return result;
}

double run_decide_throughput(const std::uint16_t* data,
                             std::size_t total_elements,
                             int threads,
                             const mans::MansParams& params) {
    if (!data || total_elements == 0) {
        return 0.0;
    }

    auto start = std::chrono::high_resolution_clock::now();
    (void)decide_use_adm(
        data,
        total_elements,
        kAdmThreshold,
        static_cast<std::uint32_t>(threads),
        params
    );
    auto end = std::chrono::high_resolution_clock::now();

    const double ms = std::chrono::duration<double, std::milli>(end - start).count();
    const double total_bytes = static_cast<double>(total_elements * sizeof(std::uint16_t));
    return ms > 0.0 ? (total_bytes / 1e6) / (ms / 1e3) : 0.0;
}

int sweep_decide_threads(const std::uint16_t* data,
                         std::size_t total_elements,
                         std::size_t chunk_elements,
                         int dims,
                         const mans::MansParams& params,
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
            line << "  [adm_decide][d" << dims << "] threads=" << threads << " ... running";
            print_progress_line(line.str(), false);
        }
        double throughput = run_decide_throughput(data, total_elements, threads, params);
        std::ostringstream line;
        line << "  [adm_decide][d" << dims << "] threads=" << threads << " ... "
             << std::fixed << std::setprecision(2) << throughput << " MB/s";
        print_progress_line(line.str(), true);

        if (csv && *csv) {
            append_csv_row(*csv, chunk_elements, "adm_decide", threads, throughput, dims);
        }
        if (throughput > best_thr) {
            best_thr = throughput;
            best_threads = threads;
        }
    }

    std::cout << "\n  [adm_decide][d" << dims << "] best threads=" << best_threads
              << " throughput=" << std::fixed << std::setprecision(2) << best_thr
              << " MB/s\n";
    return best_threads;
}

void sweep_codec_threads(const std::uint16_t* data,
                         std::size_t total_elements,
                         std::size_t chunk_elements,
                         int dims,
                         const mans::MansParams& base_params,
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
        mans::MansParams params = base_params;
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
            line << "  [codec][d" << dims << "] threads=" << threads << " ... running";
            print_progress_line(line.str(), false);
        }

        ThroughputResult result = run_compress_decompress(data, total_elements, params);
        if (!result.ok) {
            std::cerr << "\n  [codec][d" << dims << "] failed: " << result.error << "\n";
            continue;
        }

        std::ostringstream line;
        line << "  [codec][d" << dims << "] threads=" << threads << " ... "
             << "comp=" << std::fixed << std::setprecision(2) << result.comp_mbps
             << " MB/s, decomp=" << result.decomp_mbps << " MB/s";
        print_progress_line(line.str(), true);

        if (csv && *csv) {
            append_csv_row(*csv, chunk_elements, "compress", threads, result.comp_mbps, dims);
            append_csv_row(*csv, chunk_elements, "decompress", threads, result.decomp_mbps, dims);
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

    std::map<std::pair<std::size_t, int>, std::map<std::string, BestEntry>> best;
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
        std::string size_str;
        std::string mode;
        std::string threads_str;
        std::string thr_str;
        std::string dims_str;
        if (!std::getline(ss, size_str, ',')) {
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
        if (!std::getline(ss, dims_str, ',')) {
            continue;
        }

        std::size_t chunk_elements = 0;
        int threads = 0;
        int dims = 0;
        double throughput = 0.0;
        try {
            chunk_elements = static_cast<std::size_t>(std::stoull(size_str));
            threads = std::stoi(threads_str);
            throughput = std::stod(thr_str);
            dims = std::stoi(dims_str);
        } catch (const std::exception&) {
            continue;
        }
        if (dims < 1 || dims > 3) {
            continue;
        }

        auto& entry = best[std::make_pair(chunk_elements, dims)][mode];
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

    out << "chunk_elements,adm_decide_threads,compress_threads,decompress_threads,dims\n";
    for (const auto& item : best) {
        const std::size_t chunk_elements = item.first.first;
        const int dims = item.first.second;
        const auto& modes = item.second;

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
            << compress_threads << "," << decompress_threads << ","
            << dims << "\n";
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

int run_autotune_for_dims(std::uint32_t dims,
                          const mans::data_gen::SyntheticConfig& synth_cfg,
                          const std::vector<double>& data_size_mb_list,
                          int threads_min,
                          int threads_max,
                          int stride,
                          std::ofstream* csv) {
    for (double data_size_mb : data_size_mb_list) {
        std::size_t data_size_bytes = static_cast<std::size_t>(data_size_mb * 1024.0 * 1024.0);
        if (data_size_bytes < sizeof(std::uint16_t)) {
            data_size_bytes = sizeof(std::uint16_t);
        }
        std::size_t data_elements = data_size_bytes / sizeof(std::uint16_t);
        if (data_elements == 0) {
            data_elements = 1;
        }
        const std::size_t effective_data_size_bytes = data_elements * sizeof(std::uint16_t);

        mans::data_gen::GeneratedDims tune_shape{};
        try {
            tune_shape = mans::data_gen::infer_generated_dims(dims, data_elements);
        } catch (const std::exception& e) {
            std::cerr << "Failed to infer dims for data size " << effective_data_size_bytes
                      << " bytes: " << e.what() << "\n";
            return 1;
        }

        std::vector<std::uint16_t> data;
        try {
            data = mans::data_gen::generate_synthetic_by_dims<std::uint16_t>(
                kAdmThreshold, synth_cfg, tune_shape);
        } catch (const std::exception& e) {
            std::cerr << "Failed to generate synthetic data for dims=" << dims
                      << " size=" << effective_data_size_bytes << " bytes: "
                      << e.what() << "\n";
            return 1;
        }
        if (data.empty()) {
            std::cerr << "Synthetic dataset is empty (elements=0).\n";
            return 1;
        }
        const std::size_t total_elements = data.size();
        const double total_mb =
            static_cast<double>(total_elements * sizeof(std::uint16_t)) / (1024.0 * 1024.0);
        std::cout << "Dims=" << tune_shape.dims
                  << " nx=" << tune_shape.nx
                  << " ny=" << tune_shape.ny
                  << " nz=" << tune_shape.nz
                  << " | data=" << std::fixed << std::setprecision(3) << total_mb << " MB"
                  << " | elements=" << total_elements
                  << "\n";

        const mans::MansParams params_for_size = default_params(tune_shape);

        int best_decide_threads = sweep_decide_threads(
            data.data(),
            total_elements,
            total_elements,
            static_cast<int>(dims),
            params_for_size,
            threads_min,
            threads_max,
            stride,
            csv);

        sweep_codec_threads(
            data.data(),
            total_elements,
            total_elements,
            static_cast<int>(dims),
            params_for_size,
            best_decide_threads,
            threads_min,
            threads_max,
            stride,
            csv);

        std::cout << "\n";
    }

    return 0;
}

void print_usage(const char* argv0) {
    std::cout << "Usage: " << argv0
              << " [--data-size-mb-list 0.00390625,0.0078125,1,4]"
              << " [--threads-min 1] [--threads-max NPROC] [--stride 8]"
              << " [--ratio-smooth 1] [--ratio-spike 0] [--ratio-constant 0] [--ratio-random 0]"
              << " [--noise-range 20] [--seed 42]"
              << " [--csv thread_sweep.csv] [--out best_threads.csv]\n";
}

} // namespace

int main(int argc, char** argv) {
    const std::vector<double> kDefaultDataSizeMbList = {
        4.0 / 1024.0, // 4KB
        8.0 / 1024.0, // 8KB
        16.0 / 1024.0, // 16KB  
        32.0 / 1024.0, // 32KB
        64.0 / 1024.0, // 64KB
        128.0 / 1024.0, // 128KB
        256.0 / 1024.0, // 256KB
        512.0 / 1024.0, // 512KB
        1.0,          // 1MB
        4.0 ,          // 4MB
        256.0 // 256MB
    };
    std::vector<double> data_size_mb_list = kDefaultDataSizeMbList;
    int threads_min = 1;
    int threads_max = default_max_threads();
    int stride = 16;
    std::string csv_path = "thread_sweep.csv";
    std::string out_path = "best_threads.csv";

    mans::data_gen::SyntheticConfig synth_cfg;
    synth_cfg.ratio_smooth = 1.0;
    synth_cfg.ratio_spike = 0.0;
    synth_cfg.ratio_constant = 0.0;
    synth_cfg.ratio_random = 0.0;
    synth_cfg.noise_range = 20;
    synth_cfg.seed = 42;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if ((arg == "--help") || (arg == "-h")) {
            print_usage(argv[0]);
            return 0;
        }
        if ((arg == "--data-size-mb-list" || arg == "--chunk-mb-list") && i + 1 < argc) {
            std::string error;
            if (!parse_positive_double_list(argv[++i], data_size_mb_list, error)) {
                std::cerr << error << "\n";
                return 1;
            }
            continue;
        }
        if (arg == "--threads-min" && i + 1 < argc) {
            threads_min = std::stoi(argv[++i]);
            continue;
        }
        if (arg == "--threads-max" && i + 1 < argc) {
            threads_max = std::stoi(argv[++i]);
            continue;
        }
        if (arg == "--stride" && i + 1 < argc) {
            stride = std::stoi(argv[++i]);
            continue;
        }
        if (arg == "--ratio-smooth" && i + 1 < argc) {
            synth_cfg.ratio_smooth = std::stod(argv[++i]);
            continue;
        }
        if (arg == "--ratio-spike" && i + 1 < argc) {
            synth_cfg.ratio_spike = std::stod(argv[++i]);
            continue;
        }
        if (arg == "--ratio-constant" && i + 1 < argc) {
            synth_cfg.ratio_constant = std::stod(argv[++i]);
            continue;
        }
        if (arg == "--ratio-random" && i + 1 < argc) {
            synth_cfg.ratio_random = std::stod(argv[++i]);
            continue;
        }
        if (arg == "--noise-range" && i + 1 < argc) {
            synth_cfg.noise_range = std::stoi(argv[++i]);
            continue;
        }
        if (arg == "--seed" && i + 1 < argc) {
            synth_cfg.seed = static_cast<std::uint64_t>(std::stoull(argv[++i]));
            continue;
        }
        if (arg == "--csv" && i + 1 < argc) {
            csv_path = argv[++i];
            continue;
        }
        if (arg == "--out" && i + 1 < argc) {
            out_path = argv[++i];
            continue;
        }

        std::cerr << "Unknown argument: " << arg << "\n";
        print_usage(argv[0]);
        return 1;
    }

    if (data_size_mb_list.empty()) {
        std::cerr << "Invalid data size range.\n";
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

    const double ratio_sum =
        std::max(0.0, synth_cfg.ratio_smooth) +
        std::max(0.0, synth_cfg.ratio_spike) +
        std::max(0.0, synth_cfg.ratio_constant) +
        std::max(0.0, synth_cfg.ratio_random);
    if (ratio_sum <= 0.0) {
        std::cerr << "Invalid block ratios: sum must be > 0.\n";
        return 1;
    }

    std::cout << "Autotune dtype: u16\n";
    std::cout << "Data size list (MB): ";
    for (std::size_t i = 0; i < data_size_mb_list.size(); ++i) {
        if (i > 0) {
            std::cout << ",";
        }
        std::cout << data_size_mb_list[i];
    }
    std::cout << "\n";
    std::cout << "Threads: " << threads_min << " -> " << threads_max
              << " stride=" << stride << "\n";
    std::cout << "Ratios: smooth=" << synth_cfg.ratio_smooth
              << " spike=" << synth_cfg.ratio_spike
              << " constant=" << synth_cfg.ratio_constant
              << " random=" << synth_cfg.ratio_random << "\n";
    std::cout << "CSV: " << csv_path << "\n";
    std::cout << "Best CSV: " << out_path << "\n\n";

    std::ofstream csv;
    if (!open_csv_new(csv_path, csv)) {
        std::cerr << "Failed to open CSV: " << csv_path << "\n";
        return 1;
    }

    for (std::uint32_t dims = 1; dims <= 3; ++dims) {
        const int rc = run_autotune_for_dims(
            dims,
            synth_cfg,
            data_size_mb_list,
            threads_min,
            threads_max,
            stride,
            &csv);
        if (rc != 0) {
            return rc;
        }
    }

    if (!write_best_threads_csv(csv_path, out_path)) {
        return 1;
    }

    std::cout << "Done. Best threads saved to " << out_path << "\n";
    return 0;
}
