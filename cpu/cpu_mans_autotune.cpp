#include "../mans_api.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

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

void print_usage(const char* argv0) {
    std::cout << "Usage: " << argv0
              << " [--data-size-mb-list 0.00390625,0.0078125,1,4]"
              << " [--threads-min 1] [--threads-max NPROC] [--stride 8]"
              << " [--ratio-smooth 1] [--ratio-spike 0] [--ratio-constant 0] [--ratio-random 0]"
              << " [--iter 10] [--noise-range 20] [--seed 42]"
              << " [--csv thread_sweep.csv] [--out best_threads.csv]\n";
}

void write_sweep_csv(const std::vector<mans::MansAutotuneSweepRow>& rows,
                     const std::string& path) {
    std::ofstream out(path, std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open sweep CSV: " + path);
    }

    out << "chunk_elements,mode,threads,throughput_mbps,dims\n";
    for (const auto& row : rows) {
        out << row.chunk_elements << ","
            << row.mode << ","
            << row.threads << ","
            << std::fixed << std::setprecision(2) << row.throughput_mbps << ","
            << row.dims << "\n";
    }
}

void write_best_csv(const std::vector<mans::MansAutotuneBestConfig>& rows,
                    const std::string& path) {
    std::ofstream out(path, std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open best CSV: " + path);
    }

    out << "chunk_elements,compress_thread,decompress_thread,dims\n";
    for (const auto& row : rows) {
        out << row.chunk_elements << ","
            << row.compress_thread << ","
            << row.decompress_thread << ","
            << row.dims << "\n";
    }
}

} // namespace

int main(int argc, char** argv) {
    mans::MansAutotuneOptions options;
    options.verbose = true;
    std::string sweep_csv_path = "thread_sweep.csv";
    std::string best_csv_path = "best_threads.csv";
    options.synth_cfg.ratio_smooth = 1.0;
    options.synth_cfg.ratio_spike = 0.0;
    options.synth_cfg.ratio_constant = 0.0;
    options.synth_cfg.ratio_random = 0.0;
    options.synth_cfg.noise_range = 20;
    options.synth_cfg.seed = 42;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if ((arg == "--help") || (arg == "-h")) {
            print_usage(argv[0]);
            return 0;
        }
        if ((arg == "--data-size-mb-list" || arg == "--chunk-mb-list") && i + 1 < argc) {
            std::string error;
            if (!parse_positive_double_list(argv[++i], options.data_size_mb_list, error)) {
                std::cerr << error << "\n";
                return 1;
            }
            continue;
        }
        if (arg == "--threads-min" && i + 1 < argc) {
            options.threads_min = std::stoi(argv[++i]);
            continue;
        }
        if (arg == "--threads-max" && i + 1 < argc) {
            options.threads_max = std::stoi(argv[++i]);
            continue;
        }
        if (arg == "--stride" && i + 1 < argc) {
            options.stride = std::stoi(argv[++i]);
            continue;
        }
        if (arg == "--iter" && i + 1 < argc) {
            options.iter = static_cast<std::uint32_t>(std::stoul(argv[++i]));
            continue;
        }
        if (arg == "--ratio-smooth" && i + 1 < argc) {
            options.synth_cfg.ratio_smooth = std::stod(argv[++i]);
            continue;
        }
        if (arg == "--ratio-spike" && i + 1 < argc) {
            options.synth_cfg.ratio_spike = std::stod(argv[++i]);
            continue;
        }
        if (arg == "--ratio-constant" && i + 1 < argc) {
            options.synth_cfg.ratio_constant = std::stod(argv[++i]);
            continue;
        }
        if (arg == "--ratio-random" && i + 1 < argc) {
            options.synth_cfg.ratio_random = std::stod(argv[++i]);
            continue;
        }
        if (arg == "--noise-range" && i + 1 < argc) {
            options.synth_cfg.noise_range = std::stoi(argv[++i]);
            continue;
        }
        if (arg == "--seed" && i + 1 < argc) {
            options.synth_cfg.seed = static_cast<std::uint64_t>(std::stoull(argv[++i]));
            continue;
        }
        if (arg == "--csv" && i + 1 < argc) {
            sweep_csv_path = argv[++i];
            continue;
        }
        if (arg == "--out" && i + 1 < argc) {
            best_csv_path = argv[++i];
            continue;
        }

        std::cerr << "Unknown argument: " << arg << "\n";
        print_usage(argv[0]);
        return 1;
    }

    try {
        std::cout << "Autotune dtype: u16\n";
        std::cout << "Data size list (MB): ";
        for (std::size_t i = 0; i < options.data_size_mb_list.size(); ++i) {
            if (i > 0) {
                std::cout << ",";
            }
            std::cout << options.data_size_mb_list[i];
        }
        std::cout << "\n";
        std::cout << "Threads: " << options.threads_min << " -> ";
        if (options.threads_max > 0) {
            std::cout << options.threads_max;
        } else {
            std::cout << "auto";
        }
        std::cout
                  << " stride=" << options.stride
                  << " iter=" << options.iter << "\n";
        std::cout << "Ratios: smooth=" << options.synth_cfg.ratio_smooth
                  << " spike=" << options.synth_cfg.ratio_spike
                  << " constant=" << options.synth_cfg.ratio_constant
                  << " random=" << options.synth_cfg.ratio_random << "\n";
        std::cout << "CSV: " << sweep_csv_path << "\n";
        std::cout << "Best CSV: " << best_csv_path << "\n\n";

        mans::autotune(options);
        if (options.best_configs.empty()) {
            std::cerr << "No autotune results were generated.\n";
            return 1;
        }

        write_sweep_csv(options.sweep_rows, sweep_csv_path);
        write_best_csv(options.best_configs, best_csv_path);
        std::cout << "Done. Best threads saved to " << best_csv_path << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
