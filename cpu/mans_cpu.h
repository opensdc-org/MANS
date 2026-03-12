#pragma once
#include <algorithm>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
#include "../mans_defs.h"
namespace mans {
namespace cpu {

// =========================================================
// Thread CSV Helpers (shared)
// =========================================================
struct CsvThreadConfig {
    std::size_t chunk_elements = 0;
    uint32_t adm_decide_threads = 0;
    uint32_t compress_threads = 0;
    uint32_t decompress_threads = 0;
    uint32_t dims = 1;
};

inline bool load_thread_csv(const std::string& path,
                            std::vector<CsvThreadConfig>& configs,
                            std::string& error) {
    configs.clear();
    std::ifstream in(path);
    if (!in.is_open()) {
        error = "Failed to open CSV: " + path;
        return false;
    }

    std::string line;
    bool first = true;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        if (first) {
            first = false;
            if (line.find("chunk_elements") != std::string::npos) {
                continue;
            }
        }
        std::stringstream ss(line);
        std::string chunk_str;
        std::string decide_str;
        std::string comp_str;
        std::string decomp_str;
        std::string dims_str;
        if (!std::getline(ss, chunk_str, ',')) {
            continue;
        }
        if (!std::getline(ss, decide_str, ',')) {
            continue;
        }
        if (!std::getline(ss, comp_str, ',')) {
            continue;
        }
        if (!std::getline(ss, decomp_str, ',')) {
            continue;
        }
        if (!std::getline(ss, dims_str, ',')) {
            continue;
        }
        CsvThreadConfig cfg{};
        try {
            cfg.chunk_elements = static_cast<std::size_t>(std::stoull(chunk_str));
            cfg.adm_decide_threads = static_cast<uint32_t>(std::stoul(decide_str));
            cfg.compress_threads = static_cast<uint32_t>(std::stoul(comp_str));
            cfg.decompress_threads = static_cast<uint32_t>(std::stoul(decomp_str));
            cfg.dims = static_cast<uint32_t>(std::stoul(dims_str));
        } catch (...) {
            continue;
        }
        if (cfg.dims < 1 || cfg.dims > 3) {
            continue;
        }
        configs.push_back(cfg);
    }

    if (configs.empty()) {
        error = "No valid rows in CSV: " + path;
        return false;
    }
    return true;
}

inline bool find_nearest_threads(const std::vector<CsvThreadConfig>& configs,
                                 std::size_t target_elements,
                                 uint32_t target_dims,
                                 CsvThreadConfig& out) {

    if (configs.empty()) {
        return false;
    }
    std::size_t best_diff = std::numeric_limits<std::size_t>::max();
    bool found = false;
    for (const auto& cfg : configs) {
        if (cfg.dims != target_dims) {
            continue;
        }
        std::size_t diff = (cfg.chunk_elements > target_elements)
                               ? (cfg.chunk_elements - target_elements)
                               : (target_elements - cfg.chunk_elements);
        if (!found || diff < best_diff ||
            (diff == best_diff && cfg.chunk_elements < out.chunk_elements)) {
            out = cfg;
            best_diff = diff;
            found = true;
        }
    }
    return found;
}


void compress_internal(
    const void* input_data,
    size_t length,
    const MansParams& params,
    std::uint8_t* out,
    std::size_t& out_size,
    bool save_adm,
    const std::string& dump_path
);


void decompress_internal(
    const void* input_data,
    size_t length,
    const MansParams& params,
    std::uint8_t* out,
    std::size_t& out_size,
    bool save_adm,
    const std::string& dump_path
);

}
}