#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "mans_defs.h"

namespace mans::data_gen {

namespace fs = std::filesystem;

struct SyntheticConfig {
    double size_per_rank_mb = 256.0;
    double ratio_smooth = 1.0;
    double ratio_spike = 0.0;
    double ratio_constant = 0.0;
    double ratio_random = 0.0;
    int noise_range = 20;
    std::size_t block_size = 512;
    std::uint64_t seed = 42;
};

struct GeneratorConfig {
    SyntheticConfig synth{};
    std::string output_bin{};
    std::uint32_t dtype = mans::DataType::U16;
    std::uint32_t adm_threshold = 4000U;
};

struct RankDatasetInfo {
    std::size_t rank = 0;
    std::size_t elements = 0;
    fs::path output_path{};
};

struct GeneratedDims {
    std::uint32_t dims = 1;
    std::uint32_t nx = 0;
    std::uint32_t ny = 0;
    std::uint32_t nz = 0;
    std::size_t target_elements = 0;
    std::size_t logical_elements = 0;
    std::size_t block_elements = 512;
};

inline std::size_t align_up(std::size_t v, std::size_t a) {
    if (a == 0) {
        return v;
    }
    return ((v + a - 1) / a) * a;
}

inline bool mul_overflow(std::size_t a, std::size_t b, std::size_t& out) {
    if (a == 0 || b == 0) {
        out = 0;
        return false;
    }
    if (a > (std::numeric_limits<std::size_t>::max() / b)) {
        return true;
    }
    out = a * b;
    return false;
}

inline std::size_t adm_block_elements_for_dims(std::uint32_t dims) {
    if (dims == 1) {
        return 512;
    }
    if (dims == 2) {
        return 16 * 16;
    }
    if (dims == 3) {
        return 16 * 16 * 16;
    }
    throw std::runtime_error("dims must be 1, 2, or 3");
}

inline GeneratedDims infer_generated_dims(std::uint32_t dims, std::size_t target_elements) {
    if (target_elements == 0) {
        throw std::runtime_error("target_elements must be > 0");
    }
    if (dims < 1 || dims > 3) {
        throw std::runtime_error("dims must be 1, 2, or 3");
    }

    GeneratedDims out{};
    out.dims = dims;
    out.target_elements = target_elements;
    out.block_elements = adm_block_elements_for_dims(dims);

    if (dims == 1) {
        out.nx = static_cast<std::uint32_t>(
            std::min<std::size_t>(target_elements,
                                  static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())));
        out.ny = 0;
        out.nz = 0;
        out.logical_elements = out.nx;
        if (out.logical_elements < target_elements) {
            throw std::runtime_error("target_elements exceeds uint32_t range for dims=1");
        }
        return out;
    }

    if (dims == 2) {
        const std::size_t tile_x = 16;
        const std::size_t tile_y = 16;

        const long double root = std::sqrt(static_cast<long double>(target_elements));
        std::size_t nx = align_up(std::max<std::size_t>(tile_x, static_cast<std::size_t>(root)), tile_x);
        if (nx == 0) {
            nx = tile_x;
        }
        std::size_t ny = align_up((target_elements + nx - 1) / nx, tile_y);
        if (ny == 0) {
            ny = tile_y;
        }

        std::size_t logical = 0;
        if (mul_overflow(nx, ny, logical)) {
            throw std::runtime_error("dims=2 shape overflows size_t");
        }

        if (nx > std::numeric_limits<std::uint32_t>::max() ||
            ny > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("dims=2 shape exceeds uint32_t range");
        }

        out.nx = static_cast<std::uint32_t>(nx);
        out.ny = static_cast<std::uint32_t>(ny);
        out.nz = 0;
        out.logical_elements = logical;
        return out;
    }

    const std::size_t tile = 16;
    const long double root = std::cbrt(static_cast<long double>(target_elements));
    std::size_t nx = align_up(std::max<std::size_t>(tile, static_cast<std::size_t>(root)), tile);
    std::size_t ny = nx;
    std::size_t xy = 0;
    if (mul_overflow(nx, ny, xy)) {
        throw std::runtime_error("dims=3 nx*ny overflow");
    }
    if (xy == 0) {
        throw std::runtime_error("dims=3 invalid nx*ny");
    }
    std::size_t nz = align_up((target_elements + xy - 1) / xy, tile);
    if (nz == 0) {
        nz = tile;
    }

    std::size_t logical = 0;
    if (mul_overflow(xy, nz, logical)) {
        throw std::runtime_error("dims=3 shape overflows size_t");
    }
    if (nx > std::numeric_limits<std::uint32_t>::max() ||
        ny > std::numeric_limits<std::uint32_t>::max() ||
        nz > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("dims=3 shape exceeds uint32_t range");
    }

    out.nx = static_cast<std::uint32_t>(nx);
    out.ny = static_cast<std::uint32_t>(ny);
    out.nz = static_cast<std::uint32_t>(nz);
    out.logical_elements = logical;
    return out;
}

inline std::string trim_copy(const std::string& str) {
    const char* whitespace = " \t\r\n";
    const auto first = str.find_first_not_of(whitespace);
    if (first == std::string::npos) {
        return "";
    }
    const auto last = str.find_last_not_of(whitespace);
    return str.substr(first, last - first + 1);
}

inline void apply_config_kv(SyntheticConfig& cfg, const std::string& key, const std::string& val) {
    if (key == "size-per-rank-mb") {
        cfg.size_per_rank_mb = std::stod(val);
        return;
    }
    if (key == "ratio_smooth") {
        cfg.ratio_smooth = std::stod(val);
        return;
    }
    if (key == "ratio_spike") {
        cfg.ratio_spike = std::stod(val);
        return;
    }
    if (key == "ratio_constant") {
        cfg.ratio_constant = std::stod(val);
        return;
    }
    if (key == "ratio_random") {
        cfg.ratio_random = std::stod(val);
        return;
    }
    if (key == "noise_range") {
        cfg.noise_range = std::stoi(val);
        return;
    }
    if (key == "block_size") {
        cfg.block_size = static_cast<std::size_t>(std::stoull(val));
        return;
    }
    if (key == "seed") {
        cfg.seed = static_cast<std::uint64_t>(std::stoull(val));
        return;
    }
}

inline std::uint32_t parse_dtype_value(const std::string& val) {
    if (val == "u16" || val == "U16") {
        return mans::DataType::U16;
    }
    if (val == "u32" || val == "U32") {
        return mans::DataType::U32;
    }
    return static_cast<std::uint32_t>(std::stoul(val));
}

inline SyntheticConfig load_synthetic_config(const std::string& path) {
    SyntheticConfig cfg;
    if (path.empty()) {
        return cfg;
    }

    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open synthetic config: " + path);
    }

    std::string line;
    while (std::getline(in, line)) {
        const auto hash = line.find('#');
        if (hash != std::string::npos) {
            line = line.substr(0, hash);
        }

        line = trim_copy(line);
        if (line.empty()) {
            continue;
        }

        const auto eq = line.find('=');
        if (eq == std::string::npos) {
            continue;
        }

        const auto key = trim_copy(line.substr(0, eq));
        const auto val = trim_copy(line.substr(eq + 1));
        if (key.empty() || val.empty()) {
            continue;
        }

        apply_config_kv(cfg, key, val);
    }

    return cfg;
}

inline GeneratorConfig load_generator_config(const std::string& path) {
    GeneratorConfig cfg;
    if (path.empty()) {
        return cfg;
    }

    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open generator config: " + path);
    }

    std::string line;
    while (std::getline(in, line)) {
        const auto hash = line.find('#');
        if (hash != std::string::npos) {
            line = line.substr(0, hash);
        }

        line = trim_copy(line);
        if (line.empty()) {
            continue;
        }

        const auto eq = line.find('=');
        if (eq == std::string::npos) {
            continue;
        }

        const auto key = trim_copy(line.substr(0, eq));
        const auto val = trim_copy(line.substr(eq + 1));
        if (key.empty() || val.empty()) {
            continue;
        }

        if (key == "output_bin" || key == "output_name") {
            cfg.output_bin = val;
            continue;
        }
        if (key == "dtype") {
            cfg.dtype = parse_dtype_value(val);
            continue;
        }
        if (key == "adm_threshold") {
            cfg.adm_threshold = static_cast<std::uint32_t>(std::stoul(val));
            continue;
        }

        apply_config_kv(cfg.synth, key, val);
    }

    return cfg;
}

inline std::size_t aligned_total_elements(double size_per_rank_mb, std::size_t elem_size, std::size_t block_size) {
    if (elem_size == 0) {
        throw std::runtime_error("Element size cannot be 0");
    }
    const auto total_bytes = static_cast<std::size_t>(size_per_rank_mb * 1024.0 * 1024.0);
    auto elements = total_bytes / elem_size;
    if (elements == 0) {
        elements = block_size;
    }
    if (block_size == 0) {
        return elements;
    }
    const auto rem = elements % block_size;
    if (rem != 0) {
        elements += (block_size - rem);
    }
    return elements;
}

template <typename T>
inline std::size_t elements_from_file(const std::string& filename) {
    if (!fs::exists(filename)) {
        throw std::runtime_error("Input file not found: " + filename);
    }
    const auto bytes = fs::file_size(filename);
    if (bytes < sizeof(T)) {
        throw std::runtime_error("Input file too small for dtype");
    }
    return static_cast<std::size_t>(bytes / sizeof(T));
}

template <typename T>
inline std::vector<T> load_bin_slice(const std::string& filename,
                                     std::size_t offset_elems,
                                     std::size_t count_elems) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open input file: " + filename);
    }

    const auto byte_offset = static_cast<std::uint64_t>(offset_elems) * sizeof(T);
    in.seekg(static_cast<std::streamoff>(byte_offset), std::ios::beg);
    if (!in.good()) {
        throw std::runtime_error("Failed to seek input file: " + filename);
    }

    std::vector<T> data(count_elems);
    const auto byte_count = static_cast<std::uint64_t>(count_elems) * sizeof(T);
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(byte_count));
    const auto read_bytes = static_cast<std::uint64_t>(in.gcount());
    if (read_bytes != byte_count) {
        throw std::runtime_error("Short read from input file: " + filename);
    }

    return data;
}

template <typename T>
inline std::vector<T> generate_synthetic_slice(std::uint32_t adm_threshold,
                                               const SyntheticConfig& cfg,
                                               std::size_t total_elements,
                                               std::size_t slice_offset,
                                               std::size_t slice_count) {
    if (slice_count == 0) {
        return {};
    }
    if (cfg.block_size == 0) {
        throw std::runtime_error("Synthetic block_size cannot be 0");
    }

    const auto max_val = std::numeric_limits<T>::max();
    const std::vector<double> weights = {
        std::max(0.0, cfg.ratio_smooth),
        std::max(0.0, cfg.ratio_spike),
        std::max(0.0, cfg.ratio_constant),
        std::max(0.0, cfg.ratio_random),
    };
    const double weight_sum = weights[0] + weights[1] + weights[2] + weights[3];
    if (weight_sum <= 0.0) {
        throw std::runtime_error("Synthetic ratios sum to 0");
    }

    std::vector<T> out(slice_count);

    const auto slice_end = slice_offset + slice_count;
    const auto first_block = slice_offset / cfg.block_size;
    const auto last_block = (slice_end - 1) / cfg.block_size;

    for (std::size_t block = first_block; block <= last_block; ++block) {
        const auto block_start = block * cfg.block_size;
        const auto block_end = std::min(block_start + cfg.block_size, total_elements);

        const auto overlap_start = std::max(block_start, slice_offset);
        const auto overlap_end = std::min(block_end, slice_end);
        if (overlap_start >= overlap_end) {
            continue;
        }

        const auto block_seed = cfg.seed + static_cast<std::uint64_t>(block * 1315423911ULL);
        std::mt19937_64 rng(block_seed);
        std::discrete_distribution<int> type_dist(weights.begin(), weights.end());
        const int max_noise = std::max(0, cfg.noise_range);
        const int threshold_limited_noise = (adm_threshold > static_cast<std::uint32_t>(std::numeric_limits<int>::max()))
            ? max_noise
            : std::min(max_noise, static_cast<int>(adm_threshold));
        std::uniform_int_distribution<int> noise_dist(0, threshold_limited_noise);
        std::uniform_int_distribution<unsigned long long> full_range_dist(0, max_val);

        const int block_type = type_dist(rng);
        const T block_base = static_cast<T>(full_range_dist(rng));

        for (std::size_t i = overlap_start; i < overlap_end; ++i) {
            const auto local_idx = i - slice_offset;

            if (block_type == 0) {
                const int noise = noise_dist(rng);
                if (block_base > (max_val - static_cast<T>(threshold_limited_noise))) {
                    out[local_idx] = static_cast<T>(block_base - static_cast<T>(noise));
                } else {
                    out[local_idx] = static_cast<T>(block_base + static_cast<T>(noise));
                }
                continue;
            }

            if (block_type == 1) {
                out[local_idx] = block_base;
                const auto spike_pos = block_start + 1;
                if (i == spike_pos && spike_pos < block_end) {
                    const std::uint32_t spike_gap = adm_threshold + 500;
                    if (static_cast<unsigned long long>(block_base) + spike_gap > max_val) {
                        out[local_idx] = static_cast<T>(block_base - spike_gap);
                    } else {
                        out[local_idx] = static_cast<T>(block_base + spike_gap);
                    }
                }
                continue;
            }

            if (block_type == 2) {
                out[local_idx] = block_base;
                continue;
            }

            out[local_idx] = static_cast<T>(full_range_dist(rng));
        }
    }

    return out;
}

template <typename T>
inline std::vector<T> generate_synthetic_by_dims(std::uint32_t adm_threshold,
                                                 const SyntheticConfig& cfg,
                                                 const GeneratedDims& shape) {
    if (shape.target_elements == 0) {
        return {};
    }
    if (shape.dims < 1 || shape.dims > 3) {
        throw std::runtime_error("shape.dims must be 1, 2, or 3");
    }
    if (shape.nx == 0) {
        throw std::runtime_error("shape.nx must be > 0");
    }
    if (shape.dims >= 2 && shape.ny == 0) {
        throw std::runtime_error("shape.ny must be > 0 when dims>=2");
    }
    if (shape.dims == 3 && shape.nz == 0) {
        throw std::runtime_error("shape.nz must be > 0 when dims==3");
    }

    const auto max_val = std::numeric_limits<T>::max();
    const std::vector<double> weights = {
        std::max(0.0, cfg.ratio_smooth),
        std::max(0.0, cfg.ratio_spike),
        std::max(0.0, cfg.ratio_constant),
        std::max(0.0, cfg.ratio_random),
    };
    const double weight_sum = weights[0] + weights[1] + weights[2] + weights[3];
    if (weight_sum <= 0.0) {
        throw std::runtime_error("Synthetic ratios sum to 0");
    }

    const int max_noise = std::max(0, cfg.noise_range);
    const int threshold_limited_noise =
        (adm_threshold > static_cast<std::uint32_t>(std::numeric_limits<int>::max()))
            ? max_noise
            : std::min(max_noise, static_cast<int>(adm_threshold));

    std::vector<T> out(shape.target_elements);
    const std::size_t ny_eff = (shape.dims >= 2) ? static_cast<std::size_t>(shape.ny) : 1;
    const std::size_t nz_eff = (shape.dims == 3) ? static_cast<std::size_t>(shape.nz) : 1;
    const std::size_t nx = static_cast<std::size_t>(shape.nx);
    const auto idx3 = [nx, ny_eff](std::size_t x, std::size_t y, std::size_t z) -> std::size_t {
        return x + y * nx + z * nx * ny_eff;
    };

    if (shape.dims == 1) {
        const std::size_t block_size = adm_block_elements_for_dims(1);
        const std::size_t block_count = (shape.target_elements + block_size - 1) / block_size;
        for (std::size_t b = 0; b < block_count; ++b) {
            const std::size_t block_start = b * block_size;
            const std::size_t block_end = std::min(block_start + block_size, shape.target_elements);

            const auto block_seed = cfg.seed + static_cast<std::uint64_t>(b * 1315423911ULL);
            std::mt19937_64 rng(block_seed);
            std::discrete_distribution<int> type_dist(weights.begin(), weights.end());
            std::uniform_int_distribution<int> noise_dist(0, threshold_limited_noise);
            std::uniform_int_distribution<unsigned long long> full_range_dist(0, max_val);

            const int block_type = type_dist(rng);
            const T block_base = static_cast<T>(full_range_dist(rng));

            for (std::size_t i = block_start; i < block_end; ++i) {
                if (block_type == 0) {
                    const int noise = noise_dist(rng);
                    if (block_base > (max_val - static_cast<T>(threshold_limited_noise))) {
                        out[i] = static_cast<T>(block_base - static_cast<T>(noise));
                    } else {
                        out[i] = static_cast<T>(block_base + static_cast<T>(noise));
                    }
                    continue;
                }
                if (block_type == 1) {
                    out[i] = block_base;
                    if (i == block_start + 1) {
                        const std::uint32_t spike_gap = adm_threshold + 500;
                        if (static_cast<unsigned long long>(block_base) + spike_gap > max_val) {
                            out[i] = static_cast<T>(block_base - spike_gap);
                        } else {
                            out[i] = static_cast<T>(block_base + spike_gap);
                        }
                    }
                    continue;
                }
                if (block_type == 2) {
                    out[i] = block_base;
                    continue;
                }
                out[i] = static_cast<T>(full_range_dist(rng));
            }
        }
        return out;
    }

    const std::size_t blk_x = 16;
    const std::size_t blk_y = 16;
    const std::size_t blk_z = (shape.dims == 3) ? 16 : 1;

    const std::size_t grid_x = (nx + blk_x - 1) / blk_x;
    const std::size_t grid_y = (ny_eff + blk_y - 1) / blk_y;
    const std::size_t grid_z = (nz_eff + blk_z - 1) / blk_z;
    const std::size_t block_count = grid_x * grid_y * grid_z;

    for (std::size_t b = 0; b < block_count; ++b) {
        const std::size_t bx = b % grid_x;
        const std::size_t tmp = b / grid_x;
        const std::size_t by = tmp % grid_y;
        const std::size_t bz = tmp / grid_y;

        const std::size_t x0 = bx * blk_x;
        const std::size_t y0 = by * blk_y;
        const std::size_t z0 = bz * blk_z;
        const std::size_t x1 = std::min(x0 + blk_x, nx);
        const std::size_t y1 = std::min(y0 + blk_y, ny_eff);
        const std::size_t z1 = std::min(z0 + blk_z, nz_eff);

        const auto block_seed = cfg.seed + static_cast<std::uint64_t>(b * 1315423911ULL);
        std::mt19937_64 rng(block_seed);
        std::discrete_distribution<int> type_dist(weights.begin(), weights.end());
        std::uniform_int_distribution<int> noise_dist(0, threshold_limited_noise);
        std::uniform_int_distribution<unsigned long long> full_range_dist(0, max_val);

        const int block_type = type_dist(rng);
        const T block_base = static_cast<T>(full_range_dist(rng));
        std::size_t local_counter = 0;

        for (std::size_t z = z0; z < z1; ++z) {
            for (std::size_t y = y0; y < y1; ++y) {
                for (std::size_t x = x0; x < x1; ++x) {
                    const std::size_t idx = idx3(x, y, z);
                    if (idx >= shape.target_elements) {
                        ++local_counter;
                        continue;
                    }

                    if (block_type == 0) {
                        const int noise = noise_dist(rng);
                        if (block_base > (max_val - static_cast<T>(threshold_limited_noise))) {
                            out[idx] = static_cast<T>(block_base - static_cast<T>(noise));
                        } else {
                            out[idx] = static_cast<T>(block_base + static_cast<T>(noise));
                        }
                        ++local_counter;
                        continue;
                    }

                    if (block_type == 1) {
                        out[idx] = block_base;
                        if (local_counter == 1) {
                            const std::uint32_t spike_gap = adm_threshold + 500;
                            if (static_cast<unsigned long long>(block_base) + spike_gap > max_val) {
                                out[idx] = static_cast<T>(block_base - spike_gap);
                            } else {
                                out[idx] = static_cast<T>(block_base + spike_gap);
                            }
                        }
                        ++local_counter;
                        continue;
                    }

                    if (block_type == 2) {
                        out[idx] = block_base;
                        ++local_counter;
                        continue;
                    }

                    out[idx] = static_cast<T>(full_range_dist(rng));
                    ++local_counter;
                }
            }
        }
    }
    return out;
}

inline std::string make_rank_filename(const std::string& prefix,
                                      std::size_t rank,
                                      std::size_t rank_count) {
    const std::string stem = prefix.empty() ? "rank" : prefix;
    const std::size_t digits = (rank_count > 1)
        ? std::to_string(rank_count - 1).size()
        : std::size_t{1};
    std::ostringstream oss;
    oss << stem << std::setw(static_cast<int>(digits))
        << std::setfill('0') << rank << ".bin";
    return oss.str();
}

template <typename T>
inline void write_bin_file(const fs::path& output_path, const std::vector<T>& data) {
    std::ofstream out(output_path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open output: " + output_path.string());
    }
    out.write(reinterpret_cast<const char*>(data.data()),
              static_cast<std::streamsize>(data.size() * sizeof(T)));
    if (!out.good()) {
        throw std::runtime_error("Failed to write output: " + output_path.string());
    }
}

template <typename T>
inline std::vector<RankDatasetInfo> generate_rank_datasets(std::uint32_t adm_threshold,
                                                           const SyntheticConfig& cfg,
                                                           std::size_t rank_count,
                                                           const fs::path& output_dir,
                                                           const std::string& output_prefix = "",
                                                           std::size_t worker_threads = 0) {
    if (rank_count == 0) {
        throw std::runtime_error("rank_count must be >= 1");
    }
    if (cfg.size_per_rank_mb <= 0.0) {
        throw std::runtime_error("Synthetic size-per-rank-mb must be > 0");
    }
    if (cfg.block_size == 0) {
        throw std::runtime_error("Synthetic block_size cannot be 0");
    }

    fs::create_directories(output_dir);
    const std::size_t elements = aligned_total_elements(cfg.size_per_rank_mb, sizeof(T), cfg.block_size);

    std::vector<RankDatasetInfo> outputs(rank_count);
    for (std::size_t rank = 0; rank < rank_count; ++rank) {
        outputs[rank].rank = rank;
        outputs[rank].elements = elements;
        outputs[rank].output_path =
            output_dir / make_rank_filename(output_prefix, rank, rank_count);
    }

    std::size_t workers = worker_threads;
    if (workers == 0) {
        workers = static_cast<std::size_t>(std::thread::hardware_concurrency());
        if (workers == 0) {
            workers = 1;
        }
    }
    workers = std::max<std::size_t>(1, std::min(workers, rank_count));

    std::atomic<std::size_t> next_rank{0};
    std::mutex err_mu;
    std::exception_ptr first_error;
    std::atomic<bool> has_error{false};

    auto worker = [&]() {
        while (true) {
            if (has_error.load(std::memory_order_acquire)) {
                return;
            }

            const std::size_t rank = next_rank.fetch_add(1);
            if (rank >= rank_count) {
                return;
            }

            try {
                SyntheticConfig rank_cfg = cfg;
                rank_cfg.seed = cfg.seed + static_cast<std::uint64_t>(rank * 1315423911ULL);
                const auto data = generate_synthetic_slice<T>(
                    adm_threshold,
                    rank_cfg,
                    elements,
                    /*slice_offset=*/0,
                    /*slice_count=*/elements);
                write_bin_file(outputs[rank].output_path, data);
            } catch (...) {
                std::lock_guard<std::mutex> lock(err_mu);
                if (!first_error) {
                    first_error = std::current_exception();
                    has_error.store(true, std::memory_order_release);
                }
                return;
            }
        }
    };

    if (workers == 1) {
        worker();
    } else {
        std::vector<std::thread> threads;
        threads.reserve(workers);
        for (std::size_t i = 0; i < workers; ++i) {
            threads.emplace_back(worker);
        }
        for (auto& t : threads) {
            t.join();
        }
    }

    if (first_error) {
        std::rethrow_exception(first_error);
    }
    return outputs;
}

} // namespace mans::data_gen

namespace mans::h5 {
namespace data_gen = mans::data_gen;
} // namespace mans::h5
