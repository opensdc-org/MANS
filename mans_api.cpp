 #include "mans_api.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>

#ifdef MANS_ENABLE_CPU
#include "cpu/mans_cpu.h"
#endif

#ifdef MANS_ENABLE_NV
#include "nv/mans_nv.h"
#endif

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

int default_max_threads() {
    unsigned int hw = std::thread::hardware_concurrency();
    if (hw > 0) {
        return static_cast<int>(hw);
    }
    return 1;
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
    params.adm_compress_thread = 32;
    params.adm_decompress_thread = 32;
    params.dims = shape.dims;
    params.nx = shape.nx;
    params.ny = shape.ny;
    params.nz = shape.nz;
    return params;
}

#ifdef MANS_ENABLE_CPU
ThroughputResult run_compress_decompress(const std::uint16_t* data,
                                         std::size_t total_elements,
                                         const mans::MansParams& params,
                                         std::uint32_t iter) {
    ThroughputResult result;
    if (total_elements == 0) {
        result.ok = false;
        result.error = "Input is empty.";
        return result;
    }

    const std::size_t rounds = std::max<std::size_t>(1, static_cast<std::size_t>(iter));
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
    const std::size_t expected_bytes = total_elements * sizeof(std::uint16_t);
    const double total_bytes = static_cast<double>(expected_bytes);

    for (std::size_t round = 0; round < rounds; ++round) {
        std::size_t out_size = comp_buf.size();
        const auto comp_start = std::chrono::high_resolution_clock::now();
        mans::cpu::compress_internal(
            data,
            total_elements,
            params,
            comp_buf.data(),
            out_size,
            false,
            "");
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
            "");
        const auto decomp_end = std::chrono::high_resolution_clock::now();
        const double decomp_ms =
            std::chrono::duration<double, std::milli>(decomp_end - decomp_start).count();

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

        const double comp_mbps = comp_ms > 0.0 ? (total_bytes / 1e6) / (comp_ms / 1e3) : 0.0;
        const double decomp_mbps = decomp_ms > 0.0 ? (total_bytes / 1e6) / (decomp_ms / 1e3) : 0.0;
        result.comp_mbps = std::max(result.comp_mbps, comp_mbps);
        result.decomp_mbps = std::max(result.decomp_mbps, decomp_mbps);
    }

    return result;
}
#endif

void validate_autotune_options(mans::MansAutotuneOptions options) {
    if (options.data_size_mb_list.empty()) {
        throw std::runtime_error("MANS::autotune: data_size_mb_list is empty.");
    }
    if (options.dims_list.empty()) {
        throw std::runtime_error("MANS::autotune: dims_list is empty.");
    }
    if (options.threads_min <= 0) {
        throw std::runtime_error("MANS::autotune: threads_min must be > 0.");
    }
    if (options.stride <= 0) {
        throw std::runtime_error("MANS::autotune: stride must be > 0.");
    }
    if (options.iter == 0) {
        throw std::runtime_error("MANS::autotune: iter must be > 0.");
    }

    const double ratio_sum =
        std::max(0.0, options.synth_cfg.ratio_smooth) +
        std::max(0.0, options.synth_cfg.ratio_spike) +
        std::max(0.0, options.synth_cfg.ratio_constant) +
        std::max(0.0, options.synth_cfg.ratio_random);
    if (ratio_sum <= 0.0) {
        throw std::runtime_error("MANS::autotune: synthetic block ratios must sum to > 0.");
    }

    for (double value : options.data_size_mb_list) {
        if (!(value > 0.0)) {
            throw std::runtime_error("MANS::autotune: data_size_mb_list must contain positive values.");
        }
    }
    for (std::uint32_t dims : options.dims_list) {
        if (dims < 1 || dims > 3) {
            throw std::runtime_error("MANS::autotune: dims_list values must be 1, 2, or 3.");
        }
    }
}

} // namespace

namespace mans {

void compress_device(const void* input_data,
                     size_t length,
                     const MansParams& params,
                     uint8_t* out,
                     size_t& out_size) {
    if (params.backend != Backend::NVIDIA) {
        throw std::runtime_error("MANS::compress_device: only the NVIDIA backend is supported.");
    }
#ifdef MANS_ENABLE_NV
    mans::nv::compress_internal_device(input_data, length, params, out, out_size);
    return;
#else
    throw std::runtime_error("MANS::compress_device: NVIDIA backend was NOT compiled.");
#endif
}

void decompress_device(const void* input_data,
                       size_t length,
                       const MansParams& params,
                       uint8_t* out,
                       size_t& out_size) {
    if (params.backend != Backend::NVIDIA) {
        throw std::runtime_error("MANS::decompress_device: only the NVIDIA backend is supported.");
    }
#ifdef MANS_ENABLE_NV
    mans::nv::decompress_internal_device(input_data, length, params, out, out_size);
    return;
#else
    throw std::runtime_error("MANS::decompress_device: NVIDIA backend was NOT compiled.");
#endif
}

void compress(const void* input_data,
              size_t length,
              const MansParams& params,
              uint8_t* out,
              size_t& out_size) {
    if (params.backend == Backend::CPU) {
#ifdef MANS_ENABLE_CPU
        mans::cpu::compress_internal(input_data, length, params, out, out_size, false, "");
        return;
#else
        throw std::runtime_error("MANS::compress: CPU backend was NOT compiled.");
#endif
    }

    if (params.backend == Backend::NVIDIA) {
#ifdef MANS_ENABLE_NV
        mans::nv::compress_internal(input_data, length, params, out, out_size, false, "");
        return;
#else
        throw std::runtime_error("MANS::compress: NVIDIA backend was NOT compiled.");
#endif
    }

    throw std::runtime_error("MANS::compress: Unknown backend type.");
}

void decompress(const void* input_data,
                size_t length,
                const MansParams& params,
                uint8_t* out,
                size_t& out_size) {
    if (params.backend == Backend::CPU) {
#ifdef MANS_ENABLE_CPU
        mans::cpu::decompress_internal(input_data, length, params, out, out_size, false, "");
        return;
#else
        throw std::runtime_error("MANS::decompress: CPU backend was NOT compiled.");
#endif
    }

    if (params.backend == Backend::NVIDIA) {
#ifdef MANS_ENABLE_NV
        mans::nv::decompress_internal(input_data, length, params, out, out_size, false, "");
        return;
#else
        throw std::runtime_error("MANS::decompress: NVIDIA backend was NOT compiled.");
#endif
    }

    throw std::runtime_error("MANS::decompress: Unknown backend type.");
}

std::size_t get_mans_max_compress_bytes(std::size_t num_elements, const MansParams& params) {
    if (params.backend == Backend::CPU) {
#ifdef MANS_ENABLE_CPU
        return mans::cpu::get_max_compress_bytes(num_elements, params);
#else
        throw std::runtime_error("MANS::get_mans_max_compress_bytes: CPU backend was NOT compiled.");
#endif
    }

    if (params.backend == Backend::NVIDIA) {
#ifdef MANS_ENABLE_NV
        return mans::nv::get_max_compress_bytes(num_elements, params);
#else
        throw std::runtime_error("MANS::get_mans_max_compress_bytes: NVIDIA backend was NOT compiled.");
#endif
    }

    throw std::runtime_error("MANS::get_mans_max_compress_bytes: Unknown backend type.");
}

std::size_t get_mans_exact_decompress_bytes(const void* compressed_data,
                                            std::size_t compressed_len,
                                            const MansParams& params) {
    if (params.backend == Backend::CPU) {
#ifdef MANS_ENABLE_CPU
        return mans::cpu::get_exact_decompress_bytes(compressed_data, compressed_len, params);
#else
        throw std::runtime_error("MANS::get_mans_exact_decompress_bytes: CPU backend was NOT compiled.");
#endif
    }

    if (params.backend == Backend::NVIDIA) {
#ifdef MANS_ENABLE_NV
        return mans::nv::get_exact_decompress_bytes(compressed_data, compressed_len, params);
#else
        throw std::runtime_error("MANS::get_mans_exact_decompress_bytes: NVIDIA backend was NOT compiled.");
#endif
    }

    throw std::runtime_error("MANS::get_mans_exact_decompress_bytes: Unknown backend type.");
}

void autotune(MansAutotuneOptions& options) {
    validate_autotune_options(options);

    if (options.threads_max != 0 && options.threads_max < options.threads_min) {
        throw std::runtime_error("MANS::autotune: threads_max must be >= threads_min.");
    }

#ifndef MANS_ENABLE_CPU
    (void)options;
    throw std::runtime_error("MANS::autotune: CPU backend was NOT compiled.");
#else
    const int resolved_threads_max =
        (options.threads_max > 0) ? options.threads_max : default_max_threads();
    std::vector<int> thread_list =
        build_thread_list(options.threads_min, resolved_threads_max, options.stride);
    if (thread_list.empty()) {
        throw std::runtime_error("MANS::autotune: invalid thread list after resolving range.");
    }

    options.sweep_rows.clear();
    options.best_configs.clear();
    std::map<std::pair<std::size_t, int>, std::map<std::string, BestEntry>> best;

    for (std::uint32_t dims : options.dims_list) {
        for (double data_size_mb : options.data_size_mb_list) {
            std::size_t data_size_bytes = static_cast<std::size_t>(data_size_mb * 1024.0 * 1024.0);
            if (data_size_bytes < sizeof(std::uint16_t)) {
                data_size_bytes = sizeof(std::uint16_t);
            }

            std::size_t data_elements = data_size_bytes / sizeof(std::uint16_t);
            if (data_elements == 0) {
                data_elements = 1;
            }

            const mans::data_gen::GeneratedDims tune_shape =
                mans::data_gen::infer_generated_dims(dims, data_elements);
            std::vector<std::uint16_t> data =
                mans::data_gen::generate_synthetic_by_dims<std::uint16_t>(
                    kAdmThreshold, options.synth_cfg, tune_shape);
            if (data.empty()) {
                throw std::runtime_error("MANS::autotune: generated synthetic dataset is empty.");
            }

            const std::size_t total_elements = data.size();
            const mans::MansParams base_params = default_params(tune_shape);

            if (options.verbose) {
                const double total_mb =
                    static_cast<double>(total_elements * sizeof(std::uint16_t)) / (1024.0 * 1024.0);
                std::cout << "Dims=" << tune_shape.dims
                          << " nx=" << tune_shape.nx
                          << " ny=" << tune_shape.ny
                          << " nz=" << tune_shape.nz
                          << " | data=" << std::fixed << std::setprecision(3) << total_mb << " MB"
                          << " | elements=" << total_elements
                          << "\n";
            }

            for (int threads : thread_list) {
                mans::MansParams params = base_params;
                params.adm_compress_thread = static_cast<std::uint32_t>(threads);
                params.adm_decompress_thread = static_cast<std::uint32_t>(threads);

                const ThroughputResult throughput =
                    run_compress_decompress(data.data(), total_elements, params, options.iter);
                if (!throughput.ok) {
                    std::ostringstream oss;
                    oss << "MANS::autotune: dims=" << dims
                        << " elements=" << total_elements
                        << " threads=" << threads
                        << " failed: " << throughput.error;
                    throw std::runtime_error(oss.str());
                }

                options.sweep_rows.push_back(
                    MansAutotuneSweepRow{total_elements, dims, "compress",
                                         static_cast<std::uint32_t>(threads), throughput.comp_mbps});
                options.sweep_rows.push_back(
                    MansAutotuneSweepRow{total_elements, dims, "decompress",
                                         static_cast<std::uint32_t>(threads), throughput.decomp_mbps});

                auto& best_comp = best[std::make_pair(total_elements, static_cast<int>(dims))]["compress"];
                if (throughput.comp_mbps > best_comp.throughput) {
                    best_comp.throughput = throughput.comp_mbps;
                    best_comp.threads = threads;
                }
                auto& best_decomp =
                    best[std::make_pair(total_elements, static_cast<int>(dims))]["decompress"];
                if (throughput.decomp_mbps > best_decomp.throughput) {
                    best_decomp.throughput = throughput.decomp_mbps;
                    best_decomp.threads = threads;
                }

                if (options.verbose) {
                    std::cout << "  [codec][d" << dims << "] threads=" << threads
                              << " comp=" << std::fixed << std::setprecision(2) << throughput.comp_mbps
                              << " MB/s, decomp=" << throughput.decomp_mbps << " MB/s\n";
                }
            }

            if (options.verbose) {
                std::cout << "\n";
            }
        }
    }

    options.best_configs.reserve(best.size());
    for (const auto& item : best) {
        const std::size_t chunk_elements = item.first.first;
        const std::uint32_t dims = static_cast<std::uint32_t>(item.first.second);
        const auto& modes = item.second;

        std::uint32_t compress_thread = 0;
        std::uint32_t decompress_thread = 0;
        auto it_comp = modes.find("compress");
        if (it_comp != modes.end()) {
            compress_thread = static_cast<std::uint32_t>(it_comp->second.threads);
        }
        auto it_decomp = modes.find("decompress");
        if (it_decomp != modes.end()) {
            decompress_thread = static_cast<std::uint32_t>(it_decomp->second.threads);
        }

        options.best_configs.push_back(
            MansAutotuneBestConfig{chunk_elements, dims, compress_thread, decompress_thread});
    }
#endif
}

} // namespace mans
