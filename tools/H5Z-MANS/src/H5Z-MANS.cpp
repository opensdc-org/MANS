#include <vector>
#include <cstdlib> // for std::malloc, std::free, std::getenv
#include <cstring> // for std::memcpy
#include <new>     // for std::bad_alloc
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <limits>
#include <mutex>
#include <atomic>
#include <H5PLextern.h>
#include <hdf5.h>

#include "H5Z-MANS_filter_ids.h"

#include "mans_api.hpp"
#include "H5Z-MANS_config.h"
#include "cpu/mans_cpu.h"
#include "cpu/buffer_cache.h"
#include "mans_timing.h"

#if defined(H5_HAVE_PARALLEL)
#include <mpi.h>
#endif
using mans::cpu::CsvThreadConfig;
using mans::cpu::find_nearest_threads;
using mans::cpu::load_thread_csv;

namespace {
std::once_flag g_timing_dump_once;
std::once_flag g_thread_csv_warn_once;
std::once_flag g_thread_match_warn_once;
int g_mpi_rank = -1;
bool g_mpi_rank_set = false;
std::atomic<int> g_timing_iter_seen{-1};

void maybe_begin_run_from_env() {
    const char* env = std::getenv("MANS_TIMING_ITER");
    if (!env || env[0] == '\0') {
        return;
    }
    const int iter = std::atoi(env);
    if (iter <= 0) {
        return;
    }
    int prev = g_timing_iter_seen.load(std::memory_order_relaxed);
    if (iter != prev &&
        g_timing_iter_seen.compare_exchange_strong(prev, iter, std::memory_order_relaxed)) {
        #ifdef ENABLE_TIMING
        mans::TimingCollector::instance().begin_run();
        #endif
    }
}

void dump_plugin_timing() {
#ifdef ENABLE_TIMING
    if (g_mpi_rank_set && g_mpi_rank != 0) {
        return;
    }
    const char* path = std::getenv("MANS_TIMING_DUMP_PATH");
    if (!path || path[0] == '\0') {
        path = "plugin_timing.csv";
    }
    MANS_TIMING_DUMP(path);
#endif
}

void register_dump_on_exit() {
#ifdef ENABLE_TIMING
    std::call_once(g_timing_dump_once, []() {
#if defined(H5_HAVE_PARALLEL)
        int inited = 0;
        MPI_Initialized(&inited);
        if (inited) {
            MPI_Comm_rank(MPI_COMM_WORLD, &g_mpi_rank);
            g_mpi_rank_set = true;
        }
#endif
        // Ensure TimingCollector is constructed before registering atexit,
        // so dump runs before the collector is destroyed.
        mans::TimingCollector::instance();
        std::atexit(dump_plugin_timing);
    });
#endif
}

void warn_once_thread_csv_fallback(const std::string& error) {
    std::call_once(g_thread_csv_warn_once, [&]() {
        std::cerr << "[H5Z-MANS Warning] " << error
                  << ". Falling back to default threads 32,32\n";
    });
}

void warn_once_thread_match_fallback(std::size_t chunk_elements, std::uint32_t dims) {
    std::call_once(g_thread_match_warn_once, [&]() {
        std::cerr << "[H5Z-MANS Warning] No matching thread config found for chunk_elements="
                  << chunk_elements << ", dims=" << dims
                  << ". Falling back to default threads 32,32\n";
    });
}
} // namespace

using mans::h5::safe_malloc;

constexpr std::size_t kMansParamsWords = sizeof(mans::MansParams) / sizeof(unsigned int);
constexpr std::size_t kMansPublicParamWords = 1;
constexpr std::uint32_t kDefaultAdmCompressThread = 32;
constexpr std::uint32_t kDefaultAdmDecompressThread = 32;

static bool parse_mans_params_from_cd(const unsigned int* cd_values,
                                      std::size_t cd_nelmts,
                                      mans::MansParams& out) {
    if (!cd_values || cd_nelmts < kMansParamsWords) {
        return false;
    }

    std::memcpy(&out, cd_values, sizeof(out));
    if (out.mode != mans::Mode::P && out.mode != mans::Mode::R) {
        return false;
    }
    return true;
}

static bool parse_public_mode_from_cd(const unsigned int* cd_values,
                                      std::size_t cd_nelmts,
                                      std::uint32_t& mode) {
    if (!cd_values || cd_nelmts < kMansPublicParamWords) {
        return false;
    }
    mode = cd_values[0];
    return mode == mans::Mode::P || mode == mans::Mode::R;
}

static bool derive_dtype_from_type(hid_t type_id,
                                   mans::MansParams& params,
                                   std::string& error) {
    if (H5Tget_class(type_id) != H5T_INTEGER) {
        error = "datatype is not INTEGER";
        return false;
    }
    if (H5Tget_sign(type_id) != H5T_SGN_NONE) {
        error = "datatype must be unsigned";
        return false;
    }
    const size_t size = H5Tget_size(type_id);
    if (size == 2) {
        params.dtype = mans::DataType::U16;
        return true;
    }
    if (size == 4) {
        params.dtype = mans::DataType::U32;
        return true;
    }
    error = "unsupported integer width";
    return false;
}

static bool build_params_for_set_local(hid_t type_id,
                                       const unsigned int* cd_values,
                                       std::size_t cd_nelmts,
                                       mans::MansParams& out,
                                       bool& params_from_public,
                                       std::string& error) {
    params_from_public = false;

    if (cd_nelmts >= kMansParamsWords && parse_mans_params_from_cd(cd_values, cd_nelmts, out)) {
        out.backend = mans::Backend::CPU;
        return derive_dtype_from_type(type_id, out, error);
    }

    std::uint32_t mode = mans::Mode::R;
    if (!parse_public_mode_from_cd(cd_values, cd_nelmts, mode)) {
        error = "invalid public filter parameters";
        return false;
    }

    out = mans::MansParams{};
    out.backend = mans::Backend::CPU;
    out.mode = mode;
    out.adm_compress_thread = kDefaultAdmCompressThread;
    out.adm_decompress_thread = kDefaultAdmDecompressThread;
    if (!derive_dtype_from_type(type_id, out, error)) {
        return false;
    }

    params_from_public = true;
    return true;
}

static bool apply_dims_from_chunk_dims(const std::vector<hsize_t>& chunk_dims,
                                       mans::MansParams& params,
                                       std::string& error) {
    if (chunk_dims.empty()) {
        error = "chunk_dims is empty";
        return false;
    }

    const auto u32_max = static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max());
    auto to_u32 = [&](std::uint64_t v, const char* name, std::uint32_t& out) -> bool {
        if (v == 0 || v > u32_max) {
            error = std::string("invalid ") + name + " in chunk dims";
            return false;
        }
        out = static_cast<std::uint32_t>(v);
        return true;
    };

    params.dims = 1;
    params.nx = 0;
    params.ny = 0;
    params.nz = 0;

    if (chunk_dims.size() == 1) {
        return to_u32(static_cast<std::uint64_t>(chunk_dims[0]), "nx", params.nx);
    }

    if (chunk_dims.size() == 2) {
        params.dims = 2;
        if (!to_u32(static_cast<std::uint64_t>(chunk_dims[0]), "nx", params.nx)) {
            return false;
        }
        if (!to_u32(static_cast<std::uint64_t>(chunk_dims[1]), "ny", params.ny)) {
            return false;
        }
        return true;
    }

    params.dims = 3;
    if (!to_u32(static_cast<std::uint64_t>(chunk_dims[0]), "nx", params.nx)) {
        return false;
    }
    if (!to_u32(static_cast<std::uint64_t>(chunk_dims[1]), "ny", params.ny)) {
        return false;
    }

    std::uint64_t merged_z = 1;
    for (std::size_t i = 2; i < chunk_dims.size(); ++i) {
        const auto d = static_cast<std::uint64_t>(chunk_dims[i]);
        if (d == 0 || merged_z > (u32_max / d)) {
            error = "invalid nz in chunk dims";
            return false;
        }
        merged_z *= d;
    }
    if (!to_u32(merged_z, "nz", params.nz)) {
        return false;
    }
    return true;
}

static bool expected_chunk_elements_from_params(const mans::MansParams& params, std::size_t& out) {
    std::uint64_t elements = 0;
    if (params.dims <= 1) {
        if (params.nx == 0) return false;
        elements = static_cast<std::uint64_t>(params.nx);
    } else if (params.dims == 2) {
        if (params.nx == 0 || params.ny == 0) return false;
        const std::uint64_t nx = static_cast<std::uint64_t>(params.nx);
        const std::uint64_t ny = static_cast<std::uint64_t>(params.ny);
        if (nx > (std::numeric_limits<std::uint64_t>::max() / ny)) return false;
        elements = nx * ny;
    } else {
        if (params.nx == 0 || params.ny == 0 || params.nz == 0) return false;
        const std::uint64_t nx = static_cast<std::uint64_t>(params.nx);
        const std::uint64_t ny = static_cast<std::uint64_t>(params.ny);
        const std::uint64_t nz = static_cast<std::uint64_t>(params.nz);
        if (nx > (std::numeric_limits<std::uint64_t>::max() / ny)) return false;
        const std::uint64_t xy = nx * ny;
        if (xy > (std::numeric_limits<std::uint64_t>::max() / nz)) return false;
        elements = xy * nz;
    }
    if (elements == 0 || elements > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        return false;
    }
    out = static_cast<std::size_t>(elements);
    return true;
}

static void apply_effective_dims_for_chunk_elements(std::size_t actual_elements, mans::MansParams& params) {
    std::size_t expected_elements = 0;
    if (!expected_chunk_elements_from_params(params, expected_elements)) {
        return;
    }
    if (actual_elements >= expected_elements || actual_elements == 0) {
        return;
    }
    const auto u32_max = static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max());
    if (actual_elements > u32_max) {
        return;
    }
    // Tail chunks have fewer elements than the nominal chunk shape.
    // Use a deterministic 1D layout so compression writes exact per-chunk dims into header.
    params.dims = 1;
    params.nx = static_cast<std::uint32_t>(actual_elements);
    params.ny = 0;
    params.nz = 0;
}

// =========================================================
// set_local: derive internal params and auto-apply thread config
// =========================================================
static herr_t H5Z_set_local_mans(hid_t dcpl_id, hid_t type_id, hid_t space_id) {
    mans::cpu::BufferCache::instance();

    const int ndims = H5Sget_simple_extent_ndims(space_id);
    if (ndims <= 0) {
        return 0;
    }

    if (H5Pget_layout(dcpl_id) != H5D_CHUNKED) {
        return 0;
    }

    size_t cd_nelmts = 0;
    unsigned int flags = 0;
    if (H5Pget_filter_by_id2(dcpl_id, H5Z_FILTER_MANS_ID, &flags,
                             &cd_nelmts, nullptr, 0, nullptr, nullptr) < 0) {
        return 0;
    }
    if (cd_nelmts < kMansPublicParamWords) {
        return 0;
    }

    std::vector<unsigned int> cd_values(cd_nelmts, 0);
    if (H5Pget_filter_by_id2(dcpl_id, H5Z_FILTER_MANS_ID, &flags,
                             &cd_nelmts, cd_values.data(), 0, nullptr, nullptr) < 0) {
        return 0;
    }

    mans::MansParams params{};
    bool params_from_public = false;
    std::string params_error;
    if (!build_params_for_set_local(type_id, cd_values.data(), cd_nelmts,
                                    params, params_from_public, params_error)) {
        std::cerr << "[H5Z-MANS Warning] Failed to build filter params in set_local: "
                  << params_error << "\n";
        return 0;
    }

    std::vector<hsize_t> chunk_dims(static_cast<std::size_t>(ndims), 0);
    if (H5Pget_chunk(dcpl_id, ndims, chunk_dims.data()) < 0) {
        return 0;
    }

    std::size_t chunk_elements = 1;
    for (int i = 0; i < ndims; ++i) {
        if (chunk_dims[static_cast<std::size_t>(i)] == 0) {
            return 0;
        }
        chunk_elements *= static_cast<std::size_t>(chunk_dims[static_cast<std::size_t>(i)]);
    }

    {
        std::string dims_error;
        if (!apply_dims_from_chunk_dims(chunk_dims, params, dims_error)) {
            std::cerr << "[H5Z-MANS Warning] Failed to apply dims from HDF5 chunk: "
                      << dims_error << "\n";
            return 0;
        }
    }

    if (params_from_public) {
        const char* csv_env = std::getenv("MANS_THREAD_CSV");
        std::string csv_path = (csv_env && csv_env[0] != '\0') ? csv_env : "best_threads.csv";
        std::vector<CsvThreadConfig> configs;
        std::string error;
        CsvThreadConfig chosen{};
        if (!load_thread_csv(csv_path, configs, error)) {
            warn_once_thread_csv_fallback(error);
        } else if (!find_nearest_threads(
                       configs, chunk_elements, static_cast<uint32_t>(params.dims), chosen)) {
            warn_once_thread_match_fallback(chunk_elements, static_cast<uint32_t>(params.dims));
        } else {
            params.adm_compress_thread = chosen.compress_thread;
            params.adm_decompress_thread = chosen.decompress_thread;
            std::cerr << "[H5Z-MANS Info] Auto threads applied (chunk_elements="
                      << chunk_elements << ", csv_chunk_elements=" << chosen.chunk_elements
                      << "): "
                      << chosen.compress_thread << ","
                      << chosen.decompress_thread << "\n";
        }
    }

    const size_t desired_nelmts = kMansParamsWords + 1;
    std::vector<unsigned int> out_values(desired_nelmts, 0);
    std::memcpy(out_values.data(), &params, sizeof(mans::MansParams));
    out_values[kMansParamsWords] = static_cast<unsigned int>(chunk_elements);

    if (H5Pmodify_filter(dcpl_id, H5Z_FILTER_MANS_ID, flags,
                         desired_nelmts, out_values.data()) < 0) {
        return 0;
    }

    return 0;
}

// =========================================================
// Type check callback (can_apply)
// =========================================================
static htri_t H5Z_can_apply_mans(hid_t dcpl_id, hid_t type_id, hid_t space_id)
{
    if (H5Tget_class(type_id) != H5T_INTEGER) {
        std::cerr << "[H5Z-MANS Warning] Datatype is not INTEGER.\n";
        return 0;
    }
    H5T_sign_t sign = H5Tget_sign(type_id);
    if (sign != H5T_SGN_NONE) {
        std::cerr << "[H5Z-MANS Warning] Datatype must be Unsigned (UINT).\n";
        return 0;
    }
    size_t size = H5Tget_size(type_id);
    if (size != 2 && size != 4) {
        std::cerr << "[H5Z-MANS Warning] Only 2-byte (U16) or 4-byte (U32) supported. Current: " << size << "\n";
        return 0;
    }
    return 1;
}

// =========================================================
// Filter callback function: H5Z_filter_mans
// =========================================================
static size_t H5Z_filter_mans(unsigned int flags, size_t cd_nelmts,
                              const unsigned int cd_values[], size_t nbytes,
                              size_t *buf_size, void **buf)
{
    maybe_begin_run_from_env();
    const size_t required_params = kMansParamsWords;

    if (cd_nelmts < required_params) {
        std::cerr << "[H5Z-MANS Error] Filter parameter count (" << cd_nelmts
                  << ") must be at least " << required_params
                  << " (MansParams).\n";
        return 0;
    }
    mans::MansParams params{};
    if (!parse_mans_params_from_cd(cd_values, cd_nelmts, params)) {
        std::cerr << "[H5Z-MANS Error] Failed to parse MansParams.\n";
        return 0;
    }

    // Destination buffer pointer and its capacity
    void* dst_buf = nullptr;
    size_t dst_capacity = 0;
    size_t out_len = 0; // Actual size produced

    try {
        if (flags & H5Z_FLAG_REVERSE) {
            // ============================
            // Decompress Path
            // ============================
            MANS_TIMING_SCOPE("filter/decompress");
            size_t elem_size = (params.dtype == mans::DataType::U16) ? 2 : 4;
            if (elem_size == 0) {
                std::cerr << "[H5Z-MANS Error] Invalid dtype in params.\n";
                return 0;
            }

            dst_capacity = mans::get_mans_exact_decompress_bytes(*buf, nbytes, params);
            if (dst_capacity == 0 || dst_capacity % elem_size != 0) {
                std::cerr << "[H5Z-MANS Error] Failed to determine decompressed size.\n";
                return 0;
            }

            dst_buf = safe_malloc(dst_capacity);
            if (!dst_buf) return 0;

            // Pass capacity via out_len variable
            out_len = dst_capacity; 

            // Call decompress API (no vector)
            mans::decompress(*buf, nbytes, params, static_cast<uint8_t*>(dst_buf), out_len);

        } else {
            // ============================
            // Compress Path
            // ============================
            MANS_TIMING_SCOPE("filter/compress");
            // Check data alignment/size validity
            size_t elem_size = (params.dtype == mans::DataType::U16) ? 2 : 4;
            if (nbytes % elem_size != 0) {
                std::cerr << "[H5Z-MANS Error] Input buffer size (" << nbytes
                          << ") is not a multiple of element size (" << elem_size
                          << "). dtype=" << (params.dtype == mans::DataType::U16 ? "U16" : "U32")
                          << "\n";
                return 0;
            }
            size_t num_elements = nbytes / elem_size;
            apply_effective_dims_for_chunk_elements(num_elements, params);

            // Allocation: worst-case bound for MANS (covers ADM + PANS path)
            dst_capacity = mans::get_mans_max_compress_bytes(num_elements, params);
            dst_buf = safe_malloc(dst_capacity);
            if (!dst_buf) return 0;

            out_len = dst_capacity;

            // Call compress API (no vector)
            mans::compress(*buf, num_elements, params, static_cast<uint8_t*>(dst_buf), out_len);
        }

        // ==========================================
        // HDF5 Memory Replacement
        // ==========================================
        // 1. Free the input buffer provided by HDF5
        MANS_TIMING_START("filter/free_input_buf");
        if (*buf) {
            std::free(*buf);
        }
        MANS_TIMING_STOP("filter/free_input_buf");
        // 2. Point HDF5 buffer to our new buffer
        *buf = dst_buf;
        
        // 3. Update the capacity tracking
        *buf_size = dst_capacity;

        // 4. Return actual used bytes
        return out_len;

    } catch (const std::exception& e) {
        std::cerr << "[H5Z-MANS Error]: " << e.what() << "\n";
        if (dst_buf) std::free(dst_buf); // Clean up our allocation on error
        return 0;
    } catch (...) {
        std::cerr << "[H5Z-MANS Error]: Unknown exception occurred.\n";
        if (dst_buf) std::free(dst_buf);
        return 0;
    }
}

// =========================================================
// HDF5 plugin registration structure
// =========================================================
const H5Z_class2_t H5Z_MANS_CLASS[1] = {{
    H5Z_CLASS_T_VERS,       
    H5Z_FILTER_MANS_ID,     
    1,                      
    1,                      
    "H5Z-MANS",             
    H5Z_can_apply_mans,     
    H5Z_set_local_mans,     
    H5Z_filter_mans,        
}};

H5PL_type_t H5PLget_plugin_type(void) {
    return H5PL_TYPE_FILTER;
}
const void *H5PLget_plugin_info(void) {
    register_dump_on_exit();
    return H5Z_MANS_CLASS;
}
