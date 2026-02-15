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
} // namespace

// Define the Filter ID
#define H5Z_FILTER_MANS_ID 32001

using mans::h5::safe_malloc;

static bool threads_all_zero(const mans::MansParams& params) {
    return params.adm_decide_threads == 0 &&
           params.adm_center_calc_threads == 0 &&
           params.adm_encode_threads == 0 &&
           params.adm_warp_reduce_threads == 0 &&
           params.adm_fill_tail_threads == 0 &&
           params.adm_write_back_threads == 0 &&
           params.adm_restore_signals_threads == 0 &&
           params.adm_decode_values_threads == 0;
}

// =========================================================
// set_local: auto-apply thread config based on chunk size
// =========================================================
static herr_t H5Z_set_local_mans(hid_t dcpl_id, hid_t type_id, hid_t space_id) {
    (void)type_id;
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
    const size_t required_params = sizeof(mans::MansParams) / sizeof(unsigned int);
    if (cd_nelmts < required_params) {
        return 0;
    }

    std::vector<unsigned int> cd_values(cd_nelmts, 0);
    if (H5Pget_filter_by_id2(dcpl_id, H5Z_FILTER_MANS_ID, &flags,
                             &cd_nelmts, cd_values.data(), 0, nullptr, nullptr) < 0) {
        return 0;
    }

    mans::MansParams params{};
    std::memcpy(&params, cd_values.data(), sizeof(mans::MansParams));
    const bool need_auto_threads = threads_all_zero(params);

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

    const char* csv_env = std::getenv("MANS_THREAD_CSV");
    std::string csv_path = (csv_env && csv_env[0] != '\0') ? csv_env : "best_threads.csv";

    CsvThreadConfig chosen{};
    if (need_auto_threads) {
        std::vector<CsvThreadConfig> configs;
        std::string error;
        if (!load_thread_csv(csv_path, configs, error)) {
            std::cerr << "[H5Z-MANS Warning] " << error << "\n";
        } else if (!find_nearest_threads(configs, chunk_elements, chosen)) {
            std::cerr << "[H5Z-MANS Warning] No matching thread config found.\n";
        } else {
            params.adm_decide_threads = chosen.adm_decide_threads;
            params.adm_center_calc_threads = chosen.compress_threads;
            params.adm_encode_threads = chosen.compress_threads;
            params.adm_warp_reduce_threads = chosen.compress_threads;
            params.adm_fill_tail_threads = chosen.compress_threads;
            params.adm_write_back_threads = chosen.compress_threads;
            params.adm_restore_signals_threads = chosen.decompress_threads;
            params.adm_decode_values_threads = chosen.decompress_threads;
            std::cerr << "[H5Z-MANS Info] Auto threads applied (chunk_elements="
                      << chunk_elements << ", csv_chunk_elements=" << chosen.chunk_elements
                      << "): "
                      << chosen.adm_decide_threads << ","
                      << chosen.compress_threads << ","
                      << chosen.decompress_threads << "\n";
        }
    }

    const size_t desired_nelmts = std::max(cd_nelmts, required_params + 1);
    std::vector<unsigned int> out_values(desired_nelmts, 0);
    std::memcpy(out_values.data(), cd_values.data(),
                std::min(cd_nelmts, desired_nelmts) * sizeof(unsigned int));
    std::memcpy(out_values.data(), &params, sizeof(mans::MansParams));
    out_values[required_params] = static_cast<unsigned int>(chunk_elements);

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
    size_t required_params = sizeof(mans::MansParams) / sizeof(unsigned int);

    if (cd_nelmts < required_params) {
        std::cerr << "[H5Z-MANS Error] Filter parameter count (" << cd_nelmts
                  << ") must be at least " << required_params << " (MansParams).\n";
        return 0;
    }
    const mans::MansParams* params = reinterpret_cast<const mans::MansParams*>(cd_values);

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
            size_t elem_size = (params->dtype == mans::DataType::U16) ? 2 : 4;
            if (elem_size == 0) {
                std::cerr << "[H5Z-MANS Error] Invalid dtype in params.\n";
                return 0;
            }

            if (cd_nelmts > required_params) {
                const size_t chunk_elems = static_cast<size_t>(cd_values[required_params]);
                if (chunk_elems > 0) {
                    dst_capacity = chunk_elems * elem_size;
                }
            }
            if (dst_capacity == 0 && buf_size && *buf_size > 0) {
                dst_capacity = *buf_size;
            }
            if (dst_capacity == 0) {
                std::cerr << "[H5Z-MANS Error] Failed to determine decompressed size.\n";
                return 0;
            }
            
            dst_buf = safe_malloc(dst_capacity);
            if (!dst_buf) return 0;

            // Pass capacity via out_len variable
            out_len = dst_capacity; 

            // Call decompress API (no vector)
            mans::decompress(*buf, nbytes, *params, static_cast<uint8_t*>(dst_buf), out_len);

        } else {
            // ============================
            // Compress Path
            // ============================
            MANS_TIMING_SCOPE("filter/compress");
            // Check data alignment/size validity
            size_t elem_size = (params->dtype == mans::DataType::U16) ? 2 : 4;
            if (nbytes % elem_size != 0) {
                std::cerr << "[H5Z-MANS Error] Input buffer size (" << nbytes
                          << ") is not a multiple of element size (" << elem_size
                          << "). dtype=" << (params->dtype == mans::DataType::U16 ? "U16" : "U32")
                          << "\n";
                return 0;
            }
            size_t num_elements = nbytes / elem_size;

            // Allocation: worst-case bound for MANS (covers ADM + PANS path)
            dst_capacity = mans::get_mans_max_compress_bytes_p(num_elements, *params);
            dst_buf = safe_malloc(dst_capacity);
            if (!dst_buf) return 0;

            out_len = dst_capacity;

            // Call compress API (no vector)
            mans::compress(*buf, num_elements, *params, static_cast<uint8_t*>(dst_buf), out_len);
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
