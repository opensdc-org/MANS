#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include <hdf5.h>

#include "mans_data_gen.h"
#include "H5Z-MANS_config.h"
#include "cpu/mans_cpu.h"
#include "mans_timing.h"
#include "../include/sz3_config_min.h"

#if defined(H5_HAVE_PARALLEL)
#include <mpi.h>
#endif

// Filter IDs
#define FILTER_ID_NONE    32002
#define FILTER_ID_DEFLATE 1
#define FILTER_ID_ZSTD    32015
#define FILTER_ID_MANS    32001
#define FILTER_ID_SZ3     32024

static constexpr double DEFAULT_DATASET_MB = 1024.0; // 1GB
static constexpr double DEFAULT_CHUNK_MB = 8.0;
static constexpr const char* DEFAULT_OUTPUT_H5 = "hdf5_bench.h5";
static constexpr const char* TIMING_CSV_PATH = "hdf5_timing.csv";

static constexpr std::uint32_t DEFAULT_ADM_THRESHOLD = 4000U;

static const std::string RST  = "\033[0m";
static const std::string GRN  = "\033[1;32m";
static const std::string YLW  = "\033[1;33m";
static const std::string RED  = "\033[1;31m";
static const std::string BOLD = "\033[1m";

#define CHECK_H5(call)                                                              \
    do {                                                                            \
        if ((call) < 0) {                                                           \
            std::cerr << RED << "[HDF5 Error] " << #call << RST << "\n";           \
            std::exit(1);                                                           \
        }                                                                           \
    } while (0)

enum class FilterKind {
    Mans,
    Zstd,
    Sz3,
    Gzip,
    None,
};

struct Options {
    double dataset_mb = DEFAULT_DATASET_MB;
    double chunk_mb = DEFAULT_CHUNK_MB;
    std::string output_h5 = DEFAULT_OUTPUT_H5;
    FilterKind filter = FilterKind::Mans;
    std::vector<int> threads = {32, 32, 32, 32, 32, 32, 32, 32};
    bool threads_from_user = false;
};

struct TimingSums {
    double global_write_s = 0.0;
    double global_read_s = 0.0;
    double avg_write_s = 0.0;
    double avg_read_s = 0.0;
    double ratio = 0.0;
};

#if defined(H5_HAVE_PARALLEL)
struct MpiContext {
    int rank = 0;
    int ranks = 1;
    bool initialized = false;

    MpiContext(int& argc, char**& argv) {
        int is_init = 0;
        MPI_Initialized(&is_init);
        if (!is_init) {
            MPI_Init(&argc, &argv);
            initialized = true;
        }
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    }

    ~MpiContext() {
        int is_finalized = 0;
        MPI_Finalized(&is_finalized);
        if (initialized && !is_finalized) {
            MPI_Finalize();
        }
    }
};
#else
struct MpiContext {
    int rank = 0;
    int ranks = 1;

    MpiContext(int&, char**&) {}
};
#endif

static std::string filter_to_string(FilterKind kind) {
    switch (kind) {
        case FilterKind::Mans:
            return "mans";
        case FilterKind::Zstd:
            return "zstd";
        case FilterKind::Sz3:
            return "sz3";
        case FilterKind::Gzip:
            return "gzip";
        case FilterKind::None:
            return "none";
    }
    return "unknown";
}

static FilterKind parse_filter(std::string_view s) {
    if (s == "mans") {
        return FilterKind::Mans;
    }
    if (s == "zstd") {
        return FilterKind::Zstd;
    }
    if (s == "sz3") {
        return FilterKind::Sz3;
    }
    if (s == "gzip" || s == "deflate") {
        return FilterKind::Gzip;
    }
    if (s == "none") {
        return FilterKind::None;
    }
    throw std::runtime_error("Unknown filter: " + std::string(s));
}

static std::vector<int> parse_threads(const std::string& s) {
    std::vector<int> values;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) {
            throw std::runtime_error("Empty entry in --threads");
        }
        std::size_t idx = 0;
        int v = 0;
        try {
            v = std::stoi(item, &idx);
        } catch (...) {
            throw std::runtime_error("Invalid thread value: " + item);
        }
        if (idx != item.size() || v <= 0) {
            throw std::runtime_error("Invalid thread value: " + item);
        }
        values.push_back(v);
    }
    if (values.size() != 8) {
        throw std::runtime_error("--threads requires exactly 8 values");
    }
    return values;
}

static void print_usage(const char* prog) {
    std::cerr
        << "Usage: " << prog
        << " [--dataset-mb N] [--chunk-mb N] [--filter mans|zstd|sz3|gzip|none]\n"
           "             [--threads v1,v2,v3,v4,v5,v6,v7,v8] [--output file.h5]\n";
}

static Options parse_args(int argc, char** argv) {
    Options opts;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need_value = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("Missing value for ") + flag);
            }
            return argv[++i];
        };

        if (arg == "--dataset-mb") {
            opts.dataset_mb = std::stod(need_value("--dataset-mb"));
            continue;
        }
        if (arg == "--chunk-mb") {
            opts.chunk_mb = std::stod(need_value("--chunk-mb"));
            continue;
        }
        if (arg == "--filter") {
            opts.filter = parse_filter(need_value("--filter"));
            continue;
        }
        if (arg == "--threads") {
            opts.threads = parse_threads(need_value("--threads"));
            opts.threads_from_user = true;
            continue;
        }
        if (arg == "--output") {
            opts.output_h5 = need_value("--output");
            continue;
        }
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        }

        throw std::runtime_error("Unknown argument: " + arg);
    }

    if (opts.dataset_mb <= 0.0) {
        throw std::runtime_error("dataset-mb must be positive");
    }
    if (opts.chunk_mb <= 0.0) {
        throw std::runtime_error("chunk-mb must be positive");
    }

    return opts;
}

static void split_even(std::size_t total, int rank, int ranks,
                       std::size_t& offset, std::size_t& count) {
    const std::size_t base = total / static_cast<std::size_t>(ranks);
    const std::size_t rem = total % static_cast<std::size_t>(ranks);
    if (static_cast<std::size_t>(rank) < rem) {
        count = base + 1;
        offset = static_cast<std::size_t>(rank) * count;
    } else {
        count = base;
        offset = rem * (base + 1) + (static_cast<std::size_t>(rank) - rem) * base;
    }
}

static std::size_t chunk_elements(double chunk_mb, std::size_t elem_size,
                                  std::size_t total_elements) {
    const auto chunk_bytes = static_cast<std::size_t>(chunk_mb * 1024.0 * 1024.0);
    std::size_t elems = elem_size == 0 ? 0 : (chunk_bytes / elem_size);
    if (elems == 0) {
        elems = 1;
    }
    return std::min(elems, total_elements);
}

static mans::MansParams build_mans_params(const Options& opts) {
    mans::MansParams params{};
    params.backend = mans::Backend::CPU;
    params.dtype = mans::DataType::U16;
    params.adm_threshold = DEFAULT_ADM_THRESHOLD;

    if (!opts.threads_from_user) {
        params.adm_decide_threads = 0;
        params.adm_center_calc_threads = 0;
        params.adm_encode_threads = 0;
        params.adm_warp_reduce_threads = 0;
        params.adm_fill_tail_threads = 0;
        params.adm_write_back_threads = 0;
        params.adm_restore_signals_threads = 0;
        params.adm_decode_values_threads = 0;
        return params;
    }

    params.adm_decide_threads = static_cast<std::uint32_t>(opts.threads[0]);
    params.adm_center_calc_threads = static_cast<std::uint32_t>(opts.threads[1]);
    params.adm_encode_threads = static_cast<std::uint32_t>(opts.threads[2]);
    params.adm_warp_reduce_threads = static_cast<std::uint32_t>(opts.threads[3]);
    params.adm_fill_tail_threads = static_cast<std::uint32_t>(opts.threads[4]);
    params.adm_write_back_threads = static_cast<std::uint32_t>(opts.threads[5]);
    params.adm_restore_signals_threads = static_cast<std::uint32_t>(opts.threads[6]);
    params.adm_decode_values_threads = static_cast<std::uint32_t>(opts.threads[7]);
    return params;
}

static bool resolve_auto_threads(const std::string& csv_path,
                                 std::size_t chunk_elems,
                                 std::uint32_t dims,
                                 std::vector<int>& out_threads) {
    std::vector<mans::cpu::CsvThreadConfig> configs;
    std::string error;
    if (!mans::cpu::load_thread_csv(csv_path, configs, error)) {
        return false;
    }
    mans::cpu::CsvThreadConfig chosen{};
    if (!mans::cpu::find_nearest_threads(configs, chunk_elems, dims, chosen)) {
        return false;
    }
    out_threads = {
        static_cast<int>(chosen.adm_decide_threads),
        static_cast<int>(chosen.compress_threads),
        static_cast<int>(chosen.compress_threads),
        static_cast<int>(chosen.compress_threads),
        static_cast<int>(chosen.compress_threads),
        static_cast<int>(chosen.compress_threads),
        static_cast<int>(chosen.decompress_threads),
        static_cast<int>(chosen.decompress_threads)
    };
    return true;
}

static void configure_filter(hid_t dcpl,
                             const Options& opts,
                             const mans::MansParams& mans_params,
                             std::size_t chunk_elems) {
    const hsize_t chunk[1] = {static_cast<hsize_t>(chunk_elems)};
    CHECK_H5(H5Pset_chunk(dcpl, 1, chunk));

    if (opts.filter == FilterKind::None) {
        const htri_t avail = H5Zfilter_avail(FILTER_ID_NONE);
        if (!avail) {
            throw std::runtime_error("H5Z-NONE filter not available");
        }
        CHECK_H5(H5Pset_filter(dcpl, FILTER_ID_NONE, 0, 0, nullptr));
        return;
    }

    if (opts.filter == FilterKind::Gzip) {
        const unsigned int gzip_level = 6;
        CHECK_H5(H5Pset_deflate(dcpl, gzip_level));
        return;
    }

    if (opts.filter == FilterKind::Zstd) {
        const unsigned int zstd_level = 3;
        const htri_t avail = H5Zfilter_avail(FILTER_ID_ZSTD);
        if (!avail) {
            throw std::runtime_error("Zstandard filter not available");
        }
        unsigned int cd_values[1] = {zstd_level};
        CHECK_H5(H5Pset_filter(dcpl, FILTER_ID_ZSTD, H5Z_FLAG_OPTIONAL, 1, cd_values));
        return;
    }

    if (opts.filter == FilterKind::Mans) {
        const std::size_t cd_count = sizeof(mans::MansParams) / sizeof(unsigned int);
        std::vector<unsigned int> cd(cd_count, 0);
        std::memcpy(cd.data(), &mans_params, sizeof(mans::MansParams));
        CHECK_H5(H5Pset_filter(dcpl, FILTER_ID_MANS, 0, cd.size(), cd.data()));
        return;
    }

    if (opts.filter == FilterKind::Sz3) {
        const htri_t avail = H5Zfilter_avail(FILTER_ID_SZ3);
        if (!avail) {
            throw std::runtime_error("SZ3 filter not available");
        }

        SZ3::Config sz3_conf;
        static const char* sz3_ini = R"ini([GlobalSettings]
CmprAlgo = ALGO_INTERP_LORENZO
ErrorBoundMode = ABS
AbsErrorBound = 1e-3
OpenMP = YES

[AlgoSettings]
)ini";
        sz3_conf.load_ini(sz3_ini);

        std::size_t cd_nelmts = static_cast<std::size_t>(
            std::ceil(sz3_conf.size_est() / static_cast<double>(sizeof(int))));
        std::vector<unsigned int> cd_values(cd_nelmts, 0);
        auto* buffer = reinterpret_cast<unsigned char*>(cd_values.data());
        const auto conf_size_real = sz3_conf.save(buffer);
        cd_nelmts = static_cast<std::size_t>(
            std::ceil(conf_size_real / static_cast<double>(sizeof(int))));

        CHECK_H5(H5Pset_filter(dcpl, FILTER_ID_SZ3, H5Z_FLAG_MANDATORY,
                               cd_nelmts, cd_values.data()));
        return;
    }

    throw std::runtime_error("Unknown filter");
}

static hid_t create_fapl(bool use_mpio) {
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    CHECK_H5(fapl);
#if defined(H5_HAVE_PARALLEL)
    if (use_mpio) {
        CHECK_H5(H5Pset_fapl_mpio(fapl, MPI_COMM_WORLD, MPI_INFO_NULL));
    }
#else
    (void)use_mpio;
#endif
    return fapl;
}

static hid_t create_dxpl(bool use_mpio) {
    hid_t dxpl = H5Pcreate(H5P_DATASET_XFER);
    CHECK_H5(dxpl);
#if defined(H5_HAVE_PARALLEL)
    if (use_mpio) {
        CHECK_H5(H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE));
    }
#else
    (void)use_mpio;
#endif
    return dxpl;
}

static void sync_file(hid_t file, bool use_mpio) {
    void* handle = nullptr;
    CHECK_H5(H5Fget_vfd_handle(file, H5P_DEFAULT, &handle));

#if defined(H5_HAVE_PARALLEL)
    if (use_mpio) {
        MPI_File* mpif = static_cast<MPI_File*>(handle);
        MPI_File_sync(*mpif);
        return;
    }
#endif

    int fd = *static_cast<int*>(handle);
    if (::fsync(fd) != 0) {
        std::cerr << YLW << "[WARN] fsync failed" << RST << "\n";
    }
}

static void drop_cache_best_effort(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        return;
    }
#if defined(POSIX_FADV_DONTNEED)
    (void)::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
#endif
    ::close(fd);
}

static double write_hdf5(const Options& opts,
                         const mans::MansParams& params,
                         const std::vector<std::uint16_t>& local_data,
                         std::size_t total_elements,
                         std::size_t chunk_elems,
                         std::size_t rank_offset,
                         std::size_t rank_count,
                         bool use_mpio,
                         std::uint64_t& compressed_bytes_out) {
    const hid_t fapl = create_fapl(use_mpio);
    const hid_t file = H5Fcreate(opts.output_h5.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    CHECK_H5(file);
    CHECK_H5(H5Pclose(fapl));

    const hsize_t dims[1] = {static_cast<hsize_t>(total_elements)};
    const hid_t filespace = H5Screate_simple(1, dims, nullptr);
    CHECK_H5(filespace);

    const hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    CHECK_H5(dcpl);
    configure_filter(dcpl, opts, params, chunk_elems);

    const hid_t dset = H5Dcreate2(file, "dataset", H5T_NATIVE_UINT16, filespace,
                                  H5P_DEFAULT, dcpl, H5P_DEFAULT);
    CHECK_H5(dset);
    CHECK_H5(H5Pclose(dcpl));
    CHECK_H5(H5Sclose(filespace));

    const hid_t dxpl = create_dxpl(use_mpio);
    const hid_t fspace = H5Dget_space(dset);
    CHECK_H5(fspace);

    if (rank_count > 0) {
        const hsize_t file_offset[1] = {static_cast<hsize_t>(rank_offset)};
        const hsize_t file_count[1] = {static_cast<hsize_t>(rank_count)};
        CHECK_H5(H5Sselect_hyperslab(fspace, H5S_SELECT_SET, file_offset, nullptr, file_count, nullptr));

        const hid_t mspace = H5Screate_simple(1, file_count, nullptr);
        CHECK_H5(mspace);

        auto t0 = std::chrono::steady_clock::now();
        {
            MANS_TIMING_SCOPE("hdf5/write");
            CHECK_H5(H5Dwrite(dset, H5T_NATIVE_UINT16, mspace, fspace, dxpl, local_data.data()));
            CHECK_H5(H5Fflush(file, H5F_SCOPE_GLOBAL));
            MANS_TIMING_START("hdf5/sync");
            sync_file(file, use_mpio);
            MANS_TIMING_STOP("hdf5/sync");
        }
        auto t1 = std::chrono::steady_clock::now();

        compressed_bytes_out = static_cast<std::uint64_t>(H5Dget_storage_size(dset));

        CHECK_H5(H5Sclose(mspace));
        CHECK_H5(H5Sclose(fspace));
        CHECK_H5(H5Pclose(dxpl));
        CHECK_H5(H5Dclose(dset));
        CHECK_H5(H5Fclose(file));

        return std::chrono::duration<double>(t1 - t0).count();
    }

    CHECK_H5(H5Sselect_none(fspace));
    const hsize_t one[1] = {1};
    const hid_t mspace = H5Screate_simple(1, one, nullptr);
    CHECK_H5(mspace);
    CHECK_H5(H5Sselect_none(mspace));

    std::uint16_t dummy = 0;
    auto t0 = std::chrono::steady_clock::now();
    {
        MANS_TIMING_SCOPE("hdf5/write");
        CHECK_H5(H5Dwrite(dset, H5T_NATIVE_UINT16, mspace, fspace, dxpl, &dummy));
        CHECK_H5(H5Fflush(file, H5F_SCOPE_GLOBAL));
        MANS_TIMING_START("hdf5/sync");
        sync_file(file, use_mpio);
        MANS_TIMING_STOP("hdf5/sync");
    }
    auto t1 = std::chrono::steady_clock::now();

    compressed_bytes_out = static_cast<std::uint64_t>(H5Dget_storage_size(dset));

    CHECK_H5(H5Sclose(mspace));
    CHECK_H5(H5Sclose(fspace));
    CHECK_H5(H5Pclose(dxpl));
    CHECK_H5(H5Dclose(dset));
    CHECK_H5(H5Fclose(file));

    return std::chrono::duration<double>(t1 - t0).count();
}

static double read_hdf5(const Options& opts,
                        std::vector<std::uint16_t>& out,
                        std::size_t rank_offset,
                        std::size_t rank_count,
                        bool use_mpio) {
    const hid_t fapl = create_fapl(use_mpio);
    const hid_t file = H5Fopen(opts.output_h5.c_str(), H5F_ACC_RDONLY, fapl);
    CHECK_H5(file);
    CHECK_H5(H5Pclose(fapl));

    const hid_t dset = H5Dopen2(file, "dataset", H5P_DEFAULT);
    CHECK_H5(dset);

    const hid_t dxpl = create_dxpl(use_mpio);
    const hid_t fspace = H5Dget_space(dset);
    CHECK_H5(fspace);

    if (rank_count > 0) {
        const hsize_t file_offset[1] = {static_cast<hsize_t>(rank_offset)};
        const hsize_t file_count[1] = {static_cast<hsize_t>(rank_count)};
        CHECK_H5(H5Sselect_hyperslab(fspace, H5S_SELECT_SET, file_offset, nullptr, file_count, nullptr));

        const hid_t mspace = H5Screate_simple(1, file_count, nullptr);
        CHECK_H5(mspace);

        auto t0 = std::chrono::steady_clock::now();
        {
            MANS_TIMING_SCOPE("hdf5/read");
            CHECK_H5(H5Dread(dset, H5T_NATIVE_UINT16, mspace, fspace, dxpl, out.data()));
        }
        auto t1 = std::chrono::steady_clock::now();

        CHECK_H5(H5Sclose(mspace));
        CHECK_H5(H5Sclose(fspace));
        CHECK_H5(H5Pclose(dxpl));
        CHECK_H5(H5Dclose(dset));
        CHECK_H5(H5Fclose(file));

        return std::chrono::duration<double>(t1 - t0).count();
    }

    CHECK_H5(H5Sselect_none(fspace));
    const hsize_t one[1] = {1};
    const hid_t mspace = H5Screate_simple(1, one, nullptr);
    CHECK_H5(mspace);
    CHECK_H5(H5Sselect_none(mspace));

    std::uint16_t dummy = 0;
    auto t0 = std::chrono::steady_clock::now();
    {
        MANS_TIMING_SCOPE("hdf5/read");
        CHECK_H5(H5Dread(dset, H5T_NATIVE_UINT16, mspace, fspace, dxpl, &dummy));
    }
    auto t1 = std::chrono::steady_clock::now();

    CHECK_H5(H5Sclose(mspace));
    CHECK_H5(H5Sclose(fspace));
    CHECK_H5(H5Pclose(dxpl));
    CHECK_H5(H5Dclose(dset));
    CHECK_H5(H5Fclose(file));

    return std::chrono::duration<double>(t1 - t0).count();
}

static void print_config(const Options& opts,
                         std::size_t total_elements,
                         std::size_t elem_size,
                         std::size_t chunk_elems,
                         int ranks,
                         bool use_mpio) {
    const double total_mb = (static_cast<double>(total_elements) * elem_size) / 1048576.0;
    const double rank_mb = total_mb / static_cast<double>(ranks);
    const double chunk_mb = (static_cast<double>(chunk_elems) * elem_size) / 1048576.0;

    std::cout << BOLD
              << "========================================\n"
              << "   H5Z-MANS HDF5 MPI Benchmark (U16)   \n"
              << "========================================" << RST << "\n";
    std::cout << "  dataset_mb(target): " << opts.dataset_mb << "\n";
    std::cout << "  dataset_mb(actual): " << total_mb << "\n";
    std::cout << "  per_rank_mb(avg):   " << rank_mb << "\n";
    std::cout << "  chunk_mb(actual):   " << chunk_mb << "\n";
    std::cout << "  filter:             " << filter_to_string(opts.filter) << "\n";
    std::cout << "  output:             " << opts.output_h5 << "\n";
    std::cout << "  ranks:              " << ranks << (use_mpio ? " (mpi)" : " (serial)") << "\n";

    if (opts.filter == FilterKind::Mans) {
        if (opts.threads_from_user) {
            std::cout << "  threads:            ";
            for (std::size_t i = 0; i < opts.threads.size(); ++i) {
                std::cout << opts.threads[i];
                if (i + 1 != opts.threads.size()) {
                    std::cout << ",";
                }
            }
            std::cout << " (user)\n";
        } else {
            const char* csv_env = std::getenv("MANS_THREAD_CSV");
            std::string csv_path = (csv_env && csv_env[0] != '\0') ? csv_env : "best_threads.csv";
            std::vector<int> auto_threads;
            if (resolve_auto_threads(csv_path, chunk_elems, 1, auto_threads)) {
                std::cout << "  threads:            ";
                for (std::size_t i = 0; i < auto_threads.size(); ++i) {
                    std::cout << auto_threads[i];
                    if (i + 1 != auto_threads.size()) {
                        std::cout << ",";
                    }
                }
                std::cout << " (auto, " << csv_path << ")\n";
            } else {
                std::cout << "  threads:            auto (" << csv_path << ")\n";
            }
        }
    }
    std::cout << "\n";
}

static double mb_from_bytes(double bytes) {
    return bytes / 1048576.0;
}

int main(int argc, char** argv) {
    MpiContext mpi(argc, argv);

    Options opts;
    try {
        opts = parse_args(argc, argv);
    } catch (const std::exception& e) {
        if (mpi.rank == 0) {
            std::cerr << RED << "Arg parse error: " << e.what() << RST << "\n";
            print_usage(argv[0]);
        }
        return 1;
    }

#if defined(H5_HAVE_PARALLEL)
    const bool use_mpio = true;
#else
    const bool use_mpio = false;
#endif

    const std::size_t elem_size = sizeof(std::uint16_t);
    mans::h5::data_gen::SyntheticConfig synth_cfg;
    synth_cfg.size_per_rank_mb = opts.dataset_mb;

    const std::size_t total_elements = mans::h5::data_gen::aligned_total_elements(
        synth_cfg.size_per_rank_mb, elem_size, synth_cfg.block_size);

    std::size_t rank_offset = 0;
    std::size_t rank_count = 0;
    split_even(total_elements, mpi.rank, mpi.ranks, rank_offset, rank_count);

    const std::size_t chunk_elems = chunk_elements(opts.chunk_mb, elem_size, total_elements);

    if (mpi.rank == 0) {
        print_config(opts, total_elements, elem_size, chunk_elems, mpi.ranks, use_mpio);
        if (opts.threads_from_user && opts.filter != FilterKind::Mans) {
            std::cerr << YLW << "[WARN] --threads is ignored unless --filter mans" << RST << "\n";
        }
    }

#if defined(H5_HAVE_PARALLEL)
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    if (mpi.rank == 0) {
        std::cout << "[1/8] Generating data...\n";
    }

    const auto data = mans::h5::data_gen::generate_synthetic_slice<std::uint16_t>(
        DEFAULT_ADM_THRESHOLD,
        synth_cfg,
        total_elements,
        rank_offset,
        rank_count);

#if defined(H5_HAVE_PARALLEL)
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    const mans::MansParams mans_params = build_mans_params(opts);

    const double raw_bytes_total = static_cast<double>(total_elements) * elem_size;
    const double raw_mb_total = mb_from_bytes(raw_bytes_total);
    const double raw_mb_rank = raw_mb_total / static_cast<double>(mpi.ranks);

    constexpr int kIters = 11;
    TimingSums sums{};

    MANS_TIMING_RESET();

    for (int iter = 0; iter < kIters; ++iter) {
        if (mpi.rank == 0) {
            std::cout << "[" << (iter + 1) << "/" << kIters << "] Run";
            if (iter == 0) {
                std::cout << " (warmup)";
            }
            std::cout << "...\n";
        }
        {
            const std::string iter_env = std::to_string(iter + 1);
            setenv("MANS_TIMING_ITER", iter_env.c_str(), 1);
        }

        MANS_TIMING_RUN_SCOPE();
        MANS_TIMING_SCOPE("hdf5/run_total");

        std::uint64_t compressed_bytes = 0;
        double write_s = write_hdf5(opts, mans_params, data,
                                    total_elements, chunk_elems,
                                    rank_offset, rank_count,
                                    use_mpio,
                                    compressed_bytes);

#if defined(H5_HAVE_PARALLEL)
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(&compressed_bytes, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
#endif

        drop_cache_best_effort(opts.output_h5);

#if defined(H5_HAVE_PARALLEL)
        MPI_Barrier(MPI_COMM_WORLD);
#endif

        std::vector<std::uint16_t> recovered(rank_count);
        double read_s = read_hdf5(opts, recovered, rank_offset, rank_count, use_mpio);

#if defined(H5_HAVE_PARALLEL)
        MPI_Barrier(MPI_COMM_WORLD);
#endif

        bool local_ok = true;
        if (rank_count > 0) {
            local_ok = (std::memcmp(data.data(), recovered.data(), rank_count * elem_size) == 0);
        }

#if defined(H5_HAVE_PARALLEL)
        int ok_int = local_ok ? 1 : 0;
        int global_ok = 0;
        MPI_Allreduce(&ok_int, &global_ok, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
        local_ok = (global_ok != 0);
#endif

        if (!local_ok) {
            if (mpi.rank == 0) {
                std::cerr << RED << "[ERROR] Data verification failed." << RST << "\n";
            }
            return 1;
        }

        if (iter == 0) {
            continue;
        }

        double global_write_s = write_s;
        double global_read_s = read_s;
        double avg_write_s = write_s;
        double avg_read_s = read_s;

#if defined(H5_HAVE_PARALLEL)
        MPI_Allreduce(&write_s, &global_write_s, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&read_s, &global_read_s, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&write_s, &avg_write_s, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&read_s, &avg_read_s, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        avg_write_s /= static_cast<double>(mpi.ranks);
        avg_read_s /= static_cast<double>(mpi.ranks);
#endif

        const double ratio = compressed_bytes > 0
            ? (raw_bytes_total / static_cast<double>(compressed_bytes))
            : 0.0;

        sums.global_write_s += global_write_s;
        sums.global_read_s += global_read_s;
        sums.avg_write_s += avg_write_s;
        sums.avg_read_s += avg_read_s;
        sums.ratio += ratio;
    }

    if (mpi.rank == 0) {
        const double denom = static_cast<double>(kIters - 1);
        const double avg_global_write_s = sums.global_write_s / denom;
        const double avg_global_read_s = sums.global_read_s / denom;
        const double avg_rank_write_s = sums.avg_write_s / denom;
        const double avg_rank_read_s = sums.avg_read_s / denom;
        const double avg_ratio = sums.ratio / denom;

        const double global_write_thr = avg_global_write_s > 0.0 ? (raw_mb_total / avg_global_write_s) : 0.0;
        const double global_read_thr = avg_global_read_s > 0.0 ? (raw_mb_total / avg_global_read_s) : 0.0;
        const double rank_write_thr = avg_rank_write_s > 0.0 ? (raw_mb_rank / avg_rank_write_s) : 0.0;
        const double rank_read_thr = avg_rank_read_s > 0.0 ? (raw_mb_rank / avg_rank_read_s) : 0.0;

        std::cout << "\n" << BOLD << "[Summary]" << RST << "\n";
        std::cout << "  avg_ratio:            " << avg_ratio << "x\n";
        std::cout << "  global_write_s:       " << avg_global_write_s
                  << " (" << global_write_thr << " MB/s)\n";
        std::cout << "  global_read_s:        " << avg_global_read_s
                  << " (" << global_read_thr << " MB/s)\n";
        std::cout << "  rank_avg_write_s:     " << avg_rank_write_s
                  << " (" << rank_write_thr << " MB/s)\n";
        std::cout << "  rank_avg_read_s:      " << avg_rank_read_s
                  << " (" << rank_read_thr << " MB/s)\n";

        MANS_TIMING_DUMP(TIMING_CSV_PATH);
        std::cout << "  timing_csv:           " << GRN << TIMING_CSV_PATH << RST << "\n";
    }

    if (0 > H5close()) {
        if (mpi.rank == 0) {
            std::cerr << RED << "Error in H5close" << RST << "\n";
        }
        return 1;
    }

    return 0;
}
