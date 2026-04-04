#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cerrno>
#include <limits>
#include <vector>

#include <hdf5.h>
#include <mpi.h>
#include "mans_defs.h"
#include "H5Z-MANS_filter_ids.h"

#define CHECK_H5(x) do { if ((x) < 0) { std::fprintf(stderr, "HDF5 failed: %s\n", #x); std::exit(1); } } while (0)

static bool parse_positive_hsize(const char* text, hsize_t& out) {
    if (!text || *text == '\0') {
        return false;
    }
    char* end = nullptr;
    errno = 0;
    unsigned long long v = std::strtoull(text, &end, 10);
    if (errno != 0 || end == nullptr || *end != '\0' || v == 0) {
        return false;
    }
    if (v > static_cast<unsigned long long>(std::numeric_limits<hsize_t>::max())) {
        return false;
    }
    out = static_cast<hsize_t>(v);
    return true;
}

static bool product_with_overflow(const std::vector<hsize_t>& dims, hsize_t& out) {
    out = 1;
    for (hsize_t d : dims) {
        if (d == 0) return false;
        if (out > (std::numeric_limits<hsize_t>::max() / d)) {
            return false;
        }
        out *= d;
    }
    return true;
}

int main(int argc, char** argv) {
    double chunk_size_mb = 8.0;
    const char* bin_template = "datasets/rank%d.bin";
    const char* h5_template = "datasets/rank%d.h5";
    std::vector<hsize_t> cli_dims;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--chunk-size-mb") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for --chunk-size-mb\n");
                return 1;
            }
            char* end = nullptr;
            double value = std::strtod(argv[++i], &end);
            if (end == nullptr || *end != '\0' || value <= 0.0) {
                std::fprintf(stderr, "invalid --chunk-size-mb: %s\n", argv[i]);
                return 1;
            }
            chunk_size_mb = value;
        } else if (std::strcmp(argv[i], "--bin-template") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for --bin-template\n");
                return 1;
            }
            bin_template = argv[++i];
        } else if (std::strcmp(argv[i], "--h5-template") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for --h5-template\n");
                return 1;
            }
            h5_template = argv[++i];
        } else if (std::strcmp(argv[i], "--dims") == 0) {
            int consumed = 0;
            while (i + 1 < argc && std::strncmp(argv[i + 1], "--", 2) != 0) {
                hsize_t dim = 0;
                if (!parse_positive_hsize(argv[i + 1], dim)) {
                    std::fprintf(stderr, "invalid dim value: %s\n", argv[i + 1]);
                    return 1;
                }
                cli_dims.push_back(dim);
                ++i;
                ++consumed;
            }
            if (consumed == 0) {
                std::fprintf(stderr, "missing values for --dims\n");
                return 1;
            }
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
            return 1;
        }
    }

    MPI_Init(&argc, &argv);
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    auto stage = [&](const char* msg) {
        if (rank == 0) std::printf("[mans_write] %s\n", msg);
    };

    stage("loading input");
    char in[4096], out[4096];
    int in_len = std::snprintf(in, sizeof(in), bin_template, rank);
    int out_len = std::snprintf(out, sizeof(out), h5_template, rank);
    if (in_len < 0 || static_cast<size_t>(in_len) >= sizeof(in)) {
        std::fprintf(stderr, "rank %d invalid --bin-template: %s\n", rank, bin_template);
        MPI_Finalize();
        return 1;
    }
    if (out_len < 0 || static_cast<size_t>(out_len) >= sizeof(out)) {
        std::fprintf(stderr, "rank %d invalid --h5-template: %s\n", rank, h5_template);
        MPI_Finalize();
        return 1;
    }
    std::FILE* fp = std::fopen(in, "rb");
    if (!fp) { std::fprintf(stderr, "rank %d missing input: %s\n", rank, in); return 1; }
    std::fseek(fp, 0, SEEK_END);
    long bytes_l = std::ftell(fp);
    std::rewind(fp);
    if (bytes_l <= 0 || (bytes_l % (long)sizeof(std::uint16_t)) != 0) { std::fprintf(stderr, "bad input size: %s\n", in); return 1; }
    size_t bytes = (size_t)bytes_l;
    size_t elems = bytes / sizeof(std::uint16_t);
    std::vector<std::uint16_t> data(elems);
    if (std::fread(data.data(), 1, bytes, fp) != bytes) { std::fprintf(stderr, "read failed: %s\n", in); return 1; }
    std::fclose(fp);

    std::vector<hsize_t> dims = cli_dims;
    if (dims.empty()) {
        dims.push_back(static_cast<hsize_t>(elems));
    } else {
        hsize_t dims_elems = 0;
        if (!product_with_overflow(dims, dims_elems) || dims_elems != static_cast<hsize_t>(elems)) {
            std::fprintf(stderr, "rank %d --dims element count mismatch with input file: dims_elems=%llu file_elems=%llu\n",
                         rank,
                         static_cast<unsigned long long>(dims_elems),
                         static_cast<unsigned long long>(elems));
            MPI_Finalize();
            return 1;
        }
    }

    hsize_t chunk_bytes = static_cast<hsize_t>(chunk_size_mb * 1024.0 * 1024.0);
    hsize_t target_chunk_elems = chunk_bytes / static_cast<hsize_t>(sizeof(std::uint16_t));
    if (target_chunk_elems == 0) target_chunk_elems = 1;
    std::vector<hsize_t> chunk = dims;
    if (chunk.size() == 1) {
        if (chunk[0] > target_chunk_elems) chunk[0] = target_chunk_elems;
    } else {
        hsize_t tail_prod = 1;
        for (size_t i = 1; i < dims.size(); ++i) {
            if (tail_prod > (std::numeric_limits<hsize_t>::max() / dims[i])) {
                std::fprintf(stderr, "rank %d dims overflow when computing chunk shape\n", rank);
                MPI_Finalize();
                return 1;
            }
            tail_prod *= dims[i];
        }
        hsize_t lead = target_chunk_elems / tail_prod;
        if (lead == 0) lead = 1;
        if (lead > dims[0]) lead = dims[0];
        chunk[0] = lead;
    }
    if (rank == 0) std::printf("[mans_write] chunk-size-mb=%.6g\n", chunk_size_mb);

    stage("building HDF5 dataset");
    const unsigned int cd[1] = {mans::Mode::R};

    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    hid_t file = H5Fcreate(out, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);

    const int ndims = static_cast<int>(dims.size());
    hid_t space = H5Screate_simple(ndims, dims.data(), nullptr);
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    CHECK_H5(H5Pset_chunk(dcpl, ndims, chunk.data()));
    CHECK_H5(H5Pset_filter(dcpl, H5Z_FILTER_MANS_ID, 0, 1, cd));
    hid_t dset = H5Dcreate2(file, "data", H5T_NATIVE_USHORT, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    H5Pclose(dcpl);

    stage("writing data timing...");
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    CHECK_H5(H5Dwrite(dset, H5T_NATIVE_USHORT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data()));
    CHECK_H5(H5Fflush(file, H5F_SCOPE_GLOBAL));
    double t1 = MPI_Wtime();

    stage("write complete, collecting stats");
    H5Dclose(dset); H5Sclose(space); H5Fclose(file);
    double sec = t1 - t0, max_sec = 0.0;
    double local_bytes = (double)bytes, total_bytes = 0.0;
    double local_file_bytes = 0.0, total_file_bytes = 0.0;
    std::FILE* ofp = std::fopen(out, "rb");
    if (ofp) {
        std::fseek(ofp, 0, SEEK_END);
        long out_bytes_l = std::ftell(ofp);
        if (out_bytes_l > 0) local_file_bytes = (double)out_bytes_l;
        std::fclose(ofp);
    }
    MPI_Reduce(&sec, &max_sec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_bytes, &total_bytes, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_file_bytes, &total_file_bytes, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        double mib = total_bytes / (1024.0 * 1024.0);
        double bw = (max_sec > 0.0) ? (mib / max_sec) : 0.0;
        double per_rank_approx = (nprocs > 0) ? (bw / (double)nprocs) : 0.0;
        double comp_ratio_x = (total_file_bytes > 0.0) ? (total_bytes / total_file_bytes) : 0.0;
        std::printf("mans_write ranks=%d total=%.1f MiB time=%.4f s throughput=%.2f MiB/s per_rank~%.2f MiB/s comp_ratio=%.2fx\n",
                    nprocs, mib, max_sec, bw, per_rank_approx, comp_ratio_x);
    }
    MPI_Finalize();
    return 0;
}
