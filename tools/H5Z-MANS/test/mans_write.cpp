#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>

#include <hdf5.h>
#include <mpi.h>
#include "mans_defs.h"

#define FILTER_ID_MANS 32001
#define CHECK_H5(x) do { if ((x) < 0) { std::fprintf(stderr, "HDF5 failed: %s\n", #x); std::exit(1); } } while (0)

int main(int argc, char** argv) {
    double chunk_size_mb = 8.0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--chunk-size-mb") == 0) {
            char* end = nullptr;
            double value = std::strtod(argv[++i], &end);
            if (end == nullptr || *end != '\0' || value <= 0.0) {
                std::fprintf(stderr, "invalid --chunk-size-mb: %s\n", argv[i]);
                return 1;
            }
            chunk_size_mb = value;
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
    char in[64], out[64];
    std::snprintf(in, sizeof(in), "datasets/rank%d.bin", rank);
    std::snprintf(out, sizeof(out), "datasets/rank%d.h5", rank);
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

    stage("building HDF5 dataset");
    mans::MansParams p{};
    p.backend = mans::Backend::CPU; p.dtype = mans::DataType::U16; p.adm_threshold = 4000;
    p.mode=mans::Mode::R;
    p.adm_decide_threads = 16; p.adm_center_calc_threads = 32; p.adm_encode_threads = 32;
    p.adm_warp_reduce_threads = 32; p.adm_fill_tail_threads = 16; p.adm_write_back_threads = 16;
    p.adm_restore_signals_threads = 32; p.adm_decode_values_threads = 16;
    std::vector<unsigned int> cd(sizeof(p) / sizeof(unsigned int), 0);
    std::memcpy(cd.data(), &p, sizeof(p));

    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    hid_t file = H5Fcreate(out, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);

    hsize_t chunk_bytes = static_cast<hsize_t>(chunk_size_mb * 1024.0 * 1024.0);
    if (rank == 0) std::printf("[mans_write] chunk-size-mb=%.6g\n", chunk_size_mb);

    hsize_t dims[1] = {(hsize_t)elems};
    hsize_t chunk[1] = {chunk_bytes / (hsize_t)sizeof(std::uint16_t)};
    if (chunk[0] == 0 || chunk[0] > dims[0]) chunk[0] = dims[0];

    hid_t space = H5Screate_simple(1, dims, nullptr);
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    CHECK_H5(H5Pset_chunk(dcpl, 1, chunk));
    CHECK_H5(H5Pset_filter(dcpl, FILTER_ID_MANS, 0, cd.size(), cd.data()));
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
