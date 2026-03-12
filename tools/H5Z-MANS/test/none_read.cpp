#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <hdf5.h>
#include <mpi.h>

#define CHECK_H5(x) do { if ((x) < 0) { std::fprintf(stderr, "HDF5 failed: %s\n", #x); std::exit(1); } } while (0)

int main(int argc, char** argv) {
    const char* h5_template = "datasets/rank%d.h5";
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--h5-template") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for --h5-template\n");
                return 1;
            }
            h5_template = argv[++i];
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
        if (rank == 0) std::printf("[none_read] %s\n", msg);
    };

    stage("opening input hdf5");
    char in[4096];
    int in_len = std::snprintf(in, sizeof(in), h5_template, rank);
    if (in_len < 0 || static_cast<size_t>(in_len) >= sizeof(in)) {
        std::fprintf(stderr, "rank %d invalid --h5-template: %s\n", rank, h5_template);
        MPI_Finalize();
        return 1;
    }

    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    hid_t file = H5Fopen(in, H5F_ACC_RDONLY, fapl);
    H5Pclose(fapl);
    if (file < 0) {
        std::fprintf(stderr, "rank %d open failed: %s\n", rank, in);
        MPI_Finalize();
        return 1;
    }

    hid_t dset = H5Dopen2(file, "data", H5P_DEFAULT);
    if (dset < 0) {
        std::fprintf(stderr, "rank %d dataset open failed: %s:data\n", rank, in);
        H5Fclose(file);
        MPI_Finalize();
        return 1;
    }
    hid_t space = H5Dget_space(dset);
    if (space < 0) {
        std::fprintf(stderr, "rank %d dataspace open failed: %s:data\n", rank, in);
        H5Dclose(dset);
        H5Fclose(file);
        MPI_Finalize();
        return 1;
    }

    hsize_t dims[1] = {0};
    CHECK_H5(H5Sget_simple_extent_dims(space, dims, nullptr));
    size_t elems = (size_t)dims[0];
    size_t bytes = elems * sizeof(std::uint16_t);
    std::vector<std::uint16_t> data(elems);

    stage("reading data timing...");
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    CHECK_H5(H5Dread(dset, H5T_NATIVE_USHORT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data()));
    double t1 = MPI_Wtime();

    stage("read complete, collecting stats");
    H5Sclose(space);
    H5Dclose(dset);
    H5Fclose(file);

    double sec = t1 - t0, max_sec = 0.0;
    double local_bytes = (double)bytes, total_bytes = 0.0;
    MPI_Reduce(&sec, &max_sec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_bytes, &total_bytes, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        double mib = total_bytes / (1024.0 * 1024.0);
        double bw = (max_sec > 0.0) ? (mib / max_sec) : 0.0;
        double per_rank_approx = (nprocs > 0) ? (bw / (double)nprocs) : 0.0;
        std::printf("none_read ranks=%d total=%.1f MiB time=%.4f s throughput=%.2f MiB/s per_rank~%.2f MiB/s\n",
                    nprocs, mib, max_sec, bw, per_rank_approx);
    }
    MPI_Finalize();
    return 0;
}
