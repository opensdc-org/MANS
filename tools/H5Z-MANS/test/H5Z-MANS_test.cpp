#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>

#include <hdf5.h>
// Assume the config header still exists
#include "H5Z-MANS_config.h"
#include "../include/sz3_config_min.h"

// ==========================================
// 1. Constants & Definitions
// ==========================================

// Filter IDs   
#define FILTER_ID_DEFLATE 1
#define FILTER_ID_ZSTD    32015
#define FILTER_ID_MANS    32001
#define FILTER_ID_SZ3     32024  

// Simulation Control (Defaults)
const double SIM_RATIO_SMOOTH = 1.0;
const double SIM_RATIO_SPIKE  = 0.0;
const double SIM_RATIO_RANDOM = 0.0;
const int    SIM_NOISE_RANGE  = 20;
const size_t SIM_BLOCK_SIZE   = 512;

// Color Definitions
const std::string RST  = "\033[0m";
const std::string RED  = "\033[1;31m";
const std::string GRN  = "\033[1;32m";
const std::string YLW  = "\033[1;33m";
const std::string BLU  = "\033[1;36m";
const std::string BOLD = "\033[1m";

// ==========================================
// 2. Runtime Configuration Struct
// ==========================================
struct RunOptions {
    double  data_size_mb = 256.0;            // Default: 256 MB
    double  chunk_mb     = 32.0;             // Default: 32 MB
    int     filter_id    = FILTER_ID_MANS;   // Default: MANS
    std::string config_file;
    std::string output_h5;
    std::string input_bin;
};

// ==========================================
// 3. Helper Functions
// ==========================================

#define CHECK_H5(func_call) \
    do { \
        if ((func_call) < 0) { \
            std::cerr << RED << "[Error] HDF5 call failed at line " << __LINE__ << ": " #func_call << RST << "\n"; \
            std::exit(1); \
        } \
    } while (0)

namespace fs = std::filesystem;

size_t get_file_size(const std::string& filename) {
    if (fs::exists(filename)) return fs::file_size(filename);
    return 0;
}

template <typename T>
void save_bin(const std::string& filename, const std::vector<T>& data) {
    std::ofstream outfile(filename, std::ios::binary);
    if (outfile) {
        outfile.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(T));
        std::cout << "  -> Saved dump: " << GRN << filename << RST << "\n";
    }
}

template <typename T>
std::vector<T> load_bin(const std::string& filename) {
    std::ifstream infile(filename, std::ios::binary | std::ios::ate);
    if (!infile) throw std::runtime_error("Cannot open file: " + filename);
    size_t bytes = infile.tellg();
    infile.seekg(0);
    std::vector<T> data(bytes / sizeof(T));
    infile.read(reinterpret_cast<char*>(data.data()), bytes);
    return data;
}

// ==========================================
// 4. Data Generator
// ==========================================
template <typename T>
std::vector<T> generate_synthetic_data(uint32_t config_threshold, double target_size_mb) {
    size_t element_size = sizeof(T);
    size_t total_bytes = static_cast<size_t>(target_size_mb * 1024.0 * 1024.0);
    size_t num_elements = total_bytes / element_size;

    // Align to block size
    if (num_elements % SIM_BLOCK_SIZE != 0) {
        num_elements = ((num_elements / SIM_BLOCK_SIZE) + 1) * SIM_BLOCK_SIZE;
    }

    std::vector<T> data(num_elements);
    std::mt19937 rng(42);

    std::discrete_distribution<int> type_dist({SIM_RATIO_SMOOTH, SIM_RATIO_SPIKE, SIM_RATIO_RANDOM});
    std::uniform_int_distribution<int> noise_dist(0, SIM_NOISE_RANGE);
    T max_val = std::numeric_limits<T>::max();
    std::uniform_int_distribution<unsigned long long> full_range_dist(0, max_val);

    size_t num_blocks = num_elements / SIM_BLOCK_SIZE;
    size_t cnt_smooth = 0, cnt_spike = 0, cnt_random = 0;

    for (size_t b = 0; b < num_blocks; ++b) {
        size_t start = b * SIM_BLOCK_SIZE;
        size_t end = std::min(start + SIM_BLOCK_SIZE, num_elements);
        int block_type = type_dist(rng);

        if (block_type == 0) { // Smooth
            cnt_smooth++;
            T block_base = static_cast<T>(full_range_dist(rng));
            for (size_t i = start; i < end; ++i) {
                int noise = noise_dist(rng);
                if (block_base > (max_val - SIM_NOISE_RANGE)) data[i] = block_base - static_cast<T>(noise);
                else data[i] = block_base + static_cast<T>(noise);
            }
        }
        else if (block_type == 1) { // Spike
            cnt_spike++;
            T block_base = static_cast<T>(full_range_dist(rng));
            for (size_t i = start; i < end; ++i) data[i] = block_base;
            if (start + 1 < end) {
                uint32_t spike_gap = config_threshold + 500;
                if (static_cast<unsigned long long>(block_base) + spike_gap > max_val)
                    data[start + 1] = static_cast<T>(block_base - spike_gap);
                else
                    data[start + 1] = static_cast<T>(block_base + spike_gap);
            }
        }
        else { // Random
            cnt_random++;
            for (size_t i = start; i < end; ++i) data[i] = static_cast<T>(full_range_dist(rng));
        }
    }

    std::cout << "  -> [Gen] Target Size: " << target_size_mb << " MB\n";
    std::cout << "  -> [Gen] Elements:    " << num_elements << "\n";
    return data;
}

// ==========================================
// 5. Core Tester Class
// ==========================================
template <typename T>
class Tester {
public:
    Tester(const mans::h5::MansConfig& cfg, hid_t h5_type, const RunOptions& opts)
        : config(cfg), h5_native_type(h5_type), options(opts)
    {
        dump_input_file = options.output_h5 + ".input.bin";
        dump_recon_file = options.output_h5 + ".recon.bin";
    }

    void run() {
        prepare_data();
        compress_write();
        decompress_read();
        verify();
    }

private:
    mans::h5::MansConfig config;
    hid_t h5_native_type;
    RunOptions options;

    std::string dump_input_file;
    std::string dump_recon_file;

    std::vector<T> host_data;
    std::vector<T> recon_data;
    double raw_size_mb = 0;

    void prepare_data() {
        std::cout << BOLD << "[1] Preparing Data..." << RST << "\n";
        bool is_generated = false;

        if (!options.input_bin.empty()) {
            std::cout << "  -> Loading external file: " << options.input_bin << "\n";
            try {
                host_data = load_bin<T>(options.input_bin);
            } catch (std::exception& e) {
                std::cerr << YLW << "  -> [WARN] Failed to load file. Falling back to generator." << RST << "\n";
                options.input_bin = "";
            }
        }

        if (options.input_bin.empty()) {
            std::cout << "  -> Generating synthetic data (Target: " << options.data_size_mb << " MB)...\n";
            host_data = generate_synthetic_data<T>(config.get_params().adm_threshold, options.data_size_mb);
            is_generated = true;
        }

        raw_size_mb = (host_data.size() * sizeof(T)) / 1048576.0;
        std::cout << "  -> Raw Size: " << BLU << raw_size_mb << " MB" << RST << "\n";

        if (is_generated) {
            std::cout << "  -> [Auto-Save] Saving generated raw data...\n";
            save_bin(dump_input_file, host_data);
        }
    }

    void compress_write() {
        std::cout << BOLD << "[2] Writing HDF5 (Compression)..." << RST << "\n";
        std::cout << "  -> Output File: " << GRN << options.output_h5 << RST << "\n";

        hid_t f_id = H5Fcreate(options.output_h5.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        CHECK_H5(f_id);

        hsize_t dims[1] = {host_data.size()};
        hid_t s_id = H5Screate_simple(1, dims, NULL);
        hid_t p_id = H5Pcreate(H5P_DATASET_CREATE);

        // -- Change: Calculate Chunk Size in Elements from MB --
        size_t elem_size = sizeof(T);
        size_t chunk_bytes = static_cast<size_t>(options.chunk_mb * 1024.0 * 1024.0);
        hsize_t chunk_elems = chunk_bytes / elem_size;
        
        // Safety check
        if (chunk_elems == 0) chunk_elems = 1;
        if (chunk_elems > dims[0]) chunk_elems = dims[0];

        hsize_t chunk[1] = { chunk_elems };

        std::cout << "  -> Chunk Setting: " << options.chunk_mb << " MB (" << chunk[0] << " elements)\n";
        CHECK_H5(H5Pset_chunk(p_id, 1, chunk));

        // Filter selection based on options
        if (options.filter_id == FILTER_ID_DEFLATE) {
            unsigned int gzip_level = 6;
            std::cout << "  -> Filter: " << BLU << "Standard GZIP (Level " << gzip_level << ")" << RST << "\n";
            CHECK_H5(H5Pset_deflate(p_id, gzip_level));

        } else if (options.filter_id == FILTER_ID_ZSTD) {
            unsigned int zstd_level = 3;
            std::cout << "  -> Filter: " << BLU << "Zstandard (Level " << zstd_level << ")" << RST << "\n";
            htri_t avail = H5Zfilter_avail(FILTER_ID_ZSTD);
            if (!avail) {
                std::cerr << RED << "[Error] Zstandard filter (ID " << FILTER_ID_ZSTD << ") not found! " << RST << "\n";
                std::exit(1);
            }
            unsigned int cd_values[1] = { zstd_level };
            CHECK_H5(H5Pset_filter(p_id, FILTER_ID_ZSTD, H5Z_FLAG_OPTIONAL, 1, cd_values));

        } else if (options.filter_id == FILTER_ID_MANS) {
            std::cout << "  -> Filter: " << BLU << "MANS Custom (ID " << FILTER_ID_MANS << ")" << RST << "\n";
            std::vector<unsigned int> cd = config.to_cd_values();
            CHECK_H5(H5Pset_filter(p_id, FILTER_ID_MANS, 0, cd.size(), cd.data()));

        } else if (options.filter_id == FILTER_ID_SZ3) {
            std::cout << "  -> Filter: " << BLU << "SZ3 Compressor (ID " << FILTER_ID_SZ3 << ")" << RST << "\n";
            
            htri_t avail = H5Zfilter_avail(FILTER_ID_SZ3);
            if (!avail) {
                std::cerr << RED << "[Error] SZ3 filter not found in HDF5_PLUGIN_PATH!" << RST << "\n";
                std::exit(1);
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
            size_t cd_nelmts = static_cast<size_t>(
                std::ceil(sz3_conf.size_est() / 1.0 / sizeof(int)));
            std::vector<unsigned int> cd_values(cd_nelmts, 0);
            auto buffer = reinterpret_cast<unsigned char*>(cd_values.data());
            auto conf_size_real = sz3_conf.save(buffer);
            cd_nelmts = static_cast<size_t>(std::ceil(conf_size_real / 1.0 / sizeof(int)));
            CHECK_H5(H5Pset_filter(p_id, FILTER_ID_SZ3, H5Z_FLAG_MANDATORY, cd_nelmts, cd_values.data()));
            
        } else {
            std::cerr << RED << "[Error] Unknown Filter ID requested: " << options.filter_id << RST << "\n";
            std::exit(1);
        }

        hid_t d_id = H5Dcreate2(f_id, "dataset", h5_native_type, s_id, H5P_DEFAULT, p_id, H5P_DEFAULT);

        auto t1 = std::chrono::high_resolution_clock::now();
        CHECK_H5(H5Dwrite(d_id, h5_native_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, host_data.data()));
        auto t2 = std::chrono::high_resolution_clock::now();
        CHECK_H5(H5Fflush(f_id, H5F_SCOPE_GLOBAL));

        double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        double throughput = raw_size_mb / (ms/1000.0);

        std::cout << "  -> Write Time: " << BLU << ms << " ms" << RST
                  << " (" << BLU << throughput << " MB/s" << RST << ")\n";

        H5Dclose(d_id); H5Pclose(p_id); H5Sclose(s_id); H5Fclose(f_id);

        size_t c_size = get_file_size(options.output_h5);
        double ratio = (double)(host_data.size()*sizeof(T))/c_size;

        std::cout << "  -> File Size:  " << BLU << c_size / 1024.0 << " KB" << RST
                  << " (Ratio: " << BLU << ratio << "x" << RST << ")\n";
    }

    void decompress_read() {
        std::cout << BOLD << "[3] Reading HDF5 (Decompression)..." << RST << "\n";
        recon_data.resize(host_data.size());

        hid_t f_id = H5Fopen(options.output_h5.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        CHECK_H5(f_id);
        hid_t d_id = H5Dopen2(f_id, "dataset", H5P_DEFAULT);

        auto t1 = std::chrono::high_resolution_clock::now();
        CHECK_H5(H5Dread(d_id, h5_native_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, recon_data.data()));
        auto t2 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        double throughput = raw_size_mb / (ms/1000.0);

        std::cout << "  -> Read Time:  " << BLU << ms << " ms" << RST
                  << " (" << BLU << throughput << " MB/s" << RST << ")\n";

        H5Dclose(d_id); H5Fclose(f_id);
    }

    void verify() {
        std::cout << BOLD << "[4] Verifying..." << RST << "\n";
        if (host_data == recon_data) {
            std::cout << GRN << "   [SUCCESS] Data Exact Match" << RST << "\n";
        } else {
            std::cerr << RED << "   [FAILURE] Data Mismatch Detected!" << RST << "\n";
            exit(1);
        }
    }
};

// ==========================================
// 6. Main Function & Arg Parsing
// ==========================================
void print_usage(const char* prog_name) {
    std::cerr << YLW << "Usage: " << prog_name << " <config_file> <output.h5> [input.bin] [OPTIONS]" << RST << "\n";
    std::cerr << "Options:\n";
    std::cerr << "  --size <MB>        Set synthetic data size in MB (default: 256.0)\n";
    std::cerr << "  --chunk <MB>       Set chunk size in MB (default: 32.0)\n";
    std::cerr << "  --filter <name>    Set filter: mans, zstd, deflate (default: mans)\n";
}

int main(int argc, char** argv) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    RunOptions opts;
    std::vector<std::string> args;

    // Simple argument parser
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--size" && i + 1 < argc) {
            opts.data_size_mb = std::stod(argv[++i]);
        }
        else if (arg == "--chunk" && i + 1 < argc) {
            opts.chunk_mb = std::stod(argv[++i]);
        }
        else if (arg == "--filter" && i + 1 < argc) {
            std::string f = argv[++i];
            if (f == "deflate") opts.filter_id = FILTER_ID_DEFLATE;
            else if (f == "zstd") opts.filter_id = FILTER_ID_ZSTD;
            else if (f == "mans") opts.filter_id = FILTER_ID_MANS;
            else if (f == "sz3")  opts.filter_id = FILTER_ID_SZ3;
            else opts.filter_id = std::stoi(f); // Allow manual ID
        }
        else {
            args.push_back(arg);
        }
    }

    if (args.size() < 2) {
        std::cerr << RED << "Missing required arguments!" << RST << "\n";
        print_usage(argv[0]);
        return 1;
    }

    opts.config_file = args[0];
    opts.output_h5   = args[1];
    if (args.size() >= 3) opts.input_bin = args[2];

    mans::h5::MansConfig config;
    try { config.load(opts.config_file); }
    catch (std::exception& e) { std::cerr << RED << "Config Error: " << e.what() << RST << "\n"; return 1; }

    auto dtype = config.get_params().dtype;

    std::cout << BOLD << "========================================\n";
    std::cout << "    H5Z-MANS Integration Test Runner    \n";
    std::cout << "========================================" << RST << "\n";
    std::cout << " Config: " << opts.config_file << "\n";
    std::cout << " Output: " << opts.output_h5 << "\n";
    std::cout << " Size:   " << opts.data_size_mb << " MB\n";
    std::cout << " Filter: " << opts.filter_id << "\n";
    std::cout << " Chunk:  " << opts.chunk_mb << " MB\n\n";

    if (dtype == mans::DataType::U16) {
        Tester<uint16_t> t(config, H5T_NATIVE_UINT16, opts);
        t.run();
    } else {
        Tester<uint32_t> t(config, H5T_NATIVE_UINT32, opts);
        t.run();
    }

    /* Close and release resources */
    if (0 > H5close()) { 
        printf("Error in H5close\n"); 
        exit(1); 
    }
    
    std::cout << "Done\n";
    return 0;
}