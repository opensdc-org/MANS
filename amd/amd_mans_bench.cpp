// AMD MANS benchmark driver.
//
// Runs the full MANS pipeline (GPU ADM -> GPU ANS, and ANS -> ADM on decode) over
// a list of raw input files with dims=1 geometry, reports end-to-end throughput in
// GB/s (raw bytes / GPU kernel time) plus the ADM per-kernel breakdown:
//
//   * fused compress  : one decoupled kernel (adm_comp_map_ms)
//   * unfused compress: mapping kernel + thrust scan + concat kernel (separate)
//   * decompress      : one fused decode kernel (adm_decomp_decode_ms)
//
// Usage:
//   amd_mans_bench <output.csv> <input...> [--warmup N] [--iters N]
//
// The dtype is inferred from the file extension (.u2 -> uint16, .u4 -> uint32);
// the dataset label is the immediate parent directory name.

#include "../mans_api.hpp"
#include "../mans_utils.h"
#include "adm/mans_amd_adm.h"
#include "ans/mans_amd_ans.h"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

namespace {

void check_hip(hipError_t e, const char* what) {
    if (e != hipSuccess) {
        std::cerr << "[fatal] " << what << ": " << hipGetErrorString(e) << "\n";
        std::exit(1);
    }
}

// Times a synchronous device stage (the ANS stages synchronize internally).
template <typename F>
double time_stage(hipStream_t stream, F&& fn) {
    hipEvent_t start, stop;
    check_hip(hipEventCreate(&start), "hipEventCreate start");
    check_hip(hipEventCreate(&stop), "hipEventCreate stop");
    check_hip(hipEventRecord(start, stream), "hipEventRecord start");
    fn();
    check_hip(hipEventRecord(stop, stream), "hipEventRecord stop");
    check_hip(hipEventSynchronize(stop), "hipEventSynchronize stop");
    float ms = 0.0f;
    check_hip(hipEventElapsedTime(&ms, start, stop), "hipEventElapsedTime");
    check_hip(hipEventDestroy(start), "hipEventDestroy start");
    check_hip(hipEventDestroy(stop), "hipEventDestroy stop");
    return static_cast<double>(ms);
}

bool ends_with(const std::string& s, const char* suffix) {
    const std::size_t n = std::strlen(suffix);
    return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

std::string parent_dir_name(const std::string& path) {
    const std::string::size_type slash = path.find_last_of('/');
    const std::string dir = (slash == std::string::npos) ? std::string() : path.substr(0, slash);
    const std::string::size_type slash2 = dir.find_last_of('/');
    return (slash2 == std::string::npos) ? dir : dir.substr(slash2 + 1);
}

std::string base_name(const std::string& path) {
    const std::string::size_type slash = path.find_last_of('/');
    return (slash == std::string::npos) ? path : path.substr(slash + 1);
}

// "vx_1073726487_4kB.u2" -> "4kB"
std::string size_label(const std::string& path) {
    const std::string::size_type dot = path.find_last_of('.');
    const std::string base = (dot == std::string::npos) ? path : path.substr(0, dot);
    const std::string::size_type us = base.find_last_of('_');
    return (us == std::string::npos) ? base : base.substr(us + 1);
}

template <typename T>
int run_case(const std::string& path, std::ostream& csv,
             hipStream_t stream, int warmup, int iters) {
    std::vector<T> input;
    if (!mans::load_typed_file<T>(path, input) || input.empty()) {
        std::cerr << "[skip] failed to load (or empty): " << path << "\n";
        return 1;
    }
    const std::size_t num_elements = input.size();
    const std::size_t raw_bytes = num_elements * sizeof(T);
    constexpr bool u16 = std::is_same_v<T, std::uint16_t>;

    mans::MansParams params{};
    params.backend = mans::Backend::AMD;
    params.dtype = u16 ? mans::DataType::U16 : mans::DataType::U32;
    params.mode = mans::Mode::P;
    params.dims = 1;
    params.nx = static_cast<std::uint32_t>(num_elements);

    const std::size_t adm_capacity = u16
        ? mans::amd::adm::get_max_u16_payload_bytes(num_elements, params)
        : mans::amd::adm::get_max_u32_payload_bytes(num_elements, params);
    const std::size_t entropy_capacity = mans::amd::ans::get_max_compress_bytes(adm_capacity);

    T* d_input = nullptr;
    std::uint8_t* d_adm = nullptr;
    std::uint8_t* d_entropy = nullptr;
    std::uint8_t* d_adm2 = nullptr;
    T* d_output = nullptr;
    check_hip(hipMalloc(&d_input, raw_bytes), "hipMalloc input");
    check_hip(hipMalloc(&d_adm, adm_capacity), "hipMalloc adm");
    check_hip(hipMalloc(&d_entropy, entropy_capacity), "hipMalloc entropy");
    check_hip(hipMalloc(&d_adm2, adm_capacity), "hipMalloc adm2");
    check_hip(hipMalloc(&d_output, raw_bytes), "hipMalloc output");
    check_hip(hipMemcpy(d_input, input.data(), raw_bytes, hipMemcpyHostToDevice), "H2D input");

    double adm_map = 0.0, adm_scan = 0.0, adm_concat = 0.0;
    double adm_final = 0.0, adm_pack = 0.0, adm_total = 0.0;
    double adm_decode = 0.0, ans_comp = 0.0, ans_decomp = 0.0;
    bool fused = false;

    std::size_t adm_size = 0, entropy_size = 0, adm2_size = 0, out_size = 0;

    for (int i = 0; i < warmup + iters; ++i) {
        mans::amd::adm::AdmCompressTimings ct;
        adm_size = adm_capacity;
        if constexpr (u16) {
            mans::amd::adm::compress_u16_device(
                d_input, num_elements, params, d_adm, adm_size, stream, &ct);
        } else {
            mans::amd::adm::compress_u32_device(
                d_input, num_elements, params, d_adm, adm_size, stream, &ct);
        }

        entropy_size = entropy_capacity;
        const double t_ans_comp = time_stage(stream, [&]() {
            mans::amd::ans::compress_stage_device(
                d_adm, adm_size, d_entropy, entropy_capacity, entropy_size, stream);
        });

        adm2_size = adm_capacity;
        const double t_ans_decomp = time_stage(stream, [&]() {
            mans::amd::ans::decompress_stage_device(
                d_entropy, entropy_size, d_adm2, adm_capacity, adm2_size, stream);
        });

        mans::amd::adm::AdmDecompressTimings dt;
        out_size = raw_bytes;
        if constexpr (u16) {
            mans::amd::adm::decompress_u16_device(
                d_adm2, adm2_size, params, d_output, out_size, stream, &dt);
        } else {
            mans::amd::adm::decompress_u32_device(
                d_adm2, adm2_size, params, d_output, out_size, stream, &dt);
        }

        if (i >= warmup) {
            fused = ct.fused;
            adm_map += ct.map_ms;
            adm_scan += ct.scan_ms;
            adm_concat += ct.concat_ms;
            adm_final += ct.finalize_ms;
            adm_pack += ct.pack_flags_ms;
            adm_total += ct.total_ms;
            adm_decode += dt.decode_ms;
            ans_comp += t_ans_comp;
            ans_decomp += t_ans_decomp;
        }
    }

    const double inv = 1.0 / static_cast<double>(iters);
    adm_map *= inv; adm_scan *= inv; adm_concat *= inv;
    adm_final *= inv; adm_pack *= inv; adm_total *= inv;
    adm_decode *= inv; ans_comp *= inv; ans_decomp *= inv;

    std::vector<T> restored(num_elements);
    check_hip(hipMemcpy(restored.data(), d_output, raw_bytes, hipMemcpyDeviceToHost), "D2H output");
    const bool ok = (out_size == raw_bytes) && (restored == input);

    const double gb = static_cast<double>(raw_bytes) / 1e9;
    const double comp_kernel_ms = adm_total + ans_comp;
    const double decomp_kernel_ms = ans_decomp + adm_decode;
    const double comp_gbps = comp_kernel_ms > 0.0 ? gb / (comp_kernel_ms / 1000.0) : 0.0;
    const double decomp_gbps = decomp_kernel_ms > 0.0 ? gb / (decomp_kernel_ms / 1000.0) : 0.0;

    csv << parent_dir_name(path) << ','
        << (u16 ? "u2" : "u4") << ','
        << size_label(path) << ','
        << base_name(path) << ','
        << num_elements << ','
        << raw_bytes << ','
        << (fused ? "fused" : "unfused") << ','
        << adm_map << ',' << adm_scan << ',' << adm_concat << ','
        << adm_final << ',' << adm_pack << ',' << adm_total << ','
        << adm_decode << ','
        << ans_comp << ',' << ans_decomp << ','
        << comp_gbps << ',' << decomp_gbps << ','
        << (ok ? "PASS" : "FAIL") << '\n';

    std::cerr << "[done] " << path << " [" << (fused ? "fused" : "unfused") << "]"
              << " comp=" << comp_gbps << " GB/s decomp=" << decomp_gbps
              << " GB/s " << (ok ? "PASS" : "FAIL") << "\n";

    check_hip(hipFree(d_input), "hipFree input");
    check_hip(hipFree(d_adm), "hipFree adm");
    check_hip(hipFree(d_entropy), "hipFree entropy");
    check_hip(hipFree(d_adm2), "hipFree adm2");
    check_hip(hipFree(d_output), "hipFree output");
    return ok ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <output.csv> <input...> [--warmup N] [--iters N]\n";
        return 2;
    }

    const std::string csv_path = argv[1];
    int warmup = 3;
    int iters = 10;
    std::vector<std::string> files;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--warmup" && i + 1 < argc) {
            warmup = std::atoi(argv[++i]);
        } else if (a == "--iters" && i + 1 < argc) {
            iters = std::atoi(argv[++i]);
        } else {
            files.push_back(a);
        }
    }
    if (warmup < 0) warmup = 0;
    if (iters < 1) iters = 1;

    std::ofstream csv(csv_path);
    if (!csv) {
        std::cerr << "cannot open output csv: " << csv_path << "\n";
        return 1;
    }
    csv << "dataset,dtype,size,file,num_elements,raw_bytes,adm_path,"
           "adm_comp_map_ms,adm_comp_scan_ms,adm_comp_concat_ms,"
           "adm_comp_finalize_ms,adm_comp_pack_flags_ms,adm_comp_total_ms,"
           "adm_decomp_decode_ms,ans_comp_ms,ans_decomp_ms,comp_gbps,decomp_gbps,verify\n";

    hipStream_t stream = nullptr;
    int fail = 0;
    for (const auto& f : files) {
        if (ends_with(f, ".u2")) {
            fail += run_case<std::uint16_t>(f, csv, stream, warmup, iters);
        } else if (ends_with(f, ".u4")) {
            fail += run_case<std::uint32_t>(f, csv, stream, warmup, iters);
        } else {
            std::cerr << "[skip] unknown extension: " << f << "\n";
            ++fail;
        }
    }
    csv.close();
    std::cerr << "wrote " << csv_path << " (files=" << files.size()
              << ", failures=" << fail << ")\n";
    return fail == 0 ? 0 : 1;
}
