#include "pans_utils.h"
#include "CpuANSUtils.h"
#include "../../mans_timing.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<std::uint8_t> read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("failed to open input file: " + path);
    }
    std::streamsize size = in.tellg();
    if (size < 0) {
        throw std::runtime_error("failed to get input file size: " + path);
    }
    in.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> buffer(static_cast<std::size_t>(size));
    if (!in.read(reinterpret_cast<char*>(buffer.data()), size)) {
        throw std::runtime_error("failed to read input file: " + path);
    }
    return buffer;
}

std::vector<double> parse_chunks(const std::string& arg) {
    std::vector<double> chunks;
    std::stringstream ss(arg);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) {
            continue;
        }
        chunks.push_back(std::stod(item));
    }
    return chunks;
}

std::string format_chunk_label(std::size_t bytes) {
    const std::size_t kib = 1024;
    const std::size_t mib = 1024 * 1024;
    std::ostringstream out;
    if (bytes % mib == 0) {
        out << (bytes / mib) << "M";
        return out.str();
    }
    if (bytes % kib == 0) {
        out << (bytes / kib) << "K";
        return out.str();
    }
    out << bytes << "B";
    return out.str();
}

struct RunStats {
    double comp_ms = 0.0;
    double decomp_ms = 0.0;
    std::size_t comp_bytes = 0;
    bool ok = true;
};

RunStats run_once(const std::vector<std::uint8_t>& input, std::size_t chunk_bytes) {
    RunStats stats;
    const std::size_t total_bytes = input.size();
    std::size_t offset = 0;

    while (offset < total_bytes) {
        std::size_t len = std::min(chunk_bytes, total_bytes - offset);
        const std::uint8_t* in_ptr = input.data() + offset;

        std::size_t max_out =
            static_cast<std::size_t>(cpu_ans::getMaxCompressedSize(
                static_cast<uint32_t>(len)));
        std::vector<std::uint8_t> out(max_out);
        std::size_t out_len = 0;
        double comp_ms = 0.0;
        MANS_TIMING_START("pans/compress");
        pans_compress(in_ptr, len, out.data(), out_len, comp_ms);
        MANS_TIMING_STOP("pans/compress");
        stats.comp_ms += comp_ms;
        stats.comp_bytes += out_len;

        std::vector<std::uint8_t> decomp(len);
        std::size_t decomp_len = 0;
        double decomp_ms = 0.0;
        MANS_TIMING_START("pans/decompress");
        pans_decompress(out.data(), out_len, decomp.data(), decomp_len, decomp_ms);
        MANS_TIMING_STOP("pans/decompress");
        stats.decomp_ms += decomp_ms;

        if (decomp_len != len) {
            stats.ok = false;
        }

        offset += len;
    }

    return stats;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <input.bin> [--chunks 0.125,0.25,0.5,1,2,8,256] [--csv out.csv]\n";
        return 1;
    }

    std::string input_path = argv[1];
    std::string chunks_arg = "0.125,0.25,0.5,1,2,8,256";
    std::string csv_path;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--chunks" && i + 1 < argc) {
            chunks_arg = argv[++i];
        } else if (arg == "--csv" && i + 1 < argc) {
            csv_path = argv[++i];
        }
    }

    std::vector<double> chunks_mb = parse_chunks(chunks_arg);
    if (chunks_mb.empty()) {
        std::cerr << "No chunk sizes parsed from --chunks.\n";
        return 1;
    }

    std::vector<std::uint8_t> input;
    try {
        input = read_file(input_path);
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << "\n";
        return 1;
    }

    const std::size_t total_bytes = input.size();
    if (total_bytes == 0) {
        std::cerr << "Input file is empty.\n";
        return 1;
    }

    std::cout << "Input bytes: " << total_bytes << "\n";
    std::cout << "Chunks (MB): " << chunks_arg << "\n\n";

    std::ofstream csv;
    if (!csv_path.empty()) {
        csv.open(csv_path);
        if (!csv) {
            std::cerr << "Failed to open CSV output: " << csv_path << "\n";
            return 1;
        }
        csv << "chunk_label,chunk_bytes,ratio_pct,comp_mbps,decomp_mbps\n";
    }

    std::cout << std::left << std::setw(8) << "Chunk"
              << " | " << std::setw(9) << "Ratio"
              << " | " << std::setw(13) << "Comp MB/s"
              << " | " << std::setw(13) << "Decomp MB/s"
              << "\n";
    std::cout << std::string(52, '-') << "\n";

    for (double chunk_mb : chunks_mb) {
        if (chunk_mb <= 0.0) {
            std::cerr << "Skipping invalid chunk size: " << chunk_mb << "\n";
            continue;
        }

        std::size_t chunk_bytes =
            static_cast<std::size_t>(chunk_mb * 1024.0 * 1024.0);
        if (chunk_bytes == 0) {
            chunk_bytes = 1;
        }
        if (chunk_bytes % 2 != 0) {
            chunk_bytes -= 1;
        }
        if (chunk_bytes == 0) {
            chunk_bytes = 2;
        }

        constexpr int kIters = 11;
        double total_comp_ms = 0.0;
        double total_decomp_ms = 0.0;
        double total_comp_bytes = 0.0;
        bool all_ok = true;

        MANS_TIMING_RESET();
        for (int iter = 0; iter < kIters; ++iter) {
            MANS_TIMING_RUN_SCOPE();
            RunStats stats = run_once(input, chunk_bytes);
            if (!stats.ok) {
                all_ok = false;
            }
            if (iter == 0) {
                continue;
            }
            total_comp_ms += stats.comp_ms;
            total_decomp_ms += stats.decomp_ms;
            total_comp_bytes += static_cast<double>(stats.comp_bytes);
        }

        const double denom = static_cast<double>(kIters - 1);
        const double avg_comp_ms = total_comp_ms / denom;
        const double avg_decomp_ms = total_decomp_ms / denom;
        const double avg_comp_bytes = total_comp_bytes / denom;

        double ratio = 100.0 * avg_comp_bytes /
                       static_cast<double>(total_bytes);
        double comp_mbps = (static_cast<double>(total_bytes) / 1e6) /
                           (avg_comp_ms / 1e3);
        double decomp_mbps = (static_cast<double>(total_bytes) / 1e6) /
                             (avg_decomp_ms / 1e3);

        std::string label = format_chunk_label(chunk_bytes);
        std::cout << std::left << std::setw(8) << label
                  << " | " << std::setw(8) << std::fixed << std::setprecision(2)
                  << ratio << "%"
                  << " | " << std::setw(13) << std::fixed << std::setprecision(1)
                  << comp_mbps
                  << " | " << std::setw(13) << std::fixed << std::setprecision(1)
                  << decomp_mbps
                  << "\n";
        if (!all_ok) {
            std::cerr << "Warning: decompressed size mismatch for chunk "
                      << label << "\n";
        }

        if (csv) {
            csv << label << ","
                << chunk_bytes << ","
                << std::fixed << std::setprecision(2) << ratio << ","
                << std::fixed << std::setprecision(1) << comp_mbps << ","
                << std::fixed << std::setprecision(1) << decomp_mbps << "\n";
        }

        MANS_TIMING_DUMP("pans_timing.csv");
        static bool timing_path_logged = false;
        if (!timing_path_logged) {
            timing_path_logged = true;
            std::cerr << "Timing CSV path: pans_timing.csv\n";
        }
    }

    return 0;
}
