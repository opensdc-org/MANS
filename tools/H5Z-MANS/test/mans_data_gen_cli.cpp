#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string_view>
#include <stdexcept>
#include <string>
#include <vector>

#include "mans_data_gen.h"

namespace fs = std::filesystem;

struct GenOptions {
    std::string output_name;
    std::string output_dir;
    std::string config_file;
    std::string synth_config_file;

    std::optional<double> size_per_rank_mb_override;
    std::optional<double> ratio_constant_override;
    std::optional<std::uint32_t> dtype_override;
    std::optional<std::size_t> ranks_override;
    std::optional<std::size_t> jobs_override;
    std::optional<std::string> output_prefix_override;
};

static void print_usage(const char* prog) {
    std::cerr
        << "Usage:\n  " << prog
        << " [--config gen.cfg] [--output-name name.bin] [--output-dir DIR] [--output-prefix NAME]\n"
           "         [--synth-config synth.cfg] [--size-per-rank-mb MB] [--ranks N] [--jobs N]\n"
           "         [--ratio-constant R] [--dtype u16|u32]\n";
}

static std::uint32_t parse_dtype(std::string_view s) {
    if (s == "u16" || s == "U16") {
        return mans::DataType::U16;
    }
    if (s == "u32" || s == "U32") {
        return mans::DataType::U32;
    }
    return static_cast<std::uint32_t>(std::stoul(std::string(s)));
}

static std::optional<GenOptions> parse_args(int argc, char** argv) {
    GenOptions opts;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("Missing value for ") + flag);
            }
            return argv[++i];
        };

        if (arg == "--output-name") {
            opts.output_name = need_value("--output-name");
            continue;
        }
        if (arg == "--output-dir") {
            opts.output_dir = need_value("--output-dir");
            continue;
        }
        if (arg == "--output-prefix") {
            opts.output_prefix_override = need_value("--output-prefix");
            continue;
        }
        if (arg == "--config") {
            opts.config_file = need_value("--config");
            continue;
        }
        if (arg == "--synth-config") {
            opts.synth_config_file = need_value("--synth-config");
            continue;
        }
        if (arg == "--size-per-rank-mb") {
            opts.size_per_rank_mb_override = std::stod(need_value("--size-per-rank-mb"));
            continue;
        }
        if (arg == "--ratio-constant") {
            opts.ratio_constant_override = std::stod(need_value("--ratio-constant"));
            continue;
        }
        if (arg == "--ranks") {
            opts.ranks_override = static_cast<std::size_t>(std::stoull(need_value("--ranks")));
            continue;
        }
        if (arg == "--jobs") {
            opts.jobs_override = static_cast<std::size_t>(std::stoull(need_value("--jobs")));
            continue;
        }
        if (arg == "--dtype") {
            opts.dtype_override = parse_dtype(need_value("--dtype"));
            continue;
        }
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return std::nullopt;
        }

        throw std::runtime_error("Unknown argument: " + arg);
    }

    return opts;
}

static void print_block_ratio_summary(const mans::h5::data_gen::SyntheticConfig& cfg) {
    const double smooth = std::max(0.0, cfg.ratio_smooth);
    const double spike = std::max(0.0, cfg.ratio_spike);
    const double constant = std::max(0.0, cfg.ratio_constant);
    const double random = std::max(0.0, cfg.ratio_random);
    const double sum = smooth + spike + constant + random;

    std::cout << "  block_ratio_smooth:   " << (sum > 0.0 ? smooth / sum : 0.0) << "\n";
    std::cout << "  block_ratio_spike:    " << (sum > 0.0 ? spike / sum : 0.0) << "\n";
    std::cout << "  block_ratio_constant: " << (sum > 0.0 ? constant / sum : 0.0) << "\n";
    std::cout << "  block_ratio_random:   " << (sum > 0.0 ? random / sum : 0.0) << "\n";
}

template <typename T>
static void write_bin(const std::string& path, const std::vector<T>& data) {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open output: " + path);
    }
    out.write(reinterpret_cast<const char*>(data.data()),
              static_cast<std::streamsize>(data.size() * sizeof(T)));
}

template <typename T>
static int generate_and_write(const mans::h5::data_gen::SyntheticConfig& cfg,
                              std::uint32_t adm_threshold,
                              const std::string& output_path) {
    const std::size_t elem_size = sizeof(T);
    const std::size_t total_elements = mans::h5::data_gen::aligned_total_elements(
        cfg.size_per_rank_mb,
        elem_size,
        cfg.block_size);

    const auto data = mans::h5::data_gen::generate_synthetic_slice<T>(
        adm_threshold,
        cfg,
        total_elements,
        /*slice_offset=*/0,
        /*slice_count=*/total_elements);

    write_bin(output_path, data);

    const double raw_mb = (static_cast<double>(total_elements) * elem_size) / 1048576.0;
    std::cout << "Generated: " << output_path << "\n";
    std::cout << "  dtype:        " << (elem_size == 2 ? "u16" : "u32") << "\n";
    std::cout << "  elements:     " << total_elements << "\n";
    std::cout << "  raw_size_mb:  " << raw_mb << "\n";
    std::cout << "  adm_threshold:" << adm_threshold << "\n";
    print_block_ratio_summary(cfg);
    return 0;
}

template <typename T>
static int generate_ranks_and_write(const mans::h5::data_gen::SyntheticConfig& cfg,
                                    std::uint32_t adm_threshold,
                                    std::size_t rank_count,
                                    const fs::path& output_dir,
                                    const std::string& output_prefix,
                                    std::optional<std::size_t> jobs_override) {
    std::size_t workers = 0;
    if (jobs_override.has_value()) {
        workers = *jobs_override;
    }

    const auto rank_outputs = mans::h5::data_gen::generate_rank_datasets<T>(
        adm_threshold,
        cfg,
        rank_count,
        output_dir,
        output_prefix,
        workers);

    const std::size_t elem_size = sizeof(T);
    const std::size_t total_elements = rank_outputs.empty() ? 0 : rank_outputs.front().elements;
    const double raw_mb = (static_cast<double>(total_elements) * elem_size) / 1048576.0;

    std::cout << "Generated rank datasets:\n";
    std::cout << "  ranks:        " << rank_count << "\n";
    std::cout << "  dtype:        " << (elem_size == 2 ? "u16" : "u32") << "\n";
    std::cout << "  elements/rank:" << total_elements << "\n";
    std::cout << "  raw_size_mb:  " << raw_mb << "\n";
    std::cout << "  output_dir:   " << output_dir.string() << "\n";
    std::cout << "  file_prefix:  " << output_prefix << "\n";
    std::cout << "  adm_threshold:" << adm_threshold << "\n";
    print_block_ratio_summary(cfg);
    for (const auto& item : rank_outputs) {
        std::cout << "    rank " << item.rank << " -> " << item.output_path.string() << "\n";
    }
    return 0;
}

int main(int argc, char** argv) {
    GenOptions opts;
    try {
        const auto parsed = parse_args(argc, argv);
        if (!parsed.has_value()) {
            return 1;
        }
        opts = *parsed;
    } catch (const std::exception& e) {
        std::cerr << "Arg parse error: " << e.what() << "\n";
        print_usage(argv[0]);
        return 1;
    }

    mans::h5::data_gen::GeneratorConfig gen_cfg;
    const bool use_builtin_defaults = opts.config_file.empty() && opts.synth_config_file.empty();
    try {
        if (!opts.config_file.empty()) {
            gen_cfg = mans::h5::data_gen::load_generator_config(opts.config_file);
        } else {
            gen_cfg.synth = mans::h5::data_gen::load_synthetic_config(opts.synth_config_file);
        }
    } catch (const std::exception& e) {
        std::cerr << "Config error: " << e.what() << "\n";
        return 1;
    }

    if (opts.size_per_rank_mb_override.has_value()) {
        gen_cfg.synth.size_per_rank_mb = *opts.size_per_rank_mb_override;
    }
    if (gen_cfg.synth.size_per_rank_mb <= 0.0) {
        gen_cfg.synth.size_per_rank_mb = 256.0;
    }

    std::uint32_t dtype = gen_cfg.dtype;
    std::uint32_t adm_threshold = gen_cfg.adm_threshold;

    if (opts.dtype_override.has_value()) {
        dtype = *opts.dtype_override;
    }

    if (use_builtin_defaults) {
        // Keep default synthetic output ADM-friendly for the built-in ADM threshold.
        gen_cfg.synth.ratio_smooth = 1.0;
        gen_cfg.synth.ratio_spike = 0.0;
        gen_cfg.synth.ratio_constant = 0.0;
        gen_cfg.synth.ratio_random = 0.0;
        gen_cfg.synth.block_size = 512;
        const auto safe_noise = std::min<std::uint32_t>(adm_threshold, 20U);
        gen_cfg.synth.noise_range = static_cast<int>(safe_noise);
    }
    if (opts.ratio_constant_override.has_value()) {
        gen_cfg.synth.ratio_constant = *opts.ratio_constant_override;
        // Auto-adjust smooth so total weight stays near 1.0.
        const double smooth = 1.0
            - std::max(0.0, gen_cfg.synth.ratio_constant)
            - std::max(0.0, gen_cfg.synth.ratio_spike)
            - std::max(0.0, gen_cfg.synth.ratio_random);
        gen_cfg.synth.ratio_smooth = std::max(0.0, smooth);
    }

    const std::size_t ranks = opts.ranks_override.value_or(1);
    if (ranks == 0) {
        std::cerr << "Invalid --ranks value: must be >= 1.\n";
        return 1;
    }
    if (opts.jobs_override.has_value() && *opts.jobs_override == 0) {
        std::cerr << "Invalid --jobs value: must be >= 1.\n";
        return 1;
    }

    const std::string output_name = !opts.output_name.empty() ? opts.output_name : gen_cfg.output_bin;
    const bool single_output_mode = (ranks == 1 && opts.output_dir.empty());

    fs::path rank_output_dir;
    std::string rank_output_prefix = opts.output_prefix_override.value_or("rank");
    fs::path single_output_path;

    if (!single_output_mode) {
        if (!opts.output_dir.empty()) {
            rank_output_dir = fs::path(opts.output_dir);
        } else if (!output_name.empty()) {
            const fs::path candidate(output_name);
            if (candidate.has_extension()) {
                rank_output_dir = candidate.has_parent_path() ? candidate.parent_path() : fs::path(".");
                if (!opts.output_prefix_override.has_value()) {
                    const std::string stem = candidate.stem().string();
                    if (!stem.empty()) {
                        rank_output_prefix = stem;
                    }
                }
            } else {
                rank_output_dir = candidate;
            }
        } else {
            rank_output_dir = fs::path(".");
        }
    } else {
        single_output_path = output_name.empty() ? fs::path("rank0.bin") : fs::path(output_name);
    }

    try {
        if (single_output_mode && dtype == mans::DataType::U16) {
            return generate_and_write<std::uint16_t>(
                gen_cfg.synth,
                adm_threshold,
                single_output_path.string());
        }
        if (single_output_mode) {
            return generate_and_write<std::uint32_t>(
                gen_cfg.synth,
                adm_threshold,
                single_output_path.string());
        }
        if (dtype == mans::DataType::U16) {
            return generate_ranks_and_write<std::uint16_t>(
                gen_cfg.synth,
                adm_threshold,
                ranks,
                rank_output_dir,
                rank_output_prefix,
                opts.jobs_override);
        }
        return generate_ranks_and_write<std::uint32_t>(
            gen_cfg.synth,
            adm_threshold,
            ranks,
            rank_output_dir,
            rank_output_prefix,
            opts.jobs_override);
    } catch (const std::exception& e) {
        std::cerr << "Generation error: " << e.what() << "\n";
        return 1;
    }
}
