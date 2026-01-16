#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <cstddef>

#include "mans_defs.h"

namespace mans {
namespace h5 {

// Supported parameter types (for robustness, keep multi-type support logic)
enum class ParamType {
    UINT32,
    FLOAT,
    BOOL
};

// Field metadata
struct FieldMeta {
    size_t offset;  // Memory offset
    ParamType type; // Data type
};

class MansConfig {
public:
    MansConfig() {
        // 1. Clear memory
        std::memset(&params_, 0, sizeof(params_));

        // 2. Set Defaults (Important! num_threads cannot be 0)
        // Original Logic Defaults
        params_.adm_center_calc_threads     = 32;
        params_.adm_encode_threads          = 32;
        params_.adm_warp_reduce_threads     = 32;
        params_.adm_fill_tail_threads       = 16;
        params_.adm_write_back_threads      = 16;

        params_.adm_restore_signals_threads = 32;
        params_.adm_decode_values_threads   = 16;

        // 3. Register existing fields
        register_field("backend",       offsetof(mans::MansParams, backend),       ParamType::UINT32);
        register_field("dtype",         offsetof(mans::MansParams, dtype),         ParamType::UINT32);
        register_field("adm_threshold", offsetof(mans::MansParams, adm_threshold), ParamType::UINT32);

        // 4. Register new ADM thread fields
        register_field("adm_center_calc_threads",     offsetof(mans::MansParams, adm_center_calc_threads),     ParamType::UINT32);
        register_field("adm_encode_threads",          offsetof(mans::MansParams, adm_encode_threads),          ParamType::UINT32);
        register_field("adm_warp_reduce_threads",     offsetof(mans::MansParams, adm_warp_reduce_threads),     ParamType::UINT32);
        register_field("adm_fill_tail_threads",       offsetof(mans::MansParams, adm_fill_tail_threads),       ParamType::UINT32);
        register_field("adm_write_back_threads",      offsetof(mans::MansParams, adm_write_back_threads),      ParamType::UINT32);

        register_field("adm_restore_signals_threads", offsetof(mans::MansParams, adm_restore_signals_threads), ParamType::UINT32);
        register_field("adm_decode_values_threads",   offsetof(mans::MansParams, adm_decode_values_threads),   ParamType::UINT32);
    }

    void load(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) throw std::runtime_error("Config file not found: " + filepath);
        std::string line;
        while (std::getline(file, line)) {
            parse_line(line);
        }
    }

    void parse_line(std::string line) {
        // parse key=value pattern, ignore comments and blank lines
        if (auto pos = line.find('#'); pos != std::string::npos) line = line.substr(0, pos);
        if (line.empty()) return;
        auto eq_pos = line.find('=');
        if (eq_pos == std::string::npos) return;
        std::string key = trim(line.substr(0, eq_pos));
        std::string val = trim(line.substr(eq_pos + 1));
        if (key.empty()) return;
        auto it = schema_.find(key);
        if (it != schema_.end()) {
            write_to_struct(it->second, val);
        } else {
            std::cerr << "[Config Warn] Unknown key: " << key << "\n";
        }
    }

    std::vector<unsigned int> to_cd_values() const {
        // Automatically compute how many uint32 are needed
        // static_assert in mans_defs.h already ensures size is a multiple of 4
        size_t count = sizeof(mans::MansParams) / sizeof(unsigned int);
        std::vector<unsigned int> values(count);
        std::memcpy(values.data(), &params_, sizeof(mans::MansParams));
        return values;
    }
    const mans::MansParams& get_params() const { return params_; }

private:
    mans::MansParams params_;
    std::map<std::string, FieldMeta> schema_;
    void register_field(const std::string& key, size_t offset, ParamType type) {
        schema_[key] = {offset, type};
    }

    // Generic write logic (supports type conversions)
    void write_to_struct(const FieldMeta& meta, const std::string& val_str) {
        // Calculate destination address = struct base + offset
        void* target_ptr = reinterpret_cast<char*>(&params_) + meta.offset;

        try {
            switch (meta.type) {
                case ParamType::UINT32: {
                    uint32_t val = std::stoul(val_str);
                    std::memcpy(target_ptr, &val, sizeof(uint32_t));
                    break;
                }
                case ParamType::FLOAT: {
                    float val = std::stof(val_str);
                    std::memcpy(target_ptr, &val, sizeof(float));
                    break;
                }
                case ParamType::BOOL: {
                    // padding to uint32
                    std::string v = val_str;
                    std::transform(v.begin(), v.end(), v.begin(), ::tolower);
                    uint32_t val = (v == "true" || v == "1") ? 1 : 0;
                    std::memcpy(target_ptr, &val, sizeof(uint32_t));
                    break;
                }
            }
        } catch (...) {
            std::cerr << "[Config Error] Failed to parse value '" << val_str << "'\n";
        }
    }

    std::string trim(const std::string& str) {
        const char* whitespace = " \t\r\n";
        size_t first = str.find_first_not_of(whitespace);
        if (std::string::npos == first) return "";
        size_t last = str.find_last_not_of(whitespace);
        return str.substr(first, (last - first + 1));
    }
};

} // namespace h5
} // namespace mans