#pragma once

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mans_defs.h"

namespace mans {
namespace h5 {

class MansConfig {
public:
    void load(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) throw std::runtime_error("Config file not found: " + filepath);
        std::string line;
        while (std::getline(file, line)) {
            parse_line(line);
        }
    }

    void parse_line(std::string line) {
        if (auto pos = line.find('#'); pos != std::string::npos) line = line.substr(0, pos);
        line = trim(line);
        if (line.empty()) return;

        const auto eq_pos = line.find('=');
        if (eq_pos == std::string::npos) return;

        const std::string key = trim(line.substr(0, eq_pos));
        const std::string val = trim(line.substr(eq_pos + 1));
        if (key.empty()) return;

        if (key != "mode") {
            std::cerr << "[Config Warn] Unknown key: " << key << "\n";
            return;
        }

        try {
            const std::uint32_t parsed = static_cast<std::uint32_t>(std::stoul(val));
            if (parsed != mans::Mode::P && parsed != mans::Mode::R) {
                std::cerr << "[Config Error] Invalid mode: " << val << "\n";
                return;
            }
            mode_ = parsed;
        } catch (...) {
            std::cerr << "[Config Error] Failed to parse mode value '" << val << "'\n";
        }
    }

    std::vector<unsigned int> to_cd_values() const {
        return {mode_};
    }

    std::uint32_t get_mode() const { return mode_; }

private:
    std::uint32_t mode_ = mans::Mode::R;

    static std::string trim(const std::string& str) {
        const char* whitespace = " \t\r\n";
        const std::size_t first = str.find_first_not_of(whitespace);
        if (first == std::string::npos) return "";
        const std::size_t last = str.find_last_not_of(whitespace);
        return str.substr(first, last - first + 1);
    }
};

inline void* safe_malloc(std::size_t size) {
    void* ptr = std::malloc(size);
    if (!ptr) {
        std::cerr << "[H5Z-MANS Error] Memory allocation failed for size: " << size << "\n";
    }
    return ptr;
}

} // namespace h5
} // namespace mans
