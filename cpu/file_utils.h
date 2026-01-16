#ifndef FILE_UTILS_H
#define FILE_UTILS_H

#include <fstream>
#include <vector>
#include <cstdint>
#include <string>

inline bool load_u8_file(const std::string& filename, std::vector<std::uint8_t>& data) {
    std::ifstream in(filename, std::ios::binary | std::ios::ate);
    if (!in.is_open()) return false;
    std::streamsize size = in.tellg();
    if (size < 0) return false;
    in.seekg(0, std::ios::beg);
    data.resize(static_cast<std::size_t>(size));
    return static_cast<bool>(in.read(reinterpret_cast<char*>(data.data()), size));
}

inline bool load_u16_file(const std::string& filename, std::vector<std::uint16_t>& data) {
    std::ifstream in(filename, std::ios::binary | std::ios::ate);
    if (!in.is_open()) return false;
    std::streamsize size = in.tellg();
    if (size < 0) return false;
    in.seekg(0, std::ios::beg);
    data.resize(static_cast<std::size_t>(size) / sizeof(std::uint16_t));
    return static_cast<bool>(in.read(
        reinterpret_cast<char*>(data.data()), size));
}

inline bool load_u32_file(const std::string& filename, std::vector<std::uint32_t>& data) {
    std::ifstream in(filename, std::ios::binary | std::ios::ate);
    if (!in.is_open()) return false;
    std::streamsize size = in.tellg();
    if (size < 0) return false;
    in.seekg(0, std::ios::beg);
    data.resize(static_cast<std::size_t>(size) / sizeof(std::uint32_t));
    return static_cast<bool>(in.read(
        reinterpret_cast<char*>(data.data()), size));
}


inline bool save_u8_file(const std::string& filename, const std::vector<std::uint8_t>& data) {
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) return false;
    out.write(reinterpret_cast<const char*>(data.data()),
              static_cast<std::streamsize>(data.size()));
    return static_cast<bool>(out);
}

inline bool save_u16_file(const std::string& filename, const std::vector<std::uint16_t>& data) {
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) return false;
    out.write(reinterpret_cast<const char*>(data.data()),
              static_cast<std::streamsize>(data.size() * sizeof(std::uint16_t)));
    return static_cast<bool>(out);
}

inline bool save_u32_file(const std::string& filename, const std::vector<std::uint32_t>& data) {
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) return false;
    out.write(reinterpret_cast<const char*>(data.data()),
              static_cast<std::streamsize>(data.size() * sizeof(std::uint32_t)));
    return static_cast<bool>(out);
}
#endif