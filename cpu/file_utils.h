#ifndef FILE_UTILS_H
#define FILE_UTILS_H

#include <fstream>
#include <vector>
#include <cstddef>
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

inline bool load_u16_file_slice(const std::string& filename,
                                std::size_t elem_offset,
                                std::size_t elem_count,
                                std::vector<std::uint16_t>& data) {
    data.clear();
    if (elem_count == 0) {
        return true;
    }
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) return false;
    const std::streamoff byte_offset =
        static_cast<std::streamoff>(elem_offset) * static_cast<std::streamoff>(sizeof(std::uint16_t));
    in.seekg(0, std::ios::end);
    const std::streamoff size = in.tellg();
    if (size < 0 || size < byte_offset) return false;
    in.seekg(byte_offset, std::ios::beg);
    data.resize(elem_count);
    const std::streamsize bytes =
        static_cast<std::streamsize>(elem_count * sizeof(std::uint16_t));
    return static_cast<bool>(in.read(reinterpret_cast<char*>(data.data()), bytes));
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

inline bool save_u8_file(const std::string& filename,
                         const std::uint8_t* data,
                         std::size_t size) {
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) return false;
    out.write(reinterpret_cast<const char*>(data),
              static_cast<std::streamsize>(size));
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