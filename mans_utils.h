#pragma once

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "mans_defs.h"

namespace mans {

template <typename T>
inline bool load_file(const std::string& filename, std::vector<T>& data) {
    static_assert(std::is_trivially_copyable<T>::value, "load_file requires trivially copyable T");

    std::ifstream in(filename, std::ios::binary | std::ios::ate);
    if (!in.is_open()) {
        return false;
    }

    const std::streamsize size = in.tellg();
    if (size < 0) {
        return false;
    }
    if ((size % static_cast<std::streamsize>(sizeof(T))) != 0) {
        return false;
    }

    in.seekg(0, std::ios::beg);
    data.resize(static_cast<std::size_t>(size) / sizeof(T));
    return static_cast<bool>(in.read(reinterpret_cast<char*>(data.data()), size));
}

inline bool load_u8_file(const std::string& filename, std::vector<std::uint8_t>& data) {
    return load_file(filename, data);
}

inline bool load_u16_file(const std::string& filename, std::vector<std::uint16_t>& data) {
    return load_file(filename, data);
}

inline bool load_u32_file(const std::string& filename, std::vector<std::uint32_t>& data) {
    return load_file(filename, data);
}

inline bool load_u16_file_slice(const std::string& filename,
                                std::size_t elem_offset,
                                std::size_t elem_count,
                                std::vector<std::uint16_t>& data) {
    data.clear();
    if (elem_count == 0) {
        return true;
    }
    if (elem_offset > std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t) ||
        elem_count > std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t)) {
        return false;
    }

    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        return false;
    }

    const std::streamoff byte_offset =
        static_cast<std::streamoff>(elem_offset * sizeof(std::uint16_t));
    const std::streamsize bytes =
        static_cast<std::streamsize>(elem_count * sizeof(std::uint16_t));

    in.seekg(0, std::ios::end);
    const std::streamoff size = in.tellg();
    if (size < 0 || size < byte_offset || size - byte_offset < bytes) {
        return false;
    }

    in.seekg(byte_offset, std::ios::beg);
    data.resize(elem_count);
    return static_cast<bool>(in.read(reinterpret_cast<char*>(data.data()), bytes));
}

template <typename T>
inline bool save_file(const std::string& filename, const T* data, std::size_t count) {
    static_assert(std::is_trivially_copyable<T>::value, "save_file requires trivially copyable T");

    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        return false;
    }

    out.write(reinterpret_cast<const char*>(data),
              static_cast<std::streamsize>(count * sizeof(T)));
    return static_cast<bool>(out);
}

template <typename T>
inline bool save_file(const std::string& filename, const std::vector<T>& data) {
    return save_file(filename, data.data(), data.size());
}

inline bool save_u8_file(const std::string& filename, const std::vector<std::uint8_t>& data) {
    return save_file(filename, data);
}

inline bool save_u8_file(const std::string& filename,
                         const std::uint8_t* data,
                         std::size_t size) {
    return save_file(filename, data, size);
}

inline bool save_u16_file(const std::string& filename, const std::vector<std::uint16_t>& data) {
    return save_file(filename, data);
}

inline bool save_u32_file(const std::string& filename, const std::vector<std::uint32_t>& data) {
    return save_file(filename, data);
}

template <typename T>
inline bool load_typed_file(const std::string& filename, std::vector<T>& data) {
    if constexpr (std::is_same_v<T, std::uint8_t>) {
        return load_u8_file(filename, data);
    } else if constexpr (std::is_same_v<T, std::uint16_t>) {
        return load_u16_file(filename, data);
    } else if constexpr (std::is_same_v<T, std::uint32_t>) {
        return load_u32_file(filename, data);
    } else {
        static_assert(std::is_same_v<T, void>, "load_typed_file only supports uint8_t/uint16_t/uint32_t");
    }
}

template <typename T>
inline bool save_typed_file(const std::string& filename, const std::vector<T>& data) {
    if constexpr (std::is_same_v<T, std::uint8_t>) {
        return save_u8_file(filename, data);
    } else if constexpr (std::is_same_v<T, std::uint16_t>) {
        return save_u16_file(filename, data);
    } else if constexpr (std::is_same_v<T, std::uint32_t>) {
        return save_u32_file(filename, data);
    } else {
        static_assert(std::is_same_v<T, void>, "save_typed_file only supports uint8_t/uint16_t/uint32_t");
    }
}


template <typename T>
inline bool save_typed_bytes_file(const std::string& filename,
                                  const std::vector<std::uint8_t>& bytes) {
    static_assert(std::is_same_v<T, std::uint16_t> || std::is_same_v<T, std::uint32_t>,
                  "save_typed_bytes_file only supports uint16_t/uint32_t");

    if ((bytes.size() % sizeof(T)) != 0) {
        return false;
    }

    std::vector<T> typed(bytes.size() / sizeof(T));
    std::memcpy(typed.data(), bytes.data(), bytes.size());
    return save_typed_file(filename, typed);
}

inline bool parse_positive_u32(const char* text, std::uint32_t& out) {
    if (!text || *text == '\0') {
        return false;
    }
    char* end = nullptr;
    errno = 0;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno != 0 || end == nullptr || *end != '\0' || value == 0 ||
        value > static_cast<unsigned long long>(std::numeric_limits<std::uint32_t>::max())) {
        return false;
    }
    out = static_cast<std::uint32_t>(value);
    return true;
}

inline bool parse_mode(const char* text, std::uint32_t& out) {
    if (std::strcmp(text, "p") == 0 || std::strcmp(text, "P") == 0) {
        out = Mode::P;
        return true;
    }
    if (std::strcmp(text, "r") == 0 || std::strcmp(text, "R") == 0) {
        out = Mode::R;
        return true;
    }
    return false;
}

inline bool dims_product(const std::vector<std::uint32_t>& dims, std::size_t& out) {
    out = 1;
    for (std::uint32_t dim : dims) {
        if (dim == 0) {
            return false;
        }
        if (out > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(dim)) {
            return false;
        }
        out *= static_cast<std::size_t>(dim);
    }
    return true;
}

inline std::uint32_t read_le32(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0]) |
           (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) |
           (static_cast<std::uint32_t>(p[3]) << 24);
}

inline std::uint64_t read_le64(const std::uint8_t* p) {
    return static_cast<std::uint64_t>(read_le32(p)) |
           (static_cast<std::uint64_t>(read_le32(p + 4)) << 32);
}

inline void write_le64(std::uint8_t* p, std::uint64_t v) {
    p[0] = static_cast<std::uint8_t>(v & 0xFFu);
    p[1] = static_cast<std::uint8_t>((v >> 8) & 0xFFu);
    p[2] = static_cast<std::uint8_t>((v >> 16) & 0xFFu);
    p[3] = static_cast<std::uint8_t>((v >> 24) & 0xFFu);
    p[4] = static_cast<std::uint8_t>((v >> 32) & 0xFFu);
    p[5] = static_cast<std::uint8_t>((v >> 40) & 0xFFu);
    p[6] = static_cast<std::uint8_t>((v >> 48) & 0xFFu);
    p[7] = static_cast<std::uint8_t>((v >> 56) & 0xFFu);
}

inline void write_mans_header(std::uint8_t* out,
                              std::size_t raw_bytes,
                              std::uint8_t codec,
                              std::uint8_t mode,
                              std::uint8_t dims,
                              std::uint64_t nx,
                              std::uint64_t ny,
                              std::uint64_t nz) {
    if (!out) {
        throw std::invalid_argument("write_mans_header: null output");
    }
    std::memset(out, 0, kMansHeaderBytes);
    out[0] = codec;
    out[1] = mode;
    out[2] = dims;
    write_le64(out + 3, static_cast<std::uint64_t>(raw_bytes));
    // Keep the existing ABI-compatible padding before the 64-bit geometry fields.
    write_le64(out + 16, nx);
    write_le64(out + 24, ny);
    write_le64(out + 32, nz);
}

inline bool get_dtype_size(std::uint32_t dtype, std::size_t& elem_size) {
    switch (dtype) {
        case DataType::U16:
            elem_size = sizeof(std::uint16_t);
            return true;
        case DataType::U32:
            elem_size = sizeof(std::uint32_t);
            return true;
        default:
            elem_size = 0;
            return false;
    }
}

inline bool validate_mans_geometry(const MansHeader& header,
                                   std::uint32_t dtype,
                                   std::size_t raw_bytes,
                                   std::string* error = nullptr) {
    auto set_error = [&](const char* msg) {
        if (error) {
            *error = msg;
        }
    };

    std::size_t elem_size = 0;
    if (!get_dtype_size(dtype, elem_size)) {
        set_error("unsupported dtype");
        return false;
    }
    if (raw_bytes == 0 || raw_bytes % elem_size != 0) {
        set_error("raw size is not aligned to dtype");
        return false;
    }

    std::size_t expected_elements = static_cast<std::size_t>(header.nx);
    if (header.dims >= 2) {
        if (expected_elements > std::numeric_limits<std::size_t>::max() /
                                  static_cast<std::size_t>(header.ny)) {
            set_error("header geometry overflows size_t");
            return false;
        }
        expected_elements *= static_cast<std::size_t>(header.ny);
    }
    if (header.dims == 3) {
        if (expected_elements > std::numeric_limits<std::size_t>::max() /
                                  static_cast<std::size_t>(header.nz)) {
            set_error("header geometry overflows size_t");
            return false;
        }
        expected_elements *= static_cast<std::size_t>(header.nz);
    }
    if (expected_elements != raw_bytes / elem_size) {
        set_error("header geometry does not match raw size");
        return false;
    }
    return true;
}

inline bool parse_mans_header(const void* data,
                              std::size_t length,
                              MansHeader& header,
                              std::size_t& raw_bytes,
                              std::string* error = nullptr) {
    auto set_error = [&](const char* msg) {
        if (error) {
            *error = msg;
        }
    };

    raw_bytes = 0;
    if (!data) {
        set_error("compressed_data is null");
        return false;
    }
    if (length < kMansHeaderBytes) {
        set_error("compressed_len too small");
        return false;
    }

    const auto* bytes = static_cast<const std::uint8_t*>(data);
    header = MansHeader{};
    header.codec = bytes[0];
    header.mode = bytes[1];
    header.dims = bytes[2];
    std::memcpy(header.raw_bytes_le, bytes + 3, sizeof(header.raw_bytes_le));
    header.nx = read_le64(bytes + 16);
    header.ny = read_le64(bytes + 24);
    header.nz = read_le64(bytes + 32);
    if (header.codec != Codec::ADM && header.codec != Codec::RAW &&
        header.codec != Codec::AMD_ANS) {
        set_error("unknown codec");
        return false;
    }
    if (header.mode != Mode::P && header.mode != Mode::R) {
        set_error("unknown mode in header");
        return false;
    }
    if (header.dims < 1 || header.dims > 3) {
        set_error("invalid dims in header");
        return false;
    }

    const std::uint64_t u32_max =
        static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max());
    if (header.nx == 0 || header.nx > u32_max) {
        set_error("invalid nx in header");
        return false;
    }
    if (header.dims >= 2 && (header.ny == 0 || header.ny > u32_max)) {
        set_error("invalid ny in header");
        return false;
    }
    if (header.dims == 3 && (header.nz == 0 || header.nz > u32_max)) {
        set_error("invalid nz in header");
        return false;
    }

    const std::uint64_t raw_bytes_u64 = read_le64(header.raw_bytes_le);
    if (raw_bytes_u64 == 0 ||
        raw_bytes_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        set_error("invalid raw size in header");
        return false;
    }

    raw_bytes = static_cast<std::size_t>(raw_bytes_u64);
    return true;
}

inline bool parse_mans_raw_bytes(const void* data,
                                 std::size_t length,
                                 std::size_t& raw_bytes,
                                 std::string* error = nullptr) {
    MansHeader header{};
    return parse_mans_header(data, length, header, raw_bytes, error);
}

} // namespace mans
