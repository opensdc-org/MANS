#ifndef H5Z_MANS_SZ3_CONFIG_MIN_H
#define H5Z_MANS_SZ3_CONFIG_MIN_H

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

// Minimal SZ3 config encoder for cd_values, extracted from SZ3 utils.
namespace SZ3 {
using uchar = unsigned char;

#ifdef _MSC_VER
#define ALWAYS_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define ALWAYS_INLINE inline
#endif

// Endianness detection: SZ3 stores config in little-endian.
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define SZ3_BIG_ENDIAN 1
#elif defined(__BIG_ENDIAN__) || defined(__ARMEB__) || defined(__THUMBEB__) || \
    defined(__AARCH64EB__) || defined(_MIPSEB) || defined(__MIPSEB)
#define SZ3_BIG_ENDIAN 1
#else
#define SZ3_BIG_ENDIAN 0
#endif

#if SZ3_BIG_ENDIAN
#if defined(_MSC_VER)
#define SZ3_HAS_BUILTIN_BSWAP 1
#define BSWAP16(x) _byteswap_ushort(x)
#define BSWAP32(x) _byteswap_ulong(x)
#define BSWAP64(x) _byteswap_uint64(x)
#elif defined(__GNUC__) || defined(__clang__)
#define SZ3_HAS_BUILTIN_BSWAP 1
#define BSWAP16(x) __builtin_bswap16(x)
#define BSWAP32(x) __builtin_bswap32(x)
#define BSWAP64(x) __builtin_bswap64(x)
#else
#define SZ3_HAS_BUILTIN_BSWAP 0
#endif

template <typename T>
inline T byteswap(T value) {
    union {
        T val;
        uint8_t bytes[sizeof(T)];
        uint16_t u16;
        uint32_t u32;
        uint64_t u64;
    } u;
    u.val = value;
    if constexpr (sizeof(T) == 1) {
        return value;
#if SZ3_HAS_BUILTIN_BSWAP
    } else if constexpr (sizeof(T) == 2) {
        u.u16 = BSWAP16(u.u16);
    } else if constexpr (sizeof(T) == 4) {
        u.u32 = BSWAP32(u.u32);
    } else if constexpr (sizeof(T) == 8) {
        u.u64 = BSWAP64(u.u64);
#endif
    } else {
        for (size_t i = 0; i < sizeof(T) / 2; i++) {
            uint8_t tmp = u.bytes[i];
            u.bytes[i] = u.bytes[sizeof(T) - 1 - i];
            u.bytes[sizeof(T) - 1 - i] = tmp;
        }
    }
    return u.val;
}
#endif

template <class T1>
void write(T1 const var, uchar*& compressed_data_pos) {
#if SZ3_BIG_ENDIAN
    T1 le_var = byteswap(var);
    memcpy(compressed_data_pos, &le_var, sizeof(T1));
#else
    memcpy(compressed_data_pos, &var, sizeof(T1));
#endif
    compressed_data_pos += sizeof(T1);
}

template <typename T>
uint8_t vector_bit_width(const std::vector<T>& data) {
    if (data.empty()) return 0;
    T max_value = *std::max_element(data.begin(), data.end());
    uint8_t bits = 0;
    while (max_value > 0) {
        max_value >>= 1;
        ++bits;
    }
    return bits;
}

template <typename T>
void vector2bytes(const std::vector<T>& data, uint8_t bit_width, unsigned char*& c) {
    if (data.empty()) return;
    size_t current_bit = 0;
    size_t byte_index = 0;
    unsigned char current_byte = 0;

    for (T value : data) {
        size_t bits_remaining = bit_width;
        while (bits_remaining > 0) {
            size_t space_in_current_byte = 8 - (current_bit % 8);
            size_t bits_to_write = std::min(bits_remaining, space_in_current_byte);
            size_t bits_shift = (bit_width - bits_remaining);
            unsigned char bits_to_store = (value >> bits_shift) & ((1 << bits_to_write) - 1);

            current_byte |= (bits_to_store << (current_bit % 8));
            current_bit += bits_to_write;
            bits_remaining -= bits_to_write;

            if (current_bit % 8 == 0) {
                c[byte_index++] = current_byte;
                current_byte = 0;
            }
        }
    }

    if (current_bit % 8 != 0) {
        c[byte_index++] = current_byte;
    }

    c += byte_index;
}

enum EB { EB_ABS, EB_REL, EB_PSNR, EB_L2NORM, EB_ABS_AND_REL, EB_ABS_OR_REL };
enum ALGO { ALGO_LORENZO_REG, ALGO_INTERP_LORENZO, ALGO_INTERP, ALGO_NOPRED, ALGO_LOSSLESS, ALGO_BIOMD, ALGO_BIOMDXTC };
enum INTERP_ALGO { INTERP_ALGO_LINEAR, INTERP_ALGO_CUBIC };

const std::map<std::string, ALGO> ALGO_MAP = {
    {"ALGO_LORENZO_REG", ALGO_LORENZO_REG},
    {"ALGO_INTERP_LORENZO", ALGO_INTERP_LORENZO},
    {"ALGO_INTERP", ALGO_INTERP},
    {"ALGO_NOPRED", ALGO_NOPRED},
    {"ALGO_LOSSLESS", ALGO_LOSSLESS},
    {"ALGO_BIOMD", ALGO_BIOMD},
    {"ALGO_BIOMDXTC", ALGO_BIOMDXTC},
};

const std::map<std::string, EB> EB_MAP = {
    {"ABS", EB_ABS},
    {"REL", EB_REL},
    {"PSNR", EB_PSNR},
    {"NORM", EB_L2NORM},
    {"ABS_AND_REL", EB_ABS_AND_REL},
    {"ABS_OR_REL", EB_ABS_OR_REL},
};

const std::map<std::string, INTERP_ALGO> INTERP_ALGO_MAP = {
    {"INTERP_ALGO_LINEAR", INTERP_ALGO_LINEAR},
    {"INTERP_ALGO_CUBIC", INTERP_ALGO_CUBIC},
};

ALWAYS_INLINE std::string to_lower(const std::string& s) {
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(), ::tolower);
    return out;
}

template <typename EnumType>
ALWAYS_INLINE void match_enum(const std::string& input, const std::map<std::string, EnumType>& table, uint8_t& out) {
    std::string input_lc = to_lower(input);
    for (const auto& kv : table) {
        if (to_lower(kv.first) == input_lc) {
            out = static_cast<int>(kv.second);
        }
    }
}

class Config {
   public:
    template <class... Dims>
    Config(Dims... args) {
        dims = std::vector<size_t>{static_cast<size_t>(std::forward<Dims>(args))...};
        setDims(dims.begin(), dims.end());
    }

    template <class Iter>
    size_t setDims(Iter begin, Iter end) {
        auto dims_ = std::vector<size_t>(begin, end);
        dims.clear();
        for (auto dim : dims_) {
            if (dim > 1) {
                dims.push_back(dim);
            }
        }
        if (dims.empty()) {
            dims = {1};
        }
        N = static_cast<char>(dims.size());
        num = std::accumulate(dims.begin(), dims.end(), static_cast<size_t>(1), std::multiplies<size_t>());
        predDim = static_cast<uint8_t>(N);
        blockSize = (N == 1 ? 128 : (N == 2 ? 16 : 6));
        return num;
    }

    void load_ini(const std::string& ini_content) {
        std::istringstream ss(ini_content);
        std::string line, section;

        auto trim = [](std::string& s) {
            s.erase(0, s.find_first_not_of(" \t\r\n"));
            s.erase(s.find_last_not_of(" \t\r\n") + 1);
        };

        auto eq = [&](const std::string& a, const std::string& b) { return to_lower(a) == to_lower(b); };

        auto parse_bool = [&](const std::string& s) {
            auto ls = to_lower(s);
            return ls == "true" || ls == "1" || ls == "yes" || ls == "on";
        };

        while (std::getline(ss, line)) {
            trim(line);
            if (line.empty() || line[0] == '#') continue;
            if (line.front() == '[') {
                section = line.substr(1, line.find(']') - 1);
                continue;
            }

            auto sep = line.find('=');
            if (sep == std::string::npos) continue;

            std::string key = line.substr(0, sep);
            std::string value = line.substr(sep + 1);
            trim(key);
            trim(value);

            if (eq(section, "GlobalSettings")) {
                if (eq(key, "CmprAlgo"))
                    match_enum(value, ALGO_MAP, cmprAlgo);
                else if (eq(key, "ErrorBoundMode"))
                    match_enum(value, EB_MAP, errorBoundMode);
                else if (eq(key, "AbsErrorBound"))
                    absErrorBound = std::stod(value);
                else if (eq(key, "RelErrorBound"))
                    relErrorBound = std::stod(value);
                else if (eq(key, "PSNRErrorBound"))
                    psnrErrorBound = std::stod(value);
                else if (eq(key, "L2NormErrorBound"))
                    l2normErrorBound = std::stod(value);
                else if (eq(key, "OpenMP"))
                    openmp = parse_bool(value);
            } else if (eq(section, "AlgoSettings")) {
                if (eq(key, "Lorenzo"))
                    lorenzo = parse_bool(value);
                else if (eq(key, "Lorenzo2ndOrder"))
                    lorenzo2 = parse_bool(value);
                else if (eq(key, "Regression"))
                    regression = parse_bool(value);
                else if (eq(key, "Regression2ndOrder"))
                    regression2 = parse_bool(value);
                else if (eq(key, "InterpolationAlgo"))
                    match_enum(value, INTERP_ALGO_MAP, interpAlgo);
                else if (eq(key, "InterpolationDirection"))
                    interpDirection = static_cast<uint8_t>(std::stoi(value));
                else if (eq(key, "BlockSize"))
                    blockSize = std::stoi(value);
                else if (eq(key, "QuantizationBinTotal"))
                    quantbinCnt = std::stoi(value);
                else if (eq(key, "InterpolationAnchorStride"))
                    interpAnchorStride = std::stoi(value);
                else if (eq(key, "InterpolationAlpha"))
                    interpAlpha = std::stod(value);
                else if (eq(key, "InterpolationBeta"))
                    interpBeta = std::stod(value);
            }
        }
    }

    size_t save(unsigned char*& c) const {
        auto c0 = c;
        c += sizeof(uchar);

        write(N, c);
        auto bitWidth = vector_bit_width(dims);
        write(bitWidth, c);
        vector2bytes(dims, bitWidth, c);

        write(num, c);
        write(cmprAlgo, c);

        write(errorBoundMode, c);
        if (errorBoundMode == EB_ABS) {
            write(absErrorBound, c);
        } else if (errorBoundMode == EB_REL) {
            write(relErrorBound, c);
        } else if (errorBoundMode == EB_PSNR) {
            write(psnrErrorBound, c);
        } else if (errorBoundMode == EB_L2NORM) {
            write(l2normErrorBound, c);
        } else if (errorBoundMode == EB_ABS_OR_REL) {
            write(absErrorBound, c);
            write(relErrorBound, c);
        } else if (errorBoundMode == EB_ABS_AND_REL) {
            write(absErrorBound, c);
            write(relErrorBound, c);
        }

        uint8_t boolvals = (lorenzo & 1) << 7 | (lorenzo2 & 1) << 6 | (regression & 1) << 5 | (regression2 & 1) << 4 |
                           (openmp & 1) << 3;
        write(boolvals, c);

        write(dataType, c);
        write(quantbinCnt, c);
        write(blockSize, c);
        write(predDim, c);

        auto confSize = static_cast<uchar>(c - c0);
        write(confSize, c0);
        return confSize;
    }

    size_t size_est() const {
        std::vector<uchar> buffer(sizeof(Config) + 1024);
        auto buffer_pos = buffer.data();
        return save(buffer_pos);
    }

    char N = 0;
    std::vector<size_t> dims;
    size_t num = 0;

    uint8_t cmprAlgo = ALGO_INTERP_LORENZO;
    uint8_t errorBoundMode = EB_ABS;
    double absErrorBound = 1e-3;
    double relErrorBound = 0.0;
    double psnrErrorBound = 0.0;
    double l2normErrorBound = 0.0;
    bool openmp = false;

    int quantbinCnt = 65536;
    int blockSize = 0;
    uint8_t predDim = 0;
    uint8_t dataType = 0;
    bool lorenzo = true;
    bool lorenzo2 = false;
    bool regression = true;
    bool regression2 = false;
    uint8_t interpAlgo = INTERP_ALGO_CUBIC;
    uint8_t interpDirection = 0;
    int interpAnchorStride = -1;
    double interpAlpha = 1.25;
    double interpBeta = 2.0;
};

}  // namespace SZ3

#endif  // H5Z_MANS_SZ3_CONFIG_MIN_H
