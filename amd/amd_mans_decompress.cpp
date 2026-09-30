#include <iostream>
#include <string>
#include <vector>

#include "../mans_api.hpp"
#include "../mans_utils.h"

namespace {
void usage(const char* program) {
    std::cerr << "Usage: " << program << " <-u2|-u4> <input_file> <output_file>\n";
}

template <typename T>
bool run(const std::string& input_file, const std::string& output_file, mans::MansParams params) {
    std::vector<std::uint8_t> compressed;
    if (!mans::load_u8_file(input_file, compressed) || compressed.empty()) {
        std::cerr << "Failed to load compressed input: " << input_file << "\n";
        return false;
    }
    const std::size_t raw_bytes = mans::get_mans_exact_decompress_bytes(
        compressed.data(), compressed.size(), params);
    if (raw_bytes % sizeof(T) != 0) {
        std::cerr << "Header raw size is not aligned to the selected dtype.\n";
        return false;
    }
    std::vector<std::uint8_t> output(raw_bytes);
    std::size_t output_size = output.size();
    mans::decompress(compressed.data(), compressed.size(), params,
                     output.data(), output_size);
    if (output_size != raw_bytes || !mans::save_u8_file(output_file, output)) {
        std::cerr << "AMD decompression failed.\n";
        return false;
    }
    std::cout << "Decompressed " << input_file << " -> " << output_file
              << " (" << output_size << " bytes)\n";
    return true;
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        usage(argv[0]);
        return 1;
    }
    const std::string dtype = argv[1];
    const bool u16 = dtype == "-u2" || dtype == "u2";
    const bool u32 = dtype == "-u4" || dtype == "u4";
    if (!u16 && !u32) {
        usage(argv[0]);
        return 1;
    }
    mans::MansParams params{};
#ifdef MANS_ENABLE_DCU
    params.backend = mans::Backend::DCU;
#else
    params.backend = mans::Backend::AMD;
#endif
    params.dtype = u16 ? mans::DataType::U16 : mans::DataType::U32;
    params.mode = mans::Mode::P;
    try {
        return u16 ? (run<std::uint16_t>(argv[2], argv[3], params) ? 0 : 1)
                   : (run<std::uint32_t>(argv[2], argv[3], params) ? 0 : 1);
    } catch (const std::exception& error) {
        std::cerr << "AMD decompression error: " << error.what() << "\n";
        return 1;
    }
}
