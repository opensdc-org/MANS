#pragma once
// Shared constants and file I/O helpers for the HAAC ADM benchmarks.
//
// The .adm output is a flat, headerless buffer laid out as:
//   [output_lengths (gsize * int32)][centers (gsize * uint16)]
//   [codes (num_elements * uint8)][concatenated bit-signals (rest of file)]
//
// gsize = ceil(num_elements / (cmp_tblock_size * cmp_chunk)), and the last
// per-block bitstream length (needed by the decompress kernel as `last_length`)
// is recovered as:
//   last_length = bit_signals_size / cmp_tblock_size - output_lengths[gsize-1]

#include <hip/hip_runtime.h>
#include <iostream>
#include <fstream>
#include <stdint.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>

static const int cmp_tblock_size = 32; // 32 threads
static const int cmp_chunk = 16;
static const int max_bytes_signal_per_ele_16b = 2;
static const int decmp_chunk = 32;  // one decomp thread processes decmp_chunk cmp threads
static const int warp_size = 32;
static const int k = 4;
static const int kn = 16;

// Load a .u2 input file (uint16 elements) into host memory.
inline uint16_t* load_file_u16(const char* filename, size_t& size) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return nullptr;
    }
    size = file.tellg();
    file.seekg(0, std::ios::beg);
    size_t num_elements = size / sizeof(uint16_t);
    uint16_t* data = new uint16_t[num_elements];
    file.read(reinterpret_cast<char*>(data), size);
    file.close();
    return data;
}

// Load a raw byte buffer (an .adm file) into host memory.
inline uint8_t* load_file_bytes(const char* filename, size_t& size) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return nullptr;
    }
    size = file.tellg();
    file.seekg(0, std::ios::beg);
    uint8_t* data = new uint8_t[size];
    file.read(reinterpret_cast<char*>(data), size);
    file.close();
    return data;
}

// Save a byte buffer to a file.
inline void save_file(const char* filename, const uint8_t* data, size_t size) {
    std::ofstream file(filename, std::ios::binary);
    file.write(reinterpret_cast<const char*>(data), size);
    file.close();
}
