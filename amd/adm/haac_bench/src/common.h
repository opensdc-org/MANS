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

#ifdef MANS_BENCH_CUDA
#include <cuda_runtime.h>
using hipError_t = cudaError_t;
using hipEvent_t = cudaEvent_t;
using hipStream_t = cudaStream_t;
using hipMemcpyKind = cudaMemcpyKind;
static const hipError_t hipSuccess = cudaSuccess;
static const hipMemcpyKind hipMemcpyHostToDevice = cudaMemcpyHostToDevice;
static const hipMemcpyKind hipMemcpyDeviceToHost = cudaMemcpyDeviceToHost;
static const hipMemcpyKind hipMemcpyDeviceToDevice = cudaMemcpyDeviceToDevice;
inline const char* hipGetErrorString(hipError_t status) { return cudaGetErrorString(status); }
inline hipError_t hipGetLastError() { return cudaGetLastError(); }
inline hipError_t hipDeviceSynchronize() { return cudaDeviceSynchronize(); }
inline hipError_t hipMemcpy(void* dst, const void* src, size_t size, hipMemcpyKind kind) {
    return cudaMemcpy(dst, src, size, kind);
}
inline hipError_t hipMemset(void* dst, int value, size_t size) {
    return cudaMemset(dst, value, size);
}
inline hipError_t hipFree(void* ptr) { return cudaFree(ptr); }
inline hipError_t hipEventCreate(hipEvent_t* event) { return cudaEventCreate(event); }
inline hipError_t hipEventDestroy(hipEvent_t event) { return cudaEventDestroy(event); }
inline hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream = nullptr) {
    return cudaEventRecord(event, stream);
}
inline hipError_t hipEventSynchronize(hipEvent_t event) { return cudaEventSynchronize(event); }
inline hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop) {
    return cudaEventElapsedTime(ms, start, stop);
}
inline hipError_t hipStreamCreate(hipStream_t* stream) { return cudaStreamCreate(stream); }
inline hipError_t hipStreamSynchronize(hipStream_t stream) { return cudaStreamSynchronize(stream); }
inline hipError_t hipStreamDestroy(hipStream_t stream) { return cudaStreamDestroy(stream); }
template <typename T>
inline hipError_t hipMalloc(T** ptr, size_t size) {
    return cudaMalloc(reinterpret_cast<void**>(ptr), size);
}
#else
#include <hip/hip_runtime.h>
#endif
#include <iostream>
#include <fstream>
#include <stdint.h>
#include <chrono>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>

using steady_clock = std::chrono::steady_clock;
inline double elapsed_ms(const steady_clock::time_point& start,
                         const steady_clock::time_point& stop) {
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

static const int cmp_tblock_size = 32; // 32 threads
static const int cmp_chunk = 16;
// Bytes of bit-signal scratch reserved per element. A 16-bit value can need up to
// ceil(65535/126) ~ 521 output-idx bits, but the ADM code stores output_idx in a
// uint8 (max 255) and our real u2/u4 inputs need <= 117 (exafel), so 16 bytes per
// element (128 bits) covers every block in the test corpus; the pure-ADM datasets
// only need 1. The scratch stride is sized from these constants, not the payload.
static const int max_bytes_signal_per_ele_16b = 16;
static const int max_bytes_signal_per_ele_32b = 2;
static const int decmp_chunk = 32;  // one decomp thread processes decmp_chunk cmp threads
static const int warp_size = 32;
static const int k = 4;
static const int kn = 16;
static const int timing_iters = 10;  // timed kernel launches to average over

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

// Load a .u4 input file (uint32 elements) into host memory.
inline uint32_t* load_file_u32(const char* filename, size_t& size) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return nullptr;
    }
    size = file.tellg();
    file.seekg(0, std::ios::beg);
    size_t num_elements = size / sizeof(uint32_t);
    uint32_t* data = new uint32_t[num_elements];
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
