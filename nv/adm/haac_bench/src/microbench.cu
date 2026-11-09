// Synthetic NVIDIA GPU microbenchmark (integer, bitwise, device-memory copy).
// CUDA port of amd/adm/haac_bench/src/microbench.cpp so the same synthetic
// kernels run on the A100 as on the MI210.
//
// Usage: microbench [elements] [trials] [inner_iterations]

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void integer_kernel(const uint32_t* input, uint32_t* output,
                               size_t count, int iterations) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    uint32_t value = input[idx];
    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        value = value * 1664525u + 1013904223u;
    }
    output[idx] = value;
}

__global__ void bitwise_kernel(const uint32_t* input, uint32_t* output,
                               size_t count, int iterations) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    uint32_t value = input[idx];
    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        value ^= value << 13;
        value ^= value >> 17;
        value ^= value << 5;
        value = (value & 0x55555555u) | ((value & 0xaaaaaaaau) >> 1);
    }
    output[idx] = value;
}

__global__ void copy_kernel(const uint32_t* input, uint32_t* output, size_t count) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count) output[idx] = input[idx];
}

static void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(1);
    }
}

int main(int argc, char** argv) {
    const size_t count = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : (1ull << 24);
    const int trials = argc > 2 ? std::atoi(argv[2]) : 5;
    const int iterations = argc > 3 ? std::atoi(argv[3]) : 256;
    if (count == 0 || trials < 1 || iterations < 1) {
        std::fprintf(stderr, "Usage: %s [elements>0] [trials>0] [inner_iterations>0]\n", argv[0]);
        return 2;
    }

    std::vector<uint32_t> host(count);
    for (size_t i = 0; i < count; ++i) host[i] = static_cast<uint32_t>(i + 1);
    uint32_t* input = nullptr;
    uint32_t* output = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&input), count * sizeof(uint32_t)), "cudaMalloc input");
    check(cudaMalloc(reinterpret_cast<void**>(&output), count * sizeof(uint32_t)), "cudaMalloc output");
    check(cudaMemcpy(input, host.data(), count * sizeof(uint32_t), cudaMemcpyHostToDevice),
          "cudaMemcpy input");

    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "cudaEventCreate start");
    check(cudaEventCreate(&stop), "cudaEventCreate stop");
    const int block_size = 256;
    const int grid_size = static_cast<int>((count + block_size - 1) / block_size);
    std::puts("benchmark,trial,elements,inner_iterations,elapsed_ms,rate,unit");

    for (const char* benchmark : {"integer", "bitwise", "memory_copy"}) {
        for (int trial = 1; trial <= trials; ++trial) {
            if (benchmark[0] == 'i') integer_kernel<<<grid_size, block_size>>>(input, output, count, iterations);
            else if (benchmark[0] == 'b') bitwise_kernel<<<grid_size, block_size>>>(input, output, count, iterations);
            else copy_kernel<<<grid_size, block_size>>>(input, output, count);
            check(cudaGetLastError(), "microbenchmark warmup launch");
            check(cudaDeviceSynchronize(), "microbenchmark warmup sync");

            check(cudaEventRecord(start), "cudaEventRecord start");
            if (benchmark[0] == 'i') integer_kernel<<<grid_size, block_size>>>(input, output, count, iterations);
            else if (benchmark[0] == 'b') bitwise_kernel<<<grid_size, block_size>>>(input, output, count, iterations);
            else {
                for (int i = 0; i < iterations; ++i) copy_kernel<<<grid_size, block_size>>>(input, output, count);
            }
            check(cudaGetLastError(), "microbenchmark timed launch");
            check(cudaEventRecord(stop), "cudaEventRecord stop");
            check(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

            float elapsed_ms = 0;
            check(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");
            const double seconds = elapsed_ms / 1000.0;
            double rate;
            const char* unit;
            if (benchmark[0] == 'm') {
                rate = (2.0 * count * sizeof(uint32_t) * iterations / seconds) / 1.0e9;
                unit = "GB/s";
            } else {
                const int operations_per_iteration = benchmark[0] == 'i' ? 2 : 4;
                rate = (static_cast<double>(count) * iterations * operations_per_iteration / seconds) / 1.0e9;
                unit = "Gop/s";
            }
            std::printf("%s,%d,%zu,%d,%.6f,%.6f,%s\n", benchmark, trial,
                        count, iterations, elapsed_ms, rate, unit);
        }
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(input);
    cudaFree(output);
    return 0;
}
