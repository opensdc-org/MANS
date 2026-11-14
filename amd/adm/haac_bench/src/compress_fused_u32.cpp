#include "common.h"

// U32 1D fused compression: one warp per 512-element block, decoupled look-back
// prefix-sum fused into the mapping kernel. Mirrors compress_fused.cpp (uint16)
// but each thread loads four uint4 = 16 uint32, accumulates in uint64, and the
// center/values are uint32. Unconditional ADM (no raw path), matching the
// simplified uint16 benchmark so the two share the same timing methodology.
__global__ void map_values_kernel_decoupled_u32(
    const uint32_t* data, uint8_t* code, uint8_t* bit_signal, uint32_t* centers,
    int* signal_length,
    volatile int* const __restrict__ locOffset,
    volatile int* const __restrict__ cmpOffset,
    volatile int* const __restrict__ flag,
    int data_size, int shift)
{
    __shared__ unsigned int excl_sum;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = bid * blockDim.x + tid;
    const int lane = idx & 0x1f;
    const int warp = idx >> 5;

    const int base_block_start_idx = warp * cmp_chunk * cmp_tblock_size;
    const int base_thread_start_idx = base_block_start_idx + lane * cmp_chunk;
    const int base_thread_end_idx = base_thread_start_idx + cmp_chunk;

    uint8_t local_code[cmp_chunk] = {0};
    uint8_t local_output_idx[cmp_chunk] = {0};
    uint32_t vals[cmp_chunk] = {0};

    uint64_t local_sum = 0;
    uint32_t local_count = 0;
    const int end = min(base_thread_end_idx, data_size);

    // Load this thread's 16 uint32 as four vectorized uint4 once and reuse them
    // for both the center average and the per-element encoding. A boundary group
    // whose uint4 would overrun the buffer falls back to scalar loads so the
    // vectorized path never reads out of bounds (large uint32 centers would make
    // a zero-padded overread unsafe).
    if (base_thread_start_idx < end) {
        local_count = end - base_thread_start_idx;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int off = base_thread_start_idx + i * 4;
            if (off + 4 <= end) {
                const uint4 u = reinterpret_cast<const uint4*>(data)[off / 4];
                vals[i * 4 + 0] = u.x; vals[i * 4 + 1] = u.y;
                vals[i * 4 + 2] = u.z; vals[i * 4 + 3] = u.w;
            } else {
                vals[i * 4 + 0] = (off + 0 < end) ? data[off + 0] : 0u;
                vals[i * 4 + 1] = (off + 1 < end) ? data[off + 1] : 0u;
                vals[i * 4 + 2] = (off + 2 < end) ? data[off + 2] : 0u;
                vals[i * 4 + 3] = (off + 3 < end) ? data[off + 3] : 0u;
            }
        }
        #pragma unroll
        for (int i = 0; i < cmp_chunk; ++i)
            local_sum += (uint64_t)vals[i];
    }

    // Warp-reduce the sum and count.
    #pragma unroll
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0x00000000ffffffffL, local_sum, offset);
        local_count += __shfl_down_sync(0x00000000ffffffffL, local_count, offset);
    }

    // Lane 0 computes the average (uint32 center), then broadcasts it.
    uint32_t center = local_count > 0 ? (uint32_t)(local_sum / local_count) : 0u;
    center = __shfl_sync(0x00000000ffffffffL, center, 0);

    int bit_offset = 0;
    #pragma unroll
    for (int i = 0; i < cmp_chunk; ++i) {
        if (i < local_count) {
            const uint32_t val = vals[i];
            const uint32_t diff = val > center ? val - center : center - val;
            const int output_idx = (val == center) ? 1
                : (int)((diff - 1u) / 126u + 1u);
            local_code[i] = (val == center) ? (uint8_t)shift
                : (uint8_t)((val > center)
                    ? (diff + 126u - (uint32_t)output_idx * 126u) * 2u - 1u + (uint32_t)shift
                    : (diff + 126u - (uint32_t)output_idx * 126u) * 2u + (uint32_t)shift);
            local_output_idx[i] = (uint8_t)output_idx;
            bit_offset += output_idx;
        }
    }

    // Max per-thread bitstream length (bytes) across the warp.
    int max_bitstream_length = bit_offset;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        max_bitstream_length = max(max_bitstream_length,
            __shfl_down_sync(0x00000000ffffffffL, max_bitstream_length, offset));
    }
    max_bitstream_length = (max_bitstream_length + 7) / 8;  // bytes
    max_bitstream_length = __shfl_sync(0x00000000ffffffffL, max_bitstream_length, 0);

    // Write warp-level prefix-sum metadata to global memory.
    if (lane == 31) {
        signal_length[bid] = max_bitstream_length;
        centers[bid] = center;
        locOffset[warp + 1] = max_bitstream_length;
        __threadfence();
        if (warp == 0) {
            flag[0] = 2;
            __threadfence();
            flag[1] = 1;
            __threadfence();
        } else {
            flag[warp + 1] = 1;
            __threadfence();
        }
    }
    __syncthreads();

    // Global-level exclusive prefix-sum (decoupled look-back).
    if (warp == 0 && lane == 0) excl_sum = 0;
    __syncthreads();
    if (warp > 0) {
        if (!lane) {
            int lookback = warp;
            int loc_excl_sum = 0;
            while (lookback > 0) {
                int status;
                do {
                    status = flag[lookback];
                    __threadfence();
                } while (status == 0);
                if (status == 2) {
                    loc_excl_sum += cmpOffset[lookback];
                    __threadfence();
                    break;
                }
                if (status == 1) loc_excl_sum += locOffset[lookback];
                lookback--;
                __threadfence();
            }
            excl_sum = loc_excl_sum;
        }
        __syncthreads();
    }
    if (warp > 0) {
        if (!lane) cmpOffset[warp] = excl_sum;
        __threadfence();
        if (!lane) flag[warp] = 2;
        __threadfence();
    }
    __syncthreads();

    // Pack the variable-length unary signal bits into bytes via a register
    // accumulator and write straight to the concatenated-signal region.
    int write_pos = excl_sum * cmp_tblock_size + lane * max_bitstream_length;
    uint8_t* bit_out = bit_signal + write_pos;
    uint64_t acc = 0;
    int acc_bits = 0;
    int out_byte = 0;
    #pragma unroll
    for (int i = 0; i < cmp_chunk; ++i) {
        const int oi = local_output_idx[i];
        if (oi > 0) {
            acc = (acc << oi) | (1ull << (oi - 1));
            acc_bits += oi;
            while (acc_bits >= 8) {
                bit_out[out_byte++] = (uint8_t)((acc >> (acc_bits - 8)) & 0xFF);
                acc_bits -= 8;
                acc &= (1ull << acc_bits) - 1;
            }
        }
    }
    if (acc_bits > 0) {
        bit_out[out_byte] = (uint8_t)(((acc << (8 - acc_bits)) & 0xFF) | (0xFF >> acc_bits));
        for (int i = out_byte + 1; i < max_bitstream_length; ++i) bit_out[i] = 0;
    } else if (out_byte < max_bitstream_length) {
        bit_out[out_byte] = 0xFF;
        for (int i = out_byte + 1; i < max_bitstream_length; ++i) bit_out[i] = 0;
    }

    // Write the per-element codes (16 bytes per thread).
    int4* code_int2 = reinterpret_cast<int4*>(code + base_thread_start_idx);
    int4* local_code_int2 = reinterpret_cast<int4*>(local_code);
    #pragma unroll
    for (int i = 0; i < cmp_chunk / 16; ++i) {
        code_int2[i] = local_code_int2[i];
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input.u4> <output.adm>" << std::endl;
        return 1;
    }

    const char* input_file = argv[1];
    const char* output_file = argv[2];
    int shift = 1;
    const steady_clock::time_point total_start = steady_clock::now();

    std::cout << "Input file: " << input_file << std::endl;

    size_t data_size = 0;
    const steady_clock::time_point file_read_start = steady_clock::now();
    uint32_t* h_data = load_file_u32(input_file, data_size);
    const double file_read_ms = elapsed_ms(file_read_start, steady_clock::now());
    int num_elements = data_size / sizeof(uint32_t);
    if (!h_data) {
        std::cerr << "Error: Failed to load data from file: " << input_file << std::endl;
        return 1;
    }
    std::cout << "elements num: " << num_elements << std::endl;

    int bsize = cmp_tblock_size;
    int gsize = (num_elements + bsize * cmp_chunk - 1) / (bsize * cmp_chunk);

    uint32_t* d_data;
    void* d_cmpdata;
    int* d_signal_length;
    int* d_output_lengths;
    uint32_t* d_centers;
    uint8_t* d_codes;
    uint8_t* d_bit_signals;
    uint8_t* d_concatenated_signals;
    int* d_locOffset;
    int* d_flag;

    hipMalloc((void**)&d_data, data_size);
    hipMalloc((void**)&d_signal_length, gsize * sizeof(int));
    hipMemset(d_signal_length, 0, gsize * sizeof(int));
    hipMalloc((void**)&d_codes, bsize * gsize * cmp_chunk * sizeof(uint8_t));
    hipMalloc((void**)&d_locOffset, (gsize + 1) * sizeof(int));
    hipMalloc((void**)&d_flag, (gsize + 1) * sizeof(int));
    hipMalloc(&d_output_lengths, (gsize + 1) * sizeof(int));
    hipMalloc(&d_centers, gsize * sizeof(uint32_t));
    hipMalloc(&d_bit_signals, bsize * gsize * cmp_chunk * max_bytes_signal_per_ele_32b * sizeof(uint8_t));
    hipMemset(d_bit_signals, 0, bsize * gsize * cmp_chunk * max_bytes_signal_per_ele_32b * sizeof(uint8_t));
    hipMalloc(&d_concatenated_signals, bsize * gsize * cmp_chunk * max_bytes_signal_per_ele_32b * sizeof(uint8_t));
    hipMemset(d_concatenated_signals, 255, bsize * gsize * cmp_chunk * max_bytes_signal_per_ele_32b * sizeof(uint8_t));

    const steady_clock::time_point h2d_start = steady_clock::now();
    hipMemcpy(d_data, h_data, data_size, hipMemcpyHostToDevice);
    const double h2d_ms = elapsed_ms(h2d_start, steady_clock::now());

    // Warm up the same path that will be timed.
    for (int i = 0; i < 5; ++i) {
        hipMemset(d_flag, 0, (gsize + 1) * sizeof(int));
        hipMemset(d_locOffset, 0, (gsize + 1) * sizeof(int));
        hipMemset(d_output_lengths, 0, (gsize + 1) * sizeof(int));
        map_values_kernel_decoupled_u32<<<gsize, bsize>>>(
            d_data, d_codes, d_concatenated_signals,
            d_centers, d_signal_length, d_locOffset,
            d_output_lengths, d_flag, num_elements, shift);
        hipDeviceSynchronize();
    }

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipStream_t stream;
    hipStreamCreate(&stream);

    float adm_milliseconds = 0;
    float prefixsum_milliseconds = 0;
    float concat_milliseconds = 0;
    printf("=======>Use Decoupled Prefix-sum<=======\n");
    for (int i = 0; i < timing_iters; ++i) {
        hipMemset(d_flag, 0, (gsize + 1) * sizeof(int));
        hipMemset(d_locOffset, 0, (gsize + 1) * sizeof(int));
        hipMemset(d_output_lengths, 0, (gsize + 1) * sizeof(int));
        hipEventRecord(start, stream);
        map_values_kernel_decoupled_u32<<<gsize, bsize, 0, stream>>>(
            d_data, d_codes, d_concatenated_signals,
            d_centers, d_signal_length, d_locOffset,
            d_output_lengths, d_flag, num_elements, shift);
        hipEventRecord(stop, stream);
        hipEventSynchronize(stop);
        float ms = 0;
        hipEventElapsedTime(&ms, start, stop);
        adm_milliseconds += ms;
    }
    adm_milliseconds /= timing_iters;

    const float total_cmp_milliseconds =
        adm_milliseconds + prefixsum_milliseconds + concat_milliseconds;
    const float total_cmp_throughput =
        (data_size / 1024.0 / 1024 / 1024) / (total_cmp_milliseconds / 1000.0f);
    printf("Total Cmp throughput: %.2f GB/s\n", total_cmp_throughput);
    printf("ADM cost %.2f ms, prefix-sum cost %.2f ms, concat cost %.2f ms\n",
           adm_milliseconds, prefixsum_milliseconds, concat_milliseconds);

    int* h_signal_length = new int[gsize];
    int* h_output_lengths = new int[gsize];
    hipMemcpy(h_signal_length, d_signal_length, gsize * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(h_output_lengths, d_output_lengths, gsize * sizeof(int), hipMemcpyDeviceToHost);

    size_t bit_signals_size = (h_signal_length[gsize - 1] + h_output_lengths[gsize - 1]) * cmp_tblock_size * sizeof(uint8_t);
    size_t output_lengths_size = gsize * sizeof(int);
    size_t centers_size = gsize * sizeof(uint32_t);
    size_t codes_size = num_elements * sizeof(uint8_t);
    size_t total_size = output_lengths_size + centers_size + codes_size + bit_signals_size;

    hipMalloc(&d_cmpdata, total_size);

    int* d_signal_length_ptr = reinterpret_cast<int*>(d_cmpdata);
    void* d_centers_ptr = reinterpret_cast<uint8_t*>(d_cmpdata) + output_lengths_size;
    uint8_t* d_codes_ptr = reinterpret_cast<uint8_t*>(reinterpret_cast<uint8_t*>(d_cmpdata) + output_lengths_size + centers_size);
    uint8_t* d_bit_signals_ptr = reinterpret_cast<uint8_t*>(reinterpret_cast<uint8_t*>(d_cmpdata) + output_lengths_size + centers_size + codes_size);

    hipMemcpy(d_signal_length_ptr, d_output_lengths, output_lengths_size, hipMemcpyDeviceToDevice);
    hipMemcpy(d_centers_ptr, d_centers, centers_size, hipMemcpyDeviceToDevice);
    hipMemcpy(d_codes_ptr, d_codes, codes_size, hipMemcpyDeviceToDevice);
    hipMemcpy(d_bit_signals_ptr, d_concatenated_signals, bit_signals_size, hipMemcpyDeviceToDevice);

    uint8_t* h_cmpdata = new uint8_t[total_size];
    const steady_clock::time_point d2h_start = steady_clock::now();
    hipMemcpy(static_cast<uint8_t*>(h_cmpdata), d_cmpdata, total_size * sizeof(uint8_t), hipMemcpyDeviceToHost);
    const double d2h_ms = elapsed_ms(d2h_start, steady_clock::now());

    const steady_clock::time_point file_write_start = steady_clock::now();
    save_file(output_file, h_cmpdata, total_size);
    const double file_write_ms = elapsed_ms(file_write_start, steady_clock::now());

    printf("\033[0;34m-------Data Size-------\033[0m\n");
    printf("Original size: %zu Bytes\n", data_size);
    printf("Output size: %zu Bytes\n", total_size);
    printf("CR : %.2f x\n", data_size * 1.0 / total_size);
    printf("Output Length size: %zu Bytes\n", output_lengths_size);
    printf("Centers size: %zu Bytes\n", centers_size);
    printf("Codes size: %zu Bytes\n", codes_size);
    printf("Bit-signal size: %zu Bytes\n", bit_signals_size);
    printf("File read cost %.3f ms, H2D cost %.3f ms, D2H cost %.3f ms, file write cost %.3f ms\n",
           file_read_ms, h2d_ms, d2h_ms, file_write_ms);
    printf("Total wall cost %.3f ms\n",
           elapsed_ms(total_start, steady_clock::now()));

    hipFree(d_data);
    hipFree(d_cmpdata);
    hipFree(d_signal_length);
    hipFree(d_centers);
    hipFree(d_codes);
    hipFree(d_bit_signals);
    hipFree(d_concatenated_signals);
    hipFree(d_output_lengths);
    hipFree(d_locOffset);
    hipFree(d_flag);

    delete[] static_cast<uint8_t*>(h_cmpdata);
    delete[] static_cast<uint32_t*>(h_data);
    delete[] h_signal_length;
    delete[] h_output_lengths;

    hipEventDestroy(start);
    hipEventDestroy(stop);
    hipStreamDestroy(stream);

    printf("Memory has been successfully freed.\n");
    return 0;
}
