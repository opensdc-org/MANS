#include "common.h"

// U32 1D unfused decompression: decode_signals_u32 unary-decodes the bit-signal
// bytes into a per-element uint8 signal array, then restore_values_u32 rebuilds
// uint32 values from codes/center/signal. Mirrors decompress_unfused.cpp.
// decode_signals is element-type independent, so it is identical to the uint16
// version; only the restore kernel differs (uint32 center/values, unsigned
// reconstruction).

__global__ void decode_signals_u32(
    const int* output_lengths,
    const uint8_t* concatenated_signals,
    uint8_t* signals,
    int data_size,
    int last_length,
    int gsize)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_start_idx = idx * decmp_chunk;
    if (dst_start_idx >= data_size) return;

    const int lane = idx & 0x1f;
    int warp = idx >> 5;
    const int end = (output_lengths[gsize - 1] + last_length) * cmp_tblock_size;
    warp = lane < 16 ? warp * 2 : warp * 2 + 1;
    const int length = warp == gsize - 1
        ? last_length
        : output_lengths[warp + 1] - output_lengths[warp];
    int src_start_idx = output_lengths[warp] * cmp_tblock_size;
    src_start_idx += lane < 16 ? lane * length * 2 : (lane - 16) * length * 2;

    uint8_t local_signal[decmp_chunk] = {0};

    // Decode the first 16 elements (indices 0..15) from the first `length` bytes.
    {
        int signal_idx = -1;
        int run = 0;
        for (int offset_byte = 0; offset_byte < length && signal_idx < cmp_chunk; ++offset_byte) {
            const uint8_t bit_buffer = concatenated_signals[src_start_idx + offset_byte];
            #pragma unroll
            for (int bit_idx = 7; bit_idx >= 0 && signal_idx < cmp_chunk; --bit_idx) {
                if ((bit_buffer >> bit_idx) & 1) {
                    if (signal_idx >= 0 && run != 0) {
                        #pragma unroll
                        for (int j = 0; j < cmp_chunk; ++j) if (j == signal_idx) local_signal[j] = (uint8_t)run;
                    }
                    ++signal_idx;
                    run = 0;
                } else {
                    ++run;
                }
            }
        }
        if (signal_idx >= 0 && run != 0) {
            #pragma unroll
            for (int j = 0; j < cmp_chunk; ++j) if (j == signal_idx) local_signal[j] = (uint8_t)run;
        }
    }

    // Decode the next 16 elements (indices 16..31) from the second `length` bytes.
    {
        int signal_idx = cmp_chunk - 1;
        int run = 0;
        for (int offset_byte = 0; offset_byte < length && signal_idx < decmp_chunk; ++offset_byte) {
            if (src_start_idx + offset_byte + length >= end) return;
            const uint8_t bit_buffer = concatenated_signals[src_start_idx + offset_byte + length];
            #pragma unroll
            for (int bit_idx = 7; bit_idx >= 0 && signal_idx < decmp_chunk; --bit_idx) {
                if ((bit_buffer >> bit_idx) & 1) {
                    if (signal_idx >= cmp_chunk && run != 0) {
                        #pragma unroll
                        for (int j = cmp_chunk; j < decmp_chunk; ++j) if (j == signal_idx) local_signal[j] = (uint8_t)run;
                    }
                    ++signal_idx;
                    run = 0;
                } else {
                    ++run;
                }
            }
        }
        if (signal_idx >= cmp_chunk && run != 0) {
            #pragma unroll
            for (int j = cmp_chunk; j < decmp_chunk; ++j) if (j == signal_idx) local_signal[j] = (uint8_t)run;
        }
    }

    const int valid = min(decmp_chunk, data_size - dst_start_idx);
    uint8_t* out = signals + dst_start_idx;
    if (valid == decmp_chunk) {
        uint4 v0, v1;
        v0.x = static_cast<uint32_t>(local_signal[0])  | (static_cast<uint32_t>(local_signal[1]) << 8)
             | (static_cast<uint32_t>(local_signal[2]) << 16)  | (static_cast<uint32_t>(local_signal[3]) << 24);
        v0.y = static_cast<uint32_t>(local_signal[4])  | (static_cast<uint32_t>(local_signal[5]) << 8)
             | (static_cast<uint32_t>(local_signal[6]) << 16)  | (static_cast<uint32_t>(local_signal[7]) << 24);
        v0.z = static_cast<uint32_t>(local_signal[8])  | (static_cast<uint32_t>(local_signal[9]) << 8)
             | (static_cast<uint32_t>(local_signal[10]) << 16) | (static_cast<uint32_t>(local_signal[11]) << 24);
        v0.w = static_cast<uint32_t>(local_signal[12]) | (static_cast<uint32_t>(local_signal[13]) << 8)
             | (static_cast<uint32_t>(local_signal[14]) << 16) | (static_cast<uint32_t>(local_signal[15]) << 24);
        v1.x = static_cast<uint32_t>(local_signal[16]) | (static_cast<uint32_t>(local_signal[17]) << 8)
             | (static_cast<uint32_t>(local_signal[18]) << 16) | (static_cast<uint32_t>(local_signal[19]) << 24);
        v1.y = static_cast<uint32_t>(local_signal[20]) | (static_cast<uint32_t>(local_signal[21]) << 8)
             | (static_cast<uint32_t>(local_signal[22]) << 16) | (static_cast<uint32_t>(local_signal[23]) << 24);
        v1.z = static_cast<uint32_t>(local_signal[24]) | (static_cast<uint32_t>(local_signal[25]) << 8)
             | (static_cast<uint32_t>(local_signal[26]) << 16) | (static_cast<uint32_t>(local_signal[27]) << 24);
        v1.w = static_cast<uint32_t>(local_signal[28]) | (static_cast<uint32_t>(local_signal[29]) << 8)
             | (static_cast<uint32_t>(local_signal[30]) << 16) | (static_cast<uint32_t>(local_signal[31]) << 24);
        reinterpret_cast<uint4*>(out)[0] = v0;
        reinterpret_cast<uint4*>(out)[1] = v1;
    } else {
        #pragma unroll
        for (int i = 0; i < decmp_chunk; ++i) {
            if (i < valid) out[i] = local_signal[i];
        }
    }
}

__global__ void restore_values_u32(
    uint32_t* output,
    const uint32_t* centers,
    const uint8_t* codes,
    const uint8_t* signals,
    int data_size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_start_idx = idx * decmp_chunk;
    if (dst_start_idx >= data_size) return;

    const int lane = idx & 0x1f;
    const int warp = idx >> 5;
    const uint32_t center = centers[warp * 2 + (lane < 16 ? 0 : 1)];
    const int valid = min(decmp_chunk, data_size - dst_start_idx);

    if (valid == decmp_chunk) {
        #pragma unroll
        for (int base = 0; base < decmp_chunk; base += 16) {
            const int4 cv = *reinterpret_cast<const int4*>(codes + dst_start_idx + base);
            const int4 sv = *reinterpret_cast<const int4*>(signals + dst_start_idx + base);
            const uint8_t* cb = reinterpret_cast<const uint8_t*>(&cv);
            const uint8_t* sb = reinterpret_cast<const uint8_t*>(&sv);
            uint32_t res[16];
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                const uint8_t code = cb[i];
                const int diff = (code >> 1) + sb[i] * 126;
                res[i] = code & 1 ? (uint32_t)(center - (uint32_t)diff) : center + (uint32_t)diff;
            }
            int4* dst4 = reinterpret_cast<int4*>(output + dst_start_idx + base);
            int4* res4 = reinterpret_cast<int4*>(res);
            #pragma unroll
            for (int i = 0; i < 4; ++i) dst4[i] = res4[i];
        }
    } else {
        for (int i = 0; i < valid; ++i) {
            const uint8_t code = codes[dst_start_idx + i];
            const int diff = (code >> 1) + signals[dst_start_idx + i] * 126;
            output[dst_start_idx + i] = code & 1
                ? (uint32_t)(center - (uint32_t)diff)
                : center + (uint32_t)diff;
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input.adm> <original.u4>" << std::endl;
        return 1;
    }

    const char* adm_file = argv[1];
    const char* u4_file = argv[2];
    int shift = 1;
    const steady_clock::time_point total_start = steady_clock::now();

    size_t data_size = 0;
    const steady_clock::time_point file_read_start = steady_clock::now();
    uint32_t* h_original = load_file_u32(u4_file, data_size);
    const double original_read_ms = elapsed_ms(file_read_start, steady_clock::now());
    int num_elements = data_size / sizeof(uint32_t);
    if (!h_original) {
        std::cerr << "Error: Failed to load original data from file: " << u4_file << std::endl;
        return 1;
    }
    std::cout << "elements num: " << num_elements << std::endl;

    size_t adm_size = 0;
    const steady_clock::time_point adm_read_start = steady_clock::now();
    uint8_t* h_adm = load_file_bytes(adm_file, adm_size);
    const double adm_read_ms = elapsed_ms(adm_read_start, steady_clock::now());
    if (!h_adm) {
        std::cerr << "Error: Failed to load ADM data from file: " << adm_file << std::endl;
        return 1;
    }

    int bsize = cmp_tblock_size;
    int gsize = (num_elements + bsize * cmp_chunk - 1) / (bsize * cmp_chunk);
    const int decmp_block_size = 256;
    int decmp_gsize = (num_elements + decmp_block_size * decmp_chunk - 1) / (decmp_block_size * decmp_chunk);

    size_t output_lengths_size = gsize * sizeof(int);
    size_t centers_size = gsize * sizeof(uint32_t);
    size_t codes_size = num_elements * sizeof(uint8_t);
    size_t bit_signals_size = adm_size - output_lengths_size - centers_size - codes_size;

    const int* h_output_lengths = reinterpret_cast<const int*>(h_adm);
    const uint32_t* h_centers = reinterpret_cast<const uint32_t*>(h_adm + output_lengths_size);
    const uint8_t* h_codes = h_adm + output_lengths_size + centers_size;
    const uint8_t* h_signals = h_adm + output_lengths_size + centers_size + codes_size;

    int last_length = bit_signals_size / cmp_tblock_size - h_output_lengths[gsize - 1];

    uint32_t* d_decmpdata;
    uint32_t* d_centers;
    uint8_t* d_codes;
    int* d_output_lengths;
    uint8_t* d_signals;
    uint8_t* d_concatenated_signals;

    hipMalloc(&d_decmpdata, num_elements * sizeof(uint32_t));
    hipMalloc(&d_centers, centers_size);
    hipMalloc(&d_codes, codes_size);
    hipMalloc(&d_output_lengths, output_lengths_size);
    hipMalloc(&d_signals, num_elements * sizeof(uint8_t));
    hipMalloc(&d_concatenated_signals, bit_signals_size);

    const steady_clock::time_point h2d_start = steady_clock::now();
    hipMemcpy(d_centers, h_centers, centers_size, hipMemcpyHostToDevice);
    hipMemcpy(d_codes, h_codes, codes_size, hipMemcpyHostToDevice);
    hipMemcpy(d_output_lengths, h_output_lengths, output_lengths_size, hipMemcpyHostToDevice);
    hipMemcpy(d_concatenated_signals, h_signals, bit_signals_size, hipMemcpyHostToDevice);
    const double h2d_ms = elapsed_ms(h2d_start, steady_clock::now());

    hipStream_t stream;
    hipStreamCreate(&stream);

    // Warm up the same path that will be timed.
    for (int i = 0; i < 5; ++i) {
        decode_signals_u32<<<decmp_gsize, decmp_block_size, 0, stream>>>(
            d_output_lengths, d_concatenated_signals, d_signals,
            num_elements, last_length, gsize);
        restore_values_u32<<<decmp_gsize, decmp_block_size, 0, stream>>>(
            d_decmpdata, d_centers, d_codes, d_signals, num_elements);
        hipStreamSynchronize(stream);
    }

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    float decode_milliseconds = 0;
    float restore_milliseconds = 0;
    for (int i = 0; i < timing_iters; ++i) {
        hipEventRecord(start, stream);
        decode_signals_u32<<<decmp_gsize, decmp_block_size, 0, stream>>>(
            d_output_lengths, d_concatenated_signals, d_signals,
            num_elements, last_length, gsize);
        hipEventRecord(stop, stream);
        hipEventSynchronize(stop);
        float ms = 0;
        hipEventElapsedTime(&ms, start, stop);
        decode_milliseconds += ms;

        hipEventRecord(start, stream);
        restore_values_u32<<<decmp_gsize, decmp_block_size, 0, stream>>>(
            d_decmpdata, d_centers, d_codes, d_signals, num_elements);
        hipEventRecord(stop, stream);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&ms, start, stop);
        restore_milliseconds += ms;
    }
    decode_milliseconds /= timing_iters;
    restore_milliseconds /= timing_iters;

    const float decmp_milliseconds = decode_milliseconds + restore_milliseconds;
    printf("\033[0;34m-------Decmp Throughput-------\033[0m\n");
    float decmp_throughput = (data_size / 1024.0 / 1024 / 1024) / (decmp_milliseconds / 1000.0f);
    printf("Decmp throughput: %.2f GB/s\n", decmp_throughput);
    printf("\033[0;34m-------Decmp Time-------\033[0m\n");
    printf("Decode signals cost %.2f ms, restore values cost %.2f ms\n",
           decode_milliseconds, restore_milliseconds);
    printf("Total cost %.2f ms\n", decmp_milliseconds);

    uint32_t* h_decmpdata = new uint32_t[num_elements];
    const steady_clock::time_point d2h_start = steady_clock::now();
    hipMemcpy(h_decmpdata, d_decmpdata, num_elements * sizeof(uint32_t), hipMemcpyDeviceToHost);
    const double d2h_ms = elapsed_ms(d2h_start, steady_clock::now());

    const steady_clock::time_point verify_start = steady_clock::now();
    bool test = true;
    for (int i = 0; i < num_elements; ++i) {
        if (h_decmpdata[i] != h_original[i]) {
            printf("\033[0;31mFail error check!\033[0m\n");
            printf("\033[0;31mError: Data mismatch at index %d, original = %u, decompressed = %u\033[0m\n",
                   i, h_original[i], h_decmpdata[i]);
            test = false;
            break;
        }
    }
    if (test) printf("\033[0;32mPass error check!\033[0m\n");
    const double cpu_verify_ms = elapsed_ms(verify_start, steady_clock::now());
    printf("File read cost %.3f ms (original %.3f ms, ADM %.3f ms), H2D cost %.3f ms, D2H cost %.3f ms, CPU verify cost %.3f ms\n",
           original_read_ms + adm_read_ms, original_read_ms, adm_read_ms,
           h2d_ms, d2h_ms, cpu_verify_ms);
    printf("Total wall cost %.3f ms\n",
           elapsed_ms(total_start, steady_clock::now()));

    hipFree(d_decmpdata);
    hipFree(d_centers);
    hipFree(d_codes);
    hipFree(d_output_lengths);
    hipFree(d_signals);
    hipFree(d_concatenated_signals);

    delete[] h_decmpdata;
    delete[] h_original;
    delete[] h_adm;

    hipEventDestroy(start);
    hipEventDestroy(stop);

    printf("Memory has been successfully freed.\n");
    return 0;
}
