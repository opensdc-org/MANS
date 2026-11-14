#include "common.h"

// U32 1D fused decompression: single kernel unary-decodes the signal bits and
// reconstructs uint32 values from codes/center/signal. Mirrors
// decompress_fused.cpp (uint16) with uint32 elements and unsigned reconstruction.
__global__ void decompress_u32(
    uint32_t* decmp_data, uint32_t* centers, uint8_t* codes,
    int* d_output_lengths, uint8_t* d_concatenated_signals, uint8_t* d_signals,
    int data_size, int last_length, int gsize, int shift)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = bid * blockDim.x + tid;

    int lane = idx & 0x1f;
    int warp = idx >> 5;

    if (idx * decmp_chunk >= data_size) return;
    int end = (d_output_lengths[gsize - 1] + last_length) * cmp_tblock_size;

    warp = lane < 16 ? warp * 2 : warp * 2 + 1;
    int length = (warp == (gsize - 1)) ? last_length : d_output_lengths[warp + 1] - d_output_lengths[warp];

    int src_start_idx = d_output_lengths[warp] * cmp_tblock_size;
    src_start_idx = lane < 16 ? src_start_idx + lane * length * 2 : src_start_idx + (lane - 16) * length * 2;
    int dst_start_idx = idx * decmp_chunk;

    uint8_t local_signal[decmp_chunk] = {0};

    // Decode the first 16 elements (indices 0..15) from the first `length` bytes.
    {
        int signal_idx = -1;
        int run = 0;
        for (int offset_byte = 0; offset_byte < length && signal_idx < cmp_chunk; ++offset_byte) {
            const uint8_t bit_buffer = d_concatenated_signals[src_start_idx + offset_byte];
            #pragma unroll
            for (int i = 7; i >= 0 && signal_idx < cmp_chunk; --i) {
                if ((bit_buffer >> i) & 1) {
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
            if (src_start_idx + offset_byte + length > end) return;
            const uint8_t bit_buffer = d_concatenated_signals[src_start_idx + offset_byte + length];
            #pragma unroll
            for (int i = 7; i >= 0 && signal_idx < decmp_chunk; --i) {
                if ((bit_buffer >> i) & 1) {
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

    uint8_t local_codes[decmp_chunk];
    int4* local_codes_int2 = reinterpret_cast<int4*>(local_codes);
    int4* codes_int2 = reinterpret_cast<int4*>(codes + dst_start_idx);
    #pragma unroll
    for (int i = 0; i < decmp_chunk / 16; ++i) {
        local_codes_int2[i] = codes_int2[i];
    }

    uint32_t local_result[decmp_chunk];
    uint32_t center = centers[warp];

    #pragma unroll
    for (int i = 0; i < decmp_chunk; i++) {
        uint8_t code = local_codes[i];
        uint8_t signal = local_signal[i];
        int diff = (code % 2 == 1) ? ((code - 1) / 2) : (code / 2);
        diff += signal * 126;
        local_result[i] = (code % 2 == 1)
            ? (uint32_t)(center - (uint32_t)diff)
            : center + (uint32_t)diff;
    }

    int4* local_result_int4 = reinterpret_cast<int4*>(local_result);
    int4* decmp_int4 = reinterpret_cast<int4*>(decmp_data + dst_start_idx);
    #pragma unroll
    for (int i = 0; i < decmp_chunk / 4; ++i) {
        decmp_int4[i] = local_result_int4[i];
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
        decompress_u32<<<decmp_gsize, decmp_block_size, 0, stream>>>(d_decmpdata, d_centers, d_codes,
            d_output_lengths, d_concatenated_signals, d_signals,
            num_elements, last_length, gsize, shift);
        hipStreamSynchronize(stream);
    }

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    float decmp_milliseconds = 0;
    for (int i = 0; i < timing_iters; ++i) {
        hipEventRecord(start, stream);
        decompress_u32<<<decmp_gsize, decmp_block_size, 0, stream>>>(d_decmpdata, d_centers, d_codes,
            d_output_lengths, d_concatenated_signals, d_signals,
            num_elements, last_length, gsize, shift);
        hipEventRecord(stop, stream);
        hipEventSynchronize(stop);

        float ms = 0;
        hipEventElapsedTime(&ms, start, stop);
        decmp_milliseconds += ms;
    }
    decmp_milliseconds /= timing_iters;

    printf("\033[0;34m-------Decmp Throughput-------\033[0m\n");
    float decmp_throughput = (data_size / 1024.0 / 1024 / 1024) / (decmp_milliseconds / 1000.0f);
    printf("Decmp throughput: %.2f GB/s\n", decmp_throughput);
    printf("\033[0;34m-------Decmp Time-------\033[0m\n");
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
