#include "common.h"

__global__ void decode_signals_uint16(
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
    int signal_idx = -1;
    for (int offset_byte = 0; offset_byte < length && signal_idx < cmp_chunk; ++offset_byte) {
        const uint8_t bit_buffer = concatenated_signals[src_start_idx + offset_byte];
        #pragma unroll
        for (int bit_idx = 7; bit_idx >= 0 && signal_idx < cmp_chunk; --bit_idx) {
            if ((bit_buffer >> bit_idx) & 1) ++signal_idx;
            else ++local_signal[signal_idx];
        }
    }

    signal_idx = 15;
    for (int offset_byte = 0; offset_byte < length && signal_idx < decmp_chunk; ++offset_byte) {
        if (src_start_idx + offset_byte + length >= end) return;
        const uint8_t bit_buffer = concatenated_signals[src_start_idx + offset_byte + length];
        #pragma unroll
        for (int bit_idx = 7; bit_idx >= 0 && signal_idx < decmp_chunk; --bit_idx) {
            if ((bit_buffer >> bit_idx) & 1) ++signal_idx;
            else ++local_signal[signal_idx];
        }
    }

    const int valid = min(decmp_chunk, data_size - dst_start_idx);
    const int vector_end = valid & ~15;
    uint4* signal_vectors = reinterpret_cast<uint4*>(signals + dst_start_idx);
    for (int base = 0; base < vector_end; base += 16) {
        uint4 packed;
        packed.x = static_cast<uint32_t>(local_signal[base])
                 | (static_cast<uint32_t>(local_signal[base + 1]) << 8)
                 | (static_cast<uint32_t>(local_signal[base + 2]) << 16)
                 | (static_cast<uint32_t>(local_signal[base + 3]) << 24);
        packed.y = static_cast<uint32_t>(local_signal[base + 4])
                 | (static_cast<uint32_t>(local_signal[base + 5]) << 8)
                 | (static_cast<uint32_t>(local_signal[base + 6]) << 16)
                 | (static_cast<uint32_t>(local_signal[base + 7]) << 24);
        packed.z = static_cast<uint32_t>(local_signal[base + 8])
                 | (static_cast<uint32_t>(local_signal[base + 9]) << 8)
                 | (static_cast<uint32_t>(local_signal[base + 10]) << 16)
                 | (static_cast<uint32_t>(local_signal[base + 11]) << 24);
        packed.w = static_cast<uint32_t>(local_signal[base + 12])
                 | (static_cast<uint32_t>(local_signal[base + 13]) << 8)
                 | (static_cast<uint32_t>(local_signal[base + 14]) << 16)
                 | (static_cast<uint32_t>(local_signal[base + 15]) << 24);
        signal_vectors[base / 16] = packed;
    }
    for (int i = vector_end; i < valid; ++i) {
        signals[dst_start_idx + i] = local_signal[i];
    }
}

__global__ void restore_values_uint16(
    uint16_t* output,
    const uint16_t* centers,
    const uint8_t* codes,
    const uint8_t* signals,
    int data_size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_start_idx = idx * decmp_chunk;
    if (dst_start_idx >= data_size) return;

    const int lane = idx & 0x1f;
    const int warp = idx >> 5;
    const int center = centers[warp * 2 + (lane < 16 ? 0 : 1)];
    const int valid = min(decmp_chunk, data_size - dst_start_idx);
    const int vector_end = valid & ~15;

    union ByteVector {
        int4 value;
        uint8_t bytes[16];
    } code_vector, signal_vector;
    alignas(16) uint16_t result_vector[8];
    int4* output_vectors = reinterpret_cast<int4*>(output + dst_start_idx);

    for (int base = 0; base < vector_end; base += 16) {
        code_vector.value = *reinterpret_cast<const int4*>(codes + dst_start_idx + base);
        signal_vector.value = *reinterpret_cast<const int4*>(signals + dst_start_idx + base);

        #pragma unroll
        for (int group = 0; group < 2; ++group) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int offset = group * 8 + i;
                const uint8_t code = code_vector.bytes[offset];
                const int diff = (code >> 1) + signal_vector.bytes[offset] * 126;
                result_vector[i] = code & 1 ? center - diff : center + diff;
            }
            output_vectors[base / 8 + group] = *reinterpret_cast<const int4*>(result_vector);
        }
    }

    const int vector8_end = valid & ~7;
    for (int base = vector_end; base < vector8_end; base += 8) {
        const uint32_t code_word = *reinterpret_cast<const uint32_t*>(codes + dst_start_idx + base);
        const uint32_t signal_word = *reinterpret_cast<const uint32_t*>(signals + dst_start_idx + base);

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint8_t code = (code_word >> (i * 8)) & 0xff;
            const uint8_t signal = (signal_word >> (i * 8)) & 0xff;
            const int diff = (code & 1 ? (code - 1) / 2 : code / 2) + signal * 126;
            result_vector[i] = code & 1 ? center - diff : center + diff;
        }
        output_vectors[base / 8] = *reinterpret_cast<const int4*>(result_vector);
    }

    for (int i = vector8_end; i < valid; ++i) {
        const uint8_t code = codes[dst_start_idx + i];
        const int diff = (code & 1 ? (code - 1) / 2 : code / 2)
                       + signals[dst_start_idx + i] * 126;
        output[dst_start_idx + i] = code & 1 ? center - diff : center + diff;
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input.adm> <original.u2>" << std::endl;
        return 1;
    }

    const char* adm_file = argv[1];
    const char* u2_file = argv[2];
    int shift = 1;
    const steady_clock::time_point total_start = steady_clock::now();

    size_t data_size = 0;
    const steady_clock::time_point file_read_start = steady_clock::now();
    uint16_t* h_original = load_file_u16(u2_file, data_size);
    const double original_read_ms = elapsed_ms(file_read_start, steady_clock::now());
    int num_elements = data_size / sizeof(uint16_t);
    if (!h_original) {
        std::cerr << "Error: Failed to load original data from file: " << u2_file << std::endl;
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
    size_t centers_size = gsize * sizeof(uint16_t);
    size_t codes_size = num_elements * sizeof(uint8_t);
    size_t bit_signals_size = adm_size - output_lengths_size - centers_size - codes_size;

    const int* h_output_lengths = reinterpret_cast<const int*>(h_adm);
    const uint16_t* h_centers = reinterpret_cast<const uint16_t*>(h_adm + output_lengths_size);
    const uint8_t* h_codes = h_adm + output_lengths_size + centers_size;
    const uint8_t* h_signals = h_adm + output_lengths_size + centers_size + codes_size;

    int last_length = bit_signals_size / cmp_tblock_size - h_output_lengths[gsize - 1];

    uint16_t* d_decmpdata;
    uint16_t* d_centers;
    uint8_t* d_codes;
    int* d_output_lengths;
    uint8_t* d_signals;
    uint8_t* d_concatenated_signals;

    hipMalloc(&d_decmpdata, num_elements * sizeof(uint16_t));
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
        decode_signals_uint16<<<decmp_gsize, decmp_block_size, 0, stream>>>(
            d_output_lengths, d_concatenated_signals, d_signals,
            num_elements, last_length, gsize);
        restore_values_uint16<<<decmp_gsize, decmp_block_size, 0, stream>>>(
            d_decmpdata, d_centers, d_codes, d_signals, num_elements);

        hipStreamSynchronize(stream);
    }

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    float decode_milliseconds = 0;
    float restore_milliseconds = 0;
    hipEventRecord(start, stream);
    decode_signals_uint16<<<decmp_gsize, decmp_block_size, 0, stream>>>(
        d_output_lengths, d_concatenated_signals, d_signals,
        num_elements, last_length, gsize);
    hipEventRecord(stop, stream);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&decode_milliseconds, start, stop);

    hipEventRecord(start, stream);
    restore_values_uint16<<<decmp_gsize, decmp_block_size, 0, stream>>>(
        d_decmpdata, d_centers, d_codes, d_signals, num_elements);
    hipEventRecord(stop, stream);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&restore_milliseconds, start, stop);

    const float decmp_milliseconds = decode_milliseconds + restore_milliseconds;
    printf("\\033[0;34m-------Decmp Throughput-------\\033[0m\\n");
    float decmp_throughput = (data_size / 1024.0 / 1024 / 1024) / (decmp_milliseconds / 1000.0f);
    printf("Decmp throughput: %.2f GB/s\\n", decmp_throughput);
    printf("\\033[0;34m-------Decmp Time-------\\033[0m\\n");
    printf("Decode signals cost %.2f ms, restore values cost %.2f ms\\n",
           decode_milliseconds, restore_milliseconds);
    printf("Total cost %.2f ms\\n", decmp_milliseconds);

    uint16_t* h_decmpdata = new uint16_t[num_elements];
    const steady_clock::time_point d2h_start = steady_clock::now();
    hipMemcpy(h_decmpdata, d_decmpdata, num_elements * sizeof(uint16_t), hipMemcpyDeviceToHost);
    const double d2h_ms = elapsed_ms(d2h_start, steady_clock::now());

    const steady_clock::time_point verify_start = steady_clock::now();
    bool test = true;
    for (int i = 0; i < num_elements; ++i) {
        if (h_decmpdata[i] != h_original[i]) {
            printf("\033[0;31mFail error check!\033[0m\n");
            printf("\033[0;31mError: Data mismatch at index %d, original = %d, decompressed = %d\033[0m\n",
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
    hipStreamDestroy(stream);

    printf("Memory has been successfully freed.\n");
    return 0;
}
