#include "common.h"

__global__ void map_values_kernel_thrust(const uint16_t* data, uint8_t* code, uint8_t* bit_signal, uint16_t* centers, int* signal_length ,int data_size, int shift) 
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = bid * blockDim.x + tid;
    const int lane = idx & 0x1f;
    const int warp = idx >> 5;
    const int block_num = 1;   

    int base_block_start_idx;
    int base_thread_start_idx, base_thread_end_idx;
    uint4 tmp_buffer;
    
    uint8_t local_code[cmp_chunk] = {0};
    uint8_t local_output_idx[cmp_chunk] = {0};

    int diff = 0;
    int output_idx = 0;
    int local_idx = 0;
    int bit_offset = 0; // current bit position
    uint16_t currValue = 0;

    base_block_start_idx = warp * cmp_chunk * 32;
    // base_block_end_idx = base_block_start_idx + cmp_chunk * 32;
    base_thread_start_idx = base_block_start_idx + lane * cmp_chunk;
    base_thread_end_idx = base_thread_start_idx + cmp_chunk;

     // 每线程的局部和与计数
    uint32_t local_sum = 0;
    uint32_t local_count = 0;
    int end = min(base_thread_end_idx, data_size);

    // Load this thread's 16 elements as two vectorized uint4 once and reuse them
    // for both the center average and the per-element encoding, instead of 16
    // scalar __ldg loads (a large latency source at low occupancy).
    uint4 buf[2];
    if (base_thread_start_idx < end) {
        buf[0] = reinterpret_cast<const uint4*>(data)[base_thread_start_idx / 8];
        buf[1] = (base_thread_start_idx + 8 < end)
            ? reinterpret_cast<const uint4*>(data)[base_thread_start_idx / 8 + 1]
            : make_uint4(0, 0, 0, 0);
        local_count = end - base_thread_start_idx;
        #pragma unroll
        for (int b = 0; b < 2; ++b) {
            uint32_t t;
            t = buf[b].x; local_sum += (t & 0xFFFF) + (t >> 16);
            t = buf[b].y; local_sum += (t & 0xFFFF) + (t >> 16);
            t = buf[b].z; local_sum += (t & 0xFFFF) + (t >> 16);
            t = buf[b].w; local_sum += (t & 0xFFFF) + (t >> 16);
        }
    } else {
        buf[0] = make_uint4(0, 0, 0, 0);
        buf[1] = make_uint4(0, 0, 0, 0);
    }
 
     // 利用 Warp Shuffle 汇总所有线程的和与计数
    #pragma unroll
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0x00000000ffffffffL, local_sum, offset);
        local_count += __shfl_down_sync(0x00000000ffffffffL, local_count, offset);
    }

 
     // Warp 内线程 0 计算平均值，并写入 center 数组
    int center = 0;
    center = local_count > 0 ? local_sum / local_count : 0;
    center = __shfl_sync(0x00000000ffffffffL, center, 0);
 

    bool is_center;
    uint16_t remain = 0;
    uint8_t res = 0;

    #pragma unroll
    for(int j = 0; j < block_num; j++)
    {
        if(base_thread_start_idx > data_size) break;

        #pragma unroll
        for(int b = 0; b < 2; b++)
        {
            tmp_buffer = buf[b];

            currValue = static_cast<uint16_t>(tmp_buffer.x & 0xFFFF);
            is_center = (currValue == center);
            diff = (currValue > center) ? currValue - center : center - currValue;
            remain = diff % 126;
            output_idx = is_center       ?   1 
                       : (remain == 0)   ?   diff / 126 : diff / 126 + 1;

            
            res = (currValue > center) ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                                            : (diff + 126 - output_idx * 126) * 2 + shift;
            // code
            local_code[local_idx] = is_center ? shift : res;
            // signal 
            local_output_idx[local_idx] = output_idx;
            // next
            local_idx++;
            bit_offset += output_idx;
            
            currValue = static_cast<uint16_t>((tmp_buffer.x >> 16) & 0xFFFF);
            is_center = (currValue == center);
            diff = (currValue > center) ? currValue - center : center - currValue;
            remain = diff % 126;
            output_idx = is_center       ?   1 
                       : (remain == 0)   ?   diff / 126 : diff / 126 + 1;

            res = (currValue > center) ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                                            : (diff + 126 - output_idx * 126) * 2 + shift;
            // code
            local_code[local_idx] = is_center ? shift : res;
            // signal 
            local_output_idx[local_idx] = output_idx;
            // next
            local_idx++;
            bit_offset += output_idx;

            currValue = static_cast<uint16_t>(tmp_buffer.y & 0xFFFF);
            is_center = (currValue == center);
            diff = (currValue > center) ? currValue - center : center - currValue;
            remain = diff % 126;
            output_idx = is_center       ?   1 
                       : (remain == 0)   ?   diff / 126 : diff / 126 + 1;

            res = (currValue > center) ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                                            : (diff + 126 - output_idx * 126) * 2 + shift;
            // code
            local_code[local_idx] = is_center ? shift : res;
            // signal 
            local_output_idx[local_idx] = output_idx;
            // next
            local_idx++;
            bit_offset += output_idx;

            currValue = static_cast<uint16_t>((tmp_buffer.y >> 16) & 0xFFFF);
            is_center = (currValue == center);
            diff = (currValue > center) ? currValue - center : center - currValue;
            remain = diff % 126;
            output_idx = is_center       ?   1 
                       : (remain == 0)   ?   diff / 126 : diff / 126 + 1;

            res = (currValue > center) ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                                            : (diff + 126 - output_idx * 126) * 2 + shift;
            // code
            local_code[local_idx] = is_center ? shift : res;
            // signal 
            local_output_idx[local_idx] = output_idx;
            // next
            local_idx++;
            bit_offset += output_idx;

            currValue = static_cast<uint16_t>(tmp_buffer.z & 0xFFFF);
            is_center = (currValue == center);
            diff = (currValue > center) ? currValue - center : center - currValue;
            remain = diff % 126;
            output_idx = is_center       ?   1 
                       : (remain == 0)   ?   diff / 126 : diff / 126 + 1;

            res = (currValue > center) ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                                            : (diff + 126 - output_idx * 126) * 2 + shift;
            // code
            local_code[local_idx] = is_center ? shift : res;
            // signal 
            local_output_idx[local_idx] = output_idx;
            // next
            local_idx++;
            bit_offset += output_idx;

            currValue = static_cast<uint16_t>((tmp_buffer.z >> 16) & 0xFFFF);
            is_center = (currValue == center);
            diff = (currValue > center) ? currValue - center : center - currValue;
            remain = diff % 126;
            output_idx = is_center       ?   1 
                       : (remain == 0)   ?   diff / 126 : diff / 126 + 1;

            res = (currValue > center) ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                                            : (diff + 126 - output_idx * 126) * 2 + shift;
            // code
            local_code[local_idx] = is_center ? shift : res;
            // signal 
            local_output_idx[local_idx] = output_idx;
            // next
            local_idx++;
            bit_offset += output_idx;

            currValue = static_cast<uint16_t>(tmp_buffer.w & 0xFFFF);
            is_center = (currValue == center);
            diff = (currValue > center) ? currValue - center : center - currValue;
            remain = diff % 126;
            output_idx = is_center       ?   1 
                       : (remain == 0)   ?   diff / 126 : diff / 126 + 1;

            res = (currValue > center) ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                                            : (diff + 126 - output_idx * 126) * 2 + shift;
            // code
            local_code[local_idx] = is_center ? shift : res;
            // signal 
            local_output_idx[local_idx] = output_idx;
            // next
            local_idx++;
            bit_offset += output_idx;

            currValue = static_cast<uint16_t>((tmp_buffer.w >> 16) & 0xFFFF);
            is_center = (currValue == center);
            diff = (currValue > center) ? currValue - center : center - currValue;
            remain = diff % 126;
            output_idx = is_center       ?   1 
                       : (remain == 0)   ?   diff / 126 : diff / 126 + 1;

            res = (currValue > center) ? (diff + 126 - output_idx * 126) * 2 - 1 + shift
                                            : (diff + 126 - output_idx * 126) * 2 + shift;
            // code
            local_code[local_idx] = is_center ? shift : res;
            // signal 
            local_output_idx[local_idx] = output_idx;
            // next
            local_idx++;
            bit_offset += output_idx;
        }
    }

    // get max signal length in a warp, assume bitstream_length is the length of bit_signal(bytes)
    uint16_t max_bitstream_length = bit_offset;  // 初始化为当前线程的长度
    for (int offset = 16; offset > 0; offset /= 2) {
        max_bitstream_length = max(max_bitstream_length, __shfl_down_sync(0x00000000ffffffffL, max_bitstream_length, offset));
    }
    max_bitstream_length = (max_bitstream_length + 7) / 8;  //bytes
    max_bitstream_length = __shfl_sync(0x00000000ffffffffL, max_bitstream_length, 0);

    // save max length to signal_length
    if (lane == 0) {
        signal_length[bid] = max_bitstream_length;
        centers[bid] = center; 
    }


    // Pack the variable-length unary signal bits back into bytes using a register
    // accumulator and write straight to the per-thread slot. The original code
    // indexed a dynamically-addressed local array, forcing a spill to scratch
    // memory plus one serial read-modify-write per element.
    int write_pos = idx * cmp_chunk * max_bytes_signal_per_ele_16b;
    uint8_t* bit_out = bit_signal + write_pos;

    uint32_t acc = 0;
    int acc_bits = 0;
    int out_byte = 0;
    #pragma unroll
    for (int i = 0; i < cmp_chunk; ++i) {
        const int oi = local_output_idx[i];
        if (oi > 0) {
            acc = (acc << oi) | (1u << (oi - 1));
            acc_bits += oi;
            while (acc_bits >= 8) {
                bit_out[out_byte++] = (uint8_t)((acc >> (acc_bits - 8)) & 0xFF);
                acc_bits -= 8;
                acc &= (1u << acc_bits) - 1;
            }
        }
    }
    // Final partial byte: pad trailing bits with 1s (matching the original layout),
    // then zero-fill any remaining bytes up to the block's max length.
    if (acc_bits > 0) {
        bit_out[out_byte] = (uint8_t)(((acc << (8 - acc_bits)) & 0xFF) | (0xFF >> acc_bits));
        for (int i = out_byte + 1; i < max_bitstream_length; ++i) bit_out[i] = 0;
    } else if (out_byte < max_bitstream_length) {
        bit_out[out_byte] = 0xFF;
        for (int i = out_byte + 1; i < max_bitstream_length; ++i) bit_out[i] = 0;
    }

    // d_code
    int4* code_int2 = reinterpret_cast<int4*>(code + base_thread_start_idx);
    int4* local_code_int2 = reinterpret_cast<int4*>(local_code);

    #pragma unroll
    for (int i = 0; i < cmp_chunk / 16; ++i) {
        code_int2[i] = local_code_int2[i];
    }
}

__global__ void concat(
    const uint8_t* d_bit_signals,  // 原始 bitstream 数据
    const int* d_signal_length,   // 每个 warp 的 bitstream 长度
    const int* d_output_lengths,  // 前缀和数组，存放目标偏移
    uint8_t* d_concatenated_output,  // 目标拼接后数据
    int gsize,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 计算索引
    int lane = idx & 0x1f;
    int warp = idx >> 5;

    if(idx * cmp_chunk < num_elements)
    {
        int length = d_signal_length[warp];
        int bit_signal_start = idx * cmp_chunk * max_bytes_signal_per_ele_16b;
        int concat_start = d_output_lengths[warp] * cmp_tblock_size + lane * length;

        for(int i = 0; i < length; i++)
        {
            d_concatenated_output[concat_start + i] = d_bit_signals[bit_signal_start + i];
        }     
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input.u2> <output.adm>" << std::endl;
        return 1;
    }

    const char* input_file = argv[1];
    const char* output_file = argv[2];
    int shift = 1;
    const steady_clock::time_point total_start = steady_clock::now();

    std::cout << "Input file: " << input_file << std::endl;

    size_t data_size = 0;
    const steady_clock::time_point file_read_start = steady_clock::now();
    uint16_t* h_data = load_file_u16(input_file, data_size);
    const double file_read_ms = elapsed_ms(file_read_start, steady_clock::now());
    int num_elements = data_size / sizeof(uint16_t);
    if (!h_data) {
        std::cerr << "Error: Failed to load data from file: " << input_file << std::endl;
        return 1;
    }
    std::cout << "elements num: " << num_elements << std::endl;

    int bsize = cmp_tblock_size;
    int gsize = (num_elements + bsize * cmp_chunk - 1) / (bsize * cmp_chunk);

    uint16_t* d_data;
    void* d_cmpdata;
    int* d_signal_length;
    int* d_output_lengths;
    uint16_t* d_centers;
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
    hipMalloc(&d_centers, gsize * sizeof(uint16_t));
    hipMalloc(&d_bit_signals, bsize * gsize * cmp_chunk * max_bytes_signal_per_ele_16b * sizeof(uint8_t));
    hipMemset(d_bit_signals, 0, bsize * gsize * cmp_chunk * max_bytes_signal_per_ele_16b * sizeof(uint8_t));
    hipMalloc(&d_concatenated_signals, bsize * gsize * cmp_chunk * max_bytes_signal_per_ele_16b * sizeof(uint8_t));
    hipMemset(d_concatenated_signals, 255, bsize * gsize * cmp_chunk * max_bytes_signal_per_ele_16b * sizeof(uint8_t));

    const steady_clock::time_point h2d_start = steady_clock::now();
    hipMemcpy(d_data, h_data, data_size, hipMemcpyHostToDevice);
    const double h2d_ms = elapsed_ms(h2d_start, steady_clock::now());

    thrust::device_ptr<int> dev_output_lengths(d_output_lengths);
    thrust::device_ptr<int> dev_signal_length(d_signal_length);

    // Warm up the same path that will be timed.
    for (int i = 0; i < 5; ++i) {
        map_values_kernel_thrust<<<gsize, bsize>>>(
            (uint16_t*)d_data, d_codes, d_bit_signals,
            (uint16_t*)d_centers, d_signal_length, num_elements, shift);
        thrust::exclusive_scan(dev_signal_length, dev_signal_length + gsize,
                               dev_output_lengths);
        concat<<<gsize, bsize>>>(
            d_bit_signals, d_signal_length, d_output_lengths,
            d_concatenated_signals, gsize, num_elements);
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
    printf("=======>Use Thrust Prefix-sum<=======\n");
    for (int i = 0; i < timing_iters; ++i) {
        hipEventRecord(start, stream);
        map_values_kernel_thrust<<<gsize, bsize, 0, stream>>>(
            (uint16_t*)d_data, d_codes, d_bit_signals,
            (uint16_t*)d_centers, d_signal_length, num_elements, shift);
        hipEventRecord(stop, stream);
        hipEventSynchronize(stop);
        float ms = 0;
        hipEventElapsedTime(&ms, start, stop);
        adm_milliseconds += ms;

        hipEventRecord(start, stream);
        thrust::exclusive_scan(dev_signal_length, dev_signal_length + gsize,
                               dev_output_lengths);
        hipEventRecord(stop, stream);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&ms, start, stop);
        prefixsum_milliseconds += ms;

        hipEventRecord(start, stream);
        concat<<<gsize, bsize, 0, stream>>>(
            d_bit_signals, d_signal_length, d_output_lengths,
            d_concatenated_signals, gsize, num_elements);
        hipEventRecord(stop, stream);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&ms, start, stop);
        concat_milliseconds += ms;
    }
    adm_milliseconds /= timing_iters;
    prefixsum_milliseconds /= timing_iters;
    concat_milliseconds /= timing_iters;

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
    size_t centers_size = gsize * sizeof(uint16_t);
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
    delete[] static_cast<uint16_t*>(h_data);
    delete[] h_signal_length;
    delete[] h_output_lengths;

    hipEventDestroy(start);
    hipEventDestroy(stop);
    hipStreamDestroy(stream);

    printf("Memory has been successfully freed.\n");
    return 0;
}
