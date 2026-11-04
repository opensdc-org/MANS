#include "common.h"

__global__ void map_values_kernel_decoupled(
    const uint16_t* data, 
    uint8_t* code, 
    uint8_t* bit_signal, 
    uint16_t* centers, 
    int* signal_length ,
    volatile int* const __restrict__ locOffset, 
    volatile int* const __restrict__ cmpOffset, 
    volatile int* const __restrict__ flag, 
    int data_size, 
    int shift) 
{
    __shared__ unsigned int excl_sum;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = bid * blockDim.x + tid;
    const int lane = idx & 0x1f;
    const int warp = idx >> 5;
    const int block_num = cmp_chunk >> k;   

    int base_block_start_idx;
    int base_thread_start_idx, base_thread_end_idx;
    uint4 tmp_buffer;
    
    uint8_t local_code[cmp_chunk] = {0};
    // uint8_t local_signal[cmp_chunk] = {0};
    uint8_t local_bit_signal[cmp_chunk * max_bytes_signal_per_ele_16b] = {0};

    int diff = 0;
    int output_idx = 0;
    int local_idx = 0;
    int bit_offset = 0; // current bit position
    uint16_t currValue = 0;

    base_block_start_idx = warp * cmp_chunk * cmp_tblock_size;
    base_thread_start_idx = base_block_start_idx + lane * cmp_chunk;
    base_thread_end_idx = base_thread_start_idx + cmp_chunk;

     // 每线程的局部和与计数
    uint32_t local_sum = 0;
    uint32_t local_count = 0;
 
     // 每个线程遍历部分数据，计算局部和与计数
     int end = min(base_thread_end_idx, data_size);
     for (int i = base_thread_start_idx; i < end; i++) {
        local_sum += __ldg(&data[i]);  // 使用 __ldg() 让全局内存加载更高效
        local_count++;
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

    // if(warp ==1) printf("%d\n", center);
 

    bool is_center;
    uint16_t remain = 0;
    uint8_t res = 0;

    #pragma unroll
    for(int j = 0; j < block_num; j++)
    {
        base_thread_start_idx = base_block_start_idx + lane * cmp_chunk + j * kn;
        base_thread_end_idx = base_thread_start_idx + kn;
        end = min(base_thread_end_idx, data_size);
        if(base_thread_start_idx > data_size) break;

        #pragma unroll
        for(int i = base_thread_start_idx; i < end; i+=8)
        {
            tmp_buffer = reinterpret_cast<const uint4*>(data)[i / 8];

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
            local_bit_signal[bit_offset / 8] |= (1 << (7 - (bit_offset % 8)));
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
            local_bit_signal[bit_offset / 8] |= (1 << (7 - (bit_offset % 8)));
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
            local_bit_signal[bit_offset / 8] |= (1 << (7 - (bit_offset % 8)));
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
            local_bit_signal[bit_offset / 8] |= (1 << (7 - (bit_offset % 8)));
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
            local_bit_signal[bit_offset / 8] |= (1 << (7 - (bit_offset % 8)));
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
            local_bit_signal[bit_offset / 8] |= (1 << (7 - (bit_offset % 8)));
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
            local_bit_signal[bit_offset / 8] |= (1 << (7 - (bit_offset % 8)));
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
            local_bit_signal[bit_offset / 8] |= (1 << (7 - (bit_offset % 8)));
            // next
            local_idx++;
            bit_offset += output_idx;
        }
    }


    // get max signal length in a warp, assume bitstream_length is the length of bit_signal(bytes) 
    uint16_t max_bitstream_length = bit_offset;  // 初始化为当前线程的长度
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        max_bitstream_length = max(max_bitstream_length, __shfl_down_sync(0x00000000ffffffffL, max_bitstream_length, offset));
    }
    max_bitstream_length = (max_bitstream_length + 7) / 8;  //bytes
    max_bitstream_length = __shfl_sync(0x00000000ffffffffL, max_bitstream_length, 0);

    // save max length to signal_length
    // if (lane == 31) {
    //     signal_length[bid] = max_bitstream_length;
    //     centers[bid] = center; 
    // }


    int total_bits = max_bitstream_length * 8;
    int current_byte = bit_offset / 8;
    uint8_t mask = (0xFF >> (bit_offset % 8));
    (bit_offset < total_bits) ? 
        local_bit_signal[current_byte] = (bit_offset % 8 == 0) ? 0xFF : (local_bit_signal[current_byte] | mask) 
        : 0;

    // Write warp(i.e. thread-block)-level prefix-sum to global-memory.
    if(lane==31) 
    {
        signal_length[bid] = max_bitstream_length;
        centers[bid] = center; 
        locOffset[warp+1] = max_bitstream_length;
        __threadfence();
        if(warp==0)
        {
            flag[0] = 2;
            __threadfence();
            flag[1] = 1;
            __threadfence();
        }
        else
        {
            flag[warp+1] = 1;
            __threadfence();    
        }
    }
    __syncthreads();

    // Global-level prefix-sum (exclusive).
    if (warp == 0 && lane == 0) excl_sum = 0;
    __syncthreads();
    if(warp>0)
    {
        if(!lane)
        {
            // Decoupled look-back
            int lookback = warp;
            int loc_excl_sum = 0;
            while(lookback>0)
            {
                int status;
                // Local sum not end.
                do{
                    status = flag[lookback];
                    __threadfence();
                } while (status==0);
                // Lookback end.
                if(status==2)
                {
                    loc_excl_sum += cmpOffset[lookback];
                    __threadfence();
                    break;
                }
                // Continues lookback.
                if(status==1) loc_excl_sum += locOffset[lookback];
                lookback--;
                __threadfence();
            }
            excl_sum = loc_excl_sum;
        }
        __syncthreads();
    }
    
    if(warp>0)
    {
        // Update global flag.
        if(!lane) cmpOffset[warp] = excl_sum;
        __threadfence();
        if(!lane) flag[warp] = 2;
        __threadfence();  
    }
    __syncthreads();

    // Assigning compression bytes by given prefix-sum results.
    // if(!lane) base_idx = excl_sum;
    // __syncthreads();

    int write_pos = excl_sum * cmp_tblock_size + lane * max_bitstream_length;

    #pragma unroll
    for (int i = 0; i < max_bitstream_length; i++) {
        bit_signal[write_pos + i] = local_bit_signal[i];
    }

    // d_code
    int4* code_int2 = reinterpret_cast<int4*>(code + base_block_start_idx + lane * cmp_chunk);
    int4* local_code_int2 = reinterpret_cast<int4*>(local_code);

    #pragma unroll
    for (int i = 0; i < cmp_chunk / 16; ++i) {
        code_int2[i] = local_code_int2[i];
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

    std::cout << "Input file: " << input_file << std::endl;

    size_t data_size = 0;
    uint16_t* h_data = load_file_u16(input_file, data_size);
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

    hipMemcpy(d_data, h_data, data_size, hipMemcpyHostToDevice);

    // Warm up the same path that will be timed.
    for (int i = 0; i < 5; ++i) {
        hipMemset(d_flag, 0, (gsize + 1) * sizeof(int));
        hipMemset(d_locOffset, 0, (gsize + 1) * sizeof(int));
        hipMemset(d_output_lengths, 0, (gsize + 1) * sizeof(int));
        map_values_kernel_decoupled<<<gsize, bsize>>>(
            (uint16_t*)d_data, d_codes, d_concatenated_signals,
            (uint16_t*)d_centers, d_signal_length, d_locOffset,
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
    hipMemset(d_flag, 0, (gsize + 1) * sizeof(int));
    hipMemset(d_locOffset, 0, (gsize + 1) * sizeof(int));
    hipMemset(d_output_lengths, 0, (gsize + 1) * sizeof(int));
    hipEventRecord(start, stream);
    map_values_kernel_decoupled<<<gsize, bsize, 0, stream>>>(
        (uint16_t*)d_data, d_codes, d_concatenated_signals,
        (uint16_t*)d_centers, d_signal_length, d_locOffset,
        d_output_lengths, d_flag, num_elements, shift);
    hipEventRecord(stop, stream);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&adm_milliseconds, start, stop);

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
    hipMemcpy(static_cast<uint8_t*>(h_cmpdata), d_cmpdata, total_size * sizeof(uint8_t), hipMemcpyDeviceToHost);

    save_file(output_file, h_cmpdata, total_size);

    printf("\033[0;34m-------Data Size-------\033[0m\n");
    printf("Original size: %zu Bytes\n", data_size);
    printf("Output size: %zu Bytes\n", total_size);
    printf("CR : %.2f x\n", data_size * 1.0 / total_size);
    printf("Output Length size: %zu Bytes\n", output_lengths_size);
    printf("Centers size: %zu Bytes\n", centers_size);
    printf("Codes size: %zu Bytes\n", codes_size);
    printf("Bit-signal size: %zu Bytes\n", bit_signals_size);

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
