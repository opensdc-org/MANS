// uint16 version
// execute
// nvcc -O3 ./mappingV15_fused_uint16.cu -o ./mappingV15_uint16
// ./mappingV15_uint16 ../../../data/exafel/exafel_59200x388.u2 ../../../data/exafel/test.bin

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "mapping_uint16.h"
#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>


static const int cmp_tblock_size = 32; // 32 threads
static const int cmp_chunk = 16;
static const int max_bytes_signal_per_ele_16b = 2;
static const int decmp_chunk = 32;  // one decomp thread process decmp_chunk cmp thread
static const int warp_size = 32;
static const int k = 4;
static const int kn = 16;

__global__ void decompress(uint16_t* decmp_data, uint16_t* centers, uint8_t* codes, int* d_output_lengths, uint8_t* d_concatenated_signals, uint8_t* d_signals, int data_size, int last_length, int gsize, int shift)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = bid * blockDim.x + tid;

    int lane = idx & 0x1f;
    int warp = idx >> 5;

    if (idx * decmp_chunk > data_size) return;
    int end = (d_output_lengths[gsize - 1] + last_length) * cmp_tblock_size;

    warp = lane < 16 ? warp * 2 : warp * 2 + 1;
    int length = (warp == (gsize - 1)) ? last_length : d_output_lengths[warp + 1] - d_output_lengths[warp];

    // restore signal
    int src_start_idx = d_output_lengths[warp] * cmp_tblock_size;  
    src_start_idx = lane < 16 ? src_start_idx + lane * length * 2 : src_start_idx + (lane - 16) * length * 2;
    int dst_start_idx = idx * decmp_chunk;

    uint8_t bit_buffer = 0;
    int signal_idx = -1;
    int offset_byte = 0;
    bool bit = 0;

    uint8_t local_signal[decmp_chunk] = {0};

    #pragma unroll
    for (; offset_byte < length && signal_idx < cmp_chunk; offset_byte++)
    {
        bit_buffer = d_concatenated_signals[src_start_idx + offset_byte];
        #pragma unroll
        for (int i = 7; i >= 0 && signal_idx < cmp_chunk; i--)
        {
            bit = (bit_buffer >> i) & 1;
            if (bit) {
                signal_idx++;
            } else {
                local_signal[signal_idx]++;
            }
        }
    }

    offset_byte = 0;
    signal_idx = 15;
    bit_buffer = 0;
    bit = 0;
    #pragma unroll
    for (; offset_byte < length && signal_idx < decmp_chunk; offset_byte++)
    {
        if(src_start_idx + offset_byte + length > end) return;
        bit_buffer = d_concatenated_signals[src_start_idx + offset_byte + length];
        #pragma unroll
        for (int i = 7; i >= 0 && signal_idx < decmp_chunk; i--)
        {
            bit = (bit_buffer >> i) & 1;
            if (bit) {
                signal_idx++;
            } else {
                local_signal[signal_idx]++;
            }
        }
    }

    uint8_t local_codes[decmp_chunk];
    int4* local_codes_int2 = reinterpret_cast<int4*>(local_codes);
    int4* codes_int2 = reinterpret_cast<int4*>(codes + dst_start_idx);

    #pragma unroll
    for (int i = 0; i < decmp_chunk / 16; ++i) {
        local_codes_int2[i] = codes_int2[i];
    } 

    uint16_t local_result[decmp_chunk];

    int  center = lane < 16 ? centers[bid * 2] : centers[bid * 2 + 1];

    uint8_t code = 0;
    uint8_t signal = 0;
    //restore data
    for(int i = 0; i < decmp_chunk; i++)
    {
        code = local_codes[i];
        signal = local_signal[i];

        int diff = (code % 2 == 1) ? ((code - 1) / 2) : ((code) / 2);
        diff += signal * 126;
        local_result[i] = (code % 2 == 1) ? center - diff : center + diff;
    }

    int4* local_result_int4 = reinterpret_cast<int4*>(local_result);
    int4* decmp_int4 = reinterpret_cast<int4*>(decmp_data + dst_start_idx);

    #pragma unroll
    for (int i = 0; i < decmp_chunk / 8; ++i) {
        decmp_int4[i] = local_result_int4[i];
    }  
}


__global__ void map_values_kernel_thrust(const uint16_t* data,
                                         uint8_t* code,
                                         uint8_t* bit_signal,
                                         uint16_t* centers,
                                         int* signal_length,
                                         std::uint32_t* block_flags,
                                         mans::MansParams params,
                                         int data_size,
                                         int shift) 
{
    (void)block_flags;
    (void)params;
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
    // uint8_t local_signal[cmp_chunk] = {0};
    uint8_t local_bit_signal[cmp_chunk * max_bytes_signal_per_ele_16b] = {0};

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
 
     // 每个线程遍历部分数据，计算局部和与计数
     int end = min(base_thread_end_idx, data_size);
     for (int i = base_thread_start_idx; i < end; i++) {
        local_sum += __ldg(&data[i]);  // 使用 __ldg() 让全局内存加载更高效
        local_count++;
    }
 
     // 利用 Warp Shuffle 汇总所有线程的和与计数
    #pragma unroll
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        local_count += __shfl_down_sync(0xffffffff, local_count, offset);
    }

 
     // Warp 内线程 0 计算平均值，并写入 center 数组
    int center = 0;
    center = local_count > 0 ? local_sum / local_count : 0;
    center = __shfl_sync(0xffffffff, center, 0);
 

    bool is_center;
    uint16_t remain = 0;
    uint8_t res = 0;

    #pragma unroll
    for(int j = 0; j < block_num; j++)
    {
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
    for (int offset = 16; offset > 0; offset /= 2) {
        max_bitstream_length = max(max_bitstream_length, __shfl_down_sync(0xFFFFFFFF, max_bitstream_length, offset));
    }
    max_bitstream_length = (max_bitstream_length + 7) / 8;  //bytes
    max_bitstream_length = __shfl_sync(0xffffffff, max_bitstream_length, 0);

    // save max length to signal_length
    if (lane == 0) {
        signal_length[bid] = max_bitstream_length;
        centers[bid] = center; 
    }


    int total_bits = max_bitstream_length * 8;
    int current_byte = bit_offset / 8;
    uint8_t mask = (0xFF >> (bit_offset % 8));
    (bit_offset < total_bits) ? 
        local_bit_signal[current_byte] = (bit_offset % 8 == 0) ? 0xFF : (local_bit_signal[current_byte] | mask) 
        : 0;

    // write to global memory
    // d_bit_signal
    int write_pos = idx * cmp_chunk * max_bytes_signal_per_ele_16b;
    uint8_t* bit_ptr = bit_signal + write_pos;
    int2* vectorized_ptr = reinterpret_cast<int2*>(bit_ptr);
    int2* local_bit_signal_vec = reinterpret_cast<int2*>(local_bit_signal);

    int valid_vec = (max_bitstream_length + 7) / 8;
    #pragma unroll
    for (int i = 0; i < valid_vec; i++) {
        vectorized_ptr[i] = local_bit_signal_vec[i];
    }

    // d_code
    int4* code_int2 = reinterpret_cast<int4*>(code + base_thread_start_idx);
    int4* local_code_int2 = reinterpret_cast<int4*>(local_code);

    #pragma unroll
    for (int i = 0; i < cmp_chunk / 16; ++i) {
        code_int2[i] = local_code_int2[i];
    }
}

static __global__ void concat(
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

__global__ void map_values_kernel_decoupled(
    const uint16_t* data, 
    uint8_t* code, 
    uint8_t* bit_signal, 
    uint16_t* centers, 
    int* signal_length,
    std::uint32_t* block_flags,
    volatile int* const __restrict__ locOffset, 
    volatile int* const __restrict__ cmpOffset, 
    volatile int* const __restrict__ prefix_state, 
    mans::MansParams params,
    int data_size, 
    int shift) 
{
    (void)block_flags;
    (void)params;
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
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        local_count += __shfl_down_sync(0xffffffff, local_count, offset);
    }

 
     // Warp 内线程 0 计算平均值，并写入 center 数组
    int center = 0;
    center = local_count > 0 ? local_sum / local_count : 0;
    center = __shfl_sync(0xffffffff, center, 0);

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
        max_bitstream_length = max(max_bitstream_length, __shfl_down_sync(0xFFFFFFFF, max_bitstream_length, offset));
    }
    max_bitstream_length = (max_bitstream_length + 7) / 8;  //bytes
    max_bitstream_length = __shfl_sync(0xffffffff, max_bitstream_length, 0);

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
            prefix_state[0] = 2;
            __threadfence();
            prefix_state[1] = 1;
            __threadfence();
        }
        else
        {
            prefix_state[warp+1] = 1;
            __threadfence();    
        }
    }
    __syncthreads();

    // Global-level prefix-sum (exclusive).
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
                    status = prefix_state[lookback];
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
        if(!lane) prefix_state[warp] = 2;
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


namespace mans {
namespace nv {
namespace adm {

namespace {

constexpr int kDecoupledPrefixMaxGsize = 1024;

std::size_t get_flags_size(int gsize) {
    return static_cast<std::size_t>(gsize + 7) / 8;
}

bool use_decoupled_prefix_sum(int gsize) {
    return gsize <= kDecoupledPrefixMaxGsize;
}

void check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

} // namespace

void compress_u16_device(const std::uint16_t* d_input,
                         std::size_t num_elements,
                         const mans::MansParams& params,
                         std::uint8_t* d_output,
                         std::size_t& output_size,
                         cudaStream_t stream) {
    (void)params;
    if (!d_input && num_elements != 0) {
        throw std::runtime_error("compress_u16_device: d_input is null");
    }
    if (!d_output && num_elements != 0) {
        throw std::runtime_error("compress_u16_device: d_output is null");
    }
    if (num_elements == 0) {
        output_size = 0;
        return;
    }

    const int shift = 1;
    const int bsize = cmp_tblock_size;
    const int gsize = static_cast<int>((num_elements + bsize * cmp_chunk - 1) / (bsize * cmp_chunk));

    const bool use_decoupled = use_decoupled_prefix_sum(gsize);

    int* d_signal_length = nullptr;
    int* d_output_lengths = nullptr;
    std::uint16_t* d_centers = nullptr;
    std::uint8_t* d_codes = nullptr;
    std::uint8_t* d_bit_signals = nullptr;
    std::uint8_t* d_concatenated_signals = nullptr;
    std::uint32_t* d_block_flags = nullptr;
    int* d_locOffset = nullptr;
    int* d_prefix_state = nullptr;

    cudaMalloc(reinterpret_cast<void**>(&d_signal_length), static_cast<std::size_t>(gsize) * sizeof(int));
    cudaMemset(d_signal_length, 0, static_cast<std::size_t>(gsize) * sizeof(int));
    cudaMalloc(reinterpret_cast<void**>(&d_codes), static_cast<std::size_t>(bsize) * gsize * cmp_chunk * sizeof(std::uint8_t));
    cudaMalloc(reinterpret_cast<void**>(&d_output_lengths), static_cast<std::size_t>(gsize + 1) * sizeof(int));
    cudaMemset(d_output_lengths, 0, static_cast<std::size_t>(gsize + 1) * sizeof(int));
    cudaMalloc(reinterpret_cast<void**>(&d_centers), static_cast<std::size_t>(gsize) * sizeof(std::uint16_t));
    const std::size_t flags_words =
        (get_flags_size(gsize) + sizeof(std::uint32_t) - 1) / sizeof(std::uint32_t);
    cudaMalloc(reinterpret_cast<void**>(&d_block_flags), flags_words * sizeof(std::uint32_t));
    cudaMemset(d_block_flags, 0xFF, flags_words * sizeof(std::uint32_t));
    cudaMalloc(reinterpret_cast<void**>(&d_concatenated_signals), static_cast<std::size_t>(bsize) * gsize * cmp_chunk * max_bytes_signal_per_ele_16b * sizeof(std::uint8_t));
    cudaMemset(d_concatenated_signals, 255, static_cast<std::size_t>(bsize) * gsize * cmp_chunk * max_bytes_signal_per_ele_16b * sizeof(std::uint8_t));

    if (use_decoupled) {
        cudaMalloc(reinterpret_cast<void**>(&d_locOffset), static_cast<std::size_t>(gsize + 1) * sizeof(int));
        cudaMemset(d_locOffset, 0, static_cast<std::size_t>(gsize + 1) * sizeof(int));
        cudaMalloc(reinterpret_cast<void**>(&d_prefix_state), static_cast<std::size_t>(gsize + 1) * sizeof(int));
        cudaMemset(d_prefix_state, 0, static_cast<std::size_t>(gsize + 1) * sizeof(int));

        map_values_kernel_decoupled<<<gsize, bsize, sizeof(unsigned int) * 2, stream>>>(
            d_input,
            d_codes,
            d_concatenated_signals,
            d_centers,
            d_signal_length,
            d_block_flags,
            d_locOffset,
            d_output_lengths,
            d_prefix_state,
            params,
            static_cast<int>(num_elements),
            shift);
        check_cuda(cudaGetLastError(), "map_values_kernel_decoupled launch");
        check_cuda(cudaStreamSynchronize(stream), "map_values_kernel_decoupled sync");
    } else {
        cudaMalloc(reinterpret_cast<void**>(&d_bit_signals), static_cast<std::size_t>(bsize) * gsize * cmp_chunk * max_bytes_signal_per_ele_16b * sizeof(std::uint8_t));
        cudaMemset(d_bit_signals, 0, static_cast<std::size_t>(bsize) * gsize * cmp_chunk * max_bytes_signal_per_ele_16b * sizeof(std::uint8_t));

        thrust::device_ptr<int> dev_output_lengths(d_output_lengths);
        thrust::device_ptr<int> dev_signal_length(d_signal_length);

        map_values_kernel_thrust<<<gsize, bsize, 0, stream>>>(

                d_input,
                d_codes,
                d_bit_signals,
                d_centers,
                d_signal_length,
                d_block_flags,
                params,
                static_cast<int>(num_elements),
                shift);
        check_cuda(cudaGetLastError(), "map_values_kernel_thrust launch");
        check_cuda(cudaStreamSynchronize(stream), "map_values_kernel_thrust sync");
        thrust::exclusive_scan(thrust::cuda::par.on(stream), dev_signal_length, dev_signal_length + gsize, dev_output_lengths);
        check_cuda(cudaGetLastError(), "exclusive_scan launch");
        check_cuda(cudaStreamSynchronize(stream), "exclusive_scan sync");
        concat<<<gsize, bsize, 0, stream>>>(
                d_bit_signals,
                d_signal_length,
                d_output_lengths,
                d_concatenated_signals,
                gsize,
                static_cast<int>(num_elements));
        check_cuda(cudaGetLastError(), "concat launch");
        check_cuda(cudaStreamSynchronize(stream), "concat sync");
    }

    std::vector<int> signal_lengths(gsize);
    std::vector<int> output_lengths(gsize, 0);
    check_cuda(cudaMemcpyAsync(signal_lengths.data(), d_signal_length, static_cast<std::size_t>(gsize) * sizeof(int), cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync signal_lengths D2H");
    check_cuda(cudaMemcpyAsync(output_lengths.data(), d_output_lengths, static_cast<std::size_t>(gsize) * sizeof(int), cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync output_lengths D2H");
    check_cuda(cudaStreamSynchronize(stream), "signal/output lengths D2H sync");

    const std::size_t bit_signals_size =
        static_cast<std::size_t>(signal_lengths.back() + output_lengths.back()) * cmp_tblock_size;
    const std::size_t output_lengths_size = static_cast<std::size_t>(gsize) * sizeof(int);
    const std::size_t centers_size = static_cast<std::size_t>(gsize) * sizeof(std::uint16_t);
    const std::size_t flags_size = get_flags_size(gsize);
    const std::size_t codes_size = num_elements * sizeof(std::uint8_t);
    output_size = output_lengths_size + centers_size + flags_size + codes_size + bit_signals_size;

    check_cuda(cudaMemcpyAsync(d_output, d_output_lengths, output_lengths_size, cudaMemcpyDeviceToDevice, stream), "cudaMemcpyAsync packed offsets D2D");
    check_cuda(cudaMemcpyAsync(d_output + output_lengths_size, d_centers, centers_size, cudaMemcpyDeviceToDevice, stream), "cudaMemcpyAsync packed centers D2D");
    check_cuda(cudaMemcpyAsync(d_output + output_lengths_size + centers_size, d_block_flags, flags_size, cudaMemcpyDeviceToDevice, stream), "cudaMemcpyAsync packed flags D2D");
    check_cuda(cudaMemcpyAsync(d_output + output_lengths_size + centers_size + flags_size, d_codes, codes_size, cudaMemcpyDeviceToDevice, stream), "cudaMemcpyAsync packed codes D2D");
    check_cuda(cudaMemcpyAsync(d_output + output_lengths_size + centers_size + flags_size + codes_size, d_concatenated_signals, bit_signals_size, cudaMemcpyDeviceToDevice, stream), "cudaMemcpyAsync packed bits D2D");
    check_cuda(cudaStreamSynchronize(stream), "pack payload sync");

    cudaFree(d_signal_length);
    cudaFree(d_output_lengths);
    cudaFree(d_centers);
    cudaFree(d_codes);
    cudaFree(d_block_flags);
    if (d_bit_signals) cudaFree(d_bit_signals);
    cudaFree(d_concatenated_signals);
    if (d_locOffset) cudaFree(d_locOffset);
    if (d_prefix_state) cudaFree(d_prefix_state);
}

void decompress_u16_device(const std::uint8_t* d_input,
                           std::size_t input_size,
                           std::uint16_t* d_output,
                           std::size_t num_elements,
                           const mans::MansParams& params,
                           cudaStream_t stream) {
    (void)params;
    if (!d_input && input_size != 0) {
        throw std::runtime_error("decompress_u16_device: d_input is null");
    }
    if (!d_output && num_elements != 0) {
        throw std::runtime_error("decompress_u16_device: d_output is null");
    }
    if (num_elements == 0) {
        return;
    }

    const int shift = 1;
    const int bsize = cmp_tblock_size;
    const int gsize = static_cast<int>((num_elements + bsize * cmp_chunk - 1) / (bsize * cmp_chunk));
    const std::size_t output_lengths_size = static_cast<std::size_t>(gsize) * sizeof(int);
    const std::size_t centers_size = static_cast<std::size_t>(gsize) * sizeof(std::uint16_t);
    const std::size_t flags_size = get_flags_size(gsize);
    const std::size_t codes_size = num_elements * sizeof(std::uint8_t);
    const std::size_t fixed_size = output_lengths_size + centers_size + flags_size + codes_size;
    if (input_size < fixed_size) {
        throw std::runtime_error("decompress_u16_device: truncated ADM payload");
    }

    std::vector<int> output_lengths(gsize);
    check_cuda(cudaMemcpyAsync(output_lengths.data(), d_input, output_lengths_size, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync payload offsets D2H");
    check_cuda(cudaStreamSynchronize(stream), "payload offsets D2H sync");
    const std::size_t bit_signals_size = input_size - fixed_size;
    if (bit_signals_size % cmp_tblock_size != 0) {
        throw std::runtime_error("decompress_u16_device: invalid bit-signal payload size");
    }
    const int last_length = static_cast<int>(bit_signals_size / cmp_tblock_size) - output_lengths.back();
    if (last_length < 0) {
        throw std::runtime_error("decompress_u16_device: invalid last signal length");
    }

    const std::uint8_t* centers_ptr = d_input + output_lengths_size;
    const std::uint8_t* flags_ptr = centers_ptr + centers_size;
    const std::uint8_t* codes_ptr = flags_ptr + flags_size;
    const std::uint8_t* bit_signals_ptr = codes_ptr + codes_size;

    int* d_output_lengths = nullptr;
    std::uint16_t* d_centers = nullptr;
    std::uint8_t* d_codes = nullptr;
    std::uint8_t* d_concatenated_signals = nullptr;
    std::uint16_t* d_decmpdata = nullptr;

    cudaMalloc(reinterpret_cast<void**>(&d_output_lengths), output_lengths_size);
    cudaMalloc(reinterpret_cast<void**>(&d_centers), centers_size);
    cudaMalloc(reinterpret_cast<void**>(&d_codes), codes_size);
    cudaMalloc(reinterpret_cast<void**>(&d_concatenated_signals), bit_signals_size);
    cudaMalloc(reinterpret_cast<void**>(&d_decmpdata), num_elements * sizeof(std::uint16_t));

    check_cuda(cudaMemcpyAsync(d_output_lengths, d_input, output_lengths_size, cudaMemcpyDeviceToDevice, stream), "cudaMemcpyAsync output_lengths D2D");
    check_cuda(cudaMemcpyAsync(d_centers, centers_ptr, centers_size, cudaMemcpyDeviceToDevice, stream), "cudaMemcpyAsync centers D2D");
    check_cuda(cudaMemcpyAsync(d_codes, codes_ptr, codes_size, cudaMemcpyDeviceToDevice, stream), "cudaMemcpyAsync codes D2D");
    check_cuda(cudaMemcpyAsync(d_concatenated_signals, bit_signals_ptr, bit_signals_size, cudaMemcpyDeviceToDevice, stream), "cudaMemcpyAsync bit_signals D2D");

    const int decmp_gsize = static_cast<int>((num_elements + bsize * decmp_chunk - 1) / (bsize * decmp_chunk));
    decompress<<<decmp_gsize, bsize, 0, stream>>>(
        d_decmpdata,
        d_centers,
        d_codes,
        d_output_lengths,
        d_concatenated_signals,
        nullptr,
        static_cast<int>(num_elements),
        last_length,
        gsize,
        shift);
    check_cuda(cudaGetLastError(), "decompress launch");
    check_cuda(cudaStreamSynchronize(stream), "decompress sync");
    check_cuda(cudaMemcpyAsync(d_output, d_decmpdata, num_elements * sizeof(std::uint16_t), cudaMemcpyDeviceToDevice, stream), "cudaMemcpyAsync output D2D");
    check_cuda(cudaStreamSynchronize(stream), "output D2D sync");

    cudaFree(d_output_lengths);
    cudaFree(d_centers);
    cudaFree(d_codes);
    cudaFree(d_concatenated_signals);
    cudaFree(d_decmpdata);
}

std::size_t get_max_u16_payload_bytes(std::size_t num_elements,
                                      const mans::MansParams& params) {
    (void)params;
    const std::size_t gsize = (num_elements + cmp_tblock_size * cmp_chunk - 1) / (cmp_tblock_size * cmp_chunk);
    return gsize * sizeof(int) +
           gsize * sizeof(std::uint16_t) +
           get_flags_size(static_cast<int>(gsize)) +
           num_elements * sizeof(std::uint8_t) +
           num_elements * max_bytes_signal_per_ele_16b * sizeof(std::uint8_t);
}

} // namespace adm
} // namespace nv
} // namespace mans
