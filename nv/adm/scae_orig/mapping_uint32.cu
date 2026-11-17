// uint32 version
// execute
// nvcc -O3 ./mappingV15_fused_uint32.cu -o ./mappingV15_uint32
// ./mappingV15_uint32 ../../../data/kron/kron-g500-logn19_43562265.u4 ../../../data/kron/test.bin

#include <iostream>
#include <fstream>
#include <stdint.h>
#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>


static const int cmp_tblock_size = 32; // 32 threads
static const int cmp_chunk = 16;
static const int max_bytes_signal_per_ele_32b = 4;
static const int decmp_chunk = 32;  // one decomp thread process decmp_chunk cmp thread
static const int warp_size = 32;
static const int k = 4;
static const int kn = 16;

__global__ void decompress(uint32_t* decmp_data, uint32_t* centers, uint8_t* codes, int* d_output_lengths, uint8_t* d_concatenated_signals, uint8_t* d_signals, int data_size, int last_length, int gsize, int shift)
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

    uint32_t local_result[decmp_chunk];

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
    for (int i = 0; i < decmp_chunk / 4; ++i) {
        decmp_int4[i] = local_result_int4[i];
    }  
}


__global__ void map_values_kernel_thrust(const uint32_t* data, uint8_t* code, uint8_t* bit_signal, uint32_t* centers, int* signal_length ,int data_size, int shift) 
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
    // uint8_t local_signal[cmp_chunk] = {0};
    uint8_t local_bit_signal[cmp_chunk * max_bytes_signal_per_ele_32b] = {0};

    int diff = 0;
    int output_idx = 0;
    int local_idx = 0;
    int bit_offset = 0; // current bit position
    uint32_t currValue = 0;

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
    uint32_t remain = 0;
    uint8_t res = 0;

    #pragma unroll
    for(int j = 0; j < block_num; j++)
    {
        if(base_thread_start_idx > data_size) break;

        #pragma unroll
        for(int i = base_thread_start_idx; i < end; i+=4)
        {
            tmp_buffer = reinterpret_cast<const uint4*>(data)[i / 4];

            currValue = static_cast<uint32_t>(tmp_buffer.x);
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
            
            currValue = static_cast<uint32_t>(tmp_buffer.y);
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

            currValue = static_cast<uint32_t>(tmp_buffer.z);
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

            currValue = static_cast<uint32_t>(tmp_buffer.w);
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
    uint32_t max_bitstream_length = bit_offset;  // 初始化为当前线程的长度
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
    int write_pos = idx * cmp_chunk * max_bytes_signal_per_ele_32b;
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
        int bit_signal_start = idx * cmp_chunk * max_bytes_signal_per_ele_32b;
        int concat_start = d_output_lengths[warp] * cmp_tblock_size + lane * length;

        for(int i = 0; i < length; i++)
        {
            d_concatenated_output[concat_start + i] = d_bit_signals[bit_signal_start + i];
        }     
    }
}

__global__ void map_values_kernel_decoupled(
    const uint32_t* data, 
    uint8_t* code, 
    uint8_t* bit_signal, 
    uint32_t* centers, 
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
    uint8_t local_bit_signal[cmp_chunk * max_bytes_signal_per_ele_32b] = {0};

    int diff = 0;
    int output_idx = 0;
    int local_idx = 0;
    int bit_offset = 0; // current bit position
    uint32_t currValue = 0;

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
    uint32_t remain = 0;
    uint8_t res = 0;

    #pragma unroll
    for(int j = 0; j < block_num; j++)
    {
        base_thread_start_idx = base_block_start_idx + lane * cmp_chunk + j * kn;
        base_thread_end_idx = base_thread_start_idx + kn;
        end = min(base_thread_end_idx, data_size);
        if(base_thread_start_idx > data_size) break;

        #pragma unroll
        for(int i = base_thread_start_idx; i < end; i+=4)
        {
            tmp_buffer = reinterpret_cast<const uint4*>(data)[i / 4];

            currValue = static_cast<uint32_t>(tmp_buffer.x);
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
            
            currValue = static_cast<uint32_t>(tmp_buffer.y);
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

            currValue = static_cast<uint32_t>(tmp_buffer.z);
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

            currValue = static_cast<uint32_t>(tmp_buffer.w);
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
    uint32_t max_bitstream_length = bit_offset;  // 初始化为当前线程的长度
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

// Template helper function to load binary file into memory based on data type
uint32_t* load_file(const char* filename, size_t& size) {
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

// Helper function to save result to binary file
void save_file(const char* filename, const uint8_t* data, size_t size) {
    std::ofstream file(filename, std::ios::binary);
    file.write(reinterpret_cast<const char*>(data), size);
    file.close();
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input file> <output file>" << std::endl;
        return 1;
    }

    const char* input_file = argv[1];
    const char* output_file = argv[2];
    // std::string data_type = argv[3]; // 从命令行获取数据类型

    int shift = 1;

    std::cout << "Input file: " << input_file << std::endl;

    // Load input file into host memory based on data type
    size_t data_size = 0;
    uint32_t* h_data = nullptr;
    int num_elements = 0;

    h_data = load_file(input_file, data_size);
    num_elements = data_size / sizeof(uint32_t);

    if (!h_data) {
        std::cerr << "Error: Failed to load data from file: " << input_file << std::endl;
        return 1;
    }

    std::cout << "elements num: " << num_elements << std::endl;

    // 配置CUDA内核
    int bsize = cmp_tblock_size;
    int gsize = (num_elements + bsize * cmp_chunk - 1) / (bsize * cmp_chunk);

    // define 
    uint32_t* d_data;
    uint32_t* d_decmpdata;
    void* d_cmpdata;

    // compressed struct: header(signal_length & center) & code & bit_signal
    // int* d_offsets
    int* d_signal_length;
    int* d_output_lengths;
    uint32_t* d_centers;
    uint8_t* d_codes;
    uint8_t* d_bit_signals; 
    uint8_t* d_concatenated_signals;
    int* d_locOffset;
    int* d_flag;

    // memory allocation
    cudaMalloc((void**)&d_data, data_size);
    cudaMalloc((void**)&d_signal_length, gsize * sizeof(int));
    cudaMemset(d_signal_length, 0, gsize * sizeof(int));
    cudaMalloc((void**)&d_codes, bsize * gsize * cmp_chunk * sizeof(uint8_t));
    cudaMalloc((void**)&d_locOffset, (gsize + 1) * sizeof(int));
    cudaMalloc((void**)&d_flag, (gsize + 1) * sizeof(int));
    cudaMalloc(&d_output_lengths, (gsize + 1) * sizeof(int));

    cudaMalloc(&d_decmpdata, bsize * gsize * cmp_chunk * sizeof(uint32_t));

    cudaMalloc(&d_centers, gsize * sizeof(uint32_t));  
    cudaMalloc(&d_bit_signals, bsize * gsize * cmp_chunk * max_bytes_signal_per_ele_32b * sizeof(uint8_t));
    cudaMemset(d_bit_signals, 0, bsize * gsize * cmp_chunk * max_bytes_signal_per_ele_32b * sizeof(uint8_t));
    cudaMalloc(&d_concatenated_signals, bsize * gsize * cmp_chunk * max_bytes_signal_per_ele_32b * sizeof(uint8_t));
    cudaMemset(d_concatenated_signals, 255, bsize * gsize * cmp_chunk * max_bytes_signal_per_ele_32b * sizeof(uint8_t));


    // mem copy
    cudaMemcpy(d_data, h_data, data_size, cudaMemcpyHostToDevice);

    //prefix-sum initialize
    thrust::device_ptr<int> dev_output_lengths(d_output_lengths);
    thrust::device_ptr<int> dev_signal_length(d_signal_length);

    // warmup
    for(int i = 0; i < 0; i++)
    { 
        map_values_kernel_decoupled<<<gsize, bsize>>>(
            d_data, d_codes, d_bit_signals, d_centers, d_signal_length, d_locOffset, d_output_lengths, d_flag, num_elements, shift);
            cudaMemset(d_flag, 0, (gsize + 1) * sizeof(int));
        
    }

    // 设置CUDA事件用于测量时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    //CMP

    if(gsize <= 1024)
    {
        printf("\033[0;36m=======>Use Decoupled Prefix-sum<=======\033[1m\n") ;
        cudaEventRecord(start);
        map_values_kernel_decoupled<<<gsize, bsize, sizeof(unsigned int)*2, stream>>>(
            d_data, d_codes, d_concatenated_signals, d_centers, d_signal_length, d_locOffset, d_output_lengths, d_flag, num_elements, shift);
        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float adm_milliseconds = 0;
        cudaEventElapsedTime(&adm_milliseconds, start, stop);

        printf("\033[0;34m-------Cmp Throughput-------\033[0m\n") ;
        float adm_throughput = (data_size / 1024.0 / 1024 / 1024) / (adm_milliseconds / 1000.0f);  // 单位：字节每秒
        printf("ADM Kernel throughput: %.2f GB/s\n", adm_throughput);
    
        printf("\033[0;34m-------Cmp Time-------\033[0m\n") ;
        printf("Total Cmp Time: %.2f ms\n",adm_milliseconds);
    }
    else
    {
        printf("\033[0;36m=======>Use Thrust Prefix-sum<=======\033[1m\n") ;
        //adm
        cudaEventRecord(start);
        map_values_kernel_thrust<<<gsize, bsize>>>(
            d_data, d_codes, d_bit_signals, d_centers, d_signal_length, num_elements, shift);
        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float adm_milliseconds = 0;
        cudaEventElapsedTime(&adm_milliseconds, start, stop);

        // prefix-sum time
        cudaEventRecord(start);

        thrust::exclusive_scan(dev_signal_length, dev_signal_length + gsize, dev_output_lengths);  
        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float prefixsum_milliseconds = 0;
        cudaEventElapsedTime(&prefixsum_milliseconds, start, stop);

        //concat time
        cudaEventRecord(start);
        concat<<<gsize, bsize>>>(
            d_bit_signals, d_signal_length, d_output_lengths, d_concatenated_signals, gsize, num_elements);
        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float concat_milliseconds = 0;
        cudaEventElapsedTime(&concat_milliseconds, start, stop);

        printf("\033[0;34m-------Cmp Throughput-------\033[0m\n") ;
        float adm_throughput = (data_size / 1024.0 / 1024 / 1024) / (adm_milliseconds / 1000.0f);  // 单位：字节每秒
        float prefixsum_throughput = (data_size / 1024.0 / 1024 / 1024) / (prefixsum_milliseconds / 1000.0f); 
        float concat_throughput = (data_size / 1024.0 / 1024 / 1024) / (concat_milliseconds / 1000.0f); 
        float total_cmp_throughput = (data_size / 1024.0 / 1024 / 1024) / ((adm_milliseconds + prefixsum_milliseconds + concat_milliseconds) / 1000.0f); 
        printf("Total Cmp throughput: %.2f GB/s\n", total_cmp_throughput);
        printf("ADM Kernel throughput: %.2f GB/s\n", adm_throughput);
        printf("Prefix-sum Kernel throughput: %.2f GB/s\n", prefixsum_throughput);
        printf("Concat Kernel throughput: %.2f GB/s\n", concat_throughput);
    
        printf("\033[0;34m-------Cmp Time-------\033[0m\n") ;
        printf("Total Cmp Time: %.2f ms\n", (adm_milliseconds + prefixsum_milliseconds + concat_milliseconds));
        printf("ADM cost %.2f ms\n",adm_milliseconds);
        printf("Prefix-sum cost %.2f ms\n",prefixsum_milliseconds);
        printf("Concat cost %.2f ms\n",concat_milliseconds);
    }
    

    // 复制局部长度和sum长度到主机
    int* h_signal_length = new int[gsize];
    int* h_output_lengths = new int[gsize];
    cudaMemcpy(h_signal_length, d_signal_length, gsize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_lengths, d_output_lengths, gsize * sizeof(int), cudaMemcpyDeviceToHost);


    // 计算总的输出大小
    size_t bit_signals_size = (h_signal_length[gsize - 1] + h_output_lengths[gsize - 1]) * cmp_tblock_size * sizeof(uint8_t);

    //merger into d_cmpdata
    size_t output_lengths_size = gsize * sizeof(int);  // d_output_lengths
    size_t centers_size = 0;
    centers_size = gsize * sizeof(uint32_t);
    size_t codes_size = num_elements * sizeof(uint8_t);           // d_codes

    // 计算总大小
    size_t total_size = output_lengths_size + centers_size + codes_size + bit_signals_size;

    // 在设备上分配统一数组
    cudaMalloc(&d_cmpdata, total_size);

    // 计算子数组的指针位置
    int* d_signal_length_ptr = reinterpret_cast<int*>(d_cmpdata);
    void* d_centers_ptr = reinterpret_cast<uint8_t*>(d_cmpdata) + output_lengths_size;
    uint8_t* d_codes_ptr = reinterpret_cast<uint8_t*>(reinterpret_cast<uint8_t*>(d_cmpdata) + output_lengths_size + centers_size);
    uint8_t* d_bit_signals_ptr = reinterpret_cast<uint8_t*>(reinterpret_cast<uint8_t*>(d_cmpdata) + output_lengths_size + centers_size + codes_size);

    // **将已有数据拷贝到 d_cmpdata**
    cudaMemcpy(d_signal_length_ptr, d_output_lengths, output_lengths_size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_centers_ptr, d_centers, centers_size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_codes_ptr, d_codes, codes_size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_bit_signals_ptr, d_concatenated_signals, bit_signals_size, cudaMemcpyDeviceToDevice);

    uint8_t* h_cmpdata = new uint8_t[total_size];

    cudaMemcpy(static_cast<uint8_t*>(h_cmpdata), d_cmpdata, total_size * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // save
    save_file(output_file, h_cmpdata, total_size);

    
    // ----------------Decmp------------------
    // decmp kernel
    int decmp_gsize = (num_elements + bsize * decmp_chunk - 1) / (bsize * decmp_chunk);
    uint8_t* d_signals;
    cudaMalloc(&d_signals, bsize * gsize * cmp_chunk * sizeof(uint8_t));
    cudaEventRecord(start);
    decompress<<<decmp_gsize, bsize>>>(d_decmpdata, d_centers, d_codes, d_output_lengths, d_concatenated_signals, d_signals, num_elements, h_signal_length[gsize - 1], gsize, shift);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float decmp_adm_milliseconds = 0;
    cudaEventElapsedTime(&decmp_adm_milliseconds, start, stop);

    printf("\033[0;34m-------Decmp Throughput-------\033[0m\n") ;
    float decmp_throughput = (data_size / 1024.0 / 1024 / 1024) / (decmp_adm_milliseconds / 1000.0f); 
    printf("Decmp throughput: %.2f GB/s\n", decmp_throughput);

    printf("\033[0;34m-------Decmp Time-------\033[0m\n") ;
    printf("Total Decmp Time: %.2f ms\n",(decmp_adm_milliseconds));

    printf("\033[0;34m-------Data Size-------\033[0m\n");
    printf("Original size: %d Bytes\n",data_size);
    printf("Output size: %d Bytes\n",total_size);
    printf("CR : %.2f x\n",data_size * 1.0 / total_size);
    printf("Output Length size: %d Bytes\n",output_lengths_size);
    printf("Centers size: %d Bytes\n",centers_size);
    printf("Codes size: %d Bytes\n",codes_size);
    printf("Bit-signal size: %d Bytes\n",bit_signals_size);

    //verify lossless
    uint32_t* decompressed_data = nullptr;

    decompressed_data = new uint32_t[num_elements];
    cudaMemcpy(decompressed_data, d_decmpdata, num_elements * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    bool test = true;
    uint32_t* original_data = h_data;         
    for (int i = 0; i < num_elements; i++) {
        if (decompressed_data[i] != original_data[i]) {
            printf("\033[0;31mFail error check!\033[0m\n");
            printf("\033[0;31mError: Data mismatch at index %d, original = %d, decompressed = %d\033[0m\n", 
                    i, original_data[i], decompressed_data[i]);
            test = false;
            break;
        }
    }

    if(test) printf("\033[0;32mPass error check!\033[0m\n");

    // // 设定打印范围
    // int debug_start_idx = 1723;   // 可以修改为任意起始索引
    // int debug_end_idx = 1728;     // 需要检查的数据长度

    // // 申请 host 端数组
    // uint32_t* h_centers = new uint32_t[gsize];
    // uint8_t* h_codes = new uint8_t[num_elements];
    // uint8_t* h_signals = new uint8_t[num_elements];

    // // 将数据从 GPU 复制到 CPU
    // cudaMemcpy(h_centers, d_centers, gsize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_codes, d_codes, num_elements * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_signals, d_signals, num_elements * sizeof(uint8_t), cudaMemcpyDeviceToHost);


    // // 打印 Centers
    // uint32_t* original_data = static_cast<uint32_t*>(h_data);

    // printf("%d\n", h_centers[3]);
    // // printf("%d\n", h_centers[1]);
    // printf("==== datas (部分数据) ====\n");
    // for (int i = debug_start_idx; i < debug_end_idx; i++) {  // 仅打印最多16个中心值
    //     printf("[%d]: %d\n", i, original_data[i]);
    // }
    // printf("\n");

    // // 打印 Codes
    // printf("==== d_codes (部分数据) ====\n");
    // for (int i = debug_start_idx; i < debug_end_idx; i++) {
    //     printf("[%d]: %u\n", i, h_codes[i]);
    // }
    // printf("\n");

    // // 打印 Signals
    // printf("==== d_signals (部分数据) ====\n");
    // for (int i = debug_start_idx; i < debug_end_idx; i++) {
    //     printf("[%d]: %u\n", i, h_signals[i]);
    // }
    // printf("\n");

    // // 释放 CPU 端内存
    // delete[] h_centers;
    // delete[] h_codes;
    // delete[] h_signals;

    // 清理内存
    cudaFree(d_data);
    cudaFree(d_decmpdata);
    cudaFree(d_cmpdata);
    cudaFree(d_signal_length);
    cudaFree(d_centers);
    cudaFree(d_codes);
    cudaFree(d_bit_signals);
    cudaFree(d_concatenated_signals);
    cudaFree(d_output_lengths);

    delete[] static_cast<uint8_t*>(h_cmpdata);
    delete[] static_cast<uint32_t*>(h_data);
    delete[] h_signal_length;
    delete[] h_output_lengths;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);

    printf("Memory has been successfully freed.\n");

    return 0;
}