// algorithm.h
#ifndef ALGORITHM_H
#define ALGORITHM_H

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <chrono>

#include <immintrin.h>
#include <omp.h>

namespace adm {

// ---------------- global parameters ----------------
inline constexpr int cmp_tblock_size = 32;
inline constexpr int cmp_chunk = 16;
inline constexpr int decmp_chunk = 32;
inline constexpr int max_bytes_signal_per_ele_16b = 2;
inline constexpr int max_bytes_signal_per_ele_32b = 3;
inline constexpr int warp_size = 32;

// ------------- header -------------
// record metadata
struct FileHeader {
    std::uint64_t num_elements; // uint16 elements num
    std::uint64_t gsize;        // warp = ceil(num / (cmp_tblock_size * cmp_chunk))
    std::size_t len1;
    std::size_t len2;
    std::size_t len3;
    std::size_t len4;    

};

inline void compress_uint16(
    const std::vector<uint16_t>& input_data,
    std::vector<int>& output_lengths,
    std::vector<uint16_t>& centers,
    std::vector<uint8_t>& codes,
    std::vector<uint8_t>& bit_signals
) {
    int num_elements = input_data.size();
    int gsize = (num_elements + cmp_tblock_size * cmp_chunk - 1) / (cmp_tblock_size * cmp_chunk);
    int total_threads = gsize * cmp_tblock_size;

    std::vector<int> signal_length(gsize, 0);
    std::vector<int> bit_offsets(total_threads, 0);
    centers.resize(gsize);
    codes.resize(num_elements);

    // Center calculation: parallelizing and reducing unnecessary work
    #pragma omp parallel for
    for (int warp = 0; warp < gsize; ++warp) {
        int base_idx = warp * cmp_tblock_size * cmp_chunk;
        int end_idx = std::min(base_idx + cmp_tblock_size * cmp_chunk, num_elements);

        uint64_t sum = 0;
        for (int i = base_idx; i < end_idx; ++i) {
            sum += input_data[i];
        }

        int count = end_idx - base_idx;
        centers[warp] = (count > 0) ? sum / count : 0;
    }

    // Allocate temporary buffer for bit_signals
    std::vector<uint8_t> tmp_bit_signals(total_threads * cmp_chunk * max_bytes_signal_per_ele_16b, 0);

    // Encoding and setting codes, bit_signals (in temporary space)
    #pragma omp parallel for
    for (int thread_idx = 0; thread_idx < total_threads; ++thread_idx) {
        int warp = thread_idx / cmp_tblock_size;
        int lane = thread_idx % cmp_tblock_size;
        int base_idx = warp * cmp_tblock_size * cmp_chunk + lane * cmp_chunk;

        if (base_idx >= num_elements) continue;
        int center = centers[warp];

        uint8_t* bit_out = &tmp_bit_signals[thread_idx * cmp_chunk * max_bytes_signal_per_ele_16b];

        int bit_offset = 0;

        for (int i = 0; i < cmp_chunk && base_idx + i < num_elements; ++i) {
            uint16_t val = input_data[base_idx + i];
            int diff = val > center ? val - center : center - val;
            int output_len = (val == center) ? 1 : (diff + 125) / 126;
            uint8_t res = (val == center) ? 1 : ((diff + 126 - output_len * 126) * 2 + (val > center ? -1 : 0) + 1);

            codes[base_idx + i] = res;

            // Set bitstream (mark the corresponding bit)
            bit_out[bit_offset / 8] |= (1 << (7 - (bit_offset % 8)));
            bit_offset += output_len;
        }

        bit_offsets[thread_idx] = bit_offset;
    }

    // Warp-level reduction: compute signal_length[warp] deterministically
    #pragma omp parallel for
    for (int warp = 0; warp < gsize; ++warp) {
        int base_thread = warp * cmp_tblock_size;
        int end_thread = std::min(base_thread + cmp_tblock_size, total_threads);

        int max_len_bytes = 0;
        for (int t = base_thread; t < end_thread; ++t) {
            int bit_offset = bit_offsets[t];
            int length_bytes = (bit_offset + 7) / 8;
            max_len_bytes = std::max(max_len_bytes, length_bytes);
        }

        signal_length[warp] = max_len_bytes;
    }

    // Fill in the tail bits
    #pragma omp parallel for
    for (int thread_idx = 0; thread_idx < total_threads; ++thread_idx) {
        int warp = thread_idx / cmp_tblock_size;
        int bit_offset = bit_offsets[thread_idx];
        int max_len_bytes = signal_length[warp];

        if (bit_offset < max_len_bytes * 8) {
            uint8_t* bit_out = &tmp_bit_signals[thread_idx * cmp_chunk * max_bytes_signal_per_ele_16b];
            int byte_idx = bit_offset / 8;
            uint8_t mask = (bit_offset % 8 == 0) ? 0xFF : (0xFF >> (bit_offset % 8));
            bit_out[byte_idx] |= mask;
        }
    }

    // Compute prefix sum (serially)
    output_lengths.resize(gsize + 1);
    output_lengths[0] = 0;
    for (int i = 1; i <= gsize; ++i) {
        output_lengths[i] = output_lengths[i - 1] + signal_length[i - 1];
    }

    // Write back bit_signals
    int total_bit_bytes = output_lengths[gsize] * cmp_tblock_size;
    bit_signals.resize(total_bit_bytes, 0);

    #pragma omp parallel for
    for (int thread_idx = 0; thread_idx < total_threads; ++thread_idx) {
        int warp = thread_idx / cmp_tblock_size;
        int lane = thread_idx % cmp_tblock_size;
        int bit_len = signal_length[warp];
        int dst_base = output_lengths[warp] * cmp_tblock_size + lane * bit_len;

        if (dst_base + bit_len > total_bit_bytes) continue;

        const uint8_t* src = &tmp_bit_signals[thread_idx * cmp_chunk * max_bytes_signal_per_ele_16b];
        // 使用向量化指令进行批量拷贝
        #pragma omp simd
        for (int i = 0; i < bit_len; ++i) {
            bit_signals[dst_base + i] = src[i];
        }
    }
}

inline void decompress_uint16(
    const std::vector<int>& output_lengths,             // gsize
    const std::vector<uint16_t>& centers,               // gsize
    const std::vector<uint8_t>& codes,                  // num_elements
    const std::vector<uint8_t>& bit_signals,            // bitstream
    std::vector<uint16_t>& output_data                  // output: num_elements
)
{
    int num_elements = codes.size();
    int gsize = output_lengths.size();
    int total_threads = gsize * cmp_tblock_size;

    // Step 1: Restore signal[]
    std::vector<uint8_t> signals(num_elements, 0);

    #pragma omp parallel for
    for (int tid = 0; tid < total_threads; ++tid) {
        int warp = tid / cmp_tblock_size;
        int lane = tid % cmp_tblock_size;
        int idx = tid;

        if (idx * cmp_chunk >= num_elements) continue;

        int length = output_lengths[warp + 1] - output_lengths[warp];

        int src_start_idx = output_lengths[warp] * cmp_tblock_size + lane * length;
        int dst_start_idx = idx * cmp_chunk;

        uint8_t bit_buffer = 0;
        int signal_idx = -1;
        int offset_byte = 0;
        bool bit = 0;

        uint8_t local_signal[cmp_chunk] = {0};

        for (; offset_byte < length && signal_idx < cmp_chunk; offset_byte++) {
            bit_buffer = bit_signals[src_start_idx + offset_byte];
            for (int i = 7; i >= 0 && signal_idx < cmp_chunk; i--) {
                bit = (bit_buffer >> i) & 1;
                if (bit) {
                    signal_idx++;
                } else {
                    local_signal[signal_idx]++;
                }
            }
        }

        // Use a local copy to avoid accessing shared memory repeatedly
        for (int i = 0; i < cmp_chunk && dst_start_idx + i < num_elements; ++i) {
            signals[dst_start_idx + i] = local_signal[i];
        }
    }

    // Step 2: Decode values
    output_data.resize(num_elements);

    #pragma omp parallel for
    for (int tid = 0; tid < total_threads; ++tid) {
        int block_id = tid;
        int lane = block_id % warp_size;
        int bid = block_id / cmp_tblock_size;
        int base_idx = block_id * decmp_chunk;

        if (base_idx >= num_elements) continue;

        uint16_t center = (lane < 16) ? centers[bid * 2] : centers[bid * 2 + 1];

        // Use local variables to minimize memory access and reduce branch conditions
        for (int i = 0; i < decmp_chunk && base_idx + i < num_elements; ++i) {
            uint8_t code = codes[base_idx + i];
            uint8_t signal = signals[base_idx + i];

            int diff = (code % 2 == 1) ? ((code - 1) / 2) : (code / 2);
            diff += signal * 126;

            uint16_t val = (code % 2 == 1) ? center - diff : center + diff;
            output_data[base_idx + i] = val;
        }
    }
}

inline void compress_uint32(
    const std::vector<uint32_t>& input_data,
    std::vector<int>& output_lengths,
    std::vector<uint32_t>& centers,
    std::vector<uint8_t>& codes,
    std::vector<uint8_t>& bit_signals
) {
    int num_elements = input_data.size();
    int gsize = (num_elements + cmp_tblock_size * cmp_chunk - 1) / (cmp_tblock_size * cmp_chunk);
    int total_threads = gsize * cmp_tblock_size;

    std::vector<int> signal_length(gsize, 0);
    std::vector<int> bit_offsets(total_threads, 0);
    centers.resize(gsize);
    codes.resize(num_elements);

    // static const uint8_t bitmask[8] = {0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01};
    // static const uint8_t tail_mask[8] = {0xFF, 0x7F, 0x3F, 0x1F, 0x0F, 0x07, 0x03, 0x01};


    // Center calculation: parallelizing and reducing unnecessary work
    #pragma omp parallel for
    for (int warp = 0; warp < gsize; ++warp) {
        int base_idx = warp * cmp_tblock_size * cmp_chunk;
        int end_idx = std::min(base_idx + cmp_tblock_size * cmp_chunk, num_elements);

        uint64_t sum = 0;
        for (int i = base_idx; i < end_idx; ++i) {
            sum += input_data[i];
        }

        int count = end_idx - base_idx;
        centers[warp] = (count > 0) ? sum / count : 0;
    }

    // Allocate temporary buffer for bit_signals
    std::vector<uint8_t> tmp_bit_signals(total_threads * cmp_chunk * max_bytes_signal_per_ele_32b, 0);

    // Encoding and setting codes, bit_signals (in temporary space)
    #pragma omp parallel for
    for (int thread_idx = 0; thread_idx < total_threads; ++thread_idx) {
        int warp = thread_idx / cmp_tblock_size;
        int lane = thread_idx % cmp_tblock_size;
        int base_idx = warp * cmp_tblock_size * cmp_chunk + lane * cmp_chunk;

        if (base_idx >= num_elements) continue;
        int center = centers[warp];

        uint8_t* bit_out = &tmp_bit_signals[thread_idx * cmp_chunk * max_bytes_signal_per_ele_32b];

        int bit_offset = 0;

        for (int i = 0; i < cmp_chunk && base_idx + i < num_elements; ++i) {
            uint32_t val = input_data[base_idx + i];
            int diff = val > center ? val - center : center - val;
            int output_len = (val == center) ? 1 : (diff + 125) / 126;
            uint8_t res = (val == center) ? 1 : ((diff + 126 - output_len * 126) * 2 + (val > center ? -1 : 0) + 1);

            codes[base_idx + i] = res;

            // Set bitstream (mark the corresponding bit)
            // bit_out[bit_offset / 8] |= bitmask[bit_offset % 8];
            bit_out[bit_offset / 8] |= (1 << (7 - (bit_offset % 8)));
            bit_offset += output_len;
        }

        bit_offsets[thread_idx] = bit_offset;
    }

    // Warp-level reduction: compute signal_length[warp] deterministically
    #pragma omp parallel for
    for (int warp = 0; warp < gsize; ++warp) {
        int base_thread = warp * cmp_tblock_size;
        int end_thread = std::min(base_thread + cmp_tblock_size, total_threads);

        int max_len_bytes = 0;
        for (int t = base_thread; t < end_thread; ++t) {
            int bit_offset = bit_offsets[t];
            int length_bytes = (bit_offset + 7) / 8;
            max_len_bytes = std::max(max_len_bytes, length_bytes);
        }

        signal_length[warp] = max_len_bytes;
    }

    // Fill in the tail bits
    #pragma omp parallel for
    for (int thread_idx = 0; thread_idx < total_threads; ++thread_idx) {
        int warp = thread_idx / cmp_tblock_size;
        int bit_offset = bit_offsets[thread_idx];
        int max_len_bytes = signal_length[warp];

        if (bit_offset < max_len_bytes * 8) {
            uint8_t* bit_out = &tmp_bit_signals[thread_idx * cmp_chunk * max_bytes_signal_per_ele_32b];
            int byte_idx = bit_offset / 8;
            uint8_t mask = (bit_offset % 8 == 0) ? 0xFF : (0xFF >> (bit_offset % 8));
            bit_out[byte_idx] |= mask;
            // bit_out[byte_idx] |= tail_mask[bit_offset % 8];
        }
    }

    // Compute prefix sum (serially)
    output_lengths.resize(gsize + 1);
    output_lengths[0] = 0;
    for (int i = 1; i <= gsize; ++i) {
        output_lengths[i] = output_lengths[i - 1] + signal_length[i - 1];
    }

    // Write back bit_signals
    int total_bit_bytes = output_lengths[gsize] * cmp_tblock_size;
    bit_signals.resize(total_bit_bytes, 0);

    #pragma omp parallel for
    for (int thread_idx = 0; thread_idx < total_threads; ++thread_idx) {
        int warp = thread_idx / cmp_tblock_size;
        int lane = thread_idx % cmp_tblock_size;
        int bit_len = signal_length[warp];
        int dst_base = output_lengths[warp] * cmp_tblock_size + lane * bit_len;

        if (dst_base + bit_len > total_bit_bytes) continue;

        const uint8_t* src = &tmp_bit_signals[thread_idx * cmp_chunk * max_bytes_signal_per_ele_32b];
        // Use  simd for bulk copying
        #pragma omp simd
        for (int i = 0; i < bit_len; ++i) {
            bit_signals[dst_base + i] = src[i];
        }
    }
}


inline void decompress_uint32(
    const std::vector<int>& output_lengths,             // gsize
    const std::vector<uint32_t>& centers,               // gsize
    const std::vector<uint8_t>& codes,                  // num_elements
    const std::vector<uint8_t>& bit_signals,            // bitstream
    std::vector<uint32_t>& output_data                  // output: num_elements
)
{
    int num_elements = codes.size();
    int gsize = output_lengths.size();
    int total_threads = gsize * cmp_tblock_size;

    // Step 1: Restore signal[]
    std::vector<uint8_t> signals(num_elements, 0);

    #pragma omp parallel for
    for (int tid = 0; tid < total_threads; ++tid) {
        int warp = tid / cmp_tblock_size;
        int lane = tid % cmp_tblock_size;
        int idx = tid;

        if (idx * cmp_chunk >= num_elements) continue;

        int length = output_lengths[warp + 1] - output_lengths[warp];

        int src_start_idx = output_lengths[warp] * cmp_tblock_size + lane * length;
        int dst_start_idx = idx * cmp_chunk;

        uint8_t bit_buffer = 0;
        int signal_idx = -1;
        int offset_byte = 0;
        bool bit = 0;

        uint8_t local_signal[cmp_chunk] = {0};

        for (; offset_byte < length && signal_idx < cmp_chunk; offset_byte++) {
            bit_buffer = bit_signals[src_start_idx + offset_byte];
            for (int i = 7; i >= 0 && signal_idx < cmp_chunk; i--) {
                bit = (bit_buffer >> i) & 1;
                if (bit) {
                    signal_idx++;
                } else {
                    local_signal[signal_idx]++;
                }
            }
        }

        // Use a local copy to avoid accessing shared memory repeatedly
        for (int i = 0; i < cmp_chunk && dst_start_idx + i < num_elements; ++i) {
            signals[dst_start_idx + i] = local_signal[i];
        }
    }

    // Step 2: Decode values
    output_data.resize(num_elements);

    #pragma omp parallel for
    for (int tid = 0; tid < total_threads; ++tid) {
        int block_id = tid;
        int lane = block_id % warp_size;
        int bid = block_id / cmp_tblock_size;
        int base_idx = block_id * decmp_chunk;

        if (base_idx >= num_elements) continue;

        uint32_t center = (lane < 16) ? centers[bid * 2] : centers[bid * 2 + 1];

        // Use local variables to minimize memory access and reduce branch conditions
        for (int i = 0; i < decmp_chunk && base_idx + i < num_elements; ++i) {
            uint8_t code = codes[base_idx + i];
            uint8_t signal = signals[base_idx + i];

            int diff = (code % 2 == 1) ? ((code - 1) / 2) : (code / 2);
            diff += signal * 126;

            uint32_t val = (code % 2 == 1) ? center - diff : center + diff;
            output_data[base_idx + i] = val;
        }
    }
}

} // namespace adm

#endif // ALGORITHM_H
