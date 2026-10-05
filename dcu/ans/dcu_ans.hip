#include "dcu_ans.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace mans::dcu::ans {
namespace {

constexpr std::uint32_t kSymbols = 256;
constexpr std::uint32_t kBlockBytes = 4096;
constexpr std::uint32_t kLanes = 32;
constexpr std::uint32_t kProbBits = 10;
constexpr std::uint32_t kProbTotal = 1u << kProbBits;
constexpr std::uint32_t kAlignmentWords = 8;
constexpr std::uint32_t kStateStart = 1u << 15;
constexpr std::uint32_t kStateMin = 1u << 15;
constexpr std::uint32_t kStateMask = (1u << 16) - 1;
constexpr std::uint32_t kStateCheckMul = 1u << 21;
constexpr std::uint32_t kMagicVersion = 0xd00d0001u;
constexpr std::uint32_t kHeaderBytes = 32;
constexpr std::uint32_t kWarpStateBytes = 128;
constexpr std::uint32_t kDirectoryEntryBytes = 8;

void check_hip(hipError_t status, const char* what) {
    if (status != hipSuccess) {
        throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(status));
    }
}

class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { if (ptr_) (void)hipFree(ptr_); }
    void allocate(std::size_t bytes, const char* what) {
        if (ptr_) { (void)hipFree(ptr_); ptr_ = nullptr; }
        if (bytes) check_hip(hipMalloc(&ptr_, bytes), what);
    }
    void* get() const { return ptr_; }
private:
    void* ptr_ = nullptr;
};

__device__ inline std::uint32_t lane32() {
    return static_cast<std::uint32_t>(threadIdx.x) & 31u;
}

__device__ inline std::uint32_t warp_ballot32(bool predicate) {
    // Each CTA contains exactly one logical 32-lane codec warp. On wave64
    // hardware the inactive upper half contributes no bits to this mask.
    return static_cast<std::uint32_t>(__ballot(predicate));
}

__device__ inline std::uint32_t popcount32(std::uint32_t value) {
    return __popc(value);
}

struct Header {
    std::uint32_t magic_version;
    std::uint32_t num_blocks;
    std::uint32_t uncompressed_bytes;
    std::uint32_t compressed_words;
    std::uint32_t options;
    std::uint32_t checksum;
    std::uint32_t unused0;
    std::uint32_t unused1;
};
static_assert(sizeof(Header) == kHeaderBytes, "ANS header size changed");

struct DirectoryEntry { std::uint32_t lengths; std::uint32_t offset_words; };
static_assert(sizeof(DirectoryEntry) == kDirectoryEntryBytes, "ANS directory entry size changed");

__global__ void histogram_kernel(const std::uint8_t* input, std::uint32_t bytes,
                                 std::uint32_t* histogram) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < bytes) atomicAdd(histogram + input[i], 1u);
}

__global__ void normalize_kernel(const std::uint32_t* histogram,
                                 std::uint32_t bytes,
                                 std::uint16_t* probabilities,
                                 std::uint32_t* cdf,
                                 std::uint32_t* magic,
                                 std::uint32_t* shifts) {
    __shared__ std::uint32_t q[kSymbols];
    __shared__ std::uint32_t order[kSymbols];
    __shared__ std::uint32_t prefix[kSymbols];
    const std::uint32_t tid = threadIdx.x;
    if (tid < kSymbols) {
        std::uint32_t value = static_cast<std::uint32_t>(
            static_cast<float>(kProbTotal) * (static_cast<float>(histogram[tid]) / static_cast<float>(bytes)));
        if (histogram[tid] && value == 0) value = 1;
        q[tid] = value;
        order[tid] = tid;
    }
    __syncthreads();

    if (tid == 0) {
        // Match CPU's descending sort of (probability << 16) | symbol.
        for (std::uint32_t i = 1; i < kSymbols; ++i) {
            const std::uint32_t symbol = order[i];
            std::uint32_t j = i;
            while (j > 0 && ((q[order[j - 1]] << 16) | order[j - 1]) <
                                ((q[symbol] << 16) | symbol)) {
                order[j] = order[j - 1];
                --j;
            }
            order[j] = symbol;
        }
        std::int32_t total = 0;
        for (std::uint32_t i = 0; i < kSymbols; ++i) {
            prefix[i] = q[order[i]];
            total += static_cast<std::int32_t>(prefix[i]);
        }
        std::int32_t diff = static_cast<std::int32_t>(kProbTotal) - total;
        if (diff > 0) {
            const std::int32_t apply = min(diff, static_cast<std::int32_t>(kSymbols));
            while (diff > 0) {
                for (std::uint32_t i = 0; i < kSymbols; ++i) {
                    if (order[i] < static_cast<std::uint32_t>(apply)) ++prefix[i];
                }
                diff -= apply;
            }
        } else {
            diff = -diff;
            while (diff > 0) {
                std::int32_t gt_one = 0;
                for (std::uint32_t i = 0; i < kSymbols; ++i) gt_one += prefix[i] > 1;
                const std::int32_t apply = min(diff, gt_one);
                const std::int32_t start = gt_one - apply;
                for (std::int32_t i = start; i < gt_one; ++i) --prefix[i];
                diff -= apply;
            }
        }
        for (std::uint32_t i = 0; i < kSymbols; ++i) q[order[i]] = prefix[i];
        std::uint32_t running = 0;
        for (std::uint32_t i = 0; i < kSymbols; ++i) {
            const std::uint32_t p = q[i];
            prefix[i] = running;
            probabilities[i] = static_cast<std::uint16_t>(p);
            cdf[i] = running;
            running += p;
            if (p == 0) {
                magic[i] = 0;
                shifts[i] = 0;
            } else {
                const std::uint32_t shift = 32u - static_cast<std::uint32_t>(__clz(p - 1));
                const std::uint64_t magic64 = ((std::uint64_t{1} << 32) *
                    ((std::uint64_t{1} << shift) - p)) / p + 1;
                magic[i] = static_cast<std::uint32_t>(magic64);
                shifts[i] = shift;
            }
        }
    }
}

__device__ inline std::uint32_t encode_symbol(
    std::uint32_t& state, std::uint8_t symbol, std::uint32_t offset,
    std::uint16_t* output, const std::uint16_t* probabilities,
    const std::uint32_t* cdf, const std::uint32_t* magic,
    const std::uint32_t* shifts) {
    const std::uint32_t lane = lane32();
    const std::uint32_t pdf = probabilities[symbol];
    const bool write = state >= pdf * kStateCheckMul;
    const std::uint32_t ballot = warp_ballot32(write);
    const std::uint32_t lower = lane == 0 ? 0u : ((1u << lane) - 1u);
    const std::uint32_t rank = popcount32(ballot & lower);
    if (write) {
        output[offset + rank] = static_cast<std::uint16_t>(state & kStateMask);
        state >>= 16;
    }
    const std::uint32_t quotient = (__umulhi(state, magic[symbol]) + state) >> shifts[symbol];
    const std::uint32_t remainder = state - quotient * pdf;
    state = quotient * kProbTotal + remainder + cdf[symbol];
    return popcount32(ballot);
}

__global__ void encode_blocks_kernel(const std::uint8_t* input, std::uint32_t bytes,
                                     const std::uint16_t* probabilities,
                                     const std::uint32_t* cdf,
                                     const std::uint32_t* magic,
                                     const std::uint32_t* shifts,
                                     std::uint32_t stride_words,
                                     std::uint8_t* scratch,
                                     std::uint32_t* word_counts) {
    const std::uint32_t lane = lane32();
    const std::uint32_t block = blockIdx.x;
    const std::uint32_t count_blocks = bytes / kBlockBytes + (bytes % kBlockBytes != 0);
    __shared__ std::uint16_t shared_prob[kSymbols];
    __shared__ std::uint32_t shared_cdf[kSymbols];
    __shared__ std::uint32_t shared_magic[kSymbols];
    __shared__ std::uint32_t shared_shifts[kSymbols];
    for (std::uint32_t symbol = lane; symbol < kSymbols; symbol += kLanes) {
        shared_prob[symbol] = probabilities[symbol];
        shared_cdf[symbol] = cdf[symbol];
        shared_magic[symbol] = magic[symbol];
        shared_shifts[symbol] = shifts[symbol];
    }
    __syncthreads();
    if (block >= count_blocks) return;
    const std::uint32_t start = block * kBlockBytes;
    const std::uint32_t size = min(kBlockBytes, bytes - start);

    std::uint32_t state = kStateStart;
    std::uint32_t offset = 0;
    const std::uint32_t limit = (size / (kLanes * 8)) * (kLanes * 8);
    for (std::uint32_t index = lane; index < limit; index += kLanes * 8) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            offset += encode_symbol(
                state, input[start + index + j * kLanes], offset,
                reinterpret_cast<std::uint16_t*>(scratch + block * stride_words * 2 + kWarpStateBytes),
                shared_prob, shared_cdf, shared_magic, shared_shifts);
        }
    }
    const std::uint32_t aligned = (size / kLanes) * kLanes;
    for (std::uint32_t index = limit + lane; index < aligned; index += kLanes) {
        offset += encode_symbol(
            state, input[start + index], offset,
            reinterpret_cast<std::uint16_t*>(scratch + block * stride_words * 2 + kWarpStateBytes),
            shared_prob, shared_cdf, shared_magic, shared_shifts);
    }
    if (aligned < size) {
        const bool valid = lane < size - aligned;
        const bool write = valid && state >= shared_prob[input[start + aligned + (valid ? lane : 0)]] * kStateCheckMul;
        const std::uint32_t ballot = warp_ballot32(write);
        const std::uint32_t lower = lane == 0 ? 0u : ((1u << lane) - 1u);
        const std::uint32_t rank = popcount32(ballot & lower);
        if (write) {
            reinterpret_cast<std::uint16_t*>(scratch + block * stride_words * 2 + kWarpStateBytes)[offset + rank] =
                static_cast<std::uint16_t>(state & kStateMask);
            state >>= 16;
        }
        if (valid) {
            const std::uint8_t symbol = input[start + aligned + lane];
            const std::uint32_t pdf = shared_prob[symbol];
            const std::uint32_t quotient = (__umulhi(state, shared_magic[symbol]) + state) >> shared_shifts[symbol];
            const std::uint32_t remainder = state - quotient * pdf;
            state = quotient * kProbTotal + remainder + shared_cdf[symbol];
        }
        offset += popcount32(ballot);
    }
    reinterpret_cast<std::uint32_t*>(scratch + block * stride_words * 2)[lane] = state;
    if (lane == 0) word_counts[block] = offset;
}

__global__ void coalesce_kernel(const std::uint8_t* scratch,
                                const std::uint32_t* word_counts,
                                const std::uint16_t* probabilities,
                                std::uint32_t blocks, std::uint32_t input_bytes,
                                std::uint32_t stride_words,
                                std::uint32_t* prefix, std::uint8_t* output,
                                std::uint32_t* output_size) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        std::uint32_t sum = 0;
        for (std::uint32_t i = 0; i < blocks; ++i) {
            prefix[i] = sum;
            sum += (word_counts[i] + kAlignmentWords - 1) & ~(kAlignmentWords - 1);
        }
        const std::uint32_t overhead = kHeaderBytes + kSymbols * 2 +
            blocks * kWarpStateBytes + ((blocks + 1) & ~1u) * kDirectoryEntryBytes;
        Header header{kMagicVersion, blocks, input_bytes, sum, kProbBits, 0, 0, 0};
        *reinterpret_cast<Header*>(output) = header;
        *output_size = overhead + sum * 2;
        auto* probs_out = reinterpret_cast<std::uint16_t*>(output + kHeaderBytes);
        for (std::uint32_t i = 0; i < kSymbols; ++i) probs_out[i] = probabilities[i];
        auto* directory = reinterpret_cast<DirectoryEntry*>(output + kHeaderBytes + kSymbols * 2 + blocks * kWarpStateBytes);
        for (std::uint32_t i = 0; i < blocks; ++i) {
            directory[i].lengths = ((i + 1 == blocks ? header.uncompressed_bytes - i * kBlockBytes : kBlockBytes) << 16) |
                                   (word_counts[i] & 0xffffu);
            directory[i].offset_words = prefix[i];
            const auto* source = scratch + i * stride_words * 2;
            auto* states = reinterpret_cast<std::uint32_t*>(output + kHeaderBytes + kSymbols * 2 + i * kWarpStateBytes);
            const auto* source_states = reinterpret_cast<const std::uint32_t*>(source);
            for (std::uint32_t lane = 0; lane < kLanes; ++lane) states[lane] = source_states[lane];
            auto* data = output + overhead + prefix[i] * 2;
            const std::uint32_t copy_bytes = ((word_counts[i] + kAlignmentWords - 1) & ~(kAlignmentWords - 1)) * 2;
            for (std::uint32_t b = 0; b < copy_bytes; ++b) data[b] = source[kWarpStateBytes + b];
        }
    }
}

} // namespace

std::size_t get_max_compressed_size(std::size_t input_bytes) {
    if (input_bytes == 0) return kHeaderBytes;
    if (input_bytes > std::numeric_limits<std::uint32_t>::max()) return 0;
    const std::size_t blocks = input_bytes / kBlockBytes + (input_bytes % kBlockBytes != 0);
    constexpr std::size_t max_block_words =
        (kBlockBytes + kBlockBytes / 4 + 15) / 16 * 8;
    const std::size_t max_size = std::numeric_limits<std::size_t>::max();
    if (blocks > (max_size - kHeaderBytes - kSymbols * 2) /
                     (kWarpStateBytes + 2 * kDirectoryEntryBytes) ||
        blocks > max_size / max_block_words) {
        return 0;
    }
    const std::size_t directory_entries = (blocks + 1) & ~std::size_t{1};
    const std::size_t overhead = kHeaderBytes + kSymbols * 2 +
        blocks * kWarpStateBytes + directory_entries * kDirectoryEntryBytes;
    const std::size_t words = blocks * max_block_words;
    if (words > (max_size - overhead) / 2) return 0;
    return overhead + words * 2;
}

void compress_device(const std::uint8_t* input, std::size_t input_bytes,
                     std::uint8_t* output, std::size_t output_capacity,
                     std::size_t& output_bytes) {
    output_bytes = 0;
    if (!input || !output || input_bytes == 0 || input_bytes > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("DCU ANS: invalid input or size");
    }
    const auto max_size = get_max_compressed_size(input_bytes);
    if (max_size == 0 || output_capacity < max_size) {
        throw std::runtime_error("DCU ANS: output buffer is too small");
    }
    if (max_size > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("DCU ANS: compressed stream exceeds codec1 size limit");
    }
    const std::uint32_t bytes = static_cast<std::uint32_t>(input_bytes);
    const std::uint32_t blocks = bytes / kBlockBytes + (bytes % kBlockBytes != 0);
    if (blocks > std::numeric_limits<std::uint32_t>::max() / kWarpStateBytes) {
        throw std::runtime_error("DCU ANS: too many blocks");
    }
    const std::uint32_t stride_words = kLanes * 2 + (kBlockBytes + kBlockBytes / 4 + 15) / 16 * 8;
    DeviceBuffer histogram, probabilities, cdf, magic, shifts, scratch, word_counts, prefix, size_device;
    histogram.allocate(kSymbols * sizeof(std::uint32_t), "DCU ANS histogram allocation");
    probabilities.allocate(kSymbols * sizeof(std::uint16_t), "DCU ANS probability allocation");
    cdf.allocate(kSymbols * sizeof(std::uint32_t), "DCU ANS CDF allocation");
    magic.allocate(kSymbols * sizeof(std::uint32_t), "DCU ANS magic allocation");
    shifts.allocate(kSymbols * sizeof(std::uint32_t), "DCU ANS shift allocation");
    scratch.allocate(static_cast<std::size_t>(blocks) * stride_words * 2, "DCU ANS scratch allocation");
    word_counts.allocate(static_cast<std::size_t>(blocks) * sizeof(std::uint32_t), "DCU ANS lengths allocation");
    prefix.allocate(static_cast<std::size_t>(blocks + 1) * sizeof(std::uint32_t), "DCU ANS prefix allocation");
    size_device.allocate(sizeof(std::uint32_t), "DCU ANS size allocation");
    check_hip(hipMemset(histogram.get(), 0, kSymbols * sizeof(std::uint32_t)), "DCU ANS histogram memset");
    histogram_kernel<<<(bytes + 255) / 256, 256>>>(input, bytes, static_cast<std::uint32_t*>(histogram.get()));
    normalize_kernel<<<1, kSymbols>>>(static_cast<const std::uint32_t*>(histogram.get()), bytes,
        static_cast<std::uint16_t*>(probabilities.get()), static_cast<std::uint32_t*>(cdf.get()),
        static_cast<std::uint32_t*>(magic.get()), static_cast<std::uint32_t*>(shifts.get()));
    encode_blocks_kernel<<<blocks, kLanes>>>(
        input, bytes, static_cast<const std::uint16_t*>(probabilities.get()),
        static_cast<const std::uint32_t*>(cdf.get()), static_cast<const std::uint32_t*>(magic.get()),
        static_cast<const std::uint32_t*>(shifts.get()), stride_words,
        static_cast<std::uint8_t*>(scratch.get()), static_cast<std::uint32_t*>(word_counts.get()));
    coalesce_kernel<<<1, 1>>>(static_cast<const std::uint8_t*>(scratch.get()),
        static_cast<const std::uint32_t*>(word_counts.get()),
        static_cast<const std::uint16_t*>(probabilities.get()), blocks, bytes, stride_words,
        static_cast<std::uint32_t*>(prefix.get()), output, static_cast<std::uint32_t*>(size_device.get()));
    check_hip(hipGetLastError(), "DCU ANS encode launch");
    check_hip(hipDeviceSynchronize(), "DCU ANS encode synchronize");
    std::uint32_t actual = 0;
    check_hip(hipMemcpy(&actual, size_device.get(), sizeof(actual), hipMemcpyDeviceToHost), "DCU ANS size D2H");
    if (actual > output_capacity) throw std::runtime_error("DCU ANS: encoded stream exceeds output capacity");
    output_bytes = actual;
}

__global__ void decode_kernel(const std::uint8_t* input, std::uint32_t input_bytes,
                              std::uint8_t* output, std::uint32_t output_bytes,
                              std::uint32_t* error) {
    const std::uint32_t block = blockIdx.x;
    const std::uint32_t lane = threadIdx.x;
    const auto* header = reinterpret_cast<const Header*>(input);
    const std::uint32_t blocks = header->num_blocks;
    if (block >= blocks) return;

    __shared__ std::uint32_t inverse[1u << kProbBits];
    __shared__ std::uint32_t malformed;
    if (lane == 0) malformed = 0;
    __syncthreads();

    const auto* probabilities = reinterpret_cast<const std::uint16_t*>(input + kHeaderBytes);
    for (std::uint32_t symbol = lane; symbol < kSymbols; symbol += kLanes) {
        const std::uint32_t pdf = probabilities[symbol];
        std::uint32_t cdf = 0;
        for (std::uint32_t i = 0; i < symbol; ++i) cdf += probabilities[i];
        for (std::uint32_t j = 0; j < pdf; ++j) {
            inverse[cdf + j] = symbol | (pdf << 8) | (j << 20);
        }
    }
    __syncthreads();
    if (lane == 0) {
        std::uint32_t total = 0;
        for (std::uint32_t i = 0; i < kSymbols; ++i) total += probabilities[i];
        if (header->magic_version != kMagicVersion || header->options != kProbBits ||
            header->uncompressed_bytes != output_bytes || total != kProbTotal ||
            blocks != (output_bytes + kBlockBytes - 1) / kBlockBytes) malformed = 1;
        const std::uint32_t overhead = kHeaderBytes + kSymbols * 2 +
            blocks * kWarpStateBytes + ((blocks + 1) & ~1u) * kDirectoryEntryBytes;
        if (input_bytes < overhead || overhead + header->compressed_words * 2 > input_bytes) malformed = 1;
        const auto* directory = reinterpret_cast<const DirectoryEntry*>(input + kHeaderBytes + kSymbols * 2 + blocks * kWarpStateBytes);
        std::uint32_t expected_offset = 0;
        for (std::uint32_t i = 0; i < blocks; ++i) {
            const std::uint32_t expected_size = (i + 1 == blocks) ? output_bytes - i * kBlockBytes : kBlockBytes;
            if ((directory[i].lengths >> 16) != expected_size ||
                (directory[i].lengths & 0xffffu) > 0xffffu ||
                directory[i].offset_words != expected_offset) malformed = 1;
            expected_offset += ((directory[i].lengths & 0xffffu) + kAlignmentWords - 1) & ~(kAlignmentWords - 1);
        }
        if (expected_offset != header->compressed_words) malformed = 1;
    }
    __syncthreads();
    if (malformed) {
        if (lane == 0) atomicExch(error, 1u);
        return;
    }

    const auto* directory = reinterpret_cast<const DirectoryEntry*>(input + kHeaderBytes + kSymbols * 2 + blocks * kWarpStateBytes);
    const auto* states = reinterpret_cast<const std::uint32_t*>(input + kHeaderBytes + kSymbols * 2 + block * kWarpStateBytes);
    const auto entry = directory[block];
    const std::uint32_t count = entry.lengths >> 16;
    std::uint32_t compressed_words = entry.lengths & 0xffffu;
    const auto* data = reinterpret_cast<const std::uint16_t*>(input +
        kHeaderBytes + kSymbols * 2 + blocks * kWarpStateBytes +
        ((blocks + 1) & ~1u) * kDirectoryEntryBytes) + entry.offset_words;
    std::uint32_t state = states[lane];
    std::uint32_t remainder = count % kLanes;
    std::uint32_t output_offset = count - remainder;
    if (remainder != 0) {
        const bool valid = lane < remainder;
        std::uint32_t symbol = 0, pdf = 0, cdf = 0;
        const std::uint32_t packed = inverse[state & (kProbTotal - 1)];
        symbol = packed & 0xffu;
        pdf = (packed >> 8) & 0xfffu;
        cdf = packed >> 20;
        if (valid) state = pdf * (state >> kProbBits) + cdf;
        const bool read = valid && state < kStateMin;
        const std::uint32_t ballot = warp_ballot32(read);
        const std::uint32_t ge = 0xffffffffu << lane;
        const std::uint32_t rank = popcount32(ballot & ge);
        const std::uint32_t reads = popcount32(ballot);
        if (read) state = (state << 16) + data[compressed_words - rank];
        if (valid) output[block * kBlockBytes + output_offset + lane] = static_cast<std::uint8_t>(symbol);
        compressed_words -= reads;
    }
    while (output_offset > 0) {
        output_offset -= kLanes;
        const std::uint32_t packed = inverse[state & (kProbTotal - 1)];
        const std::uint32_t symbol = packed & 0xffu;
        const std::uint32_t pdf = (packed >> 8) & 0xfffu;
        const std::uint32_t cdf = packed >> 20;
        state = pdf * (state >> kProbBits) + cdf;
        const bool read = state < kStateMin;
        const std::uint32_t ballot = warp_ballot32(read);
        const std::uint32_t ge = 0xffffffffu << lane;
        const std::uint32_t rank = popcount32(ballot & ge);
        const std::uint32_t reads = popcount32(ballot);
        if (read) state = (state << 16) + data[compressed_words - rank];
        output[block * kBlockBytes + output_offset + lane] = static_cast<std::uint8_t>(symbol);
        compressed_words -= reads;
    }
}

void decompress_device(const std::uint8_t* input, std::size_t input_bytes,
                       std::uint8_t* output, std::size_t output_capacity,
                       std::size_t& output_bytes) {
    output_bytes = 0;
    if (!input || !output || input_bytes < kHeaderBytes ||
        input_bytes > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("DCU ANS: invalid compressed input");
    }
    Header header{};
    check_hip(hipMemcpy(&header, input, sizeof(header), hipMemcpyDeviceToHost), "DCU ANS header D2H");
    if (header.magic_version != kMagicVersion || header.options != kProbBits ||
        header.uncompressed_bytes == 0 || header.num_blocks !=
            (header.uncompressed_bytes + kBlockBytes - 1) / kBlockBytes) {
        throw std::runtime_error("DCU ANS: invalid codec1 header");
    }
    const std::size_t overhead = kHeaderBytes + kSymbols * 2 +
        static_cast<std::size_t>(header.num_blocks) * kWarpStateBytes +
        ((header.num_blocks + 1u) & ~1u) * kDirectoryEntryBytes;
    const std::size_t required = overhead + static_cast<std::size_t>(header.compressed_words) * 2;
    if (required > input_bytes || required < overhead) throw std::runtime_error("DCU ANS: truncated or oversized stream");
    if (header.uncompressed_bytes > output_capacity) throw std::runtime_error("DCU ANS: output buffer is too small");
    DeviceBuffer error;
    error.allocate(sizeof(std::uint32_t), "DCU ANS decode error allocation");
    check_hip(hipMemset(error.get(), 0, sizeof(std::uint32_t)), "DCU ANS decode error initialize");
    decode_kernel<<<header.num_blocks, kLanes>>>(input, static_cast<std::uint32_t>(input_bytes), output,
        header.uncompressed_bytes, static_cast<std::uint32_t*>(error.get()));
    check_hip(hipGetLastError(), "DCU ANS decode launch");
    check_hip(hipDeviceSynchronize(), "DCU ANS decode synchronize");
    std::uint32_t host_error = 0;
    check_hip(hipMemcpy(&host_error, error.get(), sizeof(host_error), hipMemcpyDeviceToHost), "DCU ANS decode validation result");
    if (host_error) throw std::runtime_error(std::string("DCU ANS: invalid codec1 payload (") + std::to_string(host_error) + ")");
    output_bytes = header.uncompressed_bytes;
}

} // namespace mans::dcu::ans
