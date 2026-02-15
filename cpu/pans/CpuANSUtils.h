#pragma once
#include <fstream>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <cassert>
#include <vector>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <numeric> 
#include <immintrin.h>
#include <thread>
#include <parallel/algorithm>
#include <avx512fintrin.h>
#include <omp.h>
#include <cstdlib>
#include <stdexcept>
#include <chrono>
#include <atomic>
#include <omp.h>
#include "../../mans_timing.h"
namespace cpu_ans {

using ANSStateT = uint32_t;
using ANSEncodedT = uint16_t;
using ANSDecodedT = uint8_t;

struct ANSWarpState { ANSStateT warpState[32]; };
struct uint2 { uint32_t x, y; };
struct uint4 { uint32_t x, y, z, w; };

inline uint32_t divDown(uint32_t a, uint32_t b) { return a / b; }
inline uint32_t getAlignmentRoundUp(size_t alignment, const void* ptr) {
    return (alignment - (reinterpret_cast<uintptr_t>(ptr) % alignment)) % alignment;
}

template <typename U, typename V>
inline auto divDown(U a, V b) -> decltype(a + b) {
  return (a / b);
}

template <typename U, typename V>
inline auto divUp(U a, V b) -> decltype(a + b) {
  return (a + b - 1) / b;
}

template <typename U, typename V>
inline auto roundDown(U a, V b) -> decltype(a + b) {
  return divDown(a, b) * b;
}

template <typename U, typename V>
inline auto roundUp(U a, V b) -> decltype(a + b) {
  return divUp(a, b) * b;
}

constexpr int kWarpSize = 32;
constexpr uint32_t kNumSymbols = 1 << (sizeof(ANSDecodedT) * 8);
constexpr uint32_t kMaxBEPSThreads = 512;
constexpr uint32_t kDefaultBlockSize = 4096;
constexpr int kANSDefaultProbBits = 10;
constexpr int kANSRequiredAlignment = 4;
constexpr int kANSStateBits = (sizeof(ANSStateT) * 8) - 1;
constexpr int kANSEncodedBits = sizeof(ANSEncodedT) * 8;
constexpr ANSStateT kANSStartState = ANSStateT(1) << (kANSStateBits - kANSEncodedBits);
constexpr ANSStateT kANSMinState = ANSStateT(1) << (kANSStateBits - kANSEncodedBits);
constexpr ANSStateT kANSEncodedMask = (ANSStateT(1) << kANSEncodedBits) - ANSStateT(1);
constexpr uint32_t kANSMagic = 0xd00d;
constexpr uint32_t kANSVersion = 0x0001;
constexpr uint32_t kBlockAlignment = 16;

struct ANSCoalescedHeader {
    static inline uint32_t getCompressedOverhead(uint32_t numBlocks) {
        int kAlignment = kBlockAlignment / sizeof(uint2);
        if (kAlignment == 0) kAlignment = 1;

        return sizeof(ANSCoalescedHeader) +
               sizeof(uint16_t) * kNumSymbols +
               sizeof(ANSWarpState) * numBlocks +
               sizeof(uint2) * roundUp(numBlocks, kAlignment);
    }
    
    inline uint32_t getTotalCompressedSize() {
        return getCompressedOverhead() +
               getTotalCompressedWords() * sizeof(ANSEncodedT);
    }

    inline uint32_t getCompressedOverhead() {
        return getCompressedOverhead(getNumBlocks());
    }

    inline float getCompressionRatio() {
        return static_cast<float>(getTotalCompressedSize()) /
               static_cast<float>(getTotalUncompressedWords() * sizeof(ANSDecodedT));
    }

    inline uint32_t getNumBlocks() { return numBlocks; }
    inline void setNumBlocks(uint32_t nb) { numBlocks = nb; }

    inline void setMagicAndVersion() {
        magicAndVersion = (kANSMagic << 16) | kANSVersion;
    }

    inline void checkMagicAndVersion() {
        assert((magicAndVersion >> 16) == kANSMagic);
        assert((magicAndVersion & 0xffffU) == kANSVersion);
    }

    inline uint32_t getTotalUncompressedWords() { return totalUncompressedWords; }
    inline void setTotalUncompressedWords(uint32_t words) { totalUncompressedWords = words; }

    inline uint32_t getTotalCompressedWords() { return totalCompressedWords; }
    inline void setTotalCompressedWords(uint32_t words) { totalCompressedWords = words; }

    inline uint32_t getProbBits() { return options & 0xf; }
    inline void setProbBits(uint32_t bits) {
        assert(bits <= 0xf);
        options = (options & 0xfffffff0U) | bits;
    }

    inline bool getUseChecksum() { return options & 0x10; }
    inline void setUseChecksum(bool uc) {
        options = (options & 0xffffffef) | (static_cast<uint32_t>(uc) << 4);
    }

    inline uint32_t getChecksum() { return checksum; }
    inline void setChecksum(uint32_t c) { checksum = c; }

    inline uint16_t* getSymbolProbs() { return reinterpret_cast<uint16_t*>(this + 1); }

    inline ANSWarpState* getWarpStates() { return reinterpret_cast<ANSWarpState*>(getSymbolProbs() + kNumSymbols); }
 
    inline uint2* getBlockWords(uint32_t numBlocks) {
        return reinterpret_cast<uint2*>(getWarpStates() + numBlocks);
    }

    inline ANSEncodedT* getBlockDataStart(uint32_t numBlocks) {
        constexpr int kAlignment = kBlockAlignment / sizeof(uint2) == 0
        ? 1 : kBlockAlignment / sizeof(uint2);
        return reinterpret_cast<ANSEncodedT*>(getBlockWords(numBlocks) + roundUp(numBlocks, kAlignment));
    }

    uint32_t magicAndVersion;
    uint32_t numBlocks;
    uint32_t totalUncompressedWords;
    uint32_t totalCompressedWords;
    uint32_t options;
    uint32_t checksum;
    uint32_t unused0;
    uint32_t unused1;
};

inline uint32_t
getRawCompBlockMaxSize(uint32_t uncompressedBlockBytes) {
  return roundUp(
      uncompressedBlockBytes + (uncompressedBlockBytes / 4), kBlockAlignment);
}

inline uint32_t getMaxBlockSizeCoalesced(uint32_t uncompressedBlockBytes) {
  return getRawCompBlockMaxSize(uncompressedBlockBytes);
}

inline uint32_t getMaxCompressedSize(uint32_t uncompressedBytes) {
  uint32_t blocks = divUp(uncompressedBytes, kDefaultBlockSize);
  size_t rawSize = ANSCoalescedHeader::getCompressedOverhead(kDefaultBlockSize);
  rawSize += (size_t)getMaxBlockSizeCoalesced(kDefaultBlockSize) * blocks;
  rawSize = roundUp(rawSize, sizeof(uint4));
  return rawSize;
}

inline uint32_t getMaxBlockSizeUnCoalesced(uint32_t uncompressedBlockBytes) {
  return sizeof(ANSWarpState) + getRawCompBlockMaxSize(uncompressedBlockBytes);
}

}