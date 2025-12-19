// pans_utils.cpp
#include "pans_utils.h"
#include "CpuANSEncode.h"
#include "CpuANSDecode.h"

#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <cstdlib>

#define PANS_PRECISION 10   // 压缩精度宏定义

using namespace cpu_ans;

// 纯压缩：一次压缩，返回 batchSize / compressedSize
void pans_compress(
    std::vector<uint8_t>& inputData,
    std::vector<uint8_t>& compressedData,
    uint32_t& batchSize,
    uint32_t& compressedSize,
    double &duration
) {
    const std::streamsize fileSize =
        static_cast<std::streamsize>(inputData.size());
    if (fileSize <= 0) {
        std::cerr << "Error: inputData is empty." << std::endl;
        batchSize = 0;
        compressedSize = 0;
        compressedData.clear();
        return;
    }

    uint8_t* inPtrs = inputData.data();
    batchSize = static_cast<uint32_t>(fileSize);
    const int precision = PANS_PRECISION;

    uint32_t* outCompressedSize = (uint32_t*)malloc(sizeof(uint32_t));
    uint8_t* encPtrs = (uint8_t*)malloc(getMaxCompressedSize(fileSize));
    ANSCoalescedHeader* headerOut = (ANSCoalescedHeader*)encPtrs;
    uint32_t maxNumCompressedBlocks;

    uint32_t maxUncompressedWords = fileSize / sizeof(ANSDecodedT);
    maxNumCompressedBlocks =
        (maxUncompressedWords + kDefaultBlockSize - 1) / kDefaultBlockSize;
    
    uint4* table = (uint4*)malloc(sizeof(uint4) * kNumSymbols);
    uint32_t* tempHistogram = (uint32_t*)malloc(sizeof(uint32_t) * kNumSymbols);
    uint32_t uncoalescedBlockStride = getMaxBlockSizeUnCoalesced(kDefaultBlockSize);
    uint8_t* compressedBlocks_host = (uint8_t*)std::aligned_alloc(
        kBlockAlignment,
        sizeof(uint8_t) * maxNumCompressedBlocks * uncoalescedBlockStride);
    uint32_t* compressedWords_host = (uint32_t*)std::aligned_alloc(
        kBlockAlignment,
        sizeof(uint32_t) * maxNumCompressedBlocks);
    uint32_t* compressedWords_host_prefix = (uint32_t*)std::aligned_alloc(
        kBlockAlignment,
        sizeof(uint32_t) * maxNumCompressedBlocks);
    uint32_t* compressedWordsPrefix_host = (uint32_t*)std::aligned_alloc(
        kBlockAlignment,
        sizeof(uint32_t) * maxNumCompressedBlocks);
    
    auto start = std::chrono::high_resolution_clock::now();  
    ansEncode(
        table,
        tempHistogram,
        precision,
        inPtrs,
        batchSize,
        encPtrs,
        outCompressedSize,
        headerOut,
        maxNumCompressedBlocks,
        uncoalescedBlockStride,
        compressedBlocks_host,
        compressedWords_host,
        compressedWords_host_prefix,
        compressedWordsPrefix_host);
    auto end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e3;
    
    auto blockWordsOut = headerOut->getBlockWords(maxNumCompressedBlocks);
    auto BlockDataStart = headerOut->getBlockDataStart(maxNumCompressedBlocks);
    
    int i = 0;
    for(; i < static_cast<int>(maxNumCompressedBlocks) - 1; i ++){
        auto uncoalescedBlock = compressedBlocks_host + i * uncoalescedBlockStride;
        for(int j = 0; j < kWarpSize; ++j){
            auto warpStateOut = (ANSWarpState*)uncoalescedBlock;
            headerOut->getWarpStates()[i].warpState[j] = (warpStateOut->warpState[j]);
        }
        blockWordsOut[i] = uint2{
            (kDefaultBlockSize << 16) | compressedWords_host[i], 
            compressedWordsPrefix_host[i]};
    }

    // Process last block
    auto uncoalescedBlock = compressedBlocks_host + i * uncoalescedBlockStride;
    for(int j = 0; j < kWarpSize; ++j){
        auto warpStateOut = (ANSWarpState*)uncoalescedBlock;
        headerOut->getWarpStates()[i].warpState[j] = (warpStateOut->warpState[j]);
    }
    
    uint32_t lastBlockWords = static_cast<uint32_t>(fileSize) % kDefaultBlockSize;
    lastBlockWords = lastBlockWords == 0 ? kDefaultBlockSize : lastBlockWords;

    blockWordsOut[i] =
        uint2{(lastBlockWords << 16) |
                  compressedWords_host[i],
              compressedWordsPrefix_host[i]};

    // 把 header + block data 全部聚合到 compressedData 里
    const uint32_t headerSize =
        headerOut->getCompressedOverhead(
            maxNumCompressedBlocks);
    uint32_t outsize = *outCompressedSize;
    compressedSize = outsize;

    compressedData.resize(outsize);

    // 先拷 header 部分
    std::memcpy(compressedData.data(),
                encPtrs,
                headerSize);

    // 依次把 block 数据拼接到 header 后面
    uint8_t* writePtr =
        compressedData.data() + headerSize;

    i = 0;
    for (; i < static_cast<int>(maxNumCompressedBlocks) - 1;
         i++) {
        auto uncoalescedBlock2 =
            compressedBlocks_host +
            i * uncoalescedBlockStride;
        uint32_t numWords = compressedWords_host[i];
        uint32_t limitEnd = divUp(numWords, kBlockAlignment / sizeof(ANSEncodedT));

        auto inT = (const uint4*)(uncoalescedBlock2 +
                                  sizeof(ANSWarpState));
        size_t bytes = (size_t)limitEnd << 4;
        std::memcpy(writePtr,
                    reinterpret_cast<const char*>(inT),
                    bytes);
        writePtr += bytes;
    }

    // Write last block data
    {
        uint32_t numWords = compressedWords_host[i];
        uint32_t limitEnd =
            divUp(numWords,
                  kBlockAlignment /
                      sizeof(ANSEncodedT));
        auto inT = (const uint4*)(uncoalescedBlock +
                                  sizeof(ANSWarpState));
        size_t bytes = (size_t)limitEnd << 4;
        std::memcpy(writePtr,
                    reinterpret_cast<const char*>(inT),
                    bytes);
        writePtr += bytes;
    }

    // Clean up
    free(outCompressedSize);
    free(encPtrs);
    free(table);
    free(tempHistogram);
    free(compressedBlocks_host);
    free(compressedWords_host);
    free(compressedWords_host_prefix);
    free(compressedWordsPrefix_host);
}

// 仅 benchmark：调用 pans_compress，多次测时
void pans_compress_and_benchmark(
    std::vector<uint8_t>& inputData,
    std::vector<uint8_t>& compressedData
) {
    uint32_t batchSize = 0;
    uint32_t compressedSize = 0;

    const std::streamsize fileSize =
        static_cast<std::streamsize>(inputData.size());

    std::cout << "encode start!" << std::endl;
    double comp_time = 1e30;
    double dur = 0.0;

    // Warmup & Benchmark loop
    for(int i = 0; i < 11; i ++){
        std::vector<uint8_t> tmp;
        uint32_t bs = 0, cs = 0;
        pans_compress(inputData, tmp, bs, cs,dur);
        if (i > 0 && comp_time > dur) comp_time = dur;  // 丢掉第 0 次作为 warmup
        if (i == 10) {
            compressedData.swap(tmp);
            batchSize = bs;
            compressedSize = cs;
        }
    }

    double c_bw = ( 1.0 * fileSize / 1e6 ) / ( (comp_time) * 1e-3 );  
    std::cout << "comp   time " << std::fixed << std::setprecision(3) << comp_time << " ms B/W "   
              << std::fixed << std::setprecision(1) << c_bw << " MB/s " << std::endl;

    if (compressedSize > 0) {
        std::printf("[pans] compress ratio: %f\n",
                    1.0 * batchSize / compressedSize);
    }
    else{
        std::cerr << "Error: compressedSize too small:" <<compressedSize<< std::endl;
    }
}

// 纯解压：一次解压，返回 batchSize / compressedSize
void pans_decompress(
    std::vector<uint8_t>& compressedData,
    std::vector<uint8_t>& decompressedData,
    uint32_t& batchSize,
    uint32_t& compressedSize,
    double &duration
) {
    if (compressedData.size() < 32) {
        std::cerr << "Error: compressedData too small."
                  << std::endl;
        decompressedData.clear();
        batchSize = 0;
        compressedSize = 0;
        return;
    }

    // 头部数据直接从 compressedData 里读
    std::vector<uint8_t> fileCompressedHead(32);
    std::memcpy(fileCompressedHead.data(),
                compressedData.data(),
                32);
    auto Header =
        (ANSCoalescedHeader*)fileCompressedHead.data();
    int totalCompressedSize =
        Header->getTotalCompressedSize();
    int bs =
        Header->getTotalUncompressedWords();

    if ((int)compressedData.size() < totalCompressedSize) {
        std::cerr
            << "Error: compressedData size less than header "
               "reported totalCompressedSize."
            << std::endl;
        decompressedData.clear();
        batchSize = 0;
        compressedSize = 0;
        return;
    }

    compressedSize = static_cast<uint32_t>(compressedData.size());
    batchSize = static_cast<uint32_t>(bs);

    uint8_t* compressedPtr =
        compressedData.data();

    const int precision = PANS_PRECISION;

    uint8_t* decPtrs =
        (uint8_t*)malloc(sizeof(uint8_t) *
                         (size_t)batchSize);
    uint32_t* symbol = (uint32_t*)std::aligned_alloc(
        kBlockAlignment,
        sizeof(uint32_t) * (1u << precision));
    uint32_t* pdf = (uint32_t*)std::aligned_alloc(
        kBlockAlignment,
        sizeof(uint32_t) * (1u << precision));
    uint32_t* cdf = (uint32_t*)std::aligned_alloc(
        kBlockAlignment,
        sizeof(uint32_t) * (1u << precision));
    
    auto start = std::chrono::high_resolution_clock::now();
    ansDecode(
        symbol,
        pdf,
        cdf,
        precision,
        compressedPtr,
        decPtrs);
    auto end = std::chrono::high_resolution_clock::now();  
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e3;


    // 输出到 vector
    decompressedData.resize((size_t)batchSize);
    std::memcpy(decompressedData.data(),
                decPtrs,
                (size_t)batchSize * sizeof(uint8_t));

    free(decPtrs);
    free(symbol);
    free(pdf);
    free(cdf);
}

// 仅 benchmark：调用 pans_decompress，多次测时
void pans_decompress_and_benchmark(
    std::vector<uint8_t>& compressedData,
    std::vector<uint8_t>& decompressedData
) {
    uint32_t batchSize = 0;
    uint32_t compressedSize = 0;



    std::cout << "decode start!" << std::endl;
    double decomp_time = 1e30;
    double dur;
    // Warmup & Benchmark loop
    for(int i = 0; i < 11; i ++){
        std::vector<uint8_t> tmp;
        uint32_t bs = 0, cs = 0;
        pans_decompress(compressedData, tmp, bs, cs,dur);
        
        if (i > 0 && decomp_time > dur) decomp_time = dur; 

        if (i == 10) {
            decompressedData.swap(tmp);
            batchSize = bs;
            compressedSize = cs;
        }
    }

    double dc_bw = (1.0 * compressedSize / 1e6) /
                   (decomp_time * 1e-3);
    std::cout << "decomp time " << std::fixed
              << std::setprecision(6) << decomp_time
              << " ms B/W " << std::fixed
              << std::setprecision(1) << dc_bw
              << " MB/s" << std::endl;
}