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

#define PANS_PRECISION 10   // define compression precision 

using namespace cpu_ans;

// tool function：raw_data or adm_compressed_data -> pans_compressed_data
void pans_compress(
    const uint8_t* inputData,
    size_t inputLen,
    uint8_t* outputData,        
    size_t &outputLen,          
    double &duration            
) {
    if (inputLen == 0) {
        std::cerr << "Error: inputData is empty." << std::endl;
        outputLen = 0;
        return;
    }

    // cast to const uint8_t* is redundant but keeps type explicit
    const uint8_t* inPtrs = inputData;
    uint32_t batchSize = static_cast<uint32_t>(inputLen);
    const int precision = PANS_PRECISION;

    uint32_t* outCompressedSize = (uint32_t*)malloc(sizeof(uint32_t));
    uint8_t* encPtrs = (uint8_t*)malloc(getMaxCompressedSize(inputLen));
    ANSCoalescedHeader* headerOut = (ANSCoalescedHeader*)encPtrs;
    uint32_t maxNumCompressedBlocks;

    uint32_t maxUncompressedWords = batchSize / sizeof(ANSDecodedT);
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
        (uint8_t*)inPtrs, 
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
    // auto BlockDataStart = headerOut->getBlockDataStart(maxNumCompressedBlocks); // Unused variable
    
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
    
    uint32_t lastBlockWords = static_cast<uint32_t>(inputLen) % kDefaultBlockSize;
    lastBlockWords = lastBlockWords == 0 ? kDefaultBlockSize : lastBlockWords;

    blockWordsOut[i] =
        uint2{(lastBlockWords << 16) |
                  compressedWords_host[i],
              compressedWordsPrefix_host[i]};

    const uint32_t headerSize =
        headerOut->getCompressedOverhead(
            maxNumCompressedBlocks);
    uint32_t outsize = *outCompressedSize;
    
    outputLen = outsize;

    // Only write if output buffer is provided
    if (outputData) {
        // First copy the header part
        std::memcpy(outputData,
                    encPtrs,
                    headerSize);

        // Append block data sequentially after the header
        uint8_t* writePtr =
            outputData + headerSize;

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

// benchmark: call pans_compress multiple times to measure time
void pans_compress_and_benchmark(
    const uint8_t* inputData,
    size_t inputLen,
    uint8_t* outputData,
    size_t &outputLen
) {
    if (inputLen == 0) {
        outputLen = 0;
        return;
    }

    // Pre-allocate internal buffer (1.5x estimate + slack)
    size_t max_buf_size = inputLen * 3 / 2 + 4096;
    std::vector<uint8_t> tmp(max_buf_size);
    std::vector<uint8_t> last_valid_output;
    
    size_t actual_compressed_size = 0;

    std::cout << "encode start!" << std::endl;
    double comp_time = 1e30;
    double dur = 0.0;

    // Warmup & Benchmark loop
    for(int i = 0; i < 11; i ++){
        size_t cs = 0;
        
        // Call compress with tmp buffer
        pans_compress(inputData, inputLen, tmp.data(), cs, dur);
        
        if (i > 0 && comp_time > dur) comp_time = dur;  // discard the 0th run as warmup
        if (i == 10) {
            actual_compressed_size = cs;
            last_valid_output.resize(cs);
            std::memcpy(last_valid_output.data(), tmp.data(), cs);
        }
    }

    // Copy result to user provided output buffer
    if (outputData && !last_valid_output.empty()) {
        std::memcpy(outputData, last_valid_output.data(), last_valid_output.size());
    }

    outputLen = last_valid_output.size();
    

    double c_bw = ( 1.0 * inputLen / 1e6 ) / ( (comp_time) * 1e-3 );  
    std::cout << "comp   time " << std::fixed << std::setprecision(3) << comp_time << " ms B/W "   
              << std::fixed << std::setprecision(1) << c_bw << " MB/s " << std::endl;

    if (actual_compressed_size > 0) {
        std::printf("[pans] compress ratio: %f\n",
                    1.0 * inputLen / actual_compressed_size);
    }
    else{
        std::cerr << "Error: compressedSize too small:" << actual_compressed_size << std::endl;
    }
}
void get_compress_and_decompressed_len(
    const uint8_t* compressedData,
    size_t &compress_len,
    size_t &decompressed_len
){

    ANSCoalescedHeader headerBuf;
    std::memcpy(&headerBuf, compressedData, 32); // header is 32 bytes
    ANSCoalescedHeader* Header = &headerBuf;
    int compress_len_header = Header->getTotalCompressedSize();
    int totalUncompressedWords = Header->getTotalUncompressedWords(); 
    decompressed_len = static_cast<size_t>(totalUncompressedWords);
    if(compress_len_header != compress_len){
        std::cerr << "Error: compress_len_header != compress_len."
                  << std::endl;
                  decompressed_len = 0;
        return;
    }
}
// tool function：pans_compressed_data -> raw_data or adm_compressed_data
void pans_decompress(
    const uint8_t* compressedData,
    size_t compressedLen,
    uint8_t* decompressedData, 
    size_t &decompressedLen,  
    double &duration
) {
    if (compressedLen < 32) {
        std::cerr << "Error: compressedData too small."
                  << std::endl;
        decompressedLen = 0;
        return;
    }

    // Read the header data directly from compressedData
    ANSCoalescedHeader headerBuf;
    std::memcpy(&headerBuf, compressedData, 32); // header is 32 bytes

    ANSCoalescedHeader* Header = &headerBuf;
    int totalCompressedSize = Header->getTotalCompressedSize();
    int totalUncompressedWords = Header->getTotalUncompressedWords(); // assuming this is in words? 

    size_t batchSize = static_cast<size_t>(totalUncompressedWords);

    if ((int)compressedLen < totalCompressedSize) {
        std::cerr
            << "Error: compressedData size less than header "
               "reported totalCompressedSize."
            << std::endl;
        decompressedLen = 0;
        return;
    }

    const int precision = PANS_PRECISION;


    uint8_t* decPtrs = (uint8_t*)malloc(sizeof(uint8_t) * batchSize);
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
        (uint8_t*)compressedData, // cast const away if api requires it
        decPtrs);
    auto end = std::chrono::high_resolution_clock::now();  
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e3;

    if (decompressedData) {
        std::memcpy(decompressedData, decPtrs, batchSize * sizeof(uint8_t));
    }
    

    decompressedLen = batchSize;


    free(decPtrs);
    free(symbol);
    free(pdf);
    free(cdf);
}

// benchmark: call pans_decompress multiple times to measure time
void pans_decompress_and_benchmark(
    const uint8_t* compressedData,
    size_t compressedLen,
    uint8_t* decompressedData,
    size_t &out_actual_len 
) {
    if (compressedLen < 32) return;

    // Parse header for stats
    ANSCoalescedHeader headerBuf;
    std::memcpy(&headerBuf, compressedData, 32);

    int totalUncompressedBytes = headerBuf.getTotalUncompressedWords(); 

    std::cout << "decode start!" << std::endl;
    double decomp_time = 1e30;
    double dur;
    size_t out_len = 0;
    
    // Warmup & Benchmark loop
    for(int i = 0; i < 11; i ++){
        pans_decompress(compressedData, compressedLen, decompressedData, out_len, dur);
        
        if (i > 0 && decomp_time > dur) decomp_time = dur; 
    }

    double dc_bw = (1.0 * out_len / 1e6) /
                   (decomp_time * 1e-3);
    std::cout << "decomp time " << std::fixed
              << std::setprecision(6) << decomp_time
              << " ms B/W " << std::fixed
              << std::setprecision(1) << dc_bw
              << " MB/s" << std::endl;

    out_actual_len = out_len;

}