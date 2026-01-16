#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include "pans_utils.h"
#include "../file_utils.h"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input.file> <output.file>" << std::endl;
        return 1;
    }
    
    std::string inputFilePath = argv[1];
    std::string outputFilePath = argv[2];
    
    std::vector<std::uint8_t> inputData;
    
    
    if(!load_u8_file(inputFilePath, inputData)) {
        std::cerr << "Failed to load input file: " << inputFilePath << std::endl;
        return 1;
    }

    
    size_t estimatedMaxSize = inputData.size() * 10 + 4096; 
    std::vector<std::uint8_t> outputData(estimatedMaxSize);

    size_t actualDecompressedSize = 0;

    
    pans_decompress_and_benchmark(
        inputData.data(),
        inputData.size(),
        outputData.data(),
        actualDecompressedSize
    );

    
    outputData.resize(actualDecompressedSize);

    
    if(!save_u8_file(outputFilePath, outputData)) {
        std::cerr << "Failed to save output file: " << outputFilePath << std::endl;
        return 1;
    }

    std::cout << "Decompression completed successfully. Size: " << actualDecompressedSize << std::endl;
    return 0;
}