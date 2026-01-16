#include <iostream>
#include <cstdint>
#include <vector>
#include <string>
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

    
    size_t maxOutputSize = inputData.size() * 3 / 2 + 4096;
    
    
    std::vector<std::uint8_t> outputData(maxOutputSize); 

    size_t actualCompressedSize = 0;

    
    pans_compress_and_benchmark(
        inputData.data(), 
        inputData.size(), 
        outputData.data(),
        actualCompressedSize
    );

    
    outputData.resize(actualCompressedSize);

    if(!save_u8_file(outputFilePath, outputData)) {
        std::cerr << "Failed to save output file: " << outputFilePath << std::endl;
        return 1;
    }

    std::cout << "Compression completed successfully. Size: " << actualCompressedSize << std::endl;
    
    
    return 0;
}