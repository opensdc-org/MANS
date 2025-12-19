#include <iostream>
#include <cstdint>
#include "pans_utils.h"
#include "../file_utils.h"
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input.file> <output.file>" << std::endl;
        return 1;
    }
    
    // 读取输入文件
    std::string inputFilePath = argv[1];
    std::string outputFilePath = argv[2];
    std::vector<std::uint8_t> inputData;
    std::vector<std::uint8_t> outputData;
    if(!load_u8_file(inputFilePath, inputData)) {
        std::cerr << "Failed to load input file: " << inputFilePath << std::endl;
        return 1;
    }

    pans_compress_and_benchmark(
        inputData, 
        outputData
    );
    if(!save_u8_file(outputFilePath, outputData)) {
        std::cerr << "Failed to save output file: " << outputFilePath << std::endl;
        return 1;
    }
    std::cout << "Compression completed successfully." << std::endl;
    
    return 0;
}