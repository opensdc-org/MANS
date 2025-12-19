cmake_minimum_required(VERSION 3.15)
project(CPUANS LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
  add_compile_options(${OpenMP_CXX_FLAGS})
  link_libraries(${OpenMP_CXX_FLAGS})
endif()

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -mavx2 -mavx512f -mavx512cd -mavx512vl -mavx512bw -fopenmp -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -march=native")
add_executable(cpuans_compress
  compress.cpp
  pans_utils.cpp
)

target_link_libraries(cpuans_compress PRIVATE
  OpenMP::OpenMP_CXX
)

add_executable(cpuans_decompress
  decompress.cpp
  pans_utils.cpp
)

target_link_libraries(cpuans_decompress PRIVATE
  OpenMP::OpenMP_CXX
)

if(APPLE)
  find_path(LLVM_DIR "llvm" PATHS "/opt/homebrew/opt/llvm" REQUIRED)
  include_directories(${LLVM_DIR}/include)
  link_directories(${LLVM_DIR}/lib)
endif()