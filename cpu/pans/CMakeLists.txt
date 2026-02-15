find_package(OpenMP REQUIRED)

if(NOT DEFINED CPU_PANS_BINARY_DIR)
  set(CPU_PANS_BINARY_DIR ${CMAKE_BINARY_DIR}/bin/cpu/pans)
endif()
add_executable(cpuans_compress
  compress.cpp
  pans_utils.cpp
)

target_link_libraries(cpuans_compress PRIVATE
  OpenMP::OpenMP_CXX
)
set_target_properties(cpuans_compress PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CPU_PANS_BINARY_DIR}
)

add_executable(cpuans_decompress
  decompress.cpp
  pans_utils.cpp
)

target_link_libraries(cpuans_decompress PRIVATE
  OpenMP::OpenMP_CXX
)
set_target_properties(cpuans_decompress PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CPU_PANS_BINARY_DIR}
)

add_executable(cpuans_bench_chunked
  bench_chunked.cpp
  pans_utils.cpp
)

target_link_libraries(cpuans_bench_chunked PRIVATE
  OpenMP::OpenMP_CXX
)
set_target_properties(cpuans_bench_chunked PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CPU_PANS_BINARY_DIR}
)

if(APPLE)
  find_path(LLVM_DIR "llvm" PATHS "/opt/homebrew/opt/llvm" REQUIRED)
  include_directories(${LLVM_DIR}/include)
  link_directories(${LLVM_DIR}/lib)
endif()
