find_package(OpenMP REQUIRED)

option(ADM_TIMING "Enable ADM timing/print output" ON)

if(NOT DEFINED CPU_ADM_BINARY_DIR)
  set(CPU_ADM_BINARY_DIR ${CMAKE_BINARY_DIR}/bin/cpu/adm)
endif()

add_executable(adm_compress
  cpuadm_compress.cpp
  adm_utils.cpp
)
set_target_properties(adm_compress PROPERTIES
  OUTPUT_NAME compress
  RUNTIME_OUTPUT_DIRECTORY ${CPU_ADM_BINARY_DIR}
)
target_compile_options(adm_compress PRIVATE
  -mavx512f -march=native
  $<$<CONFIG:Release>:-O3>
)
target_link_libraries(adm_compress PRIVATE OpenMP::OpenMP_CXX)

add_executable(adm_decompress
  cpuadm_decompress.cpp
  adm_utils.cpp
)
set_target_properties(adm_decompress PROPERTIES
  OUTPUT_NAME decompress
  RUNTIME_OUTPUT_DIRECTORY ${CPU_ADM_BINARY_DIR}
)
target_compile_options(adm_decompress PRIVATE
  -mavx512f -march=native
  $<$<CONFIG:Release>:-O3>
)
target_link_libraries(adm_decompress PRIVATE OpenMP::OpenMP_CXX)

add_executable(adm_bench_chunked
  bench_chunked.cpp
  adm_utils.cpp
)
set_target_properties(adm_bench_chunked PROPERTIES
  OUTPUT_NAME adm_bench_chunked
  RUNTIME_OUTPUT_DIRECTORY ${CPU_ADM_BINARY_DIR}
)
target_compile_definitions(adm_bench_chunked PRIVATE
  $<$<BOOL:${ADM_TIMING}>:ADM_TIMING>
)
target_compile_options(adm_bench_chunked PRIVATE
  -mavx512f -march=native
  $<$<CONFIG:Release>:-O3>
)
target_link_libraries(adm_bench_chunked PRIVATE OpenMP::OpenMP_CXX)
