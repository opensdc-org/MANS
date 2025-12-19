cmake_minimum_required(VERSION 3.15)
project(MANS_ROOT LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

message(STATUS "==========================================")
message(STATUS "  *** CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE} ***")
message(STATUS "==========================================")

# -------------------------------
# platform control
# -------------------------------
set(VALID_PLATFORMS "cpu" "nv" "amd" "all" "cpu_nv" "cpu_amd")
set(TARGET_PLATFORM "cpu_nv" CACHE STRING "Target platform to build: cpu / nv / amd / all / cpu_nv / cpu_amd")
if (NOT TARGET_PLATFORM IN_LIST VALID_PLATFORMS)
  message(FATAL_ERROR "Invalid TARGET_PLATFORM='${TARGET_PLATFORM}'. Must be one of: ${VALID_PLATFORMS}.")
endif()

# -------------------------------
# platform tag
# -------------------------------
set(BUILD_CPU FALSE)
set(BUILD_NV FALSE)
set(BUILD_AMD FALSE)

if(TARGET_PLATFORM STREQUAL "cpu" OR TARGET_PLATFORM STREQUAL "cpu_nv" OR TARGET_PLATFORM STREQUAL "cpu_amd" OR TARGET_PLATFORM STREQUAL "all")
  set(BUILD_CPU TRUE)
endif()
if(TARGET_PLATFORM STREQUAL "nv" OR TARGET_PLATFORM STREQUAL "cpu_nv" OR TARGET_PLATFORM STREQUAL "all")
  set(BUILD_NV TRUE)
endif()
if(TARGET_PLATFORM STREQUAL "amd" OR TARGET_PLATFORM STREQUAL "cpu_amd" OR TARGET_PLATFORM STREQUAL "all")
  set(BUILD_AMD TRUE)
endif()

# -------------------------------
# OpenMP（for CPU）
# -------------------------------
if(BUILD_CPU)
  find_package(OpenMP REQUIRED)
endif()

# -------------------------------
# compile MANS / CPU
# -------------------------------
if(BUILD_CPU)
  
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -march=native")#为了编译pan_utils
  # root dir：build/cpu
  set(cpu_root_binary_dir ${CMAKE_BINARY_DIR}/bin/cpu)

  # ========== ADM ==========

  # adm dir：build/cpu/adm
  set(cpu_adm_binary_dir ${cpu_root_binary_dir}/adm)

  # compress（from cpu/adm/cpuadm_compress.cpp , exec:"compress"）
  add_executable(cpuadm_compress 
    cpu/adm/cpuadm_compress.cpp
    cpu/adm/adm_utils.cpp
    )
  set_target_properties(cpuadm_compress PROPERTIES
      OUTPUT_NAME            compress
      RUNTIME_OUTPUT_DIRECTORY ${cpu_adm_binary_dir}
      ARCHIVE_OUTPUT_DIRECTORY ${cpu_adm_binary_dir}
      LIBRARY_OUTPUT_DIRECTORY ${cpu_adm_binary_dir}
  )
  target_compile_options(cpuadm_compress PRIVATE -mavx512f -fopenmp -march=native -O3)
  target_link_libraries(cpuadm_compress PRIVATE OpenMP::OpenMP_CXX)

  # decompress（from cpu/adm/cpuadm_decompress.cpp，exec: decompress）
  add_executable(cpuadm_decompress
   cpu/adm/cpuadm_decompress.cpp
   cpu/adm/adm_utils.cpp
   )
  set_target_properties(cpuadm_decompress PROPERTIES
      OUTPUT_NAME            decompress
      RUNTIME_OUTPUT_DIRECTORY ${cpu_adm_binary_dir}
      ARCHIVE_OUTPUT_DIRECTORY ${cpu_adm_binary_dir}
      LIBRARY_OUTPUT_DIRECTORY ${cpu_adm_binary_dir}
  )
  target_compile_options(cpuadm_decompress PRIVATE -mavx512f -fopenmp -march=native -O3)
  target_link_libraries(cpuadm_decompress PRIVATE OpenMP::OpenMP_CXX)

  # ========== MANS ==========

  # mans compress：build/cpu/cpu_mans_compress
  add_executable(cpu_mans_compress 
    cpu/cpu_mans_compress.cpp
    cpu/adm/adm_utils.cpp
    cpu/pans/pans_utils.cpp
  )
  set_target_properties(cpu_mans_compress PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${cpu_root_binary_dir}
  )
  target_compile_options(cpu_mans_compress PRIVATE -O3)

  # mans decompress：build/cpu/cpu_mans_decompress
  add_executable(cpu_mans_decompress
   cpu/cpu_mans_decompress.cpp
   cpu/adm/adm_utils.cpp
   cpu/pans/pans_utils.cpp
   )
  set_target_properties(cpu_mans_decompress PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${cpu_root_binary_dir}
  )
  target_compile_options(cpu_mans_decompress PRIVATE -O3)

  # ========== CPU / pans submodule ==========
  set(cpu_pans_binary_dir ${cpu_root_binary_dir}/pans)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${cpu_pans_binary_dir})
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${cpu_pans_binary_dir})
  add_subdirectory(cpu/pans)

  set_target_properties(cpuans_compress cpuans_decompress PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${cpu_pans_binary_dir}
  )

endif()

# -------------------------------
# compile mans/nv
# -------------------------------
if(BUILD_NV)
  enable_language(CUDA)
  
  # detect gpu architecture
  include(FindCUDA/select_compute_arch)
  CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS)
  string(STRIP "${INSTALLED_GPU_CCS}" INSTALLED_GPU_CCS_STRIPPED)
  string(REPLACE " " ";" INSTALLED_GPU_CCS_LIST "${INSTALLED_GPU_CCS_STRIPPED}")
  string(REPLACE "." "" DETECTED_CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_LIST}")
  message(STATUS "Auto-detected CUDA architectures: ${DETECTED_CUDA_ARCH_LIST}")

  set(nv_adm_binary_dir ${CMAKE_BINARY_DIR}/nv/adm)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/nv)

  add_executable(nv_mapping_uint16 nv/adm/mapping_uint16.cu)
  set_target_properties(nv_mapping_uint16 PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
      ARCHIVE_OUTPUT_DIRECTORY ${nv_adm_binary_dir}
      LIBRARY_OUTPUT_DIRECTORY ${nv_adm_binary_dir}
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_ARCHITECTURES "${DETECTED_CUDA_ARCH_LIST}"
  )
  target_compile_options(nv_mapping_uint16 PRIVATE -O3)

  add_executable(nv_mapping_uint32 nv/adm/mapping_uint32.cu)
  set_target_properties(nv_mapping_uint32 PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
      ARCHIVE_OUTPUT_DIRECTORY ${nv_adm_binary_dir}
      LIBRARY_OUTPUT_DIRECTORY ${nv_adm_binary_dir}
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_ARCHITECTURES "${DETECTED_CUDA_ARCH_LIST}"
  )
  target_compile_options(nv_mapping_uint32 PRIVATE -O3)

  # nv/ans submodule
  set(nv_ans_binary_dir ${CMAKE_BINARY_DIR}/nv/ans)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${nv_ans_binary_dir})
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${nv_ans_binary_dir})
  add_subdirectory(nv/ans)

endif()

# -------------------------------
# compile mans/amd (use hipcc)
# -------------------------------
if(BUILD_AMD)

  find_program(HIPCC_EXECUTABLE NAMES hipcc REQUIRED)
  if(NOT HIPCC_EXECUTABLE)
    message(FATAL_ERROR "hipcc not found. Please ensure ROCm is installed and hipcc is in PATH.")
  endif()

  # ========== amd/adm ==========

  set(AMD_ADM_BIN_DIR ${CMAKE_BINARY_DIR}/bin/amd)

  foreach(fname IN ITEMS mapping_uint16 mapping_uint32)
    set(src_file ${CMAKE_SOURCE_DIR}/amd/adm/${fname}.cpp)
    set(target_name amd_${fname})
    set(out_file ${AMD_ADM_BIN_DIR}/${target_name})

    add_custom_command(
      OUTPUT ${out_file}
      COMMAND ${HIPCC_EXECUTABLE} ${src_file}
              -DHIP_ENABLE_WARP_SYNC_BUILTINS
              -Wno-unused-result -Wno-format -O3
              -o ${out_file}
      DEPENDS ${src_file}
      COMMENT "Building AMD ADM: ${fname}.cpp"
      VERBATIM
    )

    add_custom_target(amd_adm_${fname} ALL
      DEPENDS ${out_file}
    )
  endforeach()

  # ========== amd/ans submudole ==========

  set(amd_ans_binary_dir ${CMAKE_BINARY_DIR}/amd/ans)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/amd)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${amd_ans_binary_dir})
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${amd_ans_binary_dir})
  add_subdirectory(amd/ans)

endif()

