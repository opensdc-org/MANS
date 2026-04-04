cmake_minimum_required(VERSION 3.15)

# # Default to Intel compilers unless user already specified one.
# if(NOT DEFINED CMAKE_C_COMPILER)
#   set(CMAKE_C_COMPILER icx CACHE STRING "" FORCE)
# endif()
# if(NOT DEFINED CMAKE_CXX_COMPILER)
#   set(CMAKE_CXX_COMPILER icpx CACHE STRING "" FORCE)
# endif()

project(MANS_ROOT LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

message(STATUS "==========================================")
message(STATUS "  *** CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE} ***")
message(STATUS "  *** C COMPILER: ${CMAKE_C_COMPILER} ***")
message(STATUS "  *** CXX COMPILER: ${CMAKE_CXX_COMPILER} ***")
message(STATUS "==========================================")

# =========================================================
# plugin control
# =========================================================
option(BUILD_HDF5_PLUGIN "Build HDF5 compression filter plugin (H5Z-MANS)" ON)
message(STATUS "  *** BUILD_HDF5_PLUGIN: ${BUILD_HDF5_PLUGIN} ***")
option(HDF5_PREFER_PARALLEL "Prefer parallel HDF5 if available" ON)
message(STATUS "  *** HDF5_PREFER_PARALLEL: ${HDF5_PREFER_PARALLEL} ***")
option(ENABLE_TIMING "Enable timing collection (mans_timing.h)" ON)
message(STATUS "  *** ENABLE_TIMING: ${ENABLE_TIMING} ***")

if(ENABLE_TIMING)
  add_compile_definitions(ENABLE_TIMING)
endif()

# -------------------------------
# platform control
# -------------------------------
set(VALID_PLATFORMS "cpu" "nv" "amd" "all" "cpu_nv" "cpu_amd")
set(TARGET_PLATFORM "cpu" CACHE STRING "Target platform to build: cpu / nv / amd / all / cpu_nv / cpu_amd")
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

add_library(mans_common STATIC mans_api.cpp)

set_property(TARGET mans_common PROPERTY POSITION_INDEPENDENT_CODE ON)
set_target_properties(mans_common PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# -------------------------------
# compile MANS / CPU
# -------------------------------
if(BUILD_CPU)
  add_compile_definitions(MANS_ENABLE_CPU)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")#为了编译pan_utils
  # root dir：build/cpu
  set(cpu_root_binary_dir ${CMAKE_BINARY_DIR}/bin/cpu)

  # ========== CPU / fse submodule ==========
  set(cpu_fse_binary_dir ${cpu_root_binary_dir}/fse)
  set(CPU_FSE_BINARY_DIR ${cpu_fse_binary_dir})
  add_subdirectory(cpu/fse)

  add_library(mans_cpu_core STATIC
    cpu/mans_cpu.cpp
    cpu/adm/adm_utils.cpp
    cpu/pans/pans_utils.cpp
    cpu/fse/fse_utils.cpp
  )
  # [Key] Enable PIC so it can be linked into a shared library
  set_property(TARGET mans_cpu_core PROPERTY POSITION_INDEPENDENT_CODE ON)
  set_target_properties(mans_cpu_core PROPERTIES
      ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  )
  target_compile_options(mans_cpu_core PRIVATE 
  $<$<CONFIG:Release>:-O3>
  -march=native -mavx512f)
  target_link_libraries(mans_cpu_core PRIVATE OpenMP::OpenMP_CXX PUBLIC fse_core)
  # link cpu core to common
  target_link_libraries(mans_common PRIVATE mans_cpu_core)
  # ========== ADM ==========

  # adm dir：build/cpu/adm
  set(cpu_adm_binary_dir ${cpu_root_binary_dir}/adm)
  set(CPU_ADM_BINARY_DIR ${cpu_adm_binary_dir})
  add_subdirectory(cpu/adm)

  # ========== MANS ==========

  # mans bench：build/cpu/cpu_mans_bench
  add_executable(cpu_mans_bench
    cpu/cpu_mans_bench.cpp
  )
  set_target_properties(cpu_mans_bench PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${cpu_root_binary_dir}
  )
  target_compile_options(cpu_mans_bench PRIVATE $<$<CONFIG:Release>:-O3>)
  target_link_libraries(cpu_mans_bench PRIVATE mans_common mans_cpu_core OpenMP::OpenMP_CXX)

  # mans autotune：build/cpu/cpu_mans_autotune
  add_executable(cpu_mans_autotune
    cpu/cpu_mans_autotune.cpp
  )
  set_target_properties(cpu_mans_autotune PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${cpu_root_binary_dir}
  )
  target_compile_options(cpu_mans_autotune PRIVATE $<$<CONFIG:Release>:-O3>)
  target_link_libraries(cpu_mans_autotune PRIVATE mans_common mans_cpu_core OpenMP::OpenMP_CXX)

  add_executable(cpu_mans_compress
    cpu/cpu_mans_compress.cpp
  )
  set_target_properties(cpu_mans_compress PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${cpu_root_binary_dir}
  )
  target_compile_options(cpu_mans_compress PRIVATE $<$<CONFIG:Release>:-O3>)
  target_link_libraries(cpu_mans_compress PRIVATE mans_common mans_cpu_core OpenMP::OpenMP_CXX)

  add_executable(cpu_mans_decompress
    cpu/cpu_mans_decompress.cpp
  )
  set_target_properties(cpu_mans_decompress PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${cpu_root_binary_dir}
  )
  target_compile_options(cpu_mans_decompress PRIVATE $<$<CONFIG:Release>:-O3>)
  target_link_libraries(cpu_mans_decompress PRIVATE mans_common mans_cpu_core OpenMP::OpenMP_CXX)

  
  # ========== CPU / pans submodule ==========
  set(cpu_pans_binary_dir ${cpu_root_binary_dir}/pans)
  set(CPU_PANS_BINARY_DIR ${cpu_pans_binary_dir})
  add_subdirectory(cpu/pans)

endif()

# -------------------------------
# compile mans/nv
# -------------------------------
if(BUILD_NV)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_compile_definitions(MANS_ENABLE_NV)

  add_library(mans_nv_core STATIC
    nv/mans_nv.cpp
    nv/adm/mapping_uint16.cu
    nv/adm/mapping_uint32.cu
    nv/ans/mans_nv_ans.cu
    nv/ans/src/GpuANS.cu
  )
  set_property(TARGET mans_nv_core PROPERTY POSITION_INDEPENDENT_CODE ON)
  set_target_properties(mans_nv_core PROPERTIES
      ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
  target_include_directories(mans_nv_core PUBLIC
      ${CUDAToolkit_INCLUDE_DIRS}
      ${CMAKE_CURRENT_SOURCE_DIR}/nv/ans/include
      ${CMAKE_CURRENT_SOURCE_DIR}/nv/ans/src
  )
  target_link_libraries(mans_nv_core PUBLIC CUDA::cudart)
  target_link_libraries(mans_common PRIVATE mans_nv_core)
  include(FindCUDA/select_compute_arch)
  CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS)
  string(STRIP "${INSTALLED_GPU_CCS}" INSTALLED_GPU_CCS_STRIPPED)
  string(REPLACE " " ";" INSTALLED_GPU_CCS_LIST "${INSTALLED_GPU_CCS_STRIPPED}")
  string(REPLACE "." "" DETECTED_CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_LIST}")
  string(REPLACE "+PTX" "" DETECTED_CUDA_ARCH_LIST "${DETECTED_CUDA_ARCH_LIST}")
  string(REPLACE "+ptx" "" DETECTED_CUDA_ARCH_LIST "${DETECTED_CUDA_ARCH_LIST}")
  message(STATUS "Auto-detected CUDA architectures: ${DETECTED_CUDA_ARCH_LIST}")

  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/nv)

  add_executable(nv_mans_compress nv/nv_mans_compress.cpp)
  set_target_properties(nv_mans_compress PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
  )
  target_compile_options(nv_mans_compress PRIVATE $<$<CONFIG:Release>:-O3>)
  target_link_libraries(nv_mans_compress PRIVATE mans_common)

  add_executable(nv_mans_decompress nv/nv_mans_decompress.cpp)
  set_target_properties(nv_mans_decompress PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
  )
  target_compile_options(nv_mans_decompress PRIVATE $<$<CONFIG:Release>:-O3>)
  target_link_libraries(nv_mans_decompress PRIVATE mans_common)

  add_executable(nvadm_compress nv/adm/nvadm_compress.cpp)
  set_target_properties(nvadm_compress PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
  )
  target_compile_options(nvadm_compress PRIVATE $<$<CONFIG:Release>:-O3>)
  target_link_libraries(nvadm_compress PRIVATE mans_nv_core)

  add_executable(nvadm_decompress nv/adm/nvadm_decompress.cpp)
  set_target_properties(nvadm_decompress PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}

  )
  target_compile_options(nvadm_decompress PRIVATE $<$<CONFIG:Release>:-O3>)
  target_link_libraries(nvadm_decompress PRIVATE mans_nv_core)

  # nv/ans submodule
  set(nv_ans_binary_dir ${CMAKE_BINARY_DIR}/nv/ans)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${nv_ans_binary_dir})
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${nv_ans_binary_dir})
  add_subdirectory(nv/ans)
  target_link_libraries(mans_nv_core PUBLIC gpu_ans)

endif()

# -------------------------------
# compile mans/amd (use hipcc)
# -------------------------------
if(BUILD_AMD)
  add_compile_definitions(MANS_ENABLE_AMD)
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



if(BUILD_HDF5_PLUGIN)
  add_subdirectory(tools/H5Z-MANS)
endif()