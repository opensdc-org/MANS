cmake_minimum_required(VERSION 3.15)

# # Default to Intel compilers unless user already specified one.
# if(NOT DEFINED CMAKE_C_COMPILER)
#   set(CMAKE_C_COMPILER icx CACHE STRING "" FORCE)
# endif()
# if(NOT DEFINED CMAKE_CXX_COMPILER)
#   set(CMAKE_CXX_COMPILER icpx CACHE STRING "" FORCE)
# endif()

project(MANS_ROOT LANGUAGES CXX C)

include(CTest)

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
set(VALID_PLATFORMS "cpu" "nv" "amd" "dcu" "all" "cpu_nv" "cpu_amd" "cpu_dcu")
set(TARGET_PLATFORM "cpu" CACHE STRING "Target platform to build: cpu / nv / amd / dcu / all / cpu_nv / cpu_amd / cpu_dcu")
set(DCU_HIP_COMPILER "" CACHE FILEPATH "DTK hipcc executable for DCU builds")
set(DCU_HIP_ARCH "" CACHE STRING "DTK HIP offload architecture for DCU builds")
set(DCU_HIPCUB_INCLUDE_DIR "" CACHE PATH "Optional DTK hipCUB include directory")
set(MANS_CPU_ARCH_FLAGS "-march=native -mavx2" CACHE STRING "CPU ISA flags used by CPU ADM/PANS")
if (NOT TARGET_PLATFORM IN_LIST VALID_PLATFORMS)
  message(FATAL_ERROR "Invalid TARGET_PLATFORM='${TARGET_PLATFORM}'. Must be one of: ${VALID_PLATFORMS}.")
endif()

# -------------------------------
# platform tag
# -------------------------------
set(BUILD_CPU FALSE)
set(BUILD_NV FALSE)
set(BUILD_AMD FALSE)
set(BUILD_DCU FALSE)

if(TARGET_PLATFORM STREQUAL "cpu" OR TARGET_PLATFORM STREQUAL "cpu_nv" OR
   TARGET_PLATFORM STREQUAL "cpu_amd" OR TARGET_PLATFORM STREQUAL "cpu_dcu" OR
   TARGET_PLATFORM STREQUAL "all")
  set(BUILD_CPU TRUE)
endif()
if(TARGET_PLATFORM STREQUAL "nv" OR TARGET_PLATFORM STREQUAL "cpu_nv" OR TARGET_PLATFORM STREQUAL "all")
  set(BUILD_NV TRUE)
endif()
if(TARGET_PLATFORM STREQUAL "amd" OR TARGET_PLATFORM STREQUAL "cpu_amd" OR TARGET_PLATFORM STREQUAL "all")
  set(BUILD_AMD TRUE)
endif()
if(TARGET_PLATFORM STREQUAL "dcu" OR TARGET_PLATFORM STREQUAL "cpu_dcu")
  set(BUILD_DCU TRUE)
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
  if(MANS_CPU_ARCH_FLAGS)
    separate_arguments(MANS_CPU_ARCH_FLAGS_LIST NATIVE_COMMAND "${MANS_CPU_ARCH_FLAGS}")
  else()
    set(MANS_CPU_ARCH_FLAGS_LIST)
  endif()
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
    ${MANS_CPU_ARCH_FLAGS_LIST})
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

  if(BUILD_TESTING AND BUILD_CPU)
    add_executable(mans_cross_backend_test tests/mans_cross_backend_test.cpp)
    set_target_properties(mans_cross_backend_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
    )
    target_link_libraries(mans_cross_backend_test PRIVATE mans_common CUDA::cudart)
    add_test(NAME mans_cross_backend_test COMMAND mans_cross_backend_test)
  endif()

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
# compile the shared HIP backend for AMD GPUs or DCUs
# -------------------------------
if(BUILD_AMD OR BUILD_DCU)
  if(NOT BUILD_CPU)
    message(FATAL_ERROR "The initial HIP backend reuses CPU ADM; use cpu_amd or cpu_dcu.")
  endif()

  if(BUILD_DCU AND DCU_HIP_COMPILER)
    set(HIPCC_EXECUTABLE "${DCU_HIP_COMPILER}")
  else()
    find_program(HIPCC_EXECUTABLE NAMES hipcc REQUIRED)
  endif()
  if(NOT CMAKE_HIP_COMPILER)
    set(CMAKE_HIP_COMPILER "${HIPCC_EXECUTABLE}" CACHE FILEPATH "HIP compiler")
  endif()
  if(BUILD_DCU AND DCU_HIP_ARCH)
    set(CMAKE_HIP_ARCHITECTURES "${DCU_HIP_ARCH}" CACHE STRING "HIP offload architectures")
  endif()
  enable_language(HIP)

  if(BUILD_AMD)
    add_compile_definitions(MANS_ENABLE_AMD)
    target_compile_definitions(mans_common PRIVATE MANS_ENABLE_AMD)
  endif()
  if(BUILD_DCU)
    add_compile_definitions(MANS_ENABLE_DCU)
    target_compile_definitions(mans_common PRIVATE MANS_ENABLE_DCU)
  endif()

  if(BUILD_AMD)
    # Legacy standalone ADM experiments remain opt-in targets.
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
      add_custom_target(amd_adm_${fname} DEPENDS ${out_file})
    endforeach()
  endif()

  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/amd)
  if(BUILD_DCU)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/dcu)
  endif()
  add_subdirectory(amd/ans)

  add_library(mans_amd_core STATIC
    amd/mans_amd.cpp
    amd/ans/mans_amd_ans.cpp
  )
  set_source_files_properties(
    amd/mans_amd.cpp
    amd/ans/mans_amd_ans.cpp
    PROPERTIES LANGUAGE HIP)
  set_property(TARGET mans_amd_core PROPERTY POSITION_INDEPENDENT_CODE ON)
  target_include_directories(mans_amd_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/ans/include
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/ans/src
  )
  if(BUILD_DCU AND DCU_HIPCUB_INCLUDE_DIR)
    target_include_directories(amd_gpu_ans PUBLIC ${DCU_HIPCUB_INCLUDE_DIR})
    target_include_directories(mans_amd_core PUBLIC ${DCU_HIPCUB_INCLUDE_DIR})
  endif()
  target_compile_definitions(mans_amd_core PUBLIC MANS_ENABLE_HIP_BACKEND)
  if(BUILD_DCU AND DCU_HIP_ARCH)
    set_property(TARGET mans_amd_core PROPERTY HIP_ARCHITECTURES "${DCU_HIP_ARCH}")
  endif()
  target_link_libraries(mans_amd_core PUBLIC mans_cpu_core OpenMP::OpenMP_CXX amd_gpu_ans)
  target_link_libraries(mans_common PRIVATE mans_amd_core)

  set(HIP_MANS_RUNTIME_DIR ${CMAKE_BINARY_DIR}/bin/amd)
  set(HIP_MANS_PREFIX amd)
  set(HIP_MANS_BACKEND_DEFINE MANS_ENABLE_AMD)
  set(HIP_MANS_TEST_PREFIX mans_amd)
  if(BUILD_DCU)
    set(HIP_MANS_RUNTIME_DIR ${CMAKE_BINARY_DIR}/bin/dcu)
    set(HIP_MANS_PREFIX dcu)
    set(HIP_MANS_BACKEND_DEFINE MANS_ENABLE_DCU)
    set(HIP_MANS_TEST_PREFIX mans_dcu)
  endif()

  add_executable(${HIP_MANS_PREFIX}_mans_compress amd/amd_mans_compress.cpp)
  target_compile_definitions(${HIP_MANS_PREFIX}_mans_compress PRIVATE ${HIP_MANS_BACKEND_DEFINE})
  set_target_properties(${HIP_MANS_PREFIX}_mans_compress PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${HIP_MANS_RUNTIME_DIR})
  target_link_libraries(${HIP_MANS_PREFIX}_mans_compress PRIVATE mans_common)

  add_executable(${HIP_MANS_PREFIX}_mans_decompress amd/amd_mans_decompress.cpp)
  target_compile_definitions(${HIP_MANS_PREFIX}_mans_decompress PRIVATE ${HIP_MANS_BACKEND_DEFINE})
  set_target_properties(${HIP_MANS_PREFIX}_mans_decompress PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${HIP_MANS_RUNTIME_DIR})
  target_link_libraries(${HIP_MANS_PREFIX}_mans_decompress PRIVATE mans_common)

  if(BUILD_TESTING)
    add_executable(${HIP_MANS_TEST_PREFIX}_backend_test tests/mans_amd_backend_test.cpp)
    target_compile_definitions(${HIP_MANS_TEST_PREFIX}_backend_test PRIVATE ${HIP_MANS_BACKEND_DEFINE})
    set_target_properties(${HIP_MANS_TEST_PREFIX}_backend_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
    target_link_libraries(${HIP_MANS_TEST_PREFIX}_backend_test PRIVATE mans_common)
    add_test(NAME ${HIP_MANS_TEST_PREFIX}_backend_test COMMAND ${HIP_MANS_TEST_PREFIX}_backend_test)

    add_executable(${HIP_MANS_TEST_PREFIX}_device_test tests/mans_amd_device_test.cpp)
    target_compile_definitions(${HIP_MANS_TEST_PREFIX}_device_test PRIVATE ${HIP_MANS_BACKEND_DEFINE})
    set_source_files_properties(tests/mans_amd_device_test.cpp PROPERTIES LANGUAGE HIP)
    set_target_properties(${HIP_MANS_TEST_PREFIX}_device_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
    target_link_libraries(${HIP_MANS_TEST_PREFIX}_device_test PRIVATE mans_common)
    add_test(NAME ${HIP_MANS_TEST_PREFIX}_device_test COMMAND ${HIP_MANS_TEST_PREFIX}_device_test)
  endif()
endif()



if(BUILD_HDF5_PLUGIN)
  add_subdirectory(tools/H5Z-MANS)
endif()