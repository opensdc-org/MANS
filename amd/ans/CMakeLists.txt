# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.
#set(CMAKE_MODULE_PATH "/usr/local/python3/lib/python3.8/site-packages/torch/cmake/modules" ${CMAKE_MODULE_PATH})

# AMD ANS is part of the top-level MANS project.  The HIP compiler/toolchain is
# selected by the top-level project so this directory remains usable with ROCm
# and vendor HIP toolchains alike.

add_library(amd_gpu_ans SHARED
  src/GpuANS.cpp
)
set_source_files_properties(src/GpuANS.cpp PROPERTIES LANGUAGE HIP)
set_target_properties(amd_gpu_ans PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_compile_features(amd_gpu_ans PUBLIC cxx_std_17)
if(CMAKE_HIP_ARCHITECTURES)
  set_property(TARGET amd_gpu_ans PROPERTY HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
endif()
target_include_directories(amd_gpu_ans PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(hipans_compress examples/compress.cpp)
set_source_files_properties(examples/compress.cpp PROPERTIES LANGUAGE HIP)
target_link_libraries(hipans_compress PRIVATE amd_gpu_ans)

target_compile_features(hipans_compress PRIVATE cxx_std_17)

target_include_directories(hipans_compress PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(hipans_decompress examples/decompress.cpp)
set_source_files_properties(examples/decompress.cpp PROPERTIES LANGUAGE HIP)
target_link_libraries(hipans_decompress PRIVATE amd_gpu_ans)
target_compile_features(hipans_decompress PRIVATE cxx_std_17)
target_include_directories(hipans_decompress PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
        