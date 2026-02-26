cmake_minimum_required(VERSION 3.16)

project(minimal_fse C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

add_library(fse_core STATIC
    src/debug.c
    src/entropy_common.c
    src/fse_compress.c
    src/fse_decompress.c
    src/hist.c
)
set_property(TARGET fse_core PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(fse_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(fse_core PRIVATE
        -O3
        -Wall
        -Wextra
        -Wcast-qual
        -Wcast-align
        -Wshadow
        -Wstrict-aliasing=1
        -Wswitch-enum
        -Wstrict-prototypes
        -Wundef
    )
endif()

option(FSE_BUILD_BENCH "Build FSE bench utility" OFF)
if(FSE_BUILD_BENCH)
    add_executable(fse_bench
        bench/fse_bench.c
    )
    target_link_libraries(fse_bench PRIVATE fse_core)
    target_include_directories(fse_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
endif()
