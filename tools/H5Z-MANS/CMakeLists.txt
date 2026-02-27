cmake_minimum_required(VERSION 3.15)
if(DEFINED HDF5_PREFER_PARALLEL)
    set(HDF5_PREFER_PARALLEL ${HDF5_PREFER_PARALLEL})
endif()
find_package(HDF5 REQUIRED COMPONENTS C HL)
if(HDF5_PREFER_PARALLEL)
    find_package(MPI REQUIRED)
    if(NOT HDF5_IS_PARALLEL)
        message(WARNING "HDF5_PREFER_PARALLEL=ON but the found HDF5 is serial. Set HDF5_ROOT/CMAKE_PREFIX_PATH to a parallel HDF5 install if needed.")
    endif()
endif()
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}
    ${HDF5_INCLUDE_DIRS}
)

# =========================================================
# Target 1: H5Z-MANS (Filter Plugin shared library)
# =========================================================
# Build a shared library named libH5Z-MANS.so; HDF5 will automatically load it at runtime.
add_library(H5Z-MANS SHARED src/H5Z-MANS.cpp)
target_link_libraries(H5Z-MANS PRIVATE mans_common ${HDF5_LIBRARIES} $<$<BOOL:${HDF5_PREFER_PARALLEL}>:MPI::MPI_C>)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_link_options(H5Z-MANS PRIVATE -Wl,-Bsymbolic)
endif()
set_target_properties(H5Z-MANS PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/plugins
    OUTPUT_NAME "H5Z-MANS"
    PREFIX "lib"
)

# =========================================================
# Target 1b: H5Z-NONE (Pass-through Filter Plugin)
# =========================================================
add_library(H5Z-NONE SHARED src/H5Z-NONE.cpp)
target_link_libraries(H5Z-NONE PRIVATE ${HDF5_LIBRARIES} $<$<BOOL:${HDF5_PREFER_PARALLEL}>:MPI::MPI_C>)
set_target_properties(H5Z-NONE PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/plugins
    OUTPUT_NAME "H5Z-NONE"
    PREFIX "lib"
)


# =========================================================
# Target 2: H5Z-MANS_test
# =========================================================
add_executable(H5Z-MANS_test test/H5Z-MANS_test.cpp)
target_link_libraries(H5Z-MANS_test PRIVATE ${HDF5_LIBRARIES} $<$<BOOL:${HDF5_PREFER_PARALLEL}>:MPI::MPI_C>)
set_target_properties(H5Z-MANS_test PROPERTIES
    # output to build/bin/tools
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/h5z-mans
)

# =========================================================
# Target 3: mans_data_gen (synthetic raw data generator)
# =========================================================
add_executable(mans_data_gen test/mans_data_gen_cli.cpp)
target_link_libraries(mans_data_gen PRIVATE ${HDF5_LIBRARIES} $<$<BOOL:${HDF5_PREFER_PARALLEL}>:MPI::MPI_C>)
set_target_properties(mans_data_gen PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/h5z-mans
)

# =========================================================
# Target 4: mans_write (minimal MPI independent compressed write test)
# =========================================================
add_executable(mans_write test/mans_write.cpp)
if(NOT TARGET MPI::MPI_C)
    find_package(MPI REQUIRED)
endif()
target_link_libraries(mans_write PRIVATE ${HDF5_LIBRARIES} MPI::MPI_C)
set_target_properties(mans_write PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/h5z-mans
)

# =========================================================
# Target 5: mans_read (minimal MPI compressed read test)
# =========================================================
add_executable(mans_read test/mans_read.cpp)
target_link_libraries(mans_read PRIVATE ${HDF5_LIBRARIES} MPI::MPI_C)
set_target_properties(mans_read PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/h5z-mans
)

# =========================================================
# Target 5b: mans_original_write (minimal MPI compressed write test)
# =========================================================
add_executable(mans_original_write test/mans_original_write.cpp)
target_link_libraries(mans_original_write PRIVATE ${HDF5_LIBRARIES} MPI::MPI_C)
set_target_properties(mans_original_write PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/h5z-mans
)

# =========================================================
# Target 5c: mans_original_read (minimal MPI compressed read test)
# =========================================================
add_executable(mans_original_read test/mans_original_read.cpp)
target_link_libraries(mans_original_read PRIVATE ${HDF5_LIBRARIES} MPI::MPI_C)
set_target_properties(mans_original_read PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/h5z-mans
)

# =========================================================
# Target 6: none_write (minimal MPI pass-through write test)
# =========================================================
add_executable(none_write test/none_write.cpp)
target_link_libraries(none_write PRIVATE ${HDF5_LIBRARIES} MPI::MPI_C)
set_target_properties(none_write PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/h5z-mans
)

# =========================================================
# Target 7: none_read (minimal MPI pass-through read test)
# =========================================================
add_executable(none_read test/none_read.cpp)
target_link_libraries(none_read PRIVATE ${HDF5_LIBRARIES} MPI::MPI_C)
set_target_properties(none_read PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/h5z-mans
)

# =========================================================
# Target 8: sz3_write (minimal MPI SZ3 compressed write test)
# =========================================================
add_executable(sz3_write test/sz3_write.cpp)
target_link_libraries(sz3_write PRIVATE ${HDF5_LIBRARIES} MPI::MPI_C)
set_target_properties(sz3_write PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/h5z-mans
)

# =========================================================
# Target 9: sz3_read (minimal MPI SZ3 compressed read test)
# =========================================================
add_executable(sz3_read test/sz3_read.cpp)
target_link_libraries(sz3_read PRIVATE ${HDF5_LIBRARIES} MPI::MPI_C)
set_target_properties(sz3_read PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/h5z-mans
)

# =========================================================
# Target 10: zstd_write (minimal MPI Zstd compressed write test)
# =========================================================
add_executable(zstd_write test/zstd_write.cpp)
target_link_libraries(zstd_write PRIVATE ${HDF5_LIBRARIES} MPI::MPI_C)
set_target_properties(zstd_write PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/h5z-mans
)

# =========================================================
# Target 11: zstd_read (minimal MPI Zstd compressed read test)
# =========================================================
add_executable(zstd_read test/zstd_read.cpp)
target_link_libraries(zstd_read PRIVATE ${HDF5_LIBRARIES} MPI::MPI_C)
set_target_properties(zstd_read PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/h5z-mans
)

# =========================================================
# Target 12: fse_write (minimal MPI FSE compressed write test)
# =========================================================
add_executable(fse_write test/fse_write.cpp)
target_link_libraries(fse_write PRIVATE ${HDF5_LIBRARIES} MPI::MPI_C)
set_target_properties(fse_write PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/h5z-mans
)

# =========================================================
# Target 13: fse_read (minimal MPI FSE compressed read test)
# =========================================================
add_executable(fse_read test/fse_read.cpp)
target_link_libraries(fse_read PRIVATE ${HDF5_LIBRARIES} MPI::MPI_C)
set_target_properties(fse_read PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/h5z-mans
)

# =========================================================
# Target 14: fse_ans_write (minimal MPI FSE-ANS compressed write test)
# =========================================================
add_executable(fse_ans_write test/fse_ans_write.cpp)
target_link_libraries(fse_ans_write PRIVATE ${HDF5_LIBRARIES} MPI::MPI_C dl)
set_target_properties(fse_ans_write PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/h5z-mans
)

# =========================================================
# Target 15: fse_ans_read (minimal MPI FSE-ANS compressed read test)
# =========================================================
add_executable(fse_ans_read test/fse_ans_read.cpp)
target_link_libraries(fse_ans_read PRIVATE ${HDF5_LIBRARIES} MPI::MPI_C dl)
set_target_properties(fse_ans_read PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/h5z-mans
)

# =========================================================
# Target 16: fse_huffman_write (minimal MPI FSE-Huffman compressed write test)
# =========================================================
add_executable(fse_huffman_write test/fse_huffman_write.cpp)
target_link_libraries(fse_huffman_write PRIVATE ${HDF5_LIBRARIES} MPI::MPI_C dl)
set_target_properties(fse_huffman_write PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/h5z-mans
)

# =========================================================
# Target 17: fse_huffman_read (minimal MPI FSE-Huffman compressed read test)
# =========================================================
add_executable(fse_huffman_read test/fse_huffman_read.cpp)
target_link_libraries(fse_huffman_read PRIVATE ${HDF5_LIBRARIES} MPI::MPI_C dl)
set_target_properties(fse_huffman_read PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/h5z-mans
)
