cmake_minimum_required(VERSION 3.15)
find_package(HDF5 REQUIRED COMPONENTS C HL)
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
target_link_libraries(H5Z-MANS PRIVATE mans_common ${HDF5_LIBRARIES})
set_target_properties(H5Z-MANS PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/plugins
    OUTPUT_NAME "H5Z-MANS"
    PREFIX "lib"
)


# =========================================================
# Target 2: H5Z-MANS_test
# =========================================================
add_executable(H5Z-MANS_test test/H5Z-MANS_test.cpp)
target_link_libraries(H5Z-MANS_test PRIVATE ${HDF5_LIBRARIES})
set_target_properties(H5Z-MANS_test PROPERTIES
    # output to build/bin/tools
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/h5z-mans
)