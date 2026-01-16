#include <vector>
#include <cstdlib> // for std::malloc, std::free
#include <cstring> // for std::memcpy
#include <new>     // for std::bad_alloc
#include <iostream>
#include <H5PLextern.h>

#include "mans_api.hpp"

// Define the Filter ID
#define H5Z_FILTER_MANS_ID 32001

// =========================================================
// Helper: Safe Malloc
// =========================================================
static void* safe_malloc(size_t size) {
    void* ptr = std::malloc(size);
    if (!ptr) {
        std::cerr << "[H5Z-MANS Error] Memory allocation failed for size: " << size << "\n";
    }
    return ptr;
}

// =========================================================
// Type check callback (can_apply)
// =========================================================
static htri_t H5Z_can_apply_mans(hid_t dcpl_id, hid_t type_id, hid_t space_id)
{
    if (H5Tget_class(type_id) != H5T_INTEGER) {
        std::cerr << "[H5Z-MANS Warning] Datatype is not INTEGER.\n";
        return 0;
    }
    H5T_sign_t sign = H5Tget_sign(type_id);
    if (sign != H5T_SGN_NONE) {
        std::cerr << "[H5Z-MANS Warning] Datatype must be Unsigned (UINT).\n";
        return 0;
    }
    size_t size = H5Tget_size(type_id);
    if (size != 2 && size != 4) {
        std::cerr << "[H5Z-MANS Warning] Only 2-byte (U16) or 4-byte (U32) supported. Current: " << size << "\n";
        return 0;
    }
    return 1;
}

// =========================================================
// Filter callback function: H5Z_filter_mans
// =========================================================
static size_t H5Z_filter_mans(unsigned int flags, size_t cd_nelmts,
                              const unsigned int cd_values[], size_t nbytes,
                              size_t *buf_size, void **buf)
{
    size_t required_params = sizeof(mans::MansParams) / sizeof(unsigned int);

    if (cd_nelmts < required_params) {
        std::cerr << "[H5Z-MANS Error] Filter parameter count (" << cd_nelmts
                  << ") must be at least " << required_params << " (MansParams).\n";
        return 0;
    }
    const mans::MansParams* params = reinterpret_cast<const mans::MansParams*>(cd_values);

    // Destination buffer pointer and its capacity
    void* dst_buf = nullptr;
    size_t dst_capacity = 0;
    size_t out_len = 0; // Actual size produced

    try {
        if (flags & H5Z_FLAG_REVERSE) {
            // ============================
            // Decompress Path
            // ============================
            dst_capacity = nbytes * 10;
            if (dst_capacity < 1024) dst_capacity = 1024; 
            
            dst_buf = safe_malloc(dst_capacity);
            if (!dst_buf) return 0;

            // Pass capacity via out_len variable
            out_len = dst_capacity; 

            // Call decompress API (no vector)
            mans::decompress(*buf, nbytes, *params, static_cast<uint8_t*>(dst_buf), out_len);

        } else {
            // ============================
            // Compress Path
            // ============================
            
            // Check data alignment/size validity
            size_t elem_size = (params->dtype == mans::DataType::U16) ? 2 : 4;
            if (nbytes % elem_size != 0) {
                std::cerr << "[H5Z-MANS Error] Input buffer size (" << nbytes
                          << ") is not a multiple of element size (" << elem_size
                          << "). dtype=" << (params->dtype == mans::DataType::U16 ? "U16" : "U32")
                          << "\n";
                return 0;
            }
            size_t num_elements = nbytes / elem_size;

            // Allocation: Input size + Overhead (4KB safe margin)
            dst_capacity = nbytes*2 + 1024;
            dst_buf = safe_malloc(dst_capacity);
            if (!dst_buf) return 0;

            out_len = dst_capacity;

            // Call compress API (no vector)
            mans::compress(*buf, num_elements, *params, static_cast<uint8_t*>(dst_buf), out_len);
        }

        // ==========================================
        // HDF5 Memory Replacement
        // ==========================================
        // 1. Free the input buffer provided by HDF5
        if (*buf) {
            std::free(*buf);
        }

        // 2. Point HDF5 buffer to our new buffer
        *buf = dst_buf;
        
        // 3. Update the capacity tracking
        *buf_size = dst_capacity;

        // 4. Return actual used bytes
        return out_len;

    } catch (const std::exception& e) {
        std::cerr << "[H5Z-MANS Error]: " << e.what() << "\n";
        if (dst_buf) std::free(dst_buf); // Clean up our allocation on error
        return 0;
    } catch (...) {
        std::cerr << "[H5Z-MANS Error]: Unknown exception occurred.\n";
        if (dst_buf) std::free(dst_buf);
        return 0;
    }
}

// =========================================================
// HDF5 plugin registration structure
// =========================================================
const H5Z_class2_t H5Z_MANS_CLASS[1] = {{
    H5Z_CLASS_T_VERS,       
    H5Z_FILTER_MANS_ID,     
    1,                      
    1,                      
    "H5Z-MANS",             
    H5Z_can_apply_mans,     
    NULL,                   
    H5Z_filter_mans,        
}};

H5PL_type_t H5PLget_plugin_type(void) {
    return H5PL_TYPE_FILTER;
}
const void *H5PLget_plugin_info(void) {
    return H5Z_MANS_CLASS;
}