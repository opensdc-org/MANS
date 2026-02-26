#include <H5PLextern.h>
#include <hdf5.h>

#include <cstdlib>
#include <cstring>

// =========================================================
// H5Z-NONE: pass-through filter (no compression/decompression)
// =========================================================

// Define a unique Filter ID
#define H5Z_FILTER_NONE_ID 32002

static htri_t H5Z_can_apply_none(hid_t dcpl_id, hid_t type_id, hid_t space_id)
{
    (void)dcpl_id;
    (void)type_id;
    (void)space_id;
    return 1;
}

static herr_t H5Z_set_local_none(hid_t dcpl_id, hid_t type_id, hid_t space_id)
{
    (void)dcpl_id;
    (void)type_id;
    (void)space_id;
    return 0;
}

static size_t H5Z_filter_none(unsigned int flags, size_t cd_nelmts,
                              const unsigned int cd_values[], size_t nbytes,
                              size_t *buf_size, void **buf)
{
    (void)flags;
    (void)cd_nelmts;
    (void)cd_values;

    if (!buf || !*buf || nbytes == 0) {
        return 0;
    }

    void* dst_buf = std::malloc(nbytes);
    if (!dst_buf) {
        return 0;
    }
    std::memcpy(dst_buf, *buf, nbytes);
    std::free(*buf);
    *buf = dst_buf;
    if (buf_size) {
        *buf_size = nbytes;
    }
    return nbytes;
}
// static size_t H5Z_filter_none(unsigned int flags, size_t cd_nelmts,
//     const unsigned int cd_values[], size_t nbytes,
//     size_t *buf_size, void **buf)
// {
// (void)flags;
// (void)cd_nelmts;
// (void)cd_values;
// (void)buf_size;

// // HDF5 约定：返回 0 表示失败；所以只在真正错误时返回 0
// if (!buf || !*buf) {
// return 0;
// }

// // no-op：不改变数据、不改变指针、不改变大小
// return nbytes;
// }

// =========================================================
// HDF5 plugin registration structure
// =========================================================
const H5Z_class2_t H5Z_NONE_CLASS[1] = {{
    H5Z_CLASS_T_VERS,
    H5Z_FILTER_NONE_ID,
    1,
    1,
    "H5Z-NONE",
    H5Z_can_apply_none,
    H5Z_set_local_none,
    H5Z_filter_none,
}};

H5PL_type_t H5PLget_plugin_type(void) {
    return H5PL_TYPE_FILTER;
}
const void *H5PLget_plugin_info(void) {
    return H5Z_NONE_CLASS;
}
