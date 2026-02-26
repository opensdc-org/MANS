#include <stdio.h>
#include <stdlib.h>

#include "fse.h"

static unsigned char* read_file(const char* path, size_t* size_out)
{
    FILE* f = fopen(path, "rb");
    unsigned char* buf;
    long sz;

    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }

    sz = ftell(f);
    if (sz < 0) {
        fclose(f);
        return NULL;
    }

    if (fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return NULL;
    }

    buf = (unsigned char*)malloc((size_t)sz ? (size_t)sz : 1);
    if (!buf) {
        fclose(f);
        return NULL;
    }

    if (sz > 0 && fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf);
        fclose(f);
        return NULL;
    }

    fclose(f);
    *size_out = (size_t)sz;
    return buf;
}

static int write_file(const char* path, const void* data, size_t size)
{
    FILE* f = fopen(path, "wb");
    if (!f) return 0;
    if (size > 0 && fwrite(data, 1, size, f) != size) {
        fclose(f);
        return 0;
    }
    fclose(f);
    return 1;
}

int main(int argc, char** argv)
{
    const char* input_path;
    const char* output_path;
    unsigned char* src;
    unsigned char* dst;
    size_t src_size;
    size_t dst_capacity;
    size_t csize;

    if (argc != 3) {
        fprintf(stderr, "usage: %s <input_file> <output_file>\n", argv[0]);
        return 1;
    }

    input_path = argv[1];
    output_path = argv[2];

    src = read_file(input_path, &src_size);
    if (!src) {
        fprintf(stderr, "error: failed to read input: %s\n", input_path);
        return 1;
    }

    dst_capacity = FSE_compressBound(src_size);
    if (dst_capacity == 0) {
        fprintf(stderr, "error: FSE_compressBound overflow for input size %zu\n", src_size);
        free(src);
        return 1;
    }

    dst = (unsigned char*)malloc(dst_capacity);
    if (!dst) {
        fprintf(stderr, "error: out of memory\n");
        free(src);
        return 1;
    }

    csize = FSE_compress(dst, dst_capacity, src, src_size);
    if (FSE_isError(csize)) {
        fprintf(stderr, "error: FSE_compress failed: %s\n", FSE_getErrorName(csize));
        free(src);
        free(dst);
        return 1;
    }

    if (!write_file(output_path, dst, csize)) {
        fprintf(stderr, "error: failed to write output: %s\n", output_path);
        free(src);
        free(dst);
        return 1;
    }

    printf("input:      %s\n", input_path);
    printf("output:     %s\n", output_path);
    printf("raw bytes:  %zu\n", src_size);
    printf("compressed: %zu\n", csize);
    printf("raw/compressed: %.6f\n", csize ? (double)src_size / (double)csize : 0.0);
    printf("compressed/raw: %.6f\n", src_size ? (double)csize / (double)src_size : 0.0);

    free(src);
    free(dst);
    return 0;
}
