# CPU PANS (Parallel ANS)

This directory contains the CPU PANS implementation used by MANS.

It operates on byte streams (`uint8_t`) and is typically used as the second stage after ADM.

## Build

Build from repo root (PANS targets are built when CPU platform is enabled):

```bash
cmake -S . -B build -DTARGET_PLATFORM=cpu
cmake --build build -j
```

Generated binaries:
- `build/bin/cpu/pans/cpuans_compress`
- `build/bin/cpu/pans/cpuans_decompress`
- `build/bin/cpu/pans/cpuans_bench_chunked`

## Usage

### Compress

```bash
./build/bin/cpu/pans/cpuans_compress <input.file> <output.file>
```

### Decompress

```bash
./build/bin/cpu/pans/cpuans_decompress <input.file> <output.file>
```

### Chunk Benchmark

```bash
./build/bin/cpu/pans/cpuans_bench_chunked <input.bin> [--chunks 0.125,0.25,0.5,1,2,8,256] [--csv out.csv]
```

Output CSV columns:
- `chunk_label`
- `chunk_bytes`
- `ratio_pct`
- `comp_mbps`
- `decomp_mbps`

## Notes

- Input is treated as raw bytes, not typed integers.
- OpenMP thread behavior depends on runtime environment (for example `OMP_NUM_THREADS`).
- For end-to-end MANS autotuned workflow, see top-level [README.md](../../README.md).
