# Minimal FSE (CPU)

This directory is a minimal FSE subset used by MANS CPU mode.

Included:
- FSE core compression/decompression implementation
- Required dependencies (`bitstream/hist/error/mem/debug`)
- Optional minimal bench program: `fse_bench`

Not included:
- Large original CLI/tooling set (`programs/`, fullbench, fuzzer, probagen, etc.)

## Build Inside Main Project

When building MANS with CPU enabled, `fse_core` is built automatically:

```bash
cmake -S . -B build -DTARGET_PLATFORM=cpu
cmake --build build -j
```

## Standalone Build (This Subdirectory Only)

`FSE_BUILD_BENCH` is `OFF` by default. Turn it on if you want `fse_bench`.

```bash
cmake -S cpu/fse -B build-fse -DFSE_BUILD_BENCH=ON
cmake --build build-fse -j
```

## Run Bench

```bash
./build-fse/fse_bench /path/to/input.bin /path/to/output.mfse
```

The bench prints raw/compressed size and ratios.

## Directory Layout

- `include/`: FSE headers
- `src/`: FSE core source
- `bench/fse_bench.c`: minimal file-to-file compress utility
