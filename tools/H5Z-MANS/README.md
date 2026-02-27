# H5Z-MANS: HDF5 Filter Plugin

`H5Z-MANS` is an HDF5 filter plugin for MANS compression.

## Features

- Shared-library plugin loadable via `HDF5_PLUGIN_PATH`
- MANS filter (`id=32001`) and pass-through NONE filter (`id=32002`)
- Benchmark/test executables for MANS/ZSTD/SZ3/GZIP/NONE and MPI read/write cases
- Supports unsigned integer datasets (`u16` / `u32` in MANS path)

## Build

Build from repo root with CPU backend enabled (plugin uses CPU MANS path):

```bash
cmake -S . -B build -DTARGET_PLATFORM=cpu -DBUILD_HDF5_PLUGIN=ON
cmake --build build -j
```

If you need GPU binaries too, use `TARGET_PLATFORM=cpu_nv`, `cpu_amd`, or `all`.

Main artifacts:
- `build/bin/plugins/libH5Z-MANS.so`
- `build/bin/plugins/libH5Z-NONE.so`
- `build/bin/h5z-mans/H5Z-MANS_test`
- `build/bin/h5z-mans/mans_data_gen`
- `build/bin/h5z-mans/mans_write`, `mans_read`
- `build/bin/h5z-mans/none_write`, `none_read`
- `build/bin/h5z-mans/sz3_write`, `sz3_read`
- `build/bin/h5z-mans/zstd_write`, `zstd_read`
- `build/bin/h5z-mans/fse_write`, `fse_read`
- `build/bin/h5z-mans/fse_ans_write`, `fse_ans_read`
- `build/bin/h5z-mans/fse_huffman_write`, `fse_huffman_read`

## Runtime Setup

Set plugin search path before running HDF5 tools:

```bash
export HDF5_PLUGIN_PATH=/workspace/MANS/build/bin/plugins
```

If combining with external plugins (for example SZ3), append additional paths:

```bash
export HDF5_PLUGIN_PATH="/workspace/SZ3/build/tools/H5Z-SZ3:/workspace/MANS/build/bin/plugins"
```

## Filter IDs Used in Test Tools

- `MANS`: `32001`
- `NONE`: `32002`
- `FSE`: `32028`
- `ZSTD`: `32015`
- `SZ3`: `32024`
- `GZIP/DEFLATE`: `1` (built-in HDF5)

Notes:
- `MANS`/`NONE` are provided by this repo (`libH5Z-MANS.so`, `libH5Z-NONE.so`).
- `FSE` id `32028` is used by `fse_*` test tools and requires FSE plugin availability in `HDF5_PLUGIN_PATH` (for example `libH5Zfse.so`).

## Quick Usage

### 1) Unified benchmark test

```bash
./build/bin/h5z-mans/H5Z-MANS_test \
  [--dataset-mb N] \
  [--chunk-mb N] \
  [--filter mans|zstd|sz3|gzip|none] \
  [--threads v1,v2,v3,v4,v5,v6,v7,v8] \
  [--output file.h5]
```

Notes:
- `--threads` is only used when `--filter mans`.
- default output timing CSV is `hdf5_timing.csv`.

### 2) Synthetic data generator

```bash
./build/bin/h5z-mans/mans_data_gen \
  [--config gen.cfg] [--synth-config synth.cfg] \
  [--output-dir DIR] [--output-name name.bin] [--output-prefix NAME] \
  [--size-per-rank-mb MB] [--ranks N] [--jobs N] \
  [--ratio-constant R] [--dtype u16|u32] [--adm-threshold N]
```

### 3) MPI write/read micro-tests

Writer binaries accept optional chunk size:

```bash
mpirun -n 1 ./build/bin/h5z-mans/mans_write --chunk-size-mb 4
mpirun -n 1 ./build/bin/h5z-mans/mans_read
```

Same `--chunk-size-mb` pattern applies to:
- `none_write`, `sz3_write`, `zstd_write`, `fse_write`, `fse_ans_write`, `fse_huffman_write`

## Config File for MANS Parameters

See `tools/H5Z-MANS/example.conf` for `MansParams` fields serialized into `cd_values`.

## Related

Top-level workflow and autotune guidance: [README.md](../../README.md)
