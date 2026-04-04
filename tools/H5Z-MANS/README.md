
# H5Z-MANS: HDF5 Filter Plugin

`H5Z-MANS` is an HDF5 filter plugin for MANS compression.

## Overview

This repository provides:

- An HDF5 shared-library filter plugin loadable via `HDF5_PLUGIN_PATH`
- The MANS filter (`id = 32032`)
- Benchmark and test executables for MANS, ZSTD, SZ3, GZIP, FSE, and uncompressed I/O
- MPI-based HDF5 read/write microbenchmarks
- Support for unsigned integer datasets (`u16` and `u32` in the MANS path)

## Filter IDs

The following filter IDs are used by the test tools:

- `MANS`: `32032`
- `FSE`: `32028`
- `ZSTD`: `32015`
- `SZ3`: `32024`
- `GZIP/DEFLATE`: `1` (built-in HDF5)

Notes:

- `MANS` is provided by this repository through `libH5Z-MANS.so`.
- `FSE` requires the corresponding FSE plugin to be available in `HDF5_PLUGIN_PATH` (for example, `libH5Zfse.so`).

## Build

Build from the repository root with the CPU backend enabled:

```bash
cmake -S . -B build -DTARGET_PLATFORM=cpu -DBUILD_HDF5_PLUGIN=ON
cmake --build build -j
```

To also build GPU-related binaries, set `TARGET_PLATFORM` to one of:

* `cpu_nv`
* `cpu_amd`
* `all`

### Main Artifacts

* `build/bin/plugins/libH5Z-MANS.so`
* `build/bin/plugins/libH5Z-NONE.so`
* `build/bin/h5z-mans/mans_data_gen`
* `build/bin/h5z-mans/mans_write`, `mans_read`
* `build/bin/h5z-mans/none_write`, `none_read`
* `build/bin/h5z-mans/sz3_write`, `sz3_read`
* `build/bin/h5z-mans/zstd_write`, `zstd_read`
* `build/bin/h5z-mans/fse_write`, `fse_read`
* `build/bin/h5z-mans/fse_ans_write`, `fse_ans_read`
* `build/bin/h5z-mans/fse_huffman_write`, `fse_huffman_read`



## Autotuning

`H5Z-MANS` does not expose thread counts through the public `cd_values` interface. Instead, thread counts are selected automatically in `set_local` based on chunk size and dataset mapping dimensions, using an autotuned CSV file.

### Generate the thread table

```bash
./build/bin/cpu/cpu_mans_autotune \
  --data-size-mb-list 0.00390625,0.0078125,1,4 \
  --csv ./build/thread_sweep.csv \
  --out ./build/best_threads.csv
```
### Fallback Behavior

* If `MANS_THREAD_CSV` is unset, the plugin looks for `best_threads.csv` in the current working directory.
* If no CSV is found, or no matching entry exists, the plugin prints a warning once and falls back to the default thread setting: `32,32`.

Notes:

* `cpu_mans_autotune` currently generates synthetic `u16` data internally.
* For the broader CPU autotuning and benchmarking workflow, see the top-level `README.md`.

## Quick Start

### Synthetic Data Generation

```bash
./build/bin/h5z-mans/mans_data_gen \
  [--config gen.cfg] [--synth-config synth.cfg] \
  [--output-dir DIR] [--output-name name.bin] [--output-prefix NAME] \
  [--size-per-rank-mb MB] [--ranks N] [--jobs N] \
  [--ratio-constant R] [--dtype u16|u32]
```

### MPI Write/Read Microbenchmarks

Before running:

```bash
export HDF5_PLUGIN_PATH=/workspace/MANS/build/bin/plugins
export MANS_THREAD_CSV=/workspace/MANS/build/best_threads.csv
```

Example:

```bash
mpirun -n 1 ./build/bin/h5z-mans/mans_write --chunk-size-mb 4
mpirun -n 1 ./build/bin/h5z-mans/mans_read
```

The same `--chunk-size-mb` option is supported by:

* `none_write`
* `sz3_write`
* `zstd_write`
* `fse_write`
* `fse_ans_write`
* `fse_huffman_write`

## Filter Configuration

See `tools/H5Z-MANS/example.conf` for the public MANS filter parameter file.

The public `cd_values` interface currently exposes only `mode`. Datatype, chunk geometry, and thread selection are derived internally in `set_local`.

## Related

For the full workflow and additional autotuning guidance, see the top-level [README.md](../../README.md).

```
