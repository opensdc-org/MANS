# NVIDIA HAAC ADM benchmark

This directory is the NVIDIA (CUDA) counterpart of `amd/adm/haac_bench`. It
collects the same measurements on an NVIDIA A100 that the AMD bench collects on
an Instinct MI210, so the two can be compared side by side.

The ADM kernels are shared between the two benches: `src/*.cu` here are symlinks
to `amd/adm/haac_bench/src/*.cpp`, which are written as HIP/CUDA dual-backend
code (the `MANS_BENCH_CUDA` branch of `common.h` swaps the HIP runtime calls for
CUDA). The NV build therefore compiles the **identical ADM kernels** with `nvcc`
that the AMD build compiles with `hipcc`, which is the cleanest possible
cross-vendor comparison: same algorithm, same source, different compiler and
hardware.

The four stages are:

| executable | stage | arguments |
|---|---|---|
| `compress_fused` | fused compression | `<input.u2> <output.adm>` |
| `compress_unfused` | unfused compression (mapping + scan + concat) | `<input.u2> <output.adm>` |
| `decompress_fused` | fused decompression | `<input.adm> <original.u2>` |
| `decompress_unfused` | unfused decompression (decode + restore) | `<input.adm> <original.u2>` |
| `microbench` | synthetic GPU tests | `[elements] [trials] [inner_iterations]` |

Decompression is performed on the GPU (`decompress` for fused, `decode_signals`
+ `restore_values` for unfused), matching the AMD side. All kernel timings are
CUDA-event GPU times. Each stage warms up the timed path first (5 untimed
launches), then measures `timing_iters = 10` timed launches and reports the
arithmetic mean per kernel, so the reported `*_ms` / throughput numbers are
averaged GPU-kernel times, not single-shot measurements.

## Requirements

- Linux with an NVIDIA GPU supported by the installed CUDA toolkit.
- `nvcc` (CUDA toolkit) and, optionally, `ncu` (Nsight Compute) for cache
  counter profiling.
- HAAC `.u2` input files (raw, headerless, little-endian `uint16`), the same
  files used by the AMD bench.
- A compute capability accepted by the installed compiler (default `sm_80` on an
  A100).

No ANS stage is run. These results measure ADM only.

## Input format

Identical to the AMD bench: raw, headerless `.u2` files of little-endian
unsigned 16-bit values. Element count and geometry are not parsed from the file
name; every input is treated as a flat 1D array (`num_elements = bytes / 2`),
matching the AMD bench. The scripts process every `.u2` file directly under the
input directory and ignore files in nested directories.

## Build

From this directory:

```bash
NVCC=/path/to/nvcc CUDA_ARCH=sm_80 ./scripts/build.sh
```

`CUDA_ARCH` is optional when `nvcc` has a suitable default target; `NVCC` is
required when `nvcc` is not on `PATH`. Set `BUILD_DIR` to change the binary
directory:

```bash
NVCC=/path/to/nvcc CUDA_ARCH=sm_80 BUILD_DIR="$PWD/build" ./scripts/build.sh
```

The build uses `-std=c++14 -DMANS_BENCH_CUDA` to select the CUDA path of the
shared source. Five executables are produced under `build/`.

## Run all HAAC input sizes

Pass the directory containing HAAC `.u2` files. The script runs each input
through all four stages, selects the requested GPU (default 0), saves per-run
logs and generated `.adm` files, and writes a CSV summary:

```bash
GPU_ID=0 ./scripts/run.sh /path/to/testdata/u2/haac_vx
```

An optional second argument sets the result directory:

```bash
GPU_ID=0 ./scripts/run.sh /path/to/testdata/u2/haac_vx ./results/a100-run-1
```

The results directory contains `summary.csv` and `logs/`. Generated `.adm`
files are kept in a temporary directory and removed when the script exits. The
summary has one row per input per stage, with `throughput_gbps` for that stage
and `lossless_pass` filled for the decompression stages.

## Real-input stage analysis

`analyze_real.py` is the primary NVIDIA-side workflow, mirroring the AMD one. It
runs fused/unfused compression followed by the corresponding decompression on
identical real inputs, stores each full log, keeps generated `.adm` files in a
temporary directory for the duration of the run, and writes one row per input
per stage to `stage_timings.csv`:

```bash
GPU_ID=0 ./scripts/analyze_real.py /path/to/testdata/u2/haac_vx ./results/a100-real-run-1
```

The NVIDIA and AMD benches are run on the **same input set — the HAAC `vx`
series** (10 sizes, 4 kB to 1 GB) — so results are directly comparable. Point
both scripts at a directory containing only the `vx_*.u2` files (e.g. a
symlinked `haac_vx` directory):

```bash
GPU_ID=0 ./scripts/analyze_real.py /path/to/testdata/u2/haac_vx_vx ./results/a100-real-stage-<unique-run-name>
```

Choose a fresh result path for every run. The result directory must not exist.
The CSV fields, their meanings, and the `NA` conventions are **identical to the
AMD `stage_timings.csv`** (see `amd/adm/haac_bench/README.md` for the full
field-by-field description): `total_gpu_ms` is the sum of the applicable
CUDA-event measurements, `file_read_ms`/`h2d_ms`/`d2h_ms`/`file_write_ms`/
`cpu_verify_ms` are host steady-clock durations, and `total_wall_ms` is the
end-to-end host time. Because both benches share the same source and output
format, the two `stage_timings.csv` files can be diffed or joined directly.

## Synthetic GPU microbenchmark

`microbench` is a CUDA port of the AMD microbenchmark with identical integer,
bitwise, and device-memory-copy kernels:

```bash
./build/microbench 16777216 5 256 > ./results/microbench.csv
```

It prints one CSV row per trial with CUDA-event elapsed time and a nominal
operation rate (`Gop/s`) or copy bandwidth (`GB/s`). The kernels are identical
to the AMD side, so A100 and MI210 numbers are directly comparable as synthetic
hardware probes.

## Capture system information

```bash
./scripts/system_info.sh ./results/system-a100
```

Captures `metadata.txt`, `nvidia-smi -q`, `nvcc --version`, `ncu --version`,
the NVIDIA driver version, and the relevant environment into the output
directory. Choose a new, empty output directory for each snapshot.

## Optional L2 counter profiling

`profile.sh` profiles each stage with `ncu` (Nsight Compute), matching the AMD
`profile.sh` structure. It targets only the ADM kernel(s) of each stage
(fused compression `map_values_kernel_decoupled`; unfused compression
`map_values_kernel_thrust`; fused decompression `decompress`; unfused
decompression `decode_signals_uint16` and `restore_values_uint16`):

```bash
GPU_ID=0 NCU=/path/to/ncu ./scripts/profile.sh /path/to/testdata/u2/haac_vx
```

The default metric is `lts__t_sector_hit_rate.pct` (Ampere/sm_80). Override with
`NCU_METRICS` / `NCU_HIT_RATE_METRIC` for another architecture. Raw `ncu` output
is stored under `pmc/`. `ncu` requires profiling privileges and may be slow;
treat this as a best-effort, optional measurement.

> ⚠️ **This cluster's A100 nodes cannot run `ncu` as a normal user.** The GPU
> driver has `RmProfilingAdminOnly=1` set (see `/proc/driver/nvidia/params`), so
> `ncu` fails with `ERR_NVGPUCTRPERM` ("The user does not have permission to
> access NVIDIA GPU Performance Counters") and no kernel is profiled. Fixing it
> requires an admin to set `RmProfilingAdminOnly=0` (or grant `CAP_PERFMON`).
> Until then the L2 hit-rate / DRAM-traffic / occupancy counters are unavailable
> on NV — use the offline resource-usage method below instead.

## Offline kernel resource usage (cuobjdump)

`cuobjdump --dump-resource-usage` reads the compiled `.cubin` directly and needs
**no GPU and no profiling privileges**, so it works on this cluster as-is. It
reports per-kernel `REG` / `STACK` (register spill) / `SHARED` / `LOCAL` usage —
the NV analog of the AMD `rocprof` `VGPR_Count` / `Scratch_Size` /
`LDS_Block_Size` fields.

```bash
CUOBJ=/home/bingxing2/apps/cuda/11.7.0/bin/cuobjdump
for b in compress_fused compress_unfused decompress_fused decompress_unfused; do
    $CUOBJ --dump-resource-usage build/$b
done
```

Results are saved in two symmetric trees, one per build version:

- `profile/a100-optimized/` — current (optimized) build, binaries in `build/`.
- `profile/a100-preopt/` — pre-optimization build (commit `e073eda`), binaries
  in `build_preopt/`.

Each holds `summary.csv` (one row per kernel, split out per stage like the AMD
`profile/mi210-resource-usage/summary.csv`) and `cuobjdump_resource_usage.txt`
(raw dump). The kernels listed are the 5 ADM kernels
(`map_values_kernel_decoupled`, `map_values_kernel_thrust`, `decompress`,
`decode_signals_uint16`, `restore_values_uint16`) plus the `compress_unfused`
helpers `concat`, `lookback_scan_kernel`, `init_lookback_scan_state_kernel`.
Columns: `reg_per_thread` / `stack_bytes` (spill) / `shared_bytes` (LDS) /
`local_bytes` map to the AMD `arch_vgpr` / `scr` / `lds` fields; `wgr` is the
launch block size (the `concat` and `decompress` kernels went 32 → 256 in the
optimization). Runtime counters (`occupancy_pct`, `valu_util_pct`,
`mem_stall_pct`, `fetch_size`, `write_size` in the AMD summary) are unavailable
here because `ncu` is blocked on this cluster — see the note above.

## Comparison guidance

Because the NVIDIA and AMD benches share the same ADM source, the CUDA-event
`total_gpu_ms` on A100 and the HIP-event `total_gpu_ms` on MI210 measure the
same kernels and are directly comparable as GPU ADM time (modulo
compiler/runtime differences). The same holds for the per-stage fields
(`mapping_ms`, `prefix_sum_ms`, `concat_ms`, `decode_ms`, `restore_ms`,
`fused_kernel_ms`), `kernel_throughput_gbps`, and the `microbench` synthetic
rates. Host-side fields (`file_read_ms`, `h2d_ms`, `d2h_ms`, `file_write_ms`,
`cpu_verify_ms`, `total_wall_ms`) are also directly comparable, but they are
affected by the host CPU, PCIe, and filesystem as much as by the GPU, so record
those scopes separately when interpreting small differences.

## Reproducibility

The compiler version, CUDA release, GPU model, compute capability, selected GPU,
and input dataset can all change results. Record them alongside any transferred
benchmark results using `system_info.sh`. For comparable measurements keep the
same input files, `CUDA_ARCH`, GPU selection, and result-directory naming
convention. Run the benchmark more than once when evaluating small performance
differences, and retain both `stage_timings.csv` and the per-run logs.
