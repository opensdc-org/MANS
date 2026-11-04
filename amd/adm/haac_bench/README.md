# AMD HIP HAAC ADM benchmark

This directory contains the four U16 HIP benchmark variants used for the HAAC fused/unfused ADM comparison. It is self-contained apart from ROCm/HIP and the HAAC `.u2` inputs. The sources are preserved from the benchmark runs; each executable compresses the input, writes its ADM output, decompresses it, and checks the restored U16 values against the original input.

## Requirements

- Linux with an AMD GPU supported by the installed ROCm release.
- ROCm HIP compiler (`hipcc`) and `rocprofv3` for optional L2 counter collection.
- HAAC U2 input files, such as the `.u2` files from the MANS/SC-AE dataset archive.
- A GPU architecture accepted by the installed compiler (for example, `gfx90a` on MI210).

No ANS stage is run. These results measure ADM only.

## Build

From this directory:

```bash
GPU_ARCH=gfx90a ./scripts/build.sh
```

`GPU_ARCH` is optional when the compiler has a suitable default target. Set `HIPCC` if `hipcc` is not on `PATH`, and `BUILD_DIR` to change the binary directory:

```bash
HIPCC=/opt/rocm/bin/hipcc GPU_ARCH=gfx90a BUILD_DIR="$PWD/build" ./scripts/build.sh
```

Four binaries are produced under `build/`:

- `u16_fused_compress_fused_decompress`
- `u16_fused_compress_unfused_decompress`
- `u16_unfused_compress_fused_decompress`
- `u16_unfused_compress_unfused_decompress`

## Run all HAAC input sizes

Pass the directory containing HAAC `.u2` files. The script runs every matching file with all four variants, selects the requested GPU (default 0), saves per-run logs and generated ADM files, and writes a CSV summary:

```bash
GPU_ID=0 ./scripts/run.sh /path/to/testdata/u2/haac
```

An optional second argument sets the result directory:

```bash
GPU_ID=0 ./scripts/run.sh /path/to/testdata/u2/haac ./results/mi210-rocm633
```

The results directory contains `summary.csv`, `logs/`, and `compressed/`. Each log contains the original detailed benchmark output. The script stops if a run fails or the lossless check does not pass.

To run a single input and variant directly:

```bash
HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
  ./build/u16_fused_compress_unfused_decompress input.u2 output.adm
```

## Optional L2 counter profiling

`profile.sh` runs one input and one variant through `rocprofv3`, collecting `TCC_HIT` and `TCC_MISS`:

```bash
GPU_ID=0 ./scripts/profile.sh /path/to/input.u2 \
  u16_fused_compress_fused_decompress ./profile/mi210-rocm633
```

Profiler output is saved beneath the requested directory. For each target kernel, compute:

```text
L2 hit rate (%) = 100 * TCC_HIT / (TCC_HIT + TCC_MISS)
```

Filter counter rows by the target ADM kernel name. Compression variants execute different kernels, and unfused decompression dispatches `decode_signals_uint16` and `restore_values_uint16`; do not combine unrelated runtime or setup kernels into an ADM kernel's hit rate. Counter collection can change runtime, so use `run.sh` output, not profiler runs, for throughput comparisons.

## Measurement scope

The HIP event timings reported by these sources cover GPU ADM work, not file I/O, host-device copies, ANS, or the CPU lossless comparison. Fused decompression is timed as one kernel. Split decompression is timed across the signal-decode and value-restore kernels. Fused compression reports its fused mapping kernel; unfused compression reports mapping, prefix-sum, and concatenation separately and reports their sum as compression time. Thus the unfused compression total includes the GPU scan and concat work, not just a single kernel.

The compiler version, ROCm release, GPU model, and architecture can change results. Record them alongside any transferred benchmark results (for example, `hipcc --version`, `rocminfo`, and the GPU model).
