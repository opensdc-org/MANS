# AMD HIP HAAC ADM benchmark

This directory contains the four U16 HIP benchmark executables used for the HAAC fused/unfused ADM comparison. It is self-contained apart from ROCm/HIP and the HAAC `.u2` inputs. The sources are preserved from the benchmark runs; each executable performs exactly one stage (fused or unfused compression, or fused or unfused decompression) so each kernel can be timed and profiled in isolation.

## Requirements

- Linux with an AMD GPU supported by the installed ROCm release.
- ROCm HIP compiler (`hipcc`) and `rocprofv3` for optional L2 counter collection.
- HAAC U2 input files, such as the `.u2` files from the MANS/SC-AE dataset archive.
- A GPU architecture accepted by the installed compiler (for example, `gfx1100` on the Radeon RX 7900 XTX).

No ANS stage is run. These results measure ADM only.

## Build

From this directory:

```bash
HIPCC=/opt/rocm-6.3.3/bin/hipcc GPU_ARCH=gfx1100 ./scripts/build.sh
```

`GPU_ARCH` is optional when the compiler has a suitable default target. `HIPCC` is required when `hipcc` is not on `PATH`. Set `BUILD_DIR` to change the binary directory:

```bash
HIPCC=/opt/rocm-6.3.3/bin/hipcc GPU_ARCH=gfx1100 BUILD_DIR="$PWD/build" ./scripts/build.sh
```

If the ROCm device-library bitcode is not found at `<rocm>/lib/amdgcn/bitcode` (some 6.3.3 installs place it at `<rocm>/amdgcn/bitcode`), the script detects the alternate location automatically; it can also be forced with `ROCM_DEVICE_LIB_PATH`.

Four executables are produced under `build/`:

| executable          | stage                | arguments                    |
|---------------------|----------------------|------------------------------|
| `compress_fused`    | fused compression    | `<input.u2> <output.adm>`    |
| `compress_unfused`  | unfused compression  | `<input.u2> <output.adm>`    |
| `decompress_fused`  | fused decompression  | `<input.adm> <original.u2>`  |
| `decompress_unfused`| unfused decompression| `<input.adm> <original.u2>`  |

## Run all HAAC input sizes

Pass the directory containing HAAC `.u2` files. The script runs each input through all four stages, selects the requested GPU (default 0), saves per-run logs and generated ADM files, and writes a CSV summary:

```bash
GPU_ID=0 ./scripts/run.sh /path/to/testdata/u2/haac
```

An optional second argument sets the result directory:

```bash
GPU_ID=0 ./scripts/run.sh /path/to/testdata/u2/haac ./results/mi210-rocm633
```

The results directory contains `summary.csv`, `logs/`, and `compressed/`. The summary has one row per input per stage, with `throughput_gbps` for that stage and `lossless_pass` filled for the decompression stages (compression has no lossless check). Each log contains the original detailed benchmark output. The script stops if a run fails or a lossless check does not pass.

To run a single stage directly:

```bash
HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
  ./build/compress_fused input.u2 output.adm
HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
  ./build/decompress_fused output.adm input.u2
```

## Optional L2 counter profiling

`profile.sh` collects L2 cache counters for every input and every stage in one pass, mirroring `run.sh`. It runs each stage through `rocprofv3` with the `GL2C_HIT_sum` and `GL2C_MISS_sum` counters (the gfx1100 / RDNA3 L2 counters), sums the counter values over the target ADM kernel(s) of each stage, and writes a `summary.csv` with one row per input per stage:

```bash
GPU_ID=0 ROCPROFV3=/opt/rocm-6.3.3/bin/rocprofv3 \
  ./scripts/profile.sh /path/to/testdata/u2/haac
```

An optional second argument sets the result directory:

```bash
GPU_ID=0 ROCPROFV3=/opt/rocm-6.3.3/bin/rocprofv3 \
  ./scripts/profile.sh /path/to/testdata/u2/haac ./profile/gfx1100
```

The results directory contains `summary.csv`, `logs/`, `pmc/` (raw `counter_collection.csv` files), and `compressed/`. The summary columns are `input,stage,l2_hit_rate_pct,GL2C_HIT_sum,GL2C_MISS_sum,log`, where:

```text
L2 hit rate (%) = 100 * GL2C_HIT_sum / (GL2C_HIT_sum + GL2C_MISS_sum)
```

The target kernel(s) per stage are: fused compression `map_values_kernel_decoupled`; unfused compression `map_values_kernel_thrust`; fused decompression `decompress`; unfused decompression `decode_signals_uint16` and `restore_values_uint16` (both kernels summed, since they together complete the stage). Runtime/setup kernels such as `__amd_rocclr_fillBufferAligned` and the `thrust::exclusive_scan`/`concat` helpers are excluded. Override the counters with `COUNTERS`, `HIT_COUNTER`, and `MISS_COUNTER` if profiling a different architecture.

> **Note:** on gfx1100 (Radeon RX 7900 XTX) the `GL2C_HIT`/`GL2C_MISS` counters read zero under both `rocprofv2` and `rocprofv3` (a known RDNA3 hardware/firmware limitation), so the summary will show `NA` for the hit rate on that part even though the profiling pipeline is correct. `SQ_WAVES` and other counters do report normally; only the GL2C counters are flat-zero. Use `run.sh` output, not profiler runs, for throughput comparisons.

## Measurement scope

The HIP event timings reported by these sources cover GPU ADM work, not file I/O, host-device copies, ANS, or the CPU lossless comparison. Fused decompression is timed as one kernel. Split decompression is timed across the signal-decode and value-restore kernels. Fused compression reports its fused mapping kernel; unfused compression reports mapping, prefix-sum, and concatenation separately and reports their sum as compression time. Thus the unfused compression total includes the GPU scan and concat work, not just a single kernel.

The compiler version, ROCm release, GPU model, and architecture can change results. Record them alongside any transferred benchmark results (for example, `hipcc --version`, `rocminfo`, and the GPU model).
