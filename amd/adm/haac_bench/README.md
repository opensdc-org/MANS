# AMD HIP HAAC ADM benchmark

This directory contains four U16 HIP executables for the HAAC fused/unfused ADM comparison, plus a synthetic HIP microbenchmark. The benchmark is self-contained apart from ROCm/HIP and the HAAC `.u2` inputs. Each ADM executable performs one stage (fused or unfused compression or decompression) so its work can be timed and profiled in isolation.

## Requirements

- Linux with an AMD GPU supported by the installed ROCm release.
- ROCm HIP compiler (`hipcc`) and `rocprofv3` for optional L2 counter collection.
- HAAC U2 input files, such as the `.u2` files from the MANS/SC-AE dataset archive.
- A GPU architecture accepted by the installed compiler (for example, `gfx90a` on an Instinct MI210).

No ANS stage is run. These results measure ADM only.

## Input format

The benchmark expects raw, headerless `.u2` files containing little-endian unsigned 16-bit values. The file name is not parsed to determine the element count or geometry; the executables use the file length and the data values directly. An input file must have an even number of bytes.

The benchmark scripts process every `.u2` file directly under the input directory and ignore files in nested directories. Keep separate datasets in separate input directories when comparing runs.

## Build

From this directory:

```bash
HIPCC=/opt/rocm-6.3.3/bin/hipcc GPU_ARCH=gfx90a ./scripts/build.sh
```

`GPU_ARCH` is optional when the compiler has a suitable default target. `HIPCC` is required when `hipcc` is not on `PATH`. Set `BUILD_DIR` to change the binary directory:

```bash
HIPCC=/opt/rocm-6.3.3/bin/hipcc GPU_ARCH=gfx90a BUILD_DIR="$PWD/build" ./scripts/build.sh
```

If the ROCm device-library bitcode is not found at `<rocm>/lib/amdgcn/bitcode` (some 6.3.3 installs place it at `<rocm>/amdgcn/bitcode`), the script detects the alternate location automatically; it can also be forced with `ROCM_DEVICE_LIB_PATH`.

The default build uses the `rocm` backend. An exploratory CUDA/hipify path is also available with `BACKEND=cuda`, but it is not the supported AMD benchmark path and requires both `nvcc` and `hipify-perl`:

```bash
BACKEND=cuda NVCC=/path/to/nvcc HIPIFY_PERL=/path/to/hipify-perl \
  CUDA_ARCH=sm_80 ./scripts/build.sh
```

Five executables are produced under `build/`:

| executable           | stage                 | arguments                                      |
|----------------------|-----------------------|------------------------------------------------|
| `compress_fused`     | fused compression     | `<input.u2> <output.adm>`                      |
| `compress_unfused`   | unfused compression   | `<input.u2> <output.adm>`                      |
| `decompress_fused`   | fused decompression   | `<input.adm> <original.u2>`                    |
| `decompress_unfused` | unfused decompression | `<input.adm> <original.u2>`                    |
| `microbench`         | synthetic GPU tests   | `[elements] [trials] [inner_iterations]`       |

## Run all HAAC input sizes

Pass the directory containing HAAC `.u2` files. The script runs each input through all four stages, selects the requested GPU (default 0), saves per-run logs and generated ADM files, and writes a CSV summary:

```bash
GPU_ID=0 ./scripts/run.sh /path/to/testdata/u2/haac
```

An optional second argument sets the result directory:

```bash
GPU_ID=0 ./scripts/run.sh /path/to/testdata/u2/haac ./results/mi210-rocm633
```

The results directory contains `summary.csv` and `logs/`. Generated `.adm` files are kept in a temporary directory and removed when the script exits. The summary has one row per input per stage, with `throughput_gbps` for that stage and `lossless_pass` filled for the decompression stages (compression has no lossless check). Each log contains the original detailed benchmark output. The script stops if a run fails or a lossless check does not pass. Use a new result directory for each run.

To run a single stage directly:

```bash
HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
  ./build/compress_fused input.u2 output.adm
HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
  ./build/decompress_fused output.adm input.u2
```

## Real-input stage analysis

`analyze_real.py` is the primary AMD-side workflow for comparing stage costs on identical real inputs. It runs fused/unfused compression followed by the corresponding decompression, stores each full log, keeps generated `.adm` files in a temporary directory for the duration of the run, and writes one row per input per stage to `stage_timings.csv`:

```bash
GPU_ID=0 ./scripts/analyze_real.py /path/to/testdata/u2/haac ./results/mi210-real-run-1
```

A verified MI210 run on the 7 ExaFEL input sizes used GPU 2 and produced `results/amd-real-stage-20261109-rerun1/stage_timings.csv` (28 stage rows; all 14 decompression checks passed). To repeat that dataset run with the same device selection, use:

```bash
GPU_ID=2 ./scripts/analyze_real.py /workspace/MANS/testdata/u2/exafel ./results/amd-real-stage-<unique-run-name>
```

Choose a fresh result path for every run. The result directory must not exist. Existing data is never reused or overwritten. The CSV separates kernel-only HIP-event time (`total_gpu_ms` and its applicable sub-stages) from host-observed I/O/copy timings and `total_wall_ms`; fields that do not apply or are not separately measured are `NA`. Compression output size and decompression input ADM size are recorded for each row. Temporary `.adm` files are deleted when the run exits and are not kept in the result directory. Decompression rows include the CPU lossless check result, and the run stops on a failed stage or check while retaining completed logs and rows.

For the 16 MiB ExaFEL input in the verified run, the single-run HIP-event totals were 1.57 ms (fused compression), 0.88 ms (unfused compression), 0.34 ms (fused decompression), and 0.17 ms (unfused decompression). Treat these as an example from one run, not stable performance claims; repeat runs before drawing conclusions.

For fused stages, `fused_kernel_ms` is a single fused-kernel measurement: it cannot be split into mapping, scan, concat, decode, or restore costs. Unfused compression reports mapping, prefix-sum, and concat HIP-event timings; unfused decompression reports signal decode and value restore HIP-event timings. `total_gpu_ms` is the sum of those applicable event measurements. It is not an end-to-end time.

`file_read_ms`, `h2d_ms`, `d2h_ms`, `file_write_ms`, and `cpu_verify_ms` are host `steady_clock` measurements around synchronous calls/work. In particular, the copy fields are host-observed call durations, not isolated device-link bandwidth measurements. `total_wall_ms` starts before input loading and includes allocation, warm-up, measured work, and verification as applicable; it is printed before resource cleanup. Unmeasured host metadata assembly is reported as `NA`, not inferred by subtracting other timings.

The existing `run.sh` remains a simpler throughput/lossless summary workflow; use `analyze_real.py` for phase-level comparisons. AMD HIP-event kernel throughput must not be directly compared as the same metric to NVIDIA CLI wall-clock throughput. NVIDIA-side scripts and measurements remain independent; record their timing scope separately, especially if decompression uses a CPU reference decoder.

## Synthetic GPU microbenchmark

`microbench` runs synthetic integer, bitwise, and device-memory-copy kernels. Its optional arguments are the element count, trial count, and inner iteration count:

```bash
./build/microbench [elements] [trials] [inner_iterations]
./build/microbench 16777216 5 256 > ./results/microbench.csv
```

It prints one CSV row per benchmark trial, with elapsed HIP-event time and a nominal operation rate (`Gop/s`) or copy bandwidth (`GB/s`). The copy kernel is repeated for each inner iteration; its reported bandwidth counts one read and one write per element per iteration. These synthetic results are useful for rough hardware comparisons, not as substitutes for timings of the ADM workload.

## Capture system information

Capture compiler, GPU, profiler, operating-system, and relevant environment details alongside benchmark results:

```bash
./scripts/system_info.sh ./results/system-mi210
```

The output directory contains `metadata.txt`, command output files, and `relevant_environment.txt`. Choose a new, empty output directory for each snapshot: the script creates the directory if needed and can replace files with matching names.

## Optional L2 counter profiling

`profile.sh` collects cache counters for every input and every stage in one pass, mirroring `run.sh`. Its defaults target gfx90a/Instinct MI210 and use the summed TCC counters:

```text
TCC_HIT_sum and TCC_MISS_sum
```

These counters are the appropriate defaults for the MI210 profile data in this repository. They are summed over the selected ADM kernel(s) and used to calculate a diagnostic hit rate. They are not, by themselves, proof that cache behavior causes a throughput difference.

```bash
GPU_ID=0 ROCPROFV3=/opt/rocm-6.3.3/bin/rocprofv3 \
  ./scripts/profile.sh /path/to/testdata/u2/haac
```

An optional second argument sets the result directory. It must not already exist; profiling preserves prior results by refusing to reuse an existing directory:

```bash
GPU_ID=0 ROCPROFV3=/opt/rocm-6.3.3/bin/rocprofv3 \
  ./scripts/profile.sh /path/to/testdata/u2/haac ./profile/mi210-run-1
```

The results directory contains `summary.csv`, `logs/`, and `pmc/` (raw `counter_collection.csv` files). Generated `.adm` files are kept in a temporary directory and removed when the script exits. With the defaults, the summary columns are `input,stage,l2_hit_rate_pct,TCC_HIT_sum,TCC_MISS_sum,log`, where:

```text
L2 hit rate (%) = 100 * TCC_HIT_sum / (TCC_HIT_sum + TCC_MISS_sum)
```

The target kernel(s) per stage are: fused compression `map_values_kernel_decoupled`; unfused compression `map_values_kernel_thrust`; fused decompression `decompress`; unfused decompression `decode_signals_uint16` and `restore_values_uint16` (both kernels summed, since they together complete the stage). Runtime/setup kernels such as `__amd_rocclr_fillBufferAligned` and the `thrust::exclusive_scan`/`concat` helpers are excluded. Override the counters with `COUNTERS`, `HIT_COUNTER`, and `MISS_COUNTER` when profiling another architecture. For gfx1100/Radeon RX 7900 XTX, try `COUNTERS='GL2C_HIT_sum GL2C_MISS_sum' HIT_COUNTER=GL2C_HIT_sum MISS_COUNTER=GL2C_MISS_sum`; those GL2C counters may report zero on some RDNA3 driver/firmware combinations, in which case the hit rate is reported as `NA`.

Treat the hit rate as a measurement to compare alongside throughput, raw counters, and other hardware metrics. Keep the raw `pmc/*/counter_collection.csv` files because additional counters may be needed to explain a performance difference.

## Measurement scope

The HIP event timings reported by these sources cover GPU ADM work, not file I/O, host-device copies, ANS, or the CPU lossless comparison. Fused decompression is timed as one kernel. Split decompression is timed across the signal-decode and value-restore kernels. Fused compression reports its fused mapping kernel; unfused compression reports mapping, prefix-sum, and concatenation separately and reports their sum as compression time. Thus the unfused compression total includes the GPU scan and concat work, not just a single kernel.

## Reproducibility

The compiler version, ROCm release, GPU model, architecture, selected GPU, and input dataset can change results. Record them alongside any transferred benchmark results. The `system_info.sh` helper captures this information where the corresponding tools are installed:

```bash
./scripts/system_info.sh ./results/system-mi210-run-1
```

For comparable measurements, keep the same input files, `GPU_ARCH`, GPU selection, ROCm environment, and result-directory naming convention. Run the benchmark more than once when evaluating small performance differences, and retain both `summary.csv` and the per-run logs.
