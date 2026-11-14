# 版本 A —— HIP 移植版（本次会话的第一版实现）

本目录是「从 AMD HIP 机械移植到 CUDA」的全流水线 ADM 实现的快照，用于与
「基于 sc-ae 原生 NV kernel 升级」的版本 B 做速度对比。

## 来源

- 移植源：`amd/adm/mans_amd_adm.cpp`（HIP，当前 AMD 算法，含 min/max range +
  raw fallback `kThreshold=1009/3025`、fused 阈值 `kFusedGsizeThreshold=4096`）
- 移植产物：`nv/adm/mans_nv_adm.cu`（namespace `mans::nv::adm`，13 个 `__global__`）
- driver：`nv/nv_mans_bench.cpp`（对齐 `amd/amd_mans_bench.cpp` 的 CSV 口径）
- 构建：`nvcc -O3 -std=c++17 -arch=sm_80 -I nv/ans/include
  nv/nv_mans_bench.cpp nv/adm/mans_nv_adm.cu nv/ans/mans_nv_ans.cu
  nv/ans/src/GpuANS.cu -o build-nv/nv_mans_bench`

## 结果

- `mans_bench_1d_results_interleave_nv.csv`：98 文件全部 PASS 的对齐结果
- `nv_throughput.csv`：同一份结果的 build-nv 副本

## 记录点

- 记录时 HEAD：`145cf5d0abe976cb2d7a552b1efecaddc92109e0`，branch `mi210&a100`
- 这些源文件当时为未跟踪（untracked）状态，故用文件快照留存。
