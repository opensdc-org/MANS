# NV 全流水线吞吐 benchmark（对齐 AMD 口径）

对齐 `amd/amd_mans_bench.cpp`（`build-amd-633/bin/amd/amd_mans_bench`）的完整 MANS
（ADM + ANS）吞吐测试，在 A100 上对 `hwj/testdata` 的 98 个文件测吞吐。

## 背景：NV 侧原本没有等价的全流水线 benchmark

- NV 生产 ADM（`nv/adm/mapping_uint16.cu` / `mapping_uint32.cu`）是 CPU 辅助的：
  compress 用单个 GPU `encode` kernel + CPU prefix-sum/pack（含 D2H/H2D 往返），
  decompress 是纯 CPU（`decode_reference`），没有 fused/unfused，也没有 kernel 计时。
- 因此无法直接用 CUDA event 测「纯 GPU kernel 时间」。本工作把 AMD 的 GPU-native ADM
  （`amd/adm/mans_amd_adm.cpp`，HIP）机械移植到 CUDA（`nv/adm/mans_nv_adm.cu`，
  namespace `mans::nv::adm`），再写一个镜像 driver `nv/nv_mans_bench.cpp`。

## 构建（login 节点即可，无需 GPU）

```bash
NVCC=/home/bingxing2/apps/cuda/11.7.0/bin/nvcc
mkdir -p build-nv
$NVCC -O3 -std=c++17 -arch=sm_80 -I nv/ans/include \
  nv/nv_mans_bench.cpp nv/adm/mans_nv_adm.cu \
  nv/ans/mans_nv_ans.cu nv/ans/src/GpuANS.cu \
  -o build-nv/nv_mans_bench
```

不链接旧 `nv/adm/mapping_uint16.cu`（`get_max_u16_payload_bytes` 符号冲突 + 无需）。

## 运行（GPU 节点）

```bash
export LD_LIBRARY_PATH=/home/bingxing2/apps/cuda/11.7.0/lib64:$LD_LIBRARY_PATH
cd /home/bingxing2/home/scx6546/hwj/testdata
find . -type f \( -name '*.u2' -o -name '*.u4' \) | sort > files.txt
srun --partition=gpu --gres=gpu:1 \
  /home/bingxing2/home/scx6546/hwj/MANS/build-nv/nv_mans_bench \
  /home/bingxing2/home/scx6546/hwj/MANS/build-nv/nv_throughput.csv \
  $(cat files.txt) --warmup 2 --iters 3
```

## 计时口径（与 AMD 完全一致）

1. `--warmup 2 --iters 3`：每个文件跑 warmup+iters=5 轮，前 2 轮丢弃，后 3 轮算术平均。
2. 计时范围 = **仅 GPU kernel 时间**（HIP event → CUDA event），**不含 memcpy/IO**。
3. `comp_gbps = raw_bytes/1e9 / ((adm_comp_total_ms + ans_comp_ms)/1000)`；
   `decomp_gbps = raw_bytes/1e9 / ((ans_decomp_ms + adm_decomp_decode_ms)/1000)`。
4. dtype：`.u2`→uint16，`.u4`→uint32；fused/unfused：`gsize<=4096`→fused。
5. dataset 标签 = 直接父目录名；`size` 从文件名取（如 `4kB`）。

CSV 列与 AMD 完全一致：
`dataset,dtype,size,file,num_elements,raw_bytes,adm_path,adm_comp_map_ms,
adm_comp_scan_ms,adm_comp_concat_ms,adm_comp_finalize_ms,adm_comp_pack_flags_ms,
adm_comp_total_ms,adm_decomp_decode_ms,ans_comp_ms,ans_decomp_ms,comp_gbps,
decomp_gbps,verify`

## 结果（98 文件，全部 PASS）

| dataset | n | comp (GB/s) | decomp (GB/s) |
|---|---:|---:|---:|
| big-tpch (u4) | 10 | 40.0 | 51.2 |
| haac (u2) | 60 | 23.8 | 35.5 |
| suitesparse (u4) | 8 | 10.9 | 30.6 |
| exafel (u2) | 7 | 3.5 | 11.5 |
| tomobank (u2) | 7 | 3.1 | 13.3 |
| modis (u2) | 6 | 2.0 | 7.9 |
| **全体算术平均** | 98 | **20.2** | **31.7** |
| ≥64MB 子集 | 22 | 75.7 | 88.6 |

> 小文件（4kB/16kB 等）吞吐受 kernel 启动开销主导，数值偏低；≥64MB 的 22 个文件
> 更能代表稳态吞吐（comp ~76 GB/s、decomp ~89 GB/s）。最大文件（1GB）单文件
> comp 最高 ~174 GB/s（big-tpch u4）、decomp ~132 GB/s。

## 移植时修的一个真实 bug

AMD 原 `decompress`/`decompress_u32` 1D kernel 的 raw-block 拷贝用
`reinterpret_cast<int4*>` 做 16B 向量化读。压缩后的 payload 把 signals 段放在
`len1+len2+len3+len4` 偏移处（`len3=ceil(gsize/8)` 字节，小 gsize 时是奇数），
因此该指针在 sm_80 上不是 16B 对齐 → `cudaErrorMisalignedAddress`。HIP/AMD 硬件
容忍了这种未对齐向量读，NV 直接报错。修复：把 decompress 里对 payload 的
`int4` 读（raw copy 与 codes copy）改为逐字节拷贝（`decompress_general` 本来就是
逐字节的，无需改）。结果逐位一致，verify 全 PASS。
