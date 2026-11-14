# ADM 版本 A（HIP 移植） vs 版本 B（sc-ae 原生 kernel）速度对比

## 背景

上一轮把 AMD HIP 的 ADM 全流水线机械移植到 CUDA（版本 A，`nv/adm/mans_nv_adm.cu`）。
但 `sc-ae` git tag 里其实已有 GPU-native 的 CUDA ADM kernel（`nv/adm/mapping_uint16.cu` /
`mapping_uint32.cu`），属于旧一代算法（无 raw fallback、fused 阈值 `gsize<=1024`）。

版本 B（`nv/adm/mans_nv_adm_scae.cu`）= 版本 A 的宿主脚手架（payload 布局、host wrapper、
thrust scan、concat/finalize/pack_flags、raw fallback、阈值 `kThresholdFor<T>()` = 1009/3025）
+ sc-ae 的 map kernel 信号打包方式 + **sc-ae 的 int4 向量化 decompress**。

## 两个差异点，结论相反

### 1. map kernel 信号打包：不是瓶颈

| | 版本 A（HIP 移植） | 版本 B（sc-ae） |
|---|---|---|
| 信号位写入 | 64-bit 寄存器累加器 `acc = (acc<<oi) | (1<<(oi-1))`，再按字节 flush | 逐位写 `local_bit_signal[bit_offset/8] |= (1 << (7 - bit_offset%8))`，再整体按字节拷贝 |

两版位流完全一致（lossless 100%）。用字节级 decompress 时两版吞吐持平
（comp geomean B/A≈1.0），即 **信号打包不是 map kernel 的瓶颈**。

### 2. decompress 访存方式：真正的瓶颈

原始 sc-ae 的 1D `decompress`/`decompress_u32` 用 `reinterpret_cast<int4*>` 做 16B
向量化拷贝（raw copy / codes 读 / result 写）。机械移植时为了绕开 sm_80 未对齐 int4
的 `cudaErrorMisalignedAddress`，我把这三处改成了**逐字节**拷贝——这正是 AMD MI210
decompress（165–323 GB/s）明显快于 A100（62–132 GB/s，约慢 2.5x）的原因。

修复方式：恢复 int4 向量化访存，并在 payload 组装时把 codes/signals 段 **16B 对齐**
（`codes_off = align16(len1+len2+len3)`、`signals_off = align16(codes_off+len4)`，
padding 间隙 memset 0），`local_codes`/`local_result` 加 `__align__(16)`。这样 int4
读写在 sm_80 上严格对齐，向量化与对齐都不再是问题。

## 结果（A100-PCIE-40GB，sm_80，`--warmup 2 --iters 3`，98 文件）

数据文件（标签：版本 A = 移植版本，版本 B = sc版本）：
- **移植版本**（版本 A）：`mans_bench_1d_results_interleave_nv_ported.csv`
- **sc版本**（版本 B，int4 向量化 decompress）：`mans_bench_1d_results_interleave_nv_scae.csv`
- 逐文件对比：`nv_adm_ported_vs_scae.csv`
- AMD MI210 基线：`mans_bench_1d_results_interleave.csv`

**98 文件全部 PASS（无损还原一致），行序/文件集合完全一致。**

### 几何均值（geomean）

| metric | AMD MI210 | 版本 A | 版本 B | B/A |
|---|---|---|---|---|
| comp_gbps | 2.806 | 2.625 | 2.401 | 0.915 |
| decomp_gbps | 4.335 | 5.887 | 9.236 | 1.569 |
| adm_decode GB/s | 19.68 | 14.37 | 38.72 | 2.695 |

`adm_decode GB/s` = `raw_bytes / adm_decomp_decode_ms`，即 ADM decode kernel 本身的吞吐。
向量化后版本 B 的 ADM decode 吞吐是版本 A 的 **2.7x**（geomean；中位数 2.14x），
且超过了 AMD MI210 的 ADM decode 吞吐（38.7 vs 19.7 GB/s）。

## 结论

- **unary 信号打包方式不是 ADM 的吞吐瓶颈**（寄存器累加器 vs 逐位 local 数组，两版位流一致、map 阶段持平）。
- **decompress 的访存方式才是瓶颈**：逐字节拷贝 vs int4 向量化拷贝，向量化后 ADM decode
  快 2.7x。AMD MI210 上的原始 sc-ae 一直用 int4 向量化，机械移植成逐字节后 A100 反而
  「看起来」比 MI210 慢——这不是硬件差异，而是移植时丢掉了向量化优化。
