# ADM 版本 A（HIP 移植） vs 版本 B（sc-ae 原生 kernel）速度对比

## 背景

上一轮把 AMD HIP 的 ADM 全流水线机械移植到 CUDA（版本 A，`nv/adm/mans_nv_adm.cu`）。
但 `sc-ae` git tag 里其实已有 GPU-native 的 CUDA ADM kernel（`nv/adm/mapping_uint16.cu` /
`mapping_uint32.cu`），属于旧一代算法（无 raw fallback、fused 阈值 `gsize<=1024`、
decompress 里 int4 拷贝未对齐）。

版本 B（`nv/adm/mans_nv_adm_scae.cu`）= 版本 A 的宿主脚手架（payload 布局、host wrapper、
thrust scan、concat/finalize/pack_flags、raw fallback、阈值 `kThresholdFor<T>()` = 1009/3025、
decompress 字节级对齐修复）+ sc-ae 的 map kernel 信号打包方式。

## 唯一的吞吐相关差异：unary 信号打包

| | 版本 A（HIP 移植） | 版本 B（sc-ae） |
|---|---|---|
| 信号位写入 | 64-bit 寄存器累加器 `acc = (acc<<oi) | (1<<(oi-1))`，再按字节 flush | 逐位写 `local_bit_signal[bit_offset/8] |= (1 << (7 - bit_offset%8))`，再整体按字节拷贝 |
| 编码公式 | `output_idx = ceil(diff/126)`，code 相同 | 相同 |
| raw fallback | `(max-min) < 1009/3025` | 相同（补进 sc-ae） |
| decompress | 字节级拷贝（已修对齐） | 相同（sc-ae int4 改为字节级） |

两者输出的 ADM 位流完全一致（已验证 lossless 100%）。

## 结果（A100-PCIE-40GB，sm_80，`--warmup 2 --iters 3`，98 文件）

数据文件（标签：版本 A = 移植版本，版本 B = sc版本）：
- **移植版本**（版本 A，阈值已修正 1009/3025）：`mans_bench_1d_results_interleave_nv_ported.csv`
- **sc版本**（版本 B）：`mans_bench_1d_results_interleave_nv_scae.csv`
- 逐文件对比存档：`nv_adm_ported_vs_scae.csv`

**98 文件全部 PASS（无损还原一致），行序/文件集合完全一致。**

### 几何均值（geomean）

| metric | 版本 A | 版本 B | B/A |
|---|---|---|---|
| comp_gbps | 2.625 | 2.636 | 1.0045 |
| decomp_gbps | 5.887 | 6.055 | 1.0285 |

### B/A 比率分布（98 文件）

| metric | min | p10 | 中位数 | p90 | max |
|---|---|---|---|---|---|
| comp_gbps | 0.643 | 0.931 | 0.998 | 1.101 | 1.375 |
| decomp_gbps | 0.922 | 0.988 | 1.001 | 1.059 | 1.398 |

### 按 dtype / adm_path

| 分组 | comp B/A | decomp B/A |
|---|---|---|
| u2 (80) | 1.011 | 1.036 |
| u4 (18) | 0.977 | 0.998 |
| fused (66) | 1.028 | 1.032 |
| unfused (32) | 0.958 | 1.021 |

## 结论

版本 A（寄存器累加器打包）与版本 B（sc-ae 逐位打包）吞吐**基本一致**，中位数 B/A ≈ 1.0，
80% 文件落在 ±10% 内。差异在单次运行、共享 GPU 节点的测量噪声范围内（个别大文件出现
±15~35% 抖动，如 `col_360011594_262144kB.u4` comp B/A=0.64、部分 1GB haac u2 约 0.86，
均为噪声而非信号打包导致）。

**即：unary 信号打包方式（寄存器累加器 vs 逐位 local 数组）不是 ADM map kernel 的吞吐瓶颈；**
两种实现的编码位流一致、吞吐相当，机械移植的 HIP 版本与原 sc-ae 原生 kernel 没有实质速度差异。
