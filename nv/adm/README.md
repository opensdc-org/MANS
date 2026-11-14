# nv/adm —— ADM 的三种 GPU kernel 实现

本目录（及其子目录）包含三代 ADM GPU 代码，用于论文里 A100 / MI210 的 ADM 编解码对比。
三种实现的核心区别在 **unary 信号打包方式** 与 **是否带 raw fallback**，编码公式一致。

## 代码变体

| 目录/文件 | namespace | 是什么 | 信号打包 | raw fallback / 阈值 |
|---|---|---|---|---|
| `scae_orig/mapping_uint16.cu`<br>`scae_orig/mapping_uint32.cu` | 全局（无 namespace） | **原始 sc-ae GPU kernel**（`sc-ae` git tag 原样提取） | 逐位写 `local_bit_signal[bit_offset/8] \|= (1 << (7-bit_offset%8))` | 无；fused 阈值 `gsize<=1024`；decompress 用 int4 拷贝（sm_80 上未对齐会报错） |
| `mans_nv_adm.cu` | `mans::nv::adm` | **移植版本**（从 AMD HIP `amd/adm/mans_amd_adm.cpp` 机械移植） | 64-bit 寄存器累加器 `acc = (acc<<oi) \| (1<<(oi-1))` | 有；`kThresholdFor<T>()` = 1009/3025；fused 阈值 4096 |
| `mans_nv_adm_scae.cu` | `mans::nv::adm_scae` | **sc版本**（以 `scae_orig` 原始 kernel 为基础升级） | 逐位写（沿用 sc-ae） | 有；补 raw fallback + 阈值 1009/3025 + decompress 对齐修复 |

- `scae_orig/`：**原始 sc-ae GPU kernel**，旧算法（纯 center + unary，4 个 kernel：
  `decompress` / `map_values_kernel_thrust` / `concat` / `map_values_kernel_decoupled`），
  带独立 `main()`，可 `nvcc -O3 scae_orig/mapping_uint16.cu -o mapping_uint16` 单独编译运行。
  注意其 `decompress` 用 `reinterpret_cast<int4*>` 做 16B 拷贝，signals 段奇数偏移时
  sm_80 会 `cudaErrorMisalignedAddress`。
- `mapping_uint16.cu` / `mapping_uint32.cu`（本目录，181/… 行）：当前**生产版**，已把 4 个
  kernel 重构为单个 templated `compress_kernel<T>`，decompress 纯 CPU。与 `scae_orig` 不是一回事。

## 测试 driver

| driver | 调用哪个实现 | 二进制 |
|---|---|---|
| `nv/nv_mans_bench.cpp` | 移植版本（`mans::nv::adm`） | `build-nv/nv_mans_bench` |
| `nv/nv_mans_bench_scae.cpp` | sc版本（`mans::nv::adm_scae`） | `build-nv/nv_mans_bench_scae` |

构建（login 节点免 GPU）：
```bash
nvcc -O3 -std=c++17 -arch=sm_80 -I nv/ans/include \
  nv/nv_mans_bench.cpp      nv/adm/mans_nv_adm.cu      nv/ans/mans_nv_ans.cu nv/ans/src/GpuANS.cu -o build-nv/nv_mans_bench
nvcc -O3 -std=c++17 -arch=sm_80 -I nv/ans/include \
  nv/nv_mans_bench_scae.cpp nv/adm/mans_nv_adm_scae.cu nv/ans/mans_nv_ans.cu nv/ans/src/GpuANS.cu -o build-nv/nv_mans_bench_scae
```

## 结果 CSV（MANS 根目录）—— 哪个是 sc版本 / 移植版本

| 文件 | 平台 / 实现 | 标签 |
|---|---|---|
| `mans_bench_1d_results_interleave.csv` | AMD MI210 基线（对照） | AMD |
| `mans_bench_1d_results_interleave_nv_ported.csv` | **移植版本**（HIP 移植，阈值已修正 1009/3025） | **移植版本** |
| `mans_bench_1d_results_interleave_nv_scae.csv` | **sc版本**（sc-ae 原始 kernel 升级） | **sc版本** |
| `mans_bench_1d_results_interleave_nv.csv` | 移植版本（旧阈值 3500，**已被 ported 取代**，仅留档） | 移植版本（旧） |
| `nv_adm_ported_vs_scae.csv` | 逐文件 **移植版本 vs sc版本** 对比 | 对比 |

98 文件全部 PASS（无损还原一致）。对比结论见 `nv/adm_ab_comparison.md`。
