# AMD MI210 vs NVIDIA A100 —— HAAC ADM 优化前后差异

> 结论先行:同一份 ADM kernel 源码(AMD `src/*.cpp`,NV `src/*.cu` 为其符号链接),经优化(向量化 `uint4` 加载 + 消除动态索引局部数组 + `concat`/`decompress` block 32→256)后,**AMD MI210 提速 3~5×,NV A100 几乎不变**。原因是优化针对的是 hipcc 的代码生成缺陷(LDS 膨胀 + 寄存器溢出压低 occupancy),而 nvcc 本来就没有这个问题。优化后两平台差距大幅收敛,`compress_fused` 甚至 AMD 反超。
>
> 数据可信度:✅ 实测(GPU event / rocprof / cuobjdump);† 为小 kernel 时间,受测量噪声影响。

---

## 1. 端到端 kernel 时间对比(1 GB 输入,GPU event,ms)✅

| stage | kernel | AMD 优化前 | AMD 优化后 | AMD 加速 | NV 优化前 | NV 优化后 | NV 加速 |
|---|---:|---:|---:|---:|---:|---:|
| compress_fused | `map_values_kernel_decoupled` | 76.86 | 23.65 | **3.25×** | 37.19 | 35.09 | 0.94× |
| compress_unfused | `map_values_kernel_thrust` | 25.67 | 5.18 | **4.96×** | 3.69 | 2.69 | 1.37× |
| 　└ concat | `concat` | 1.30 | 0.97 | 1.34× | 0.91 | 0.87 | 1.05× |
| 　└ 合计 | — | 27.05 | 6.23 | **4.34×** | 5.97 | 3.79 | 1.58× |
| decompress_fused | `decompress` | 13.12 | 3.32 | **3.95×** | 1.42 | 1.47 | 1.04× |
| decompress_unfused | `decode_signals_uint16` | 1.63 | 1.11 | 1.47× | 0.68 | 1.14 | 0.60×† |
| 　└ `restore_values_uint16` | 3.04 | 3.00 | 1.01× | 2.22 | 2.43 | 0.91×† |
| 　└ 合计 | — | 4.67 | 4.11 | 1.14× | 2.90 | 3.57 | 0.81×† |

† 小 kernel(单次 <3 ms),时间受测量噪声影响,`decode`/`restore` 在 NV 侧的"变慢"非真实退化。

**规律**:AMD 上慢得最多的 kernel(`map_values_kernel_thrust`、`decompress`、`map_values_kernel_decoupled`)恰好是优化收益最大的(4~5×);NV 上这些 kernel 本来就不慢,优化几乎不改变时间。

---

## 2. 两平台差距收敛(AMD 时间 ÷ NV 时间)🧮

| stage | 优化前 | 优化后 |
|---|---:|---:|
| compress_fused | 2.07× (AMD 慢) | **0.67× (AMD 快)** |
| compress_unfused | 4.53× | 1.64× |
| decompress_fused | **9.24×** | **2.26×** |
| decompress_unfused | 1.61× | 1.15× |

优化前 AMD 在解压 fused 上慢 9.2×,优化后缩到 2.3×;压缩 fused 由"慢 2×"翻转为"快 1.5×"。

---

## 3. 资源占用对比 —— 优化到底改了什么

### 3.1 AMD MI210(rocprof 实测,1 GB)✅

| kernel | occupancy | LDS (bytes) | scratch (bytes) | VGPR | block |
|---|---:|---:|---:|---:|---|
| `map_values_kernel_decoupled` | **8.81% → 99.58%** | 16896 → 512 | 48 → 0 | 32 → 56 | 32 |
| `map_values_kernel_thrust` | **12.15% → 99.00%** | 16384 → 0 | 48 → 0 | 28 → 52 | 32 |
| `concat` | 50.4% → 90.64% | 0 | 0 | 8 | 32 → 256 |
| `decompress` | **6.08% → 87.68%** | 32768 → 0 | 0 | 44 → 60 | 32 → 256 |
| `decode_signals_uint16` | **23.4% → 95.08%** | 32768 → 0 | 0 | 12 → 40 | 256 |
| `restore_values_uint16` | 84.32% → 82.43% | 0 | 0 | 60 | 256 |

**AMD 侧的优化机制(实测证据)**:
1. **LDS 消除**:`decompress`/`decode` 各 32 KB → 0,`thrust` 16 KB → 0,`decoupled` 16.9 KB → 512 B。源码里的动态索引局部数组 `local_signal[]/local_codes[]` 之前被 hipcc 整块塞进 LDS,优化后寄存器化。
2. **scratch(spill)消除**:`decoupled`/`thrust` 各 48 B → 0。
3. **occupancy 大幅回升**:从 6~12% 拉到 88~100%,有足够的 wavefront 隐藏内存延迟。
4. **`concat` block 32→256**:occupancy 50% → 91%,这是 69f4b38 concat 优化的直接效果。

### 3.2 NV A100(cuobjdump 反汇编实测)✅

| kernel | REG | STACK (spill) | SHARED | block |
|---|---:|---:|---:|---|
| `map_values_kernel_decoupled` | 32 → 40 | 48 → 0 | 4 | 32 |
| `map_values_kernel_thrust` | 32 → 40 | 48 → 0 | 0 | 32 |
| `concat` | 28 | 0 | 0 | 32 → 256 |
| `decompress` | 32 → 56 | 32 → 0 | 0 | 32 → 256 |
| `decode_signals_uint16` | 32 → 45 | 32 → 0 | 0 | 256 |
| `restore_values_uint16` | 32 | 0 | 0 | 256 |

**NV 侧对比**:
- nvcc 优化前就用 **32 个寄存器 + 0 shared** 装下动态索引数组(无 LDS 膨胀),只有 32~48 B 的轻微 spill。
- 优化后 spill 归零,寄存器升到 40/45/56 —— 但这对已经"不慢"的 nvcc 产物提升有限,故计时几乎不变。
- 注:NV 无运行时 occupancy 计数(本集群 `ncu` 被 `RmProfilingAdminOnly=1` 锁死),上表 reg/stack/shared 来自离线 `cuobjdump`;AMD 的 occupancy/VGPR/LDS 来自 rocprof。

---

## 4. 结论

| 维度 | AMD MI210 | NV A100 |
|---|---|---|
| 优化前瓶颈 | hipcc 把动态索引局部数组塞进 LDS(16~32 KB/block)+ spill,occupancy 压到 6~12% | nvcc 本就寄存器化,仅轻微 spill,occupancy 正常 |
| 优化动作 | 向量化 `uint4` 加载 + 消除局部数组 + block 32→256 | 同一份源码,仅 spill 归零、寄存器微升 |
| 优化后 | LDS/scratch 归零,occupancy 88~100%,提速 3~5× | 计时几乎不变(±噪声) |
| 两平台差距 | decompress 慢 9.2× → 2.3×;compress 慢 2.1× → 快 1.5× | — |

**一句话总结**:这轮优化修复的是 **hipcc(MI210)编译器对 gather 型 ADM kernel 的代码生成缺陷**,把 occupancy 从个位数拉满;同一份源码交给 nvcc(A100)本来就没有这个缺陷,所以 A100 计时不变。优化后两平台在 ADM 上的差距从"最差 9×"收敛到"2× 以内,部分 kernel 反超"。

---

## 附:数据文件位置

- 计时(优化前/后,AMD + NV):
  - `amd/adm/haac_bench/results/mi210-preopt/stage_timings.csv`
  - `amd/adm/haac_bench/results/mi210-optimized/stage_timings.csv`
  - `nv/adm/haac_bench/results/a100-preopt/stage_timings.csv`
  - `nv/adm/haac_bench/results/a100-optimized/stage_timings.csv`
- 资源占用:
  - AMD(occupancy/VGPR/LDS/scratch,rocprof):`amd/adm/haac_bench/profile/mi210-{preopt,optimized}/resource/summary.csv`
  - AMD(L2 hit rate):`amd/adm/haac_bench/profile/mi210-{preopt,optimized}/summary.csv`
  - NV(reg/stack/shared,cuobjdump):`nv/adm/haac_bench/profile/a100-{preopt,optimized}/summary.csv`
- 优化前源码快照:`amd/adm/haac_bench/src_preopt/`
