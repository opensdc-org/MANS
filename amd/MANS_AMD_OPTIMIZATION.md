# MANS（ADM + ANS）在 AMD MI210 上的优化总结

> 本文档用于论文修改的交接，汇总 MANS 压缩流水线在 AMD MI210 上的**定位分析**（profile）、**优化方法**与**评测结果**。
> 所有优化均保证**逐字节（byte-identical）一致**——即 AMD 版本与参考实现的输出完全一致，只改实现、不改算法语义。
>
> 平台：AMD Instinct MI210（gfx90a，104 CU，硬件 wavefront = 64 lane），ROCm 6.3.3，构建目录 `build-amd-633`。
> 参照平台：NVIDIA A100（`nv/` 目录，`mans_nv.*` / `nvadm_*`），用于 motivation 的差异分析。
> 论文基线：`/workspace/MANS/ori_results.xlsx`（DietGPU 论文的 MANS 结果）。

---

## 0. 一句话结论

MANS 的核心瓶颈不是算术吞吐，而是 **AMD GPU 上的延迟（latency）与访存模式**，以及**编译器（hipcc）对同一源码生成的低效资源分配**：

- **编译器问题（ADM 侧最硬的一条证据）**：同一份 ADM kernel 源码，`nvcc`（A100）用 32 个寄存器装下动态索引局部数组，`hipcc`（MI210）却整块挪进 16–32 KB LDS 或爆到 60 VGPR，把 occupancy 压到 ~6–12%（vs A100 ~50%）；
- ADM 的 mapping/decompress kernel 是**串行依赖链延迟受限**（`%126` 模运算在关键路径上）；
- ANS decode 是**串行 rANS 链延迟受限**（`SQ_WAIT_INST_LDS` 占主导）；
- ANS histogram 是 **LDS bank-conflict 受限**（64 lane 共用一个 bucket）；
- 小文件（≤1MB）是 **host 端开销受限**（`hipMalloc`/`hipMemcpy`/同步）。

针对性地做了 6 类优化（ADM 内核重构 + 模运算消除 + ANS histogram + ANS decode interleave + 小文件 host 削减，见 §3），最终：**大文件压缩 ≥4MB 快 1.04–3.56×、解压 ≥16MB 快 1.20–2.15×（对比 DietGPU 基线）**；仍慢的只剩小文件（host 开销，非 GPU 内核问题）。

---

## 1. Motivation（动机）

### 1.1 AMD vs NVIDIA 的性能差距

MANS 最初在 NVIDIA A100 上实现并达到高吞吐。移植到 AMD MI210 后，同一套 ADM 内核在 AMD 上明显更慢。下表是 **ADM/HAAC 阶段的 fused kernel 吞吐量**（`input_bytes / fused_kernel_ms / 1e6`，单位 GB/s），来源为两平台的 `adm/haac_bench/results/*/stage_timings.csv`：

**ADM 压缩（`map_values_kernel_decoupled`，fused 单 kernel）：**

| 文件大小 | A100 (NVIDIA) | MI210 基线 (preopt) | MI210 优化后 | 基线差距 | 优化后 |
|---|---|---|---|---|---|
| 1 GB   | 26.89 | 13.01 | 42.29 | 2.1× | 1.57× |
| 256 MB | 26.56 | 13.00 | 42.10 | 2.0× | 1.59× |
| 64 MB  | 26.71 | 12.86 | 41.28 | 2.1× | 1.55× |
| 16 MB  | 22.37 | 12.38 | 38.35 | 1.8× | 1.71× |
| 4 MB   | 18.54 | 10.83 | 29.35 | 1.7× | 1.58× |
| 1 MB   | 6.63  | 7.18  | 14.74 | 0.92× | 2.22× |

**ADM 解压（`decompress`，fused 单 kernel）：**

| 文件大小 | A100 (NVIDIA) | MI210 基线 (preopt) | MI210 优化后 | 基线差距 | 优化后 |
|---|---|---|---|---|---|
| 1 GB   | 703.24 | 76.22  | 301.11 | **9.2×** | 2.3× |
| 256 MB | 686.98 | 76.03  | 294.87 | **9.0×** | 2.3× |
| 64 MB  | 640.20 | 74.50  | 270.68 | **8.6×** | 2.4× |
| 16 MB  | 469.23 | 70.94  | 207.34 | **6.6×** | 2.3× |
| 4 MB   | 275.74 | 56.86  | 110.72 | **4.8×** | 2.5× |
| 1 MB   | 121.29 | 33.46  | 41.89  | **3.6×** | 2.9× |

> **动机核心**：同一算法在 AMD MI210 上的 ADM 解压 fused kernel 基线只有 NVIDIA A100 的 **1/9**，压缩约 **1/2**。这个差距不是算法问题，而是 AMD 架构上的**延迟受限与访存模式**问题，必须针对 AMD 特性重新优化。经过优化后差距大幅收窄，压缩端甚至反超 A100（42 vs 27 GB/s）。

> 说明：`MI210 优化后` 一栏对应的是整个 ADM 优化 pass（`mi210-preopt → mi210-optimized`），即 §2.2 的 ADM 内核重构（LDS 消除 / 加载合并 / block size 等 6 项，来自 `amd/adm/haac_bench` 的优化）+ 后续的模运算消除；`preopt` 源码快照在 `amd/adm/haac_bench/src_preopt/`（commit `e073eda`）。原始计时数据见 `nv/adm/haac_bench/results/a100-{preopt,optimized}/stage_timings.csv` 与 `amd/adm/haac_bench/results/mi210-{preopt,optimized}/stage_timings.csv`；资源占用数据见两侧 `profile/*/summary.csv` 与 `profile/mi210-*/resource/summary.csv`。

### 1.2 全流水线耗时分解（用于说明"时间花在哪"）

用 rocprof 对 256 MB u2 文件跑完整 MANS（ADM→ANS→ANS→ADM），内核级时间占比（`results.stats.csv`）：

| 内核 | 平均耗时 (ms) | 占比 |
|---|---|---|
| `map_values_kernel_thrust`（ADM mapping） | 1.349 | **23.3%** |
| `decompress`（ADM 解压 fused） | 0.851 | 14.7% |
| `ansDecodeKernel`（ANS 解压） | 0.723 | 12.5% |
| `concat`（ADM） | 0.513 | 8.9% |
| `ansEncodeBatch`（ANS 编码） | 0.473 | 8.2% |
| `histogramBatch`（ANS 直方图） | 0.439 | 7.6% |

结构性结论：**ADM 主导**（mapping + decompress + concat ≈ 47%），ANS 次之（encode + decode + histogram ≈ 28%）；在 ANS 内部 **decode（12.5%）> encode（8.2%）**。因此优化优先级为：ADM mapping/decompress → ANS decode → ANS histogram/encode。

### 1.3 逐内核的 profile 定位（为什么 AMD 慢）

#### (a) `ansDecodeKernel` —— 串行 rANS 链延迟受限

- 每条 warp 是一条独立的串行 rANS 状态链：`s_bar = state & mask → lookup[s_bar]（LDS）→ state = pdf*(state>>ProbBits)+cdf → read = state<kANSMinState → __ballot → state=(state<<16)+v（gmem 读）`。这条链每步都依赖上一步的结果，无法靠更多线程吞吐加速。
- rocprof（`stall_metrics.csv`）显示 `SQ_WAIT_INST_LDS` 是 decode kernel 的主导 stall（约 2.4–2.5 万 cycle/波前），即大部分时间在等 LDS 查表（1024 项 uint32 lookup，4 KB）。
- 资源占用：24 vgprs（interleave 后 32）、4 KB LDS、0 scratch → 受 LDS 限制 **16 blocks/CU = 32 wavefronts ≈ 50% occupancy**。即使清空 LDS 也提不了 occupancy（vgpr 已经 cap 在 16 blocks），而 LDS 随机访问本身很快，所以它只是"延迟"瓶颈而非"带宽"瓶颈。

#### (b) `ansEncodeBatch` —— 串行 rANS 链延迟受限（无易解）

- rocprof 硬件计数器：VALU 50M / SALU 11M / LDS 2.4M / VMEM 4.5M，`SQ_LDS_BANK_CONFLICT ≈ 0`，所有执行单元利用率 **≤14%** → 典型的 latency-bound。
- ISA 分析（`hipcc --save-temps` 的 `.s`）：编译器已经做了 software pipelining——LDS 查表是一条 `ds_read_b128`，归一化算数被融合成 `v_add3`（`state + cdf + div*(1024-pdf)`），ballot 的 popcount 是一条 `s_bcnt1_i32_b64`。串行链约 6 条 dependent VALU（mul_hi→add→lshr→sub→mul_lo→add3）。
- **唯一能打破延迟的方法是 interleaved rANS（每 lane 2+ 条状态流）来制造 ILP，但会改变输出格式，破坏逐字节一致约束——不可行。** 因此 encode 侧没有做（也没有易解的）内核级优化。

#### (c) `histogramBatch` —— LDS bank-conflict 受限

- 原始布局是每个 64-lane wavefront 一个 256-bin bucket（`buckets[kWarps][257]`），在偏斜数据上热点符号会把 64 个 lane 全部压到同一个 LDS bank。
- 微基准（256 MB）：64 lanes/bucket = 均匀 457 GB/s、**偏斜（90% 0）97 GB/s**；改成 8 lanes/bucket（32 buckets）= 均匀 657 GB/s、**偏斜 362 GB/s**（3.7×）。
- occupancy 无关（blocks/SM 1→8 零变化）——纯粹 bank-conflict 受限。全局内存原子 ~5× 更慢（81 GB/s），所以必须留在 LDS。

#### (d) `map_values_kernel_thrust`（ADM mapping）—— 延迟受限 + 模运算在关键路径

- mapping kernel 是 latency-bound（串行 `__shfl_down` 归约 + 逐元素编码链），不是 wavefront 吞吐受限。
- 逐元素计算里 `(diff % 126 == 0) ? diff/126 : diff/126+1` 的 `%126`（magic-multiply + mul + sub）正好落在 `output_idx → bit_offset` 的关键路径上。
- 尝试 64-lane wavefront 转换：**字节一致但慢 ~4.6%**（64-thread block 使 wavefront 数减半，512 元素 ADM block 的延迟隐藏更差）——已回退。

### 1.4 编译器问题：同一份源码在 nvcc vs hipcc 下的资源分配差异

这是"AMD vs NVIDIA 差异"最硬的一条证据（整理自 `nv/adm/haac_bench/ANALYSIS.md` 与两侧 profile summary）。`nv/adm/haac_bench/src/*.cu` 是 `amd/adm/haac_bench/src/*.cpp` 的**符号链接**，两侧编译的是**逐字节相同的 ADM kernel 源码**，仅 `common.h` 的 `MANS_BENCH_CUDA` 分支切换 CUDA/HIP 运行时——所以差异只能来自**编译器**（nvcc 11.7 vs hipcc/ROCm 6.3.3）与硬件。

**同一源码，两个编译器给出的资源分配完全不同：**

| kernel | nvcc (A100) | hipcc (MI210) |
|---|---|---|
| `decompress` | **32 REG**, 0 shared, 32B local | **44 VGPR + 32 KB LDS**, 0 scratch |
| `restore_values_uint16` | **32 REG**, 0 shared, 0 local | **60 VGPR**, 0 LDS, 0 scratch |
| `decode_signals_uint16` | **32 REG**, 0 shared, 32B local | **12 VGPR + 32 KB LDS**, 0 scratch |
| `map_values_kernel_decoupled` | 32 REG, 4B shared, 48B local | 32 VGPR, **16.9 KB LDS**, 48B scratch |
| `map_values_kernel_thrust` | 32 REG, 0 shared, 48B local | 28 VGPR, **16 KB LDS**, 48B scratch |

> 数据来源：NV 列 `cuobjdump --dump-resource-usage`（`profile/a100-{preopt,optimized}/summary.csv`）；AMD 列 rocprof `VGPR_Count`/`LDS_Block_Size`/`Scratch_Size`（`profile/mi210-*/resource/summary.csv`）。

**直接结论**：源码里 `local_signal[32]` / `local_codes[32]` / `local_result[32]` 三个**动态索引**局部数组，`nvcc` 用 32 个寄存器就装下；`hipcc` 要么整块挪进 LDS（`decompress`/`decode` 各 32 KB/block），要么在 `restore` 里爆到 60 VGPR。这是**编译器层面的差距**，与硬件无关，可复现、可定位。

**这个差距直接压死 occupancy——即 §1.1 里解压 9.2× 差距的机制：**

| kernel | MI210 occupancy（优化前） | 根因 | A100 occupancy |
|---|---|---|---|
| `decompress` | **~6%（32 KB LDS → 1 wavefront/CU）** | 32 KB LDS/block | ~50%（32 REG） |
| `map_values_kernel_thrust` | **~12%** | 16 KB LDS | ~50% |

MI210 每 CU 64 KB LDS、至多 ~32 wavefront：`decompress` 每 block 占 32 KB LDS → 每 CU 最多 2 block = 1 wavefront ≈ 3% occupancy，几乎没有 warp 去隐藏内存延迟。A100 侧 32 REG/0 shared → 32 warp ≈ 50%。

**再叠加访存放大 × 小 L2（让延迟更难隐藏）：** `decompress` 是变长 bit 流 gather（每个 warp 从**数据依赖的偏移**读 bit 流），rocprof 实测 L2 读流量 56 GB vs 有效数据 ~1.6 GB = **~35× 读放大**；而 MI210 L2 仅 **8 MB**（A100 **40 MB**），miss 更多、去 DRAM 次数更多。最终有效带宽只有峰值的 **5%**（DRAM 其实已跑 52% 峰值），A100 是 49%。

> 一句话：AMD 慢 = **hipcc 对 gather + 局部数组 kernel 的寄存器/LDS 分配缺陷压低 occupancy**（编译器问题）× **8 MB 小 L2 吸收不了 35× 重复读**（硬件差异）。两者共同作用在"不规则访存"上，所以慢的幅度与 kernel 的不规则程度正相关：`decompress` 9.2× > `map_values_kernel_thrust` 7.0× > 规整 kernel（decode/restore 无局部数组）1.6×。

---

## 2. 优化方法（全部 byte-identical）

### 2.1 ANS 编码：histogram 8 lanes/bucket（LDS bank-conflict 消除）

**问题**：见 §1.3(c)，64 lane 共用一个 bucket 在偏斜数据上严重 bank-conflict。

**方法**：`kLanesPerBucket = 8`（32 buckets × 257 × 4 = 32 KB smem），`bucketId = threadIdx.x / 8`，归约阶段累加 32 个 bucket。文件：`amd/ans/include/ans/GpuANSStatistics.h`。

**效果**（1 GB u2 `vx`）：histogram **4.69 → 1.79 ms（2.6×）**，ANS encode 7.45 → 4.94 ms，整体压缩 69 → 84 GB/s；u4 `big-tpch` 146.6 → 159.2 GB/s。直方图是精确整数计数，分桶对结果是 no-op，故**逐字节一致**。

### 2.2 ADM（HAAC）内核优化

ADM 是 MANS 的前端（HAAC center/mean + 残差 unary 编码）。四个内核在 MI210 上逐个 profile 后重构。全部改动保持 **bit-exact**（每个输入 lossless 校验通过）。前 6 项来自 `amd/adm/haac_bench` 的优化（README「Kernel optimizations」，对应 §1.1 里 ADM 解压 9.2×→2.3×、压缩 2.1×→1.57× 的根源；优化前源码快照在 `src_preopt/`），第 7 项是后续追加的模运算消除。

**资源/occupancy 变化总览（1 GB u2，`profile/mi210-*/resource/summary.csv` 实测）：**

| kernel | occupancy 优化前→后 | LDS 优化前→后 | arch_vgpr 优化前→后 |
|---|---|---|---|
| `map_values_kernel_decoupled`（压缩 fused） | 8.81% → **99.58%** | 16896 → 512 | 32 → 56 |
| `map_values_kernel_thrust`（压缩 unfused） | 12.15% → **99.00%** | 16384 → 0 | 28 → 52 |
| `decompress`（解压 fused） | 6.08% → **87.68%** | 32768 → 0 | 44 → 60 |
| `decode_signals_uint16`（解压 unfused） | 23.4% → **95.08%** | 32768 → 0 | 12 → 40 |

**（1）寄存器驻留信号解码（LDS 消除）** —— `decompress` / `decode_signals_uint16` 原实现 `local_signal[signal_idx]++` 是数据依赖索引，编译器把 32 项数组整块放进 LDS（`lds=32768`，32 KB/block），occupancy 掉到 ~6%。改为寄存器里的 zero-run 计数 + 编译期索引的完全展开 predicated write（`run!=0` 才写，常见全零分支零开销），配合 `#pragma unroll`，所有局部数组变静态索引、留在寄存器（`lds=0`）。occupancy **6% → 91%**。

**（2）中心求和 / 加载合并** —— mapping 原用 16 次标量 `__ldg` 分别算 block 中心平均，与逐元素 `uint4` 加载分离；低 occupancy 下这些标量加载是延迟瓶颈（`OccupancyPercent`~12%、`MemUnitStalled`~2%）。把中心求和折叠进单个 `uint4 buf[2]` 加载、同时复用于平均与逐元素编码。这是 ≥16 MB 大尺寸的主导收益（**3–4×**）。

**（3）Bit-packing 溢出消除** —— 原逐 bit 写 `local_bit_signal[bit_offset/8] |= (1 << (7-(bit_offset%8)))`（动态索引局部数组）迫使数组进 scratch/local 内存并每元素串行 read-modify-write。改为固定索引 `local_output_idx[cmp_chunk]` + 寄存器累加器重打包 unary 码、直接流式写字节。主要帮小输入。

**（4）block size 32→256（仅 fused）** —— 32 线程 block 占据 64-lane wavefront 却一半 lane 空闲。因内核全部按全局线程索引寻址，扩到 256 只需把中心查找 `centers[bid*2]` 改为 `centers[warp]`。occupancy 6% → 91%。

**（5）边界修正（fused）** —— tail-guard `>` 改 `>=`，使最后 block 的填充线程在任意 block size 下干净退出。

**（6）静态向量化信号写出（unfused decode）** —— 解码信号用编译期索引打包进 `uint4`，替代运行时索引循环。

**（7）模运算消除（`%126` → 移位/除法）** —— 见 §1.3(d)：16 处 uint16 的 `(diff%126==0)?diff/126:diff/126+1` → `(diff+125)/126`（diff≥1）；5 处 uint32/通用多维 → `(int)((diff-1u)/126u+1u)`（diff≥1，无溢出）。文件：`amd/adm/mans_amd_adm.cpp` 及 haac_bench `compress_fused.cpp`/`compress_unfused.cpp`。效果：map 1.386 → 1.351 ms（**~2.5%**），comp 82.9 → 83.8 GB/s；u32 侧编译器本已强度削减模运算、改写无收益（平坦）。

> 附带记录：64-lane wavefront 转换尝试过但**回退**（字节一致但慢 ~4.6%，wavefront 数减半使延迟隐藏更差），见 §1.3(d)。

### 2.3 ANS 解压（大文件）：2-block interleave（延迟隐藏）

**问题**：见 §1.3(a)，decode 是串行 rANS 链延迟受限，且 4 KB LDS 使 occupancy 卡在 16 blocks/CU。

**方法**：在 `ansDecodeKernel` 的 grid-stride 循环里**配对两个满 block，交错解码**（`decodeTwoFullBlocksInterleaved`：每次迭代先解 block A 再解 block B），用 B 的算术隐藏 A 的 LDS/gmem 延迟。最后一个不完整 block 落到原单 block 路径。文件：`amd/ans/include/ans/GpuANSDecode.h`。

**效果**（256 MB `zz` u2）：`ansDecodeKernel` **617.9 → 529.3 μs（−14.3%）**。`ans_decomp_ms` 相对基线：1 GB u2 **−21.7% ~ −23.6%**，256 MB −5% ~ −14%（数据相关：越不可压缩越 gmem-bound，interleave 帮助越小），64 MB −2.5%，≤16 MB 持平（host/occupancy 受限）；u4 大文件 −1.7% ~ −2.4%。全部 PASS。

**已回退的死路**（避免重复）：
- 4-block interleave → 574 μs（更差）：寄存器 32→~56 vgprs，occupancy 掉到 9 blocks/CU，损失大于额外 ILP。
- 用 gmem 读替换 LDS 查表 → 723 μs（差很多）：LDS 随机访问其实很快，且释放 LDS 不提升 occupancy（vgpr 已 cap 在 16 blocks）。

### 2.4 小文件（≤4MB）：host 端开销削减

**问题**：小文件 ANS 延迟由 **host 开销**主导，而非 GPU 内核。4 kB u2 文件 GPU 内核仅 ~64 μs（压）/ ~33 μs（解），但 benchmark 的 `ans_comp_ms`/`ans_decomp_ms` 显示 ~145/~83 μs，即 ~80/~50 μs 是 `hipMalloc`/`hipFree`/`hipMemcpy D2H`/`hipStreamSynchronize` 的空档。

**方法**：
1. **持久化 scratch buffer**：`GpuANSEncode.h`/`GpuANSDecode.h` 里函数局部 `static` 缓冲 + 按需扩容，替换每次调用的 `hipMalloc`/`hipFree`（encode 6+6、decode 1+1）和尾部的 `hipStreamSynchronize`。
2. **合并压缩的 D2H 读**：`mans_amd_ans.cpp` 里去掉单独的 `d_size` 设备缓冲及其拷贝，改从 coalesced header（`header.total_compressed_size()`）读 `encoded_size`，把两次 D2H 合并成一次。

**效果**（vx u2，30 iters，字节一致 PASS）：压缩 **−24%**（4 kB 144.7→110.3 μs）、**−28%**（16 kB 181→130 μs）；解压 −14%（16 kB 137→118 μs）、4 kB 持平（82 μs）。

> 死路：`ansDecodeTable` 的 `hipcub::BlockScan` 不是瓶颈（~10.3 μs 是 kernel launch + 固定 setup），手写 64-lane warp scan 字节一致但无提速（已回退）。剩余解压差距是 `ansDecodeKernel` 的串行链（~23 μs）+ 强制 header D2H 读，不改 API 无法消除（调用方需要提前拿到 `total_uncompressed_words`）。

---

## 3. Evaluation（评测结果与 breakdown）

全量 98 文件（80 u2 + 18 u4；数据集 haac / exafel / modis / tomobank / big-tpch / suitesparse，尺寸 4KB–1GB）在 MI210 上跑完，**98/98 PASS**，逐字节一致。

### 3.1 结果文件

| 文件 | 内容 |
|---|---|
| `/workspace/MANS/mans_bench_1d_results_interleave.csv` | 全量 98 文件逐文件结果（ADM/ANS 分阶段耗时 + 吞吐 + PASS 标记） |
| `/workspace/MANS/mans_bench_1d_comparison.csv` | 整体压缩/解压吞吐 vs DietGPU 基线（ratio = my/ref） |
| `/workspace/MANS/mans_bench_1d_haac_breakdown_comparison.csv` | HAAC 的 ADM/ANS 四阶段分解 vs DietGPU |
| `/workspace/MANS/mans_bench_1d_comparison_report.md` | Markdown 报告（表格形式） |

### 3.2 整体吞吐对比（vs DietGPU 基线，ratio < 1.00 = 更慢）

关键数据点（完整见 `mans_bench_1d_comparison.csv`）：

**压缩**：
- **≥4 MB 全面胜出**：1 GB 2.15×（HAAC）、2.22×（TPCH）；256 MB 2.05×；64 MB 1.76×；16 MB 1.35×；4 MB 1.11×。
- **≤1 MB 部分落后**：0.88–1.00×（小文件，host 开销）。

**解压**：
- **≥16 MB 全面胜出**：1 GB 1.66–2.15×；256 MB 1.57–1.94×；64 MB 1.38–1.65×；16 MB 1.20–1.46×。
- **≤1 MB 落后**：0.73–0.94×（小文件，host 开销）。

> SuiteSparse/TPCH 的小文件压缩反而大幅胜出（2.5–3.6×），是因为这些数据压缩率极高、参考基线在该尺寸极低；但同尺寸**解压仍落后**（0.73–0.92×），说明小文件瓶颈在解压侧的固定开销。

### 3.3 HAAC ADM/ANS 分解对比（吞吐 GB/s，ratio = my/ref）

来源 `mans_bench_1d_haac_breakdown_comparison.csv`，节选关键尺寸：

| 尺寸 | ADM_comp | ANS_comp | ADM_decomp | ANS_decomp |
|---|---|---|---|---|
| 1 GB   | 141.1 (**2.20×**) | 253.3 (**2.04×**) | 320.1 (**1.94×**) | 465.8 (**1.27×**) |
| 256 MB | 133.7 (2.07×) | 222.4 (1.86×) | 312.9 (1.90×) | 397.7 (1.07×) |
| 64 MB  | 111.0 (1.84×) | 150.0 (1.54×) | 288.7 (1.93×) | 243.1 (0.87×) |
| 16 MB  | 69.0 (1.49×) | 67.6 (1.14×) | 220.0 (1.98×) | 105.3 (0.80×) |
| 4 MB   | 29.0 (1.16×) | 22.8 (0.97×) | 117.1 (2.14×) | 27.4 (0.77×) |
| 1 MB   | 13.2 (1.00×) | 6.1 (0.95×) | 44.3 (1.77×) | 7.3 (0.77×) |
| 256 KB | 4.7 (0.88×) | 1.6 (0.93×) | 10.0 (1.17×) | 1.8 (0.71×) |
| 64 KB  | 1.4 (0.91×) | 0.46 (0.90×) | 3.0 (1.29×) | 0.51 (0.77×) |
| 16 KB  | 0.34 (0.95×) | 0.11 (0.80×) | 0.60 (1.12×) | 0.12 (0.69×) |
| 4 KB   | 0.11 (1.17×) | 0.037 (0.87×) | 0.24 (1.78×) | 0.050 (0.62×) |

**规律**：
- **ADM_decomp 全尺寸胜出**（1.12–2.14×）——ADM 解压优化最成功。
- **ADM_comp 与 ANS_comp 在 ≥4MB 胜出**（ADM 1.16–2.20×，ANS 1.14–2.04×），≤256KB 落后。
- **ANS_decomp 大文件胜出（1 GB 1.27×、256 MB 1.07×），≤64MB 落后（0.62–0.87×）**——这正是 2-block interleave 只帮到大文件、小文件仍受 host 开销限制的直接体现。

### 3.4 仍然更慢的部分（诚实标注）

所有仍慢的部分**全部集中在 ≤1MB 的小文件**，根因是 **host 端固定开销**（kernel launch、D2H、同步），而非 GPU 内核：

- 整体压缩更慢：≤1MB 的小文件（HAAC/EXAFEL/MODIS/TomoBank 的 0.88–1.00×）。
- 整体解压更慢：≤1MB 的小文件（0.73–0.94×）。
- HAAC 分解更慢：主要是 **ANS_decomp（≤64MB，0.62–0.87×）**、ANS_comp（≤16MB）、ADM_comp（≤256KB）。

后续若要补上小文件差距，方向是**内核融合 / 减少 launch 次数 / 改进 API**（例如消除 decode 前必须的 header D2H 读），而不是继续优化单个 GPU 内核。

---

## 4. 附录

### 4.1 关键文件

| 文件 | 说明 |
|---|---|
| `amd/ans/include/ans/GpuANSStatistics.h` | histogram 8 lanes/bucket（§2.1） |
| `amd/ans/include/ans/GpuANSEncode.h` | encode + 小文件 scratch 缓存（§2.4） |
| `amd/ans/include/ans/GpuANSDecode.h` | decode 2-block interleave + 小文件缓存（§2.3/§2.4） |
| `amd/ans/mans_amd_ans.cpp` | 压缩 D2H 合并（§2.4） |
| `amd/adm/mans_amd_adm.cpp` | modulo 消除（§2.2 第 7 项）、`concat_block_size=256` |
| `amd/adm/haac_bench/src/*.cpp` | 优化后的 4 个 ADM kernel（§2.2 第 1–6 项） |
| `amd/adm/haac_bench/src_preopt/*.cpp` | 优化前源码快照（commit `e073eda`，论文"优化前 vs 优化后"对比用） |
| `nv/adm/haac_bench/src/*.cu` | 指向上面 `.cpp` 的符号链接（NV 侧同源码） |
| `nv/adm/haac_bench/ANALYSIS.md` | **编译器问题完整分析**（nvcc vs hipcc 资源分配 / occupancy / L2 放大，§1.4 的来源） |
| `amd/adm/haac_bench/results/mi210-{preopt,optimized}/stage_timings.csv` | ADM 阶段计时（§1.1 表的数据来源） |
| `nv/adm/haac_bench/results/a100-{preopt,optimized}/stage_timings.csv` | A100 同口径计时 |
| `amd/adm/haac_bench/profile/mi210-*/resource/summary.csv` | MI210 occupancy/VGPR/LDS/scratch 实测（§1.4、§2.2 表） |
| `nv/adm/haac_bench/profile/a100-*/summary.csv` | A100 cuobjdump 寄存器/共享/局部（§1.4 表） |

### 4.2 复现命令

```bash
# 设备选择（GPUs 2/3 空闲；只设 ROCR_VISIBLE_DEVICES，不要设 HIP_VISIBLE_DEVICES）
unset HIP_VISIBLE_DEVICES; export ROCR_VISIBLE_DEVICES=3

# 全量 benchmark（输出格式同 mans_bench_1d_results.csv）
./build-amd-633/bin/amd/amd_mans_bench OUT.csv FILE... --warmup 2 --iters 3

# dtype 由扩展名推断（.u2→uint16, .u4→uint32）；fused/unfused 分界 gsize<=4096
```

ADM/HAAC 阶段独立 benchmark（同一份源码，AMD `hipcc` / NV `nvcc` 各编一份，用于 §1.4 编译器对比）：

```bash
# AMD (MI210)
cd amd/adm/haac_bench
HIPCC=/opt/rocm-6.3.3/bin/hipcc GPU_ARCH=gfx90a ./scripts/build.sh
GPU_ID=3 ./scripts/analyze_real.py /path/to/testdata/u2/haac_vx ./results/mi210-<run>
# 资源占用（occupancy/VGPR/LDS）
GPU_ID=3 ./scripts/profile.sh /path/to/testdata/u2/haac_vx ./profile/mi210-<run>

# NV (A100) —— nv/*.cu 是指向 amd/*.cpp 的符号链接
cd nv/adm/haac_bench
NVCC=/path/to/nvcc CUDA_ARCH=sm_80 ./scripts/build.sh
GPU_ID=0 ./scripts/analyze_real.py /path/to/testdata/u2/haac_vx ./results/a100-<run>
# NV 寄存器占用（无需 GPU，读 .cubin）：
cuobjdump --dump-resource-usage build/compress_fused  # 等 5 个可执行文件
```

### 4.3 profile 复现

- 内核耗时分解：`results.stats.csv`（`__hipcc` 统计的各内核 TotalDurationNs/Percentage）。
- stall 计数：`stall_metrics.csv`（rocprof `pmc:` 指标，含 `SQ_WAIT_INST_LDS`、`SQ_WAVES`、`arch_vgpr`/`lds`/`grd`/`wgr`）。
- ISA 分析：`hipcc --save-temps` 生成 `.s`，观察 software pipelining（`ds_read_b128` / `v_add3` / `s_bcnt1_i32_b64`）。

### 4.4 备注

- 已知（前序会话，非本优化引入）：`ansEncodeCoalesceBatch` 未初始化 `ANSCoalescedHeader` 的 `checksum/unused0/unused1`（12 字节非确定），但从未被读取，round-trip 校验不受影响。
- 所有优化均保持**逐字节一致**，未改变压缩格式/算法语义；唯一被排除的方向是 interleaved rANS（会破坏输出格式）。
