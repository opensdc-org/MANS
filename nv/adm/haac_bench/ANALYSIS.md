# AMD MI210 vs NVIDIA A100 —— HAAC ADM 性能差异分析

> 结论先行:**同一份 ADM kernel 源码**(NV 侧 `src/*.cu` 是对 AMD 侧 `.cpp` 的符号链接,两侧编译的是逐字节相同的 kernel),MI210 在解压上比 A100 慢 9.2×、压缩慢 2~7×。根因不是算法,而是 **hipcc 编译器对 gather 型 kernel 生成的代码 occupancy 被压死 + MI210 L2 只有 A100 的 1/5**。
>
> 数据可信度标注(全文沿用):
> - ✅ 实测(profiler / 反汇编 / GPU event)
> - 🧮 由实测数据 + 公开规格推算
> - 📐 公开硬件规格
> - ❌ 缺失(未实测,原因见 §7)

---

## 1. 背景与方法

- **对比对象**:AMD Instinct MI210(`gfx90a`)vs NVIDIA A100-PCIE-40GB(`sm_80`)。
- **源码一致性**:`nv/adm/haac_bench/src/*` 是 `amd/adm/haac_bench/src/*.cpp` 的符号链接,`common.h` 通过 `#ifdef MANS_BENCH_CUDA` 把 `hip*` 映射到 `cuda*`。两侧编译的是**同一份 ADM kernel 源码**,只是编译器不同(nvcc 11.7 vs hipcc/ROCm)。
- **测试数据**:HAAC `vx` 系列(4 kB ~ 1 GB,10 个尺寸),两侧同源。
- **计时口径**:kernel 时间一律用 GPU event(`hipEvent`/`cudaEvent`),warmup 5 次 + 10 次取平均,不含 host 侧 memcpy/文件 IO/运行时启动。

---

## 2. 端到端 kernel 时间对比 ✅

以 1 GB 输入为准(此时传输与启动开销可忽略,kernel 时间主导):

| 阶段 / kernel | AMD MI210 (ms) | NV A100 (ms) | 慢倍数 |
|---|---:|---:|---:|
| `decompress`(解压 fused) | 13.12 | 1.42 | **9.2×** |
| `map_values_kernel_thrust`(压缩 unfused) | 25.71 | 3.69 | **7.0×** |
| `map_values_kernel_decoupled`(压缩 fused) | 76.88 | 37.19 | 2.1× |
| `decode`+`restore`(解压 unfused) | 4.67 | 2.90 | 1.6× |

> 数据来源:`amd/.../results/mi210-vx-real-stage/stage_timings.csv` 与 `nv/.../results/a100-preopt/stage_timings.csv` 的 `fused_kernel_ms` / `mapping_ms` / `decode_ms` / `restore_ms` 字段(HIP/CUDA event,实测)。

**规律**:慢得最多的(`decompress` 9.2×、`map_values_kernel_thrust` 7.0×)都是**变长 bit 流 gather / 查表拼接**且**携带局部数组**的 kernel;而访存规整、无局部数组的(`decode`/`restore`/`decoupled` 的主体)只慢 1.6~2.1×。差距随"不规则访存 + 局部数组"程度递增。

---

## 3. 根因一:编译器代码质量差异 ✅

对同一份源码,两个编译器给出的资源分配完全不同(这是全篇最硬的一条证据):

| kernel | nvcc(A100)寄存器/shared/local | hipcc(MI210)VGPR/LDS/scratch |
|---|---|---|
| `decompress` | **32 REG**, 0 shared, 32B local | **44 VGPR + 32 KB LDS**, 0 scratch |
| `restore_values_uint16` | **32 REG**, 0 shared, 0 local | **60 VGPR**, 0 LDS, 0 scratch |
| `decode_signals_uint16` | **32 REG**, 0 shared, 32B local | **12 VGPR + 32 KB LDS**, 0 scratch |
| `map_values_kernel_decoupled` | 32 REG, 4B shared, 48B local | 32 VGPR, **16.9 KB LDS**, 48B scratch |
| `map_values_kernel_thrust` | 32 REG, 0 shared, 48B local | 28 VGPR, **16 KB LDS**, 48B scratch |

- ✅ NV 列:`cuobjdump --dump-resource-usage` 反汇编 `build/*` 二进制所得(`REG`/`SHARED`/`LOCAL`)。
- ✅ AMD 列:`rocprof` 的 `counter_collection.csv`(`VGPR_Count`/`SGPR_Count`/`LDS_Block_Size`/`Scratch_Size`)。
- 🧮 `scratch` 与 `LOCAL`(STACK)都表示寄存器溢出到片外内存的字节数/线程。

**直接结论**:源码里的 `local_signal[32]` / `local_codes[32]` / `local_result[32]` 三个**动态索引**局部数组,`nvcc` 用 32 个寄存器就装下,`hipcc` 要么把它们整块挪进 LDS(shared,`decompress`/`decode` 各 32 KB/block),要么在 `restore` 里直接爆到 60 个 VGPR。这是**编译器(而非硬件)层面的差距**,可复现、可定位。

---

## 4. 根因二:occupancy 被 LDS / VGPR 压死 🧮

由 §3 的实测资源占用 + 📐 硬件规格推算。

**MI210 侧**(规格来源:rocprof `agent_info.csv`,实测)📐:
- 每 CU **LDS = 64 KB**,`Max_Waves_Per_Simd = 8`(4 SIMD/CU → 每 CU 至多 ~32 wavefront),`Wave_Front_Size = 64`。

`decompress` 每 block(32 线程)占 **32 KB LDS** → 每 CU 最多 **2 个 block = 64 线程 = 1 个 wavefront**。

| | occupancy(推算) |
|---|---|
| MI210 `decompress` | 1 wavefront / 32 ≈ **~3%** |
| A100 `decompress` | 32 REG + 0 shared → 受 32 block/SM 限制 → 32 warp / 64 ≈ **~50%** |

> 🧮 推算依据:MI210 每 CU 64 KB LDS 与至多 32 wavefront(实测 agent_info)+ `decompress` 每 block 32 KB LDS(实测 rocprof)。A100 每 SM 至多 32 block、64 warp(公开规格)+ 32 REG/0 shared(实测 cuobjdump)。**occupancy 的精确百分比未实测**(❌ 需 profiler 计数器,见 §7)。

**结论**:AMD 侧 `decompress`/`decode` 的 32 KB LDS 把同时驻留的 wavefront 压到 ~1 个,几乎没有足够的 warp 去隐藏内存延迟;`restore` 的 60 VGPR 同理。这正是"有局部数组的 kernel 慢 7~9×、没局部数组的慢 1.6×"的机制解释。

---

## 5. 根因三:gather 访存 × 小 L2 ✅(源码)+ ✅(AMD 计数)+ 📐(规格)

### 5.1 访存模式:变长 bit 流 gather ✅(源码证据)

`decompress_fused.cpp` 内,`decompress` 的核心读是数据依赖的:

```c
int src_start_idx = d_output_lengths[warp] * cmp_tblock_size;   // 起始偏移:数据依赖
int length        = d_output_lengths[warp+1] - d_output_lengths[warp]; // 长度:数据依赖
...
bit_buffer = d_concatenated_signals[src_start_idx + offset_byte]; // 变长 bit 流随机读
```

每个 warp 从**由压缩数据决定的偏移**去读 bit 流,是非连续、变长的 gather。

### 5.2 AMD 侧 L2 访存量 ✅(rocprof `TCC_HIT`/`TCC_MISS`,1 GB 解压实测)

| 指标 | 实测值 |
|---|---:|
| L2 sector 总访问 | 437.7 M 次 = **56 GB** 读流量 |
| 有效数据量(读 adm ~0.6 GB + 写 1 GB) | ~1.6 GB |
| **读放大** | **~35×** |
| L2 miss 率 | 18.0%(`TCC_MISS/(HIT+MISS)`) |
| 落到 DRAM 的读流量 | 10.1 GB |

35× 的重复读全靠 L2 吸收。而 L2 容量(📐 公开规格):

| GPU | L2 |
|---|---:|
| MI210 (`gfx90a`) | **8 MB** |
| A100 (`sm_80`) | **40 MB** |

同样的 gather 负载,MI210 的 8 MB L2 miss 更多、去 DRAM 次数更多,而每次 miss 的延迟又因 §4 的低 occupancy 无法隐藏——三者相乘。

---

## 6. 带宽利用率:AMD "算得动、搬不动" 🧮(由实测时间 + 实测计数推算)

以 1 GB 解压为例:

| 指标 | AMD MI210 | NV A100 |
|---|---:|---:|
| kernel 时间(✅ event) | 13.09 ms | 1.42 ms |
| **有效带宽**(输出 1 GB ÷ 时间) | **82 GB/s = 5% 峰值** | 756 GB/s = 49% 峰值 |
| **DRAM 实际带宽**(✅ AMD 实测 11.1 GB ÷ 时间) | **848 GB/s = 52% 峰值** | ❌ 无实测 |

- 📐 峰值带宽:MI210 HBM2e ≈ 1.6 TB/s,A100 ≈ 1.55 TB/s(公开规格)。
- 🧮 AMD 的 DRAM 流量 = 读 10.1 GB(`TCC_MISS×128B`,实测)+ 写 1 GB ≈ 11.1 GB。

**关键洞察**:AMD 的 DRAM 其实**已经跑到 52% 峰值带宽**,但有效产出只有 5%——90% 的带宽被 35× 的重复读吃掉。所以这不是"算力不够",而是**访存效率被编译器产物 + 小 L2 双重拖死**。NV 侧 49% 的有效带宽利用率说明 A100 把绝大部分带宽花在了有用的数据上。

---

## 7. 结论

| 根因 | 数据支撑 | 可信度 |
|---|---|---|
| ① hipcc 对 gather+局部数组 kernel 生成低效代码(32 KB LDS / 60 VGPR) | rocprof + cuobjdump 反汇编 | ✅ 实测 |
| ② occupancy 被压到 ~3%(vs A100 ~50%) | 由 ① 的资源数据 + 硬件规格推算 | 🧮 推算 |
| ③ gather 变长访存(35× 读放大)× L2 8 MB vs 40 MB | 源码 + rocprof TCC 计数 + 公开规格 | ✅(源码/计数)+ 📐(规格) |
| ④ 有效带宽仅 5% 峰值,但 DRAM 已 52% | 由实测时间 + 实测计数推算 | 🧮 推算 |

**一句话总结**:AMD(MI210)比 NV(A100)慢,是**编译器(hipcc)对变长 gather 型 ADM kernel 的寄存器/LDS 分配缺陷压低 occupancy,叠加 MI210 只有 A100 1/5 的 L2 去吸收 35× 的重复读**——两个因素都作用在"不规则访存"这一共同点上,因此慢的幅度与 kernel 的不规则程度正相关(decompress 9.2× > mapping 7.0× > 规整 kernel 1.6×)。

---

## 8. 缺失数据与补测路径 ❌

以下结论点当前**没有实测数据**,为推断,需要 profiler 计数器才能坐实:

1. **NV 侧的 L2 hit rate / DRAM 流量** —— `ncu` 被 GPU 节点驱动参数 `RmProfilingAdminOnly=1` 锁死,普通用户无权限(❌ `ERR_NVGPUCTRPERM`)。
2. **两侧实测 occupancy** —— 需 `sm__warps_active`(NV)/ 对应 AMD 计数器。
3. **AMD `decompress` 32 KB LDS 的精确成因** —— 源码可见局部数组仅 ~128 B/线程,rocprof 报 32 KB/block,可能含编译器对动态索引数组的对齐/动态分配膨胀,未深究。

**补测方法**:由管理员在 GPU 节点设置 `RmProfilingAdminOnly=0`(或授予 `CAP_PERFMON`),然后运行本目录下已就绪的 `ncu_prof.sh` 即可拿到 NV 侧 L2 hit rate / occupancy 对照。AMD 侧可直接复用 `profile/` 下既有 `rocprof` 流程(注意 `compress_*` 阶段要过滤掉 `__amd_rocclr_copyBuffer`/`fillBufferAligned` 这些 memcpy/memset 底层 kernel,只统计 ADM kernel)。

---

### 附:关键数据文件位置

- 端到端计时(优化前):`amd/.../results/mi210-vx-real-stage/stage_timings.csv`、`nv/.../results/a100-preopt/stage_timings.csv`
- 端到端计时(优化后):`nv/.../results/a100-optimized/stage_timings.csv`(以及 AMD 侧 `mi210-final` 等)
- AMD 硬件计数器:`amd/.../profile/mi210/pmc/*/*_counter_collection.csv`(TCC_HIT/MISS、VGPR/SGPR/LDS)
- NV 寄存器占用(优化前/后):`nv/.../profile/a100-preopt/summary.csv` 与 `nv/.../profile/a100-optimized/summary.csv`(本文 §3 数据来源,`cuobjdump --dump-resource-usage` 所得)
