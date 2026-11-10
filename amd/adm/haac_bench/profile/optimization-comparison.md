# MI210 (gfx90a) 优化前 vs 优化后 profile 对比

对比对象：

- **优化前**：`profile/mi210/`（L2 命中率）+ 导出到 `/tmp/haac_pre/` 的旧代码二进制（资源占用、耗时）
- **优化后**：`profile/mi210-opt/`（L2 命中率）+ `profile/mi210-resource-usage/`（资源占用）+ `results/mi210-final/`（耗时）

输入统一为 `vx_1073726487_1048576kB.u2`（1 GB，536,870,912 个 uint16 元素）。资源占用为 15 次 dispatch 均值。

---

## 1. 资源占用对比（8 kernel，1 GB 输入）

| kernel | stage | 指标 | 优化前 | 优化后 | 变化 |
|---|---|---|---|---|---|
| `map_values_kernel_decoupled` | compress_fused | occupancy % | 8.81 | 99.58 | **+90.8** |
| | | VALU % | 32.49 | 21.12 | −11.4 |
| | | MemStall % | 2.15 | 2.68 | +0.5 |
| | | wgr / lds / scr | 32 / 16896 / 48 | 32 / 512 / 0 | lds、scr 基本消除 |
| `map_values_kernel_thrust` | compress_unfused | occupancy % | 12.15 | 99.00 | **+86.9** |
| | | VALU % | 43.86 | 43.21 | −0.7 |
| | | MemStall % | 5.15 | 12.30 | +7.2 |
| | | lds / scr | 16384 / 48 | 0 / 0 | 完全消除 |
| `init_lookback_scan_state_kernel` | compress_unfused | occupancy % | 0.10 | 0.10 | 不变 |
| `lookback_scan_kernel` | compress_unfused | occupancy % | 22.84 | 23.68 | +0.8 |
| | | lds | 14336 | 14336 | 不变（thrust 内部） |
| `concat` | compress_unfused | occupancy % | 50.40 | 90.64 | **+40.2** |
| | | VALU % | 50.00 | 100.00 | **+50** |
| | | MemStall % | 13.77 | 35.78 | +22.0 |
| | | wgr | 32 | 256 | 32→256 |
| `decompress` | decompress_fused | occupancy % | 6.08 | 87.68 | **+81.6** |
| | | VALU % | 50.00 | 100.00 | **+50** |
| | | MemStall % | 2.83 | 69.69 | +66.9 |
| | | wgr / lds | 32 / 32768 | 256 / 0 | lds 完全消除 |
| `decode_signals_uint16` | decompress_unfused | occupancy % | 23.40 | 95.08 | **+71.7** |
| | | VALU % | 100.00 | 100.00 | 不变 |
| | | MemStall % | 8.24 | 16.85 | +8.6 |
| | | lds | 32768 | 0 | 完全消除 |
| `restore_values_uint16` | decompress_unfused | occupancy % | 84.32 | 82.43 | −1.9 |
| | | VALU % | 100.00 | 100.00 | 不变 |
| | | MemStall % | 78.41 | 75.99 | −2.4 |

**要点**

- 所有经过优化的 kernel（映射、decode、concat、fused decompress）occupancy 都从 <25% 提升到 82%+，`lds`/`scr` 局部内存与寄存器溢出被消除。
- `MemUnitStalled` 在解压类 kernel 里明显上升（decompress 2.83→69.69，concat 13.77→35.78）：这不是退步，而是 occupancy 抬升后更多 wavefront 同时竞争内存单元，原本被低 occupancy 掩盖的内存延迟暴露出来——解压本质上是内存密集（memstall 高是这类 kernel 的正常画像）。
- `restore_values_uint16` 和两个 thrust scan kernel 本来就没有局部内存问题，改动后基本不变，符合预期。

---

## 2. 耗时对比（HIP-event kernel 时间，1 GB 输入）

| stage | 阶段 | 优化前 (ms) | 优化后 (ms) | 加速 |
|---|---|---|---|---|
| compress_fused | fused_kernel | 76.88 | 23.65 | **3.25×** |
| compress_unfused | mapping | 25.71 | 5.17 | **4.97×** |
| | prefix_sum | 0.08 | 0.08 | 1.00× |
| | concat | 1.30 | 0.97 | **1.34×** |
| | **total_gpu** | **27.09** | **6.22** | **4.36×** |
| decompress_fused | fused_kernel | 13.12 | 3.32 | **3.95×** |
| decompress_unfused | decode | 1.63 | 1.11 | 1.47× |
| | restore | 3.04 | 2.89 | 1.05× |
| | **total_gpu** | **4.67** | **4.00** | 1.17× |

> 注：`concat` 优化后的 0.97 ms 是重测值；`results/mi210-final/stage_timings.csv` 中 concat 仍为 1.30 ms（该 CSV 在 concat 优化前生成）。compress_unfused 优化后 total_gpu 由 mapping 5.17 + prefix 0.08 + concat 0.97 得到。

**要点**

- 压缩侧是主要收益：fused mapping 3.25×、unfused mapping 4.97×，直接来自 `center-sum/load 合并` 与 `bit-packing spill 消除`（资源占用表里 lds 16896→512、scr 48→0）。
- concat 通过块大小 32→256 获得 1.34×（occupancy 50→91%，VALU 50→100%）。
- 解压侧 fused 3.95× 受益于 lds 消除 + 块大小 32→256；unfused 解压只提升 1.17×，因为 `restore_values_uint16` 本就是内存密集且未大改，`decode_signals_uint16` 虽 lds 消除但整体占比小。

---

## 3. L2 命中率对比（1 GB 输入）

| stage | 优化前 L2 % | 优化后 L2 % | Δ |
|---|---|---|---|
| compress_fused | 90.79 | 89.44 | −1.35 |
| compress_unfused | 80.66 | 69.13 | −11.52 |
| decompress_fused | 82.66 | 88.97 | **+6.32** |
| decompress_unfused | 74.95 | 75.17 | +0.22 |

**要点**

- L2 命中率不是吞吐差异的因果证明（见 `README.md`「Optional L2 counter profiling」），只能作为旁证。
- `compress_unfused` 命中率下降 11.5pt，但吞吐反而提升 4.36×：优化后 mapping 不再反复读中心值到 L2（原来 16 次标量 `__ldg` 触发的缓存行为改变），且访存模式从 scratch/局部内存改为寄存器流式写出。
- `decompress_fused` 命中率上升 6.3pt，与 lds 消除后访存更规整一致。

---

## 4. 数据文件索引

| 数据 | 路径 |
|---|---|
| 优化前 L2 命中率 | `profile/mi210/summary.csv` |
| 优化后 L2 命中率 | `profile/mi210-opt/summary.csv` |
| 优化后资源占用（8 kernel） | `profile/mi210-resource-usage/summary.csv` |
| 优化前资源占用原始 | `/tmp/pre_cmp_fused.csv`、`/tmp/pre_cmp_unfused.csv`、`/tmp/pre_dec_fused.csv`、`/tmp/pre_dec_unfused.csv` |
| 优化前耗时 | `/tmp/haac_pre/amd/adm/haac_bench/results/mi210-vx-real-stage/stage_timings.csv` |
| 优化后耗时 | `results/mi210-final/stage_timings.csv` |
| 优化前后资源对比 CSV | `/tmp/pre_vs_post_resource.csv` |
