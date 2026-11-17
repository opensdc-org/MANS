# 优化前的 ADM kernel 源码快照

本目录保存 **优化前**(commit `e073eda`,即优化提交 `62242c5` 的父提交)的 4 个
ADM kernel 源码,与当前工作区 `src/` 目录下的**优化后**版本并存,用于论文里
"优化前 vs 优化后"的代码对比。

| 文件 | 说明 |
|---|---|
| `compress_fused.cpp` | 融合压缩 → `map_values_kernel_decoupled` |
| `compress_unfused.cpp` | 非融合压缩 → `map_values_kernel_thrust` |
| `decompress_fused.cpp` | 融合解压 → `decompress` |
| `decompress_unfused.cpp` | 非融合解压 → `decode_signals_uint16` + `restore_values_uint16` |

## 来源

- 优化前 commit:`e073eda`(`update mi210`)
- 优化 commit:`62242c5`(`optimized mi210`)——改动即这 4 个文件的 kernel 实现
  (向量化 `uint4` 加载 + 消除动态索引局部数组,消除寄存器溢出)。
- 导出命令:
  ```bash
  git show e073eda:amd/adm/haac_bench/src/<file>.cpp > src_preopt/<file>.cpp
  ```

## 与优化后版本对比

```bash
cd amd/adm/haac_bench
diff -u src_preopt/decompress_fused.cpp src/decompress_fused.cpp   # 以此类推
# 或直接看优化提交的源码 diff:
git show 62242c5 -- src/decompress_fused.cpp
```

## 注意

NV 侧 `nv/adm/haac_bench/src/*.cu` 是指向 `amd/adm/haac_bench/src/*.cpp` 的
符号链接,因此本快照同时代表 NV 侧的优化前源码(两侧编译的是同一份 kernel,
仅 `common.h` 的 `MANS_BENCH_CUDA` 分支切换 CUDA/HIP 运行时)。
