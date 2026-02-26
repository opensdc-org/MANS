# Minimal FSE (CMake)

这个目录是从原始工程中抽离出的最小 FSE 核心子集，仅保留：

- FSE 核心压缩/解压实现
- 必要依赖（bitstream/hist/error/mem/debug）
- 一个最简 bench：`fse_bench`

不包含：

- 原 `programs/` 下复杂命令行工具
- fullbench/fuzzer/probagen 等测试工具
- HUF 压缩实现源码

## 目录结构

- `include/`：FSE 核心头文件
- `src/`：FSE 核心实现
- `bench/fse_bench.c`：最简压缩程序（输入原始文件，输出压缩文件）

## 构建

在仓库根目录执行：

```bash
cmake -S . -B build
cmake --build build -j
```

或仅构建子目录：

```bash
cmake -S fse -B build
cmake --build build -j
```

## 运行

```bash
./build/fse_bench /path/to/input.bin /path/to/output.mfse
```

输出会包含：

- raw 大小
- compressed 大小
- `raw/compressed`
- `compressed/raw`

注意：

- 分块与 `raw` / `rle` 回退逻辑已下沉到库中 `FSE_compress()` / `FSE_decompress()`。
- `bench` 只负责文件读写和调用 `FSE_compress()`。
