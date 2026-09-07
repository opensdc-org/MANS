# MANS: Efficient and Portable ANS Encoding for Multi-Byte Integer Data on CPUs and GPUs

[![C++ Version](https://img.shields.io/badge/C++-17%2B-blue.svg)](https://isocpp.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-12.6-yellow.svg)
![ROCm](https://img.shields.io/badge/ROCm-supported-red.svg)

**MANS** is a high-performance compression framework designed to make **Asymmetric Numeral Systems (ANS)** efficient and portable for **multi-byte integer data** across CPUs, NVIDIA GPUs, and AMD GPUs.

At the core of MANS is **ADM (Adaptive Data Mapping)** — a lightweight, distribution-aware transformation that maps 16/32-bit integers into a compact 8-bit domain, significantly improving both compression ratio and throughput while maintaining full numerical fidelity. ⚡

This framework is tailored for high-volume scientific and HPC workloads such as photon science, large-scale simulations, and sensor-array pipelines, where multi-byte integer data is ubiquitous and processed in small slices. 🔬🚀

(C) 2025 by Institute of Computing Technology, Chinese Academy of Sciences.

**Developers**: Wenjing Huang (Lead Developer, Designer of ADM), Jinwu Yang (CPU and GPU Implementations/Optimizations of Parallel ANS Encoders), Shengquan Yin (HIP Version of ANS Encoder)

**Contributors**: Dingwen Tao (Supervisor), Guangming Tan

---

## ✨ Key Features

* **🚀 High-Efficiency Compression**  
  Achieves up to **1.24×** higher compression ratio than standard ANS and **2.37×** higher than 16-bit Huffman.

* **⚡ Cross-Platform High-Performance**  
  Optimized for:
  - NVIDIA GPUs (CUDA)
  - AMD GPUs (HIP/ROCm)
  - Multi-core CPUs with OpenMP + SIMD

* **🧩 Adaptive Data Mapping (ADM)**  
  Converts multi-byte integers into effective 8-bit symbols with **<1% overhead**.

* **🔀 Flexible CPU Modes**  
  - **-p mode**: portable, GPU-consistent ANS implementation  
  - **-r mode**: maximum compression ratio using FSE-ANS

* **📦 Lightweight & Easy Integration**  
  Minimal dependencies, C++17 compatible, and CMake-based build system.

---

## 📈 Compression Ratio Performance

MANS consistently delivers strong compression ratios across real-world scientific datasets:

<img src="figure/CR.png" width="50%">

### **CPU (-r mode)**
- **2.37×** higher compression ratio vs. FSE-ANS  
- **1.32×** higher compression ratio vs. 16-bit Huffman  
- Up to **2.09×** improvement on quantization-based datasets  
- More stable performance on small slices compared to 16-bit Huffman  

### **CPU (-p mode)**
- **1.24×** higher compression ratio than FSE-ANS  
- Slightly lower than -r mode due to parallel ANS design  
- Maintains consistency with GPU behavior

---

## 📊 Throughput Performance

MANS provides strong, consistent performance across diverse platforms:

![Performance Plot](figure/THR.png)

### **Intel(R) Xeon(R) Gold 5220S**
- **1.92×** faster compression vs. FSE-ANS  
- **2.04×** faster decompression vs. FSE-ANS  

### **NVIDIA A100 GPU**
- **45.14×** faster compression vs. nvCOMP Huffman  
- Up to **288.45×** faster decompression compared to CPU portable mode  

### **AMD MI210 GPU**
- Up to **90.42×** faster compression and **135.86×** faster decompression vs. CPU  
- 0.52× CUDA compression throughput; 0.47× CUDA decompression throughput

---

## ⚙️ Requirements

- **CMake ≥ 3.15**  
- **C++17** compiler  
- **OpenMP** (for CPU parallelization)  
- **CUDA 12.6** (for NVIDIA GPUs)  
- **ROCm** (for AMD GPUs)  
- Git  
- Recommended OS: **Ubuntu 22.04+**

---

## 🔧 Building

### **1️⃣ Clone the Repository**

```shell
git clone https://github.com/ewTomato/MANS.git
```

### **2️⃣ Configure & Build**

Build CPU and NVIDIA support with the cross-backend tests enabled:

```shell
cd MANS
cmake -S . -B build \
  -DTARGET_PLATFORM=cpu_nv \
  -DBUILD_TESTING=ON \
  -DBUILD_HDF5_PLUGIN=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Available `TARGET_PLATFORM` values:

- `cpu` — CPU-only build
- `nv` — NVIDIA-only build
- `cpu_nv` — CPU + NVIDIA build
- `amd` — AMD-only build
- `cpu_amd` — CPU + AMD build
- `all` — CPU + NVIDIA + AMD build

The cross-backend test target is built when both `BUILD_TESTING=ON` and CPU/NVIDIA support are enabled.

---

## 🚀 Usage

MANS exposes complete MANS streams through `cpu_mans_compress`, `cpu_mans_decompress`, `nv_mans_compress`, and `nv_mans_decompress`. A complete stream has the following layout:

```text
[MansHeader][PANS/ANS payload]
```

The header records the codec, mode, raw byte count, and effective 1D/2D/3D geometry. ADM uses block-level decisions: each block independently selects ADM or RAW according to `range < 3500`.

### **Complete MANS stream: CPU and NVIDIA**

`-u2` selects unsigned 16-bit input and `-u4` selects unsigned 32-bit input. Use `--dims` to describe the logical shape; the product of the dimensions must equal the number of input elements.

CPU P-mode compression:

```bash
./build/bin/cpu/cpu_mans_compress \
  -u2 input.u2 output.mans \
  --mode p --dims 17 17
```

CPU decompression:

```bash
./build/bin/cpu/cpu_mans_decompress \
  -u2 output.mans restored.u2
```

NVIDIA compression:

```bash
./build/bin/nv/nv_mans_compress \
  -u2 input.u2 output.cuda.mans \
  --mode p --dims 17 17
```

NVIDIA decompression:

```bash
./build/bin/nv/nv_mans_decompress \
  -u2 output.cuda.mans restored.cuda.u2
```

The complete streams are interoperable in P mode:

```bash
# CPU-compressed stream decoded by NVIDIA
./build/bin/nv/nv_mans_decompress \
  -u2 output.mans restored-by-cuda.u2

# NVIDIA-compressed stream decoded by CPU
./build/bin/cpu/cpu_mans_decompress \
  -u2 output.cuda.mans restored-by-cpu.u2
```

NVIDIA currently supports P mode only. CPU R mode remains available for CPU-only compression and decompression, but R-mode streams are not accepted by the NVIDIA backend.

### **CPU: autotune first, then auto-pick threads**

`cpu_mans_autotune` generates synthetic **u16** datasets internally and sweeps all three mappings (`dims=1/2/3`) in one run.
- data-size list is configurable by `--data-size-mb-list`
- output CSV contains a `dims` column

1) Run autotune and generate thread CSV:
```bash
./build/bin/cpu/cpu_mans_autotune \
  --data-size-mb-list 0.00390625,0.0078125,1,4 \
  --csv ./build/thread_sweep.csv --out ./build/best_threads.csv
```

Optional: control synthetic block-type ratios (smooth/spike/constant/random):
```bash
./build/bin/cpu/cpu_mans_autotune \
  --ratio-smooth 1.0 --ratio-spike 0.0 --ratio-constant 0.0 --ratio-random 0.0
```

2) Run bench without `--threads`; it auto-loads CSV and selects nearest thread config by input size:
```bash
cd build
./bin/cpu/cpu_mans_bench -u2 /path/to/input_u16.bin --mode r --dims 1 134217728 --csv bench.csv
```
Auto-load order:
- `MANS_THREAD_CSV` env var (if set)
- otherwise `best_threads.csv` in current working directory

Debug warning for ADM bypass:
```bash
MANS_WARN_IF_NO_ADM=1 ./bin/cpu/cpu_mans_bench -u2 /path/to/input_u16.bin --mode r --dims 1 134217728
```

### **Low-level NVIDIA ADM/ANS tools**

The standalone ADM and ANS targets are useful for component-level benchmarking. They do not replace the complete MANS API flow above and should not be used when testing CPU/NVIDIA stream interoperability.

### **AMD GPU**

AMD ADM/ANS tools remain separate component-level targets. The complete CPU/NVIDIA P-mode interoperability described above applies to the `cpu_nv` build; AMD support follows its own HIP pipeline.

### **Cross-backend verification**

Run the CPU/NVIDIA P-mode interoperability test after building with `-DBUILD_TESTING=ON`:

```bash
ctest --test-dir build --output-on-failure
```

The test covers:

- CPU compression -> NVIDIA decompression;
- NVIDIA compression -> CPU decompression;
- U16 and U32 data;
- 1D, 2D, and 3D shapes;
- partial 1D blocks and non-aligned 2D/3D tiles;
- block-level ADM and RAW decisions;
- ANS block boundaries and representative constant, narrow-range, wide-range, and random data.

For CUDA memory checking:

```bash
compute-sanitizer --tool memcheck \
  build/tests/mans_cross_backend_test
```

### **HDF5 Filter Plugin: H5Z-MANS**

See [tools/H5Z-MANS/README.md](tools/H5Z-MANS/README.md) for detailed instructions on building and using the HDF5 filter plugin for MANS.
---

## 📁 Project Structure

```
MANS/
 ├── amd/              # ADM, ANS, GPU kernels(AMD version)
 ├── build/            # CMake build directory (generated by user)
 ├── cpu/              # ADM, PANS(CPU version)
 ├── nv/               # ADM, ANS, GPU kernels(NVIDIA version)
 ├── testdata/ 
 ├── tools/            # test scripts and tools(hdf5 filter)
 └── README.md
 ...
```
---

## 📜 License

MANS is released under the **MIT License**.  
Please see the `LICENSE` file for full details.

---

## 📚 Citation

If you use **MANS** in your research or software, please cite our work:

```
@inproceedings{huang2025mans,
  title={MANS: Efficient and Portable ANS Encoding for Multi-Byte Integer Data on CPUs and GPUs},
  author={Huang, Wenjing and Yang, Jinwu and Yin, Shengquan and Li, Haoxu and Gu, Yida and Liu, Zedong and Jing, Xing and Wei, Zheng and Fu, Shiyuan and Hu, Hao and others},
  booktitle={Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis},
  pages={1299--1314},
  year={2025},
  DOI={10.1145/3712285.3759825}
}
```

More details and related materials will coming soon.
