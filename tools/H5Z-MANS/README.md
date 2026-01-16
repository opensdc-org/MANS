
# H5Z-MANS: HDF5 Compression Filter Plugin for MANS

## Features

* **Dynamic Loading:** Built as a shared library (`.so` / `.dll`), allowing usage without recompiling HDF5 applications.
* **CPU Backend:** Multi-threaded acceleration using OpenMP (AVX512 optimizations).
* **Configurable:** Runtime tuning via filter parameters (`cd_values`) generated from a config file.
* **Data Type Support:** Unsigned 16-bit and 32-bit integer datasets only.

---

## Build Instructions

This plugin is built as part of the main MANS project. Ensure the `BUILD_HDF5_PLUGIN` option is enabled (default: ON) and HDF5 is installed.

```bash
mkdir build && cd build

# Configure (Ensure BUILD_HDF5_PLUGIN is ON)
cmake .. -DBUILD_HDF5_PLUGIN=ON 

# Build
make -j

```

### Build Artifacts

After a successful build, the artifacts will be located at:

* **Plugin Library:** `build/bin/plugins/libH5Z-MANS.so`
* **Test Utility:** `build/bin/h5z-mans/H5Z-MANS_test`

---

## Usage

### Environment Setup (Crucial)

HDF5 requires the `HDF5_PLUGIN_PATH` environment variable to locate external filters. You must point this to the directory containing `libH5Z-MANS.so`.

```bash
# Assuming you are in the project root/build directory
export HDF5_PLUGIN_PATH=$(pwd)/bin/plugins

```

### Filter ID

* **H5Z-MANS filter ID:** `32001`

### HDF5 Notes

* H5Z-MANS only supports **unsigned integer** datasets with element size **2 bytes (U16)** or **4 bytes (U32)**.
* If your application does not find the plugin, verify `HDF5_PLUGIN_PATH` and ensure the file name is `libH5Z-MANS.so`.

## Testing & Verification

A dedicated test tool `H5Z-MANS_test` is provided to verify compression integrity (Bit-Exact check) and measure compression ratio.

### 1. Create a Configuration File

See [example.conf](example.conf). The config controls backend, dtype, ADM threshold, and thread counts used to populate `cd_values`.

### 2. Run the Test

```bash
# 1. Export Plugin Path
export HDF5_PLUGIN_PATH=<path of build>/bin/plugins

# 2. Run Test
# Usage: ./bin/h5z-mans/H5Z-MANS_test <config_file> <output.h5> [input.bin] [OPTIONS]


Options:
  --size <MB>        Set synthetic data size in MB (default: 256.0)
  --chunk <MB>       Set chunk size in MB (default: 32.0)
  --filter <name>    Set filter: mans, zstd, deflate, sz3 (default: mans)
```

Notes:
* If `input.bin` is not provided, synthetic data is generated.
* SZ3 config format can be referenced from `https://github.com/szcompressor/SZ3/blob/master/tools/sz3/sz3.config`.


