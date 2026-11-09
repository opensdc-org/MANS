#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bench_dir=$(cd -- "$script_dir/.." && pwd)
repo_root=$(cd -- "$bench_dir/../../.." && pwd)
amd_src="$repo_root/amd/adm/haac_bench/src"

compiler=${NVCC:-nvcc}
build_dir=${BUILD_DIR:-"$bench_dir/build"}
cuda_arch=${CUDA_ARCH:-sm_80}
command -v "$compiler" >/dev/null || { echo "CUDA compiler not found: $compiler" >&2; exit 2; }

# The four ADM stage executables share their source with the AMD haac_bench
# (nv/adm/haac_bench/src/*.cu are symlinks to amd/adm/haac_bench/src/*.cpp).
# Those sources are HIP/CUDA dual-backend: defining MANS_BENCH_CUDA compiles the
# CUDA path of common.h instead of the HIP path, so the identical ADM kernels are
# built with nvcc. The synthetic microbenchmark is a native CUDA port.
targets=(compress_fused compress_unfused decompress_fused decompress_unfused microbench)
mkdir -p "$build_dir"

for name in "${targets[@]}"; do
    source="$bench_dir/src/$name.cu"
    if [[ ! -e "$source" ]]; then
        echo "Missing source: $source" >&2
        exit 2
    fi
    echo "Building $name with $compiler (arch $cuda_arch)"
    "$compiler" -O3 -std=c++14 -arch="$cuda_arch" -DMANS_BENCH_CUDA \
        -I"$bench_dir/src" -I"$amd_src" \
        "$source" -o "$build_dir/$name"
done

echo "Binaries: $build_dir"
