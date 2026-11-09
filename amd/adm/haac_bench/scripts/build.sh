#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bench_dir=$(cd -- "$script_dir/.." && pwd)
repo_root=$(cd -- "$bench_dir/../../.." && pwd)

backend=${BACKEND:-rocm}
if [[ "$backend" != rocm && "$backend" != cuda ]]; then
    echo "BACKEND must be rocm or cuda (got: $backend)" >&2
    exit 2
fi

if [[ "$backend" == cuda ]]; then
    compiler=${NVCC:-nvcc}
    hipify=${HIPIFY_PERL:-hipify-perl}
    build_dir=${BUILD_DIR:-"$bench_dir/build/cuda"}
    cuda_arch=${CUDA_ARCH:-sm_80}
    command -v "$compiler" >/dev/null || { echo "CUDA compiler not found: $compiler" >&2; exit 2; }
    command -v "$hipify" >/dev/null || { echo "hipify-perl not found: $hipify" >&2; exit 2; }
else
    compiler=${HIPCC:-hipcc}
    build_dir=${BUILD_DIR:-"$bench_dir/build"}
    command -v "$compiler" >/dev/null || { echo "HIP compiler not found: $compiler" >&2; exit 2; }
fi

# The four pure-stage executables plus the synthetic hardware microbenchmark.
targets=(compress_fused compress_unfused decompress_fused decompress_unfused microbench)
mkdir -p "$build_dir"

if [[ "$backend" == rocm ]]; then
    declare -a arch_flag=()
    if [[ -n "${GPU_ARCH:-}" ]]; then
        arch_flag+=(--offload-arch="$GPU_ARCH")
    fi

    # ROCm installs the device-library bitcode in one of these locations.
    declare -a devlib_flag=()
    if [[ -n "${ROCM_DEVICE_LIB_PATH:-}" ]]; then
        devlib_flag+=(--rocm-device-lib-path="$ROCM_DEVICE_LIB_PATH")
    else
        rocm_root=$(dirname "$(dirname "$(command -v "$compiler")")")
        for candidate in "$rocm_root/lib/amdgcn/bitcode" "$rocm_root/amdgcn/bitcode"; do
            if [[ -f "$candidate/ockl.bc" ]]; then
                devlib_flag+=(--rocm-device-lib-path="$candidate")
                break
            fi
        done
    fi
fi

for name in "${targets[@]}"; do
    source="$bench_dir/src/$name.cpp"
    if [[ ! -f "$source" ]]; then
        echo "Missing source: $source" >&2
        exit 2
    fi

    if [[ "$backend" == rocm ]]; then
        echo "Building $name with $compiler ${GPU_ARCH:+(arch $GPU_ARCH)}"
        "$compiler" -O3 -std=c++14 "${arch_flag[@]}" "${devlib_flag[@]}" \
            -DHIP_ENABLE_WARP_SYNC_BUILTINS \
            -I"$repo_root/amd/adm" -I"$bench_dir/src" \
            "$source" -o "$build_dir/$name"
    else
        tmp_source=$(mktemp --suffix=.cu)
        trap 'rm -f "$tmp_source"' EXIT
        echo "Translating and building $name with $compiler ($cuda_arch)"
        "$hipify" "$source" > "$tmp_source"
        "$compiler" -O3 -std=c++14 -arch="$cuda_arch" \
            -I"$repo_root/amd/adm" -I"$bench_dir/src" \
            "$tmp_source" -o "$build_dir/$name"
        rm -f "$tmp_source"
        trap - EXIT
    fi
done

echo "Backend: $backend"
echo "Binaries: $build_dir"
