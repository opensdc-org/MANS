#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bench_dir=$(cd -- "$script_dir/.." && pwd)
repo_root=$(cd -- "$bench_dir/../../.." && pwd)

hipcc=${HIPCC:-hipcc}
build_dir=${BUILD_DIR:-"$bench_dir/build"}

declare -a arch_flag=()
if [[ -n "${GPU_ARCH:-}" ]]; then
    arch_flag+=(--offload-arch="$GPU_ARCH")
fi

# ROCm 6.3.3 in this environment installs the device-library bitcode under
# <rocm>/amdgcn/bitcode rather than the usual <rocm>/lib/amdgcn/bitcode, so pass
# the device-lib path explicitly when the standard location is missing.
declare -a devlib_flag=()
if [[ -n "${ROCM_DEVICE_LIB_PATH:-}" ]]; then
    devlib_flag+=(--rocm-device-lib-path="$ROCM_DEVICE_LIB_PATH")
else
    rocm_root=$(dirname "$(dirname "$(command -v "$hipcc")")")
    for candidate in "$rocm_root/lib/amdgcn/bitcode" "$rocm_root/amdgcn/bitcode"; do
        if [[ -f "$candidate/ockl.bc" ]]; then
            devlib_flag+=(--rocm-device-lib-path="$candidate")
            break
        fi
    done
fi

# The four pure-stage executables (compress/decompress x fused/unfused).
targets=(compress_fused compress_unfused decompress_fused decompress_unfused)

mkdir -p "$build_dir"
for name in "${targets[@]}"; do
    source="$bench_dir/src/$name.cpp"
    if [[ ! -f "$source" ]]; then
        echo "Missing source: $source" >&2
        exit 2
    fi
    echo "Building $name with $hipcc ${GPU_ARCH:+(arch $GPU_ARCH)}"
    "$hipcc" -O3 -std=c++14 "${arch_flag[@]}" "${devlib_flag[@]}" \
        -DHIP_ENABLE_WARP_SYNC_BUILTINS \
        -I"$repo_root/amd/adm" \
        "$source" -o "$build_dir/$name"
done

echo "Binaries: $build_dir"
