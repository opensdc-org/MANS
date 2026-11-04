#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bench_dir=$(cd -- "$script_dir/.." && pwd)
repo_root=$(cd -- "$bench_dir/../../.." && pwd)

hipcc=${HIPCC:-hipcc}
build_dir=${BUILD_DIR:-"$bench_dir/build"}

declare -A arch_flag=()
if [[ -n "${GPU_ARCH:-}" ]]; then
    arch_flag+=(--offload-arch="$GPU_ARCH")
fi

mkdir -p "$build_dir"
for source in "$bench_dir"/src/*.cpp; do
    name=$(basename "$source" .cpp)
    echo "Building $name with $hipcc ${GPU_ARCH:+(arch $GPU_ARCH)}"
    "$hipcc" -O3 -std=c++14 "${arch_flag[@]}" \
        -DHIP_ENABLE_WARP_SYNC_BUILTINS \
        -I"$repo_root/amd/adm" \
        "$source" -o "$build_dir/$name"
done

echo "Binaries: $build_dir"
