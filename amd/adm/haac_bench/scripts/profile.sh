#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bench_dir=$(cd -- "$script_dir/.." && pwd)
build_dir=${BUILD_DIR:-"$bench_dir/build"}
profiler=${ROCPROFV3:-rocprofv3}
input=${1:-}
variant=${2:-}
output_dir=${3:-"$bench_dir/profile/$(date +%Y%m%d_%H%M%S)"}
gpu=${GPU_ID:-0}

if [[ -z "$input" || ! -f "$input" || -z "$variant" ]]; then
    echo "Usage: $0 <input.u2> <variant-name> [profile-output-directory]" >&2
    echo "Variants: u16_fused_compress_fused_decompress, u16_fused_compress_unfused_decompress," >&2
    echo "          u16_unfused_compress_fused_decompress, u16_unfused_compress_unfused_decompress" >&2
    exit 2
fi
binary="$build_dir/$variant"
if [[ ! -x "$binary" ]]; then
    echo "Missing executable: $binary; run scripts/build.sh first." >&2
    exit 2
fi
mkdir -p "$output_dir"

export CUDA_VISIBLE_DEVICES="$gpu"
export HIP_VISIBLE_DEVICES="$gpu"
export ROCR_VISIBLE_DEVICES="$gpu"

base=$(basename "$input" .u2)
"$profiler" --pmc TCC_HIT TCC_MISS --output-format csv \
    --output-directory "$output_dir/$base-$variant" \
    -- "$binary" "$input" "$output_dir/$base-$variant.adm" \
    2>&1 | tee "$output_dir/$base-$variant.log"

echo "Profile results: $output_dir/$base-$variant"
