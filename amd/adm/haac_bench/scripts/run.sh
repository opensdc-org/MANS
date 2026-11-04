#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bench_dir=$(cd -- "$script_dir/.." && pwd)
build_dir=${BUILD_DIR:-"$bench_dir/build"}
data_dir=${1:-${DATA_DIR:-}}
result_dir=${2:-${RESULT_DIR:-"$bench_dir/results/$(date +%Y%m%d_%H%M%S)"}}
gpu=${GPU_ID:-0}

if [[ -z "$data_dir" || ! -d "$data_dir" ]]; then
    echo "Usage: $0 <haac-u2-directory> [result-directory]" >&2
    exit 2
fi
if [[ ! -d "$build_dir" ]]; then
    echo "Missing binaries in $build_dir; run scripts/build.sh first." >&2
    exit 2
fi

shopt -s nullglob
inputs=("$data_dir"/*.u2)
if ((${#inputs[@]} == 0)); then
    echo "No .u2 files found in $data_dir" >&2
    exit 2
fi

mkdir -p "$result_dir/logs" "$result_dir/compressed"
summary="$result_dir/summary.csv"
printf 'input,variant,input_bytes,compression_gbps,decompression_gbps,lossless_pass,log\n' > "$summary"

export CUDA_VISIBLE_DEVICES="$gpu"
export HIP_VISIBLE_DEVICES="$gpu"
export ROCR_VISIBLE_DEVICES="$gpu"

variants=(
    u16_fused_compress_fused_decompress
    u16_fused_compress_unfused_decompress
    u16_unfused_compress_fused_decompress
    u16_unfused_compress_unfused_decompress
)

for input in "${inputs[@]}"; do
    input_name=$(basename "$input")
    input_bytes=$(stat -c '%s' "$input")
    for variant in "${variants[@]}"; do
        binary="$build_dir/$variant"
        if [[ ! -x "$binary" ]]; then
            echo "Missing executable: $binary" >&2
            exit 2
        fi
        stem="${input_name%.u2}_$variant"
        output="$result_dir/compressed/$stem.adm"
        log="$result_dir/logs/$stem.log"
        echo "[$variant] $input_name"
        if ! "$binary" "$input" "$output" 2>&1 | tee "$log"; then
            echo "Benchmark failed; see $log" >&2
            exit 1
        fi
        cmp_gbps=$(awk '/Total Cmp throughput:/ { value=$4 } END { if (value != "") print value }' "$log")
        decmp_gbps=$(awk '/Decmp throughput:/ { value=$3 } END { if (value != "") print value }' "$log")
        if ! grep -q 'Pass error check!' "$log"; then
            echo "Lossless verification did not pass; see $log" >&2
            exit 1
        fi
        printf '"%s",%s,%s,%s,%s,yes,"%s"\n' \
            "$input_name" "$variant" "$input_bytes" "${cmp_gbps:-NA}" \
            "${decmp_gbps:-NA}" "$log" >> "$summary"
    done
done

echo "Summary: $summary"
