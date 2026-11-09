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

mkdir -p "$result_dir/logs"
temp_adm_dir=$(mktemp -d "${TMPDIR:-/tmp}/haac-adm.XXXXXX")
trap 'rm -rf -- "$temp_adm_dir"' EXIT
summary="$result_dir/summary.csv"
printf 'input,stage,input_bytes,throughput_gbps,lossless_pass,log\n' > "$summary"

export CUDA_VISIBLE_DEVICES="$gpu"
export HIP_VISIBLE_DEVICES="$gpu"
export ROCR_VISIBLE_DEVICES="$gpu"

for input in "${inputs[@]}"; do
    input_name=$(basename "$input")
    input_bytes=$(stat -c '%s' "$input")

    # Compression stages: input.u2 -> compressed.adm
    for comp in fused unfused; do
        binary="$build_dir/compress_$comp"
        if [[ ! -x "$binary" ]]; then
            echo "Missing executable: $binary" >&2
            exit 2
        fi
        adm="$temp_adm_dir/${input_name%.u2}_$comp.adm"
        log="$result_dir/logs/${input_name%.u2}_compress_$comp.log"
        echo "[compress_$comp] $input_name"
        if ! "$binary" "$input" "$adm" 2>&1 | tee "$log"; then
            echo "Compression failed; see $log" >&2
            exit 1
        fi
        gbps=$(awk '/Total Cmp throughput:/ { print $4 }' "$log")
        printf '"%s",compress_%s,%s,%s,n/a,"%s"\n' \
            "$input_name" "$comp" "$input_bytes" "${gbps:-NA}" "$log" >> "$summary"
    done

    # Decompression stages: compressed.adm + original.u2 -> lossless check
    for decmp in fused unfused; do
        binary="$build_dir/decompress_$decmp"
        if [[ ! -x "$binary" ]]; then
            echo "Missing executable: $binary" >&2
            exit 2
        fi
        adm="$temp_adm_dir/${input_name%.u2}_$decmp.adm"
        log="$result_dir/logs/${input_name%.u2}_decompress_$decmp.log"
        echo "[decompress_$decmp] $input_name"
        if ! "$binary" "$adm" "$input" 2>&1 | tee "$log"; then
            echo "Decompression failed; see $log" >&2
            exit 1
        fi
        gbps=$(awk '/Decmp throughput:/ { print $3 }' "$log")
        if grep -q 'Pass error check!' "$log"; then
            pass=yes
        else
            pass=no
        fi
        printf '"%s",decompress_%s,%s,%s,%s,"%s"\n' \
            "$input_name" "$decmp" "$input_bytes" "${gbps:-NA}" "$pass" "$log" >> "$summary"
        if [[ "$pass" != "yes" ]]; then
            echo "Lossless verification did not pass; see $log" >&2
            exit 1
        fi
    done
done

echo "Summary: $summary"
