#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bench_dir=$(cd -- "$script_dir/.." && pwd)
build_dir=${BUILD_DIR:-"$bench_dir/build"}
profiler=${ROCPROFV3:-rocprofv3}
# L2 counters for gfx90a (AMD Instinct MI210); summed across TCC instances.
# For gfx1100 (RX 7900 XTX) use GL2C_HIT_sum / GL2C_MISS_sum instead.
counters=${COUNTERS:-"TCC_HIT_sum TCC_MISS_sum"}
hit_counter=${HIT_COUNTER:-TCC_HIT_sum}
miss_counter=${MISS_COUNTER:-TCC_MISS_sum}
data_dir=${1:-${DATA_DIR:-}}
result_dir=${2:-${RESULT_DIR:-"$bench_dir/profile/$(date +%Y%m%d_%H%M%S)"}}
gpu=${GPU_ID:-0}

if [[ -z "$data_dir" || ! -d "$data_dir" ]]; then
    echo "Usage: $0 <haac-u2-directory> [profile-output-directory]" >&2
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

if [[ -e "$result_dir" ]]; then
    echo "Profile output directory already exists; choose a new directory: $result_dir" >&2
    exit 2
fi
mkdir -p "$result_dir/logs" "$result_dir/pmc"
temp_adm_dir=$(mktemp -d "${TMPDIR:-/tmp}/haac-adm.XXXXXX")
trap 'rm -rf -- "$temp_adm_dir"' EXIT
summary="$result_dir/summary.csv"
printf 'input,stage,l2_hit_rate_pct,%s,%s,log\n' "$hit_counter" "$miss_counter" > "$summary"

export CUDA_VISIBLE_DEVICES="$gpu"
export HIP_VISIBLE_DEVICES="$gpu"
export ROCR_VISIBLE_DEVICES="$gpu"

# ADM kernel(s) whose L2 traffic belongs to each stage. Unfused decompression
# runs two kernels that together complete the stage, so both are summed.
declare -A kernel_regex=(
    [compress_fused]=map_values_kernel_decoupled
    [compress_unfused]=map_values_kernel_thrust
    [decompress_fused]=decompress
    [decompress_unfused]='decode_signals_uint16|restore_values_uint16'
)

profile_one() {
    local stage=$1
    shift
    local bin_args=("$@")
    local binary="$build_dir/$stage"
    local tag="${input_name%.u2}_$stage"
    local log="$result_dir/logs/$tag.log"
    local out="$result_dir/pmc/$tag"

    echo "[$stage] $input_name"
    "$profiler" --pmc $counters --output-format csv \
        --output-directory "$out" \
        -- "$binary" "${bin_args[@]}" 2>&1 | tee "$log" >/dev/null

    local csv
    csv=$(find "$out" -name '*counter_collection.csv' -print -quit)
    if [[ -z "$csv" ]]; then
        echo "  WARN: no counter CSV produced for $stage; see $log" >&2
        printf '"%s",%s,NA,NA,NA,"%s"\n' "$input_name" "$stage" "$log" >> "$summary"
        return
    fi

    local hit miss rate
    read -r hit miss rate <<< "$(python3 -I "$script_dir/parse_pmc.py" \
        "$csv" "${kernel_regex[$stage]}" "$hit_counter" "$miss_counter")"
    printf '"%s",%s,%s,%s,%s,"%s"\n' \
        "$input_name" "$stage" "$rate" "$hit" "$miss" "$log" >> "$summary"
}

for input in "${inputs[@]}"; do
    input_name=$(basename "$input")

    # Compression stages (also produce the .adm consumed by decompression).
    for comp in fused unfused; do
        stage="compress_$comp"
        if [[ ! -x "$build_dir/$stage" ]]; then
            echo "Missing executable: $build_dir/$stage" >&2
            exit 2
        fi
        adm="$temp_adm_dir/${input_name%.u2}_$comp.adm"
        profile_one "$stage" "$input" "$adm"
    done

    # Decompression stages (consume the .adm produced above).
    for decmp in fused unfused; do
        stage="decompress_$decmp"
        if [[ ! -x "$build_dir/$stage" ]]; then
            echo "Missing executable: $build_dir/$stage" >&2
            exit 2
        fi
        adm="$temp_adm_dir/${input_name%.u2}_$decmp.adm"
        profile_one "$stage" "$adm" "$input"
    done
done

echo "Summary: $summary"
