#!/usr/bin/env bash
# Optional L2/cache counter profiling for the NVIDIA HAAC ADM stages using
# NVIDIA Nsight Compute (ncu). Mirrors amd/adm/haac_bench/scripts/profile.sh:
# each stage is profiled against its ADM kernel(s) only, and the raw per-metric
# CSV is kept under pmc/ so extra counters can be inspected later.
#
# ncu requires profiling privileges (and, on some systems, CAP_SYS_ADMIN / a
# persistent-mode GPU). Metric names vary by architecture; the default targets
# Ampere (sm_80) L2 hit rate.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bench_dir=$(cd -- "$script_dir/.." && pwd)
build_dir=${BUILD_DIR:-"$bench_dir/build"}
profiler=${NCU:-ncu}
metrics=${NCU_METRICS:-"lts__t_sector_hit_rate.pct"}
hit_rate_metric=${NCU_HIT_RATE_METRIC:-"lts__t_sector_hit_rate.pct"}
data_dir=${1:-${DATA_DIR:-}}
result_dir=${2:-${RESULT_DIR:-"$bench_dir/profile/$(date +%Y%m%d_%H%M%S)"}}
gpu=${GPU_ID:-0}

if [[ -z "$data_dir" || ! -d "$data_dir" ]]; then
    echo "Usage: $0 <haac-u2-directory> [profile-output-directory]" >&2
    exit 2
fi
if ! command -v "$profiler" >/dev/null 2>&1; then
    echo "Profiler not found: $profiler" >&2
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
temp_adm_dir=$(mktemp -d "${TMPDIR:-/tmp}/nv-haac-adm.XXXXXX")
trap 'rm -rf -- "$temp_adm_dir"' EXIT
summary="$result_dir/summary.csv"
printf 'input,stage,l2_hit_rate_pct,log\n' > "$summary"

export CUDA_VISIBLE_DEVICES="$gpu"

# ADM kernel(s) whose L2 traffic belongs to each stage. Unfused decompression
# runs two kernels that together complete the stage, so both are matched.
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
    local pmc="$result_dir/pmc/$tag.csv"

    echo "[$stage] $input_name"
    if ! "$profiler" --csv --metrics "$metrics" \
        --kernel-name "regex:${kernel_regex[$stage]}" \
        -- "$binary" "${bin_args[@]}" >"$pmc" 2>"$log"; then
        echo "  WARN: ncu failed for $stage; see $log" >&2
        printf '"%s",%s,NA,"%s"\n' "$input_name" "$stage" "$log" >> "$summary"
        return
    fi

    local rate
    rate=$(awk -F',' -v m="$hit_rate_metric" '
        { for (i = 1; i <= NF; i++) {
            if ($i == "\"" m "\"" || $i == m) {
                v = $(i + 1); gsub(/"/, "", v); gsub(/%/, "", v);
                if (v ~ /^[0-9]+(\.[0-9]+)?$/) { print v; exit }
            }
        } }
    ' "$pmc")
    [[ -z "$rate" ]] && rate=NA
    printf '"%s",%s,%s,"%s"\n' "$input_name" "$stage" "$rate" "$log" >> "$summary"
}

for input in "${inputs[@]}"; do
    input_name=$(basename "$input")

    for comp in fused unfused; do
        stage="compress_$comp"
        if [[ ! -x "$build_dir/$stage" ]]; then
            echo "Missing executable: $build_dir/$stage" >&2
            exit 2
        fi
        adm="$temp_adm_dir/${input_name%.u2}_$comp.adm"
        profile_one "$stage" "$input" "$adm"
    done

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
echo "Note: raw ncu output is under $result_dir/pmc/. Metric names are arch-specific;"
echo "adjust NCU_METRICS / NCU_HIT_RATE_METRIC for your target architecture."
