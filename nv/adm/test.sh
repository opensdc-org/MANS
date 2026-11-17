#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
build_dir=${BUILD_DIR:-"$repo_root/build/bin/nv"}
result_dir=${RESULT_DIR:-"$repo_root/nv/adm/results/nvadm_$(date +%Y%m%d_%H%M%S)_$$"}
time_bin=${TIME_BIN:-/usr/bin/time}
input_path=${1:-}

usage() {
    printf 'Usage: %s <input.u2|input-directory> --dims x [y z] [--output-dir DIR] [--build-dir DIR]\n' "$0" >&2
}

if [[ -z "$input_path" ]]; then
    usage
    exit 2
fi
shift

dims=()
while (($#)); do
    case "$1" in
        --dims)
            if ((${#dims[@]} != 0)); then
                echo "--dims may only be specified once." >&2
                exit 2
            fi
            shift
            while (($#)) && [[ "$1" != --* ]]; do
                dims+=("$1")
                shift
            done
            ;;
        --output-dir)
            (($# >= 2)) || { usage; exit 2; }
            result_dir=$2
            shift 2
            ;;
        --build-dir)
            (($# >= 2)) || { usage; exit 2; }
            build_dir=$2
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 2
            ;;
    esac
done

if ((${#dims[@]} < 1 || ${#dims[@]} > 3)); then
    echo "Provide one to three positive dimensions with --dims." >&2
    usage
    exit 2
fi
expected_elements=1
for dim in "${dims[@]}"; do
    if [[ ! "$dim" =~ ^[1-9][0-9]*$ ]] || ((${#dim} > 10)) || ((10#$dim > 4294967295)); then
        echo "Invalid dimension: $dim" >&2
        exit 2
    fi
    if ((expected_elements > 9223372036854775807 / 10#$dim)); then
        echo "Dimension product exceeds the supported range." >&2
        exit 2
    fi
    expected_elements=$((expected_elements * 10#$dim))
done

if [[ ! -x "$build_dir/nvadm_compress" || ! -x "$build_dir/nvadm_decompress" ]]; then
    echo "Missing NVIDIA ADM executables in $build_dir; build targets nvadm_compress and nvadm_decompress first." >&2
    exit 2
fi
if ! command -v "$time_bin" >/dev/null 2>&1; then
    echo "Wall-clock timer not found: $time_bin" >&2
    exit 2
fi

if [[ -d "$input_path" ]]; then
    input_dir=$(cd -- "$input_path" && pwd)
    shopt -s nullglob
    inputs=("$input_dir"/*.u2)
else
    if [[ ! -f "$input_path" ]]; then
        echo "Input file not found: $input_path" >&2
        exit 2
    fi
    input_dir=$(cd -- "$(dirname -- "$input_path")" && pwd)
    inputs=("$input_dir/$(basename -- "$input_path")")
fi
if ((${#inputs[@]} == 0)); then
    echo "No .u2 inputs found in $input_path" >&2
    exit 2
fi

# Each run gets a fresh directory so previous outputs and summaries are never replaced.
if [[ -e "$result_dir" ]]; then
    echo "Output directory already exists; choose a new directory: $result_dir" >&2
    exit 2
fi
mkdir -p "$(dirname -- "$result_dir")"
mkdir -- "$result_dir"
result_dir=$(cd -- "$result_dir" && pwd)
mkdir -- "$result_dir/logs" "$result_dir/compressed" "$result_dir/decompressed"
summary="$result_dir/summary.csv"
printf 'input,stage,input_bytes,output_bytes,wall_seconds,input_gbps,lossless_pass,log\n' > "$summary"

csv_quote() {
    local value=${1//\"/\"\"}
    printf '"%s"' "$value"
}

for input in "${inputs[@]}"; do
    input_name=$(basename -- "$input")
    stem=${input_name%.u2}
    input_bytes=$(stat -c '%s' -- "$input")
    if ((input_bytes % 2 != 0)); then
        echo "Input is not a whole number of uint16 values: $input" >&2
        exit 1
    fi
    file_elements=$((input_bytes / 2))
    if ((file_elements != expected_elements)); then
        printf 'Dimension mismatch for %s: --dims has %s elements, file has %s uint16 values.\n' \
            "$input_name" "$expected_elements" "$file_elements" >&2
        exit 1
    fi

    adm="$result_dir/compressed/$stem.adm"
    compress_log="$result_dir/logs/${stem}_compress.log"
    compress_time="$result_dir/logs/${stem}_compress.seconds"
    echo "[nvadm_compress] $input_name"
    if ! "$time_bin" -f '%e' -o "$compress_time" \
        "$build_dir/nvadm_compress" -u2 "$input" "$adm" --dims "${dims[@]}" \
        >"$compress_log" 2>&1; then
        echo "Compression failed; see $compress_log" >&2
        exit 1
    fi
    compress_seconds=$(<"$compress_time")
    adm_bytes=$(stat -c '%s' -- "$adm")
    compress_gbps=$(awk -v bytes="$input_bytes" -v seconds="$compress_seconds" \
        'BEGIN { if (seconds > 0) printf "%.6f", bytes / seconds / 1e9; else print "NA" }')
    printf '%s,compress,%s,%s,%s,%s,n/a,%s\n' \
        "$(csv_quote "$input_name")" "$input_bytes" "$adm_bytes" "$compress_seconds" \
        "$compress_gbps" "$(csv_quote "$compress_log")" >> "$summary"

    roundtrip="$result_dir/decompressed/$stem.roundtrip.u2"
    decompress_log="$result_dir/logs/${stem}_decompress.log"
    decompress_time="$result_dir/logs/${stem}_decompress.seconds"
    echo "[nvadm_decompress] $input_name"
    if ! "$time_bin" -f '%e' -o "$decompress_time" \
        "$build_dir/nvadm_decompress" -u2 "$adm" "$roundtrip" --dims "${dims[@]}" \
        >"$decompress_log" 2>&1; then
        echo "Decompression failed; see $decompress_log" >&2
        exit 1
    fi
    decompress_seconds=$(<"$decompress_time")
    roundtrip_bytes=$(stat -c '%s' -- "$roundtrip")
    decompress_gbps=$(awk -v bytes="$input_bytes" -v seconds="$decompress_seconds" \
        'BEGIN { if (seconds > 0) printf "%.6f", bytes / seconds / 1e9; else print "NA" }')
    if cmp -s -- "$input" "$roundtrip"; then
        lossless=yes
    else
        lossless=no
    fi
    printf '%s,decompress,%s,%s,%s,%s,%s,%s\n' \
        "$(csv_quote "$input_name")" "$input_bytes" "$roundtrip_bytes" "$decompress_seconds" \
        "$decompress_gbps" "$lossless" "$(csv_quote "$decompress_log")" >> "$summary"
    if [[ "$lossless" != yes ]]; then
        echo "Lossless verification failed; see $decompress_log and $roundtrip" >&2
        exit 1
    fi
done

echo "Summary: $summary"
echo "Note: wall-clock timings cover each complete CLI invocation, including file I/O, host/device transfers, CPU-side work, and GPU work."
