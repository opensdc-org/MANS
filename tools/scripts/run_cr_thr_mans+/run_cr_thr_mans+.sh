#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# --------------------------
# Constants to edit
# --------------------------
U2_DATASET_ROOT="${U2_DATASET_ROOT:-/hwj/data/testdata/u2}"
U4_DATASET_ROOT="${U4_DATASET_ROOT:-/hwj/data/testdata/u4}"
FORCE_1D_DATASET_ROOT="${FORCE_1D_DATASET_ROOT:-/public/share/acnnprvuzd/MANS/datasets/1d_test}"
WORKDIR="${WORKDIR:-/hwj/project/MANSplus/build}"
MODE="${MODE:-r}"
# --------------------------
# Constants to edit
# --------------------------
ALGO="${ALGO:-mans+_${MODE}}"
BENCH_BIN="${BENCH_BIN:-./bin/cpu/cpu_mans_bench}"
CHUNKS_MB="${CHUNKS_MB:-0}"
THRESHOLD="${THRESHOLD:-4000}"
WARMUP="${WARMUP:-5}"
ITER="${ITER:-10}"

CSV_FILE="${CSV_FILE:-${SCRIPT_DIR}/${ALGO}_results.csv}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/log}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${ALGO}.log}"


mkdir -p "${LOG_DIR}"
: > "${LOG_FILE}"

printf 'dataset_folder,dataset_name,file_size_bytes,algo,input_type,ratio,comp_mbps,decomp_mbps,error\n' > "${CSV_FILE}"

dataset_files=()
dataset_types=()
dataset_force_1d=()

read_dims_from_meta() {
    local meta_file="$1"
    python3 - "${meta_file}" <<'PY'
import json
import sys

meta_path = sys.argv[1]
with open(meta_path, "r", encoding="utf-8") as f:
    obj = json.load(f)

raw = obj.get("chosen_block_shape_zyx")
if not isinstance(raw, list) or len(raw) < 1 or len(raw) > 3:
    sys.exit(1)

vals = []
for v in raw:
    if isinstance(v, bool):
        sys.exit(1)
    if not isinstance(v, (int, float)):
        sys.exit(1)
    iv = int(v)
    if iv <= 0:
        sys.exit(1)
    vals.append(iv)

# metadata is z,y,x; bench needs x,y,z
vals = list(reversed(vals))
print(f"{len(vals)} " + " ".join(str(x) for x in vals))
PY
}

resolve_dims_args() {
    local dataset_file="$1"
    local input_type="$2"
    local file_size_bytes="$3"
    local force_1d="${4:-0}"

    local bytes_per_elem=0
    case "${input_type}" in
        -u2|u2) bytes_per_elem=2 ;;
        -u4|u4) bytes_per_elem=4 ;;
        *)
            echo "[error] unsupported input type: ${input_type}" | tee -a "${LOG_FILE}"
            return 1
            ;;
    esac
    if (( file_size_bytes % bytes_per_elem != 0 )); then
        echo "[error] file size is not aligned to dtype bytes: ${dataset_file} (${file_size_bytes} bytes, dtype_bytes=${bytes_per_elem})" | tee -a "${LOG_FILE}"
        return 1
    fi
    local elem_count=$((file_size_bytes / bytes_per_elem))

    # u4 benchmark uses 1D shape only (no 3D mapping path)
    case "${input_type}" in
        -u4|u4)
            printf '1 %s\n' "${elem_count}"
            return 0
            ;;
    esac

    local meta_file="${dataset_file}.json"
    if [[ -f "${meta_file}" ]]; then
        local dims_line=""
        if dims_line="$(read_dims_from_meta "${meta_file}" 2>/dev/null || true)" && [[ -n "${dims_line}" ]]; then
            local -a tokens=()
            read -r -a tokens <<< "${dims_line}"
            local prod=1
            local idx
            for ((idx=1; idx<${#tokens[@]}; idx++)); do
                prod=$((prod * tokens[idx]))
            done
            if (( prod == elem_count )); then
                if [[ "${force_1d}" == "1" && "${tokens[0]}" == "3" ]]; then
                    printf '1 %s\n' "${elem_count}"
                else
                    echo "${dims_line}"
                fi
                return 0
            fi
            echo "[warn] metadata dims product mismatch, fallback to 1D: ${dataset_file}" | tee -a "${LOG_FILE}"
        else
            echo "[warn] failed to parse dims from metadata, fallback to 1D: ${meta_file}" | tee -a "${LOG_FILE}"
        fi
    fi

    printf '1 %s\n' "${elem_count}"
}

add_dataset_files() {
    local root="$1"
    local input_type="$2"
    local force_1d="${3:-0}"

    if [[ ! -d "${root}" ]]; then
        echo "[warn] dataset root not found, skip: ${root}" | tee -a "${LOG_FILE}"
        return
    fi

    local files=()
    local pattern="*"
    case "${input_type}" in
        -u2|u2) pattern="*.u2" ;;
        -u4|u4) pattern="*.u4" ;;
    esac
    while IFS= read -r -d '' file; do
        files+=("${file}")
    done < <(find "${root}" -type f -name "${pattern}" -print0 | sort -z)

    if [[ ${#files[@]} -eq 0 ]]; then
        echo "[warn] no files found under, skip: ${root}" | tee -a "${LOG_FILE}"
        return
    fi

    for file in "${files[@]}"; do
        dataset_files+=("${file}")
        dataset_types+=("${input_type}")
        dataset_force_1d+=("${force_1d}")
    done
}

add_dataset_files "${U2_DATASET_ROOT}" "-u2"
add_dataset_files "${FORCE_1D_DATASET_ROOT}" "-u2" "1"
add_dataset_files "${U4_DATASET_ROOT}" "-u4"

if [[ ${#dataset_files[@]} -eq 0 ]]; then
    echo "[done] no datasets to run. csv=${CSV_FILE}" | tee -a "${LOG_FILE}"
    exit 0
fi

if [[ ! -x "${WORKDIR}/${BENCH_BIN#./}" ]]; then
    echo "[error] bench binary not found or not executable: ${WORKDIR}/${BENCH_BIN#./}" | tee -a "${LOG_FILE}"
    exit 1
fi

echo "[info] datasets: ${#dataset_files[@]}" | tee -a "${LOG_FILE}"
echo "[info] u2_root: ${U2_DATASET_ROOT}" | tee -a "${LOG_FILE}"
echo "[info] FORCE_1D_DATASET_ROOT: ${FORCE_1D_DATASET_ROOT}" | tee -a "${LOG_FILE}"
echo "[info] u4_root: ${U4_DATASET_ROOT}" | tee -a "${LOG_FILE}"
echo "[info] csv: ${CSV_FILE}" | tee -a "${LOG_FILE}"
echo "[info] log: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "[info] warmup: ${WARMUP}" | tee -a "${LOG_FILE}"
echo "[info] iter: ${ITER}" | tee -a "${LOG_FILE}"

for i in "${!dataset_files[@]}"; do
    dataset_file="${dataset_files[$i]}"
    input_type="${dataset_types[$i]}"
    force_1d="${dataset_force_1d[$i]}"
    dataset_folder="$(basename -- "$(dirname -- "${dataset_file}")")"
    if [[ "${force_1d}" == "1" ]]; then
        dataset_folder="1d-${dataset_folder}"
    fi
    dataset_name="$(basename -- "${dataset_file}")"
    file_size_bytes="$(stat -c%s "${dataset_file}")"
    dims_args="$(resolve_dims_args "${dataset_file}" "${input_type}" "${file_size_bytes}" "${force_1d}")"
    if [[ -z "${dims_args}" ]]; then
        err="resolve_dims_failed"
        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
            "${dataset_folder}" \
            "${dataset_name}" \
            "${file_size_bytes}" \
            "${ALGO}" \
            "${input_type}" \
            "" "" "" \
            "${err}" >> "${CSV_FILE}"
        echo "[warn] skip dataset due to dim parse failure: ${dataset_file}" | tee -a "${LOG_FILE}"
        continue
    fi
    read -r -a dims_tokens <<< "${dims_args}"

    tmp_csv="$(mktemp /tmp/cpu_mans_bench_uall.XXXXXX.csv)"
    tmp_log="$(mktemp /tmp/cpu_mans_bench_uall.XXXXXX.log)"

    echo "[run] (${input_type}) ${dataset_file} --warmup ${WARMUP} --runs ${ITER} --dims ${dims_args}" | tee -a "${LOG_FILE}"
    if (cd "${WORKDIR}" && "${BENCH_BIN}" "${input_type}" "${dataset_file}" --mode "${MODE}" --threshold "${THRESHOLD}" --warmup "${WARMUP}" --runs "${ITER}" --dims "${dims_tokens[@]}" --csv "${tmp_csv}") > "${tmp_log}" 2>&1; then
        cat "${tmp_log}" >> "${LOG_FILE}"
        wrote_rows=0
        while IFS=, read -r chunk_label chunk_bytes ratio comp_mbps decomp_mbps; do
            if [[ -z "${chunk_label}" ]]; then
                continue
            fi
            printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
                "${dataset_folder}" \
                "${dataset_name}" \
                "${file_size_bytes}" \
                "${ALGO}" \
                "${input_type}" \
                "${ratio}" \
                "${comp_mbps}" \
                "${decomp_mbps}" \
                "" >> "${CSV_FILE}"
            wrote_rows=1
        done < <(awk 'NR > 1' "${tmp_csv}")

        if [[ "${wrote_rows}" -eq 0 ]]; then
            err="empty_result_csv"
            printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
                "${dataset_folder}" \
                "${dataset_name}" \
                "${file_size_bytes}" \
                "${ALGO}" \
                "${input_type}" \
                "" "" "" \
                "${err}" >> "${CSV_FILE}"
            echo "[warn] empty result rows: ${dataset_file}" | tee -a "${LOG_FILE}"
        fi
    else
        cat "${tmp_log}" >> "${LOG_FILE}"
        err="$(awk '
            /Decompression mismatch|Compression failed|Invalid|Out of memory|Segmentation fault|core dumped|^\[Error\]/ { msg=$0 }
            END { if (msg != "") print msg; else print "bench_command_failed" }
        ' "${tmp_log}" | tail -n 1)"
        err="${err//,/;}"
        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
            "${dataset_folder}" \
            "${dataset_name}" \
            "${file_size_bytes}" \
            "${ALGO}" \
            "${input_type}" \
            "" "" "" \
            "${err}" >> "${CSV_FILE}"
        echo "[warn] bench failed: ${dataset_file}" | tee -a "${LOG_FILE}"
    fi

    rm -f "${tmp_csv}" "${tmp_log}" "${tmp_csv}".*.timing.csv
done
echo "[done] finished. csv=${CSV_FILE}" | tee -a "${LOG_FILE}"
