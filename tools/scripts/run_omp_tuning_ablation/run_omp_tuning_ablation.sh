#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# --------------------------
# Constants to edit
# --------------------------
U2_DATASET_ROOT="${U2_DATASET_ROOT:-/workspace/MANS/datasets/testdata/u2}"
U4_DATASET_ROOT="${U4_DATASET_ROOT:-/workspace/MANS/datasets/testdata/u4}"
WORKDIR="${WORKDIR:-/workspace/MANS/build}"
MODE="${MODE:-p}"
# --------------------------
# Constants to edit
# --------------------------
ALGO="${ALGO:-mans+_${MODE}}"
BENCH_BIN="${BENCH_BIN:-./bin/cpu/cpu_mans_bench}"
CHUNKS_MB="${CHUNKS_MB:-0}"
THRESHOLD="${THRESHOLD:-4000}"

get_cpu_threads() {
    if command -v nproc >/dev/null 2>&1; then
        nproc
    else
        getconf _NPROCESSORS_ONLN
    fi
}

build_threads_spec() {
    local threads="$1"
    local count="$2"

    local spec=""
    local idx
    for ((idx = 1; idx <= count; idx++)); do
        if [[ "${idx}" -gt 1 ]]; then
            spec+=","
        fi
        spec+="${threads}"
    done

    printf '%s\n' "${spec}"
}

count_csv_values() {
    local csv="$1"
    local -a items=()
    local count=0
    IFS=',' read -r -a items <<< "${csv}"
    for item in "${items[@]}"; do
        item="${item//[[:space:]]/}"
        [[ -z "${item}" ]] && continue
        ((count += 1))
    done
    printf '%s\n' "${count}"
}

normalize_threads_spec() {
    local spec="$1"
    local stages="$2"
    local count
    count="$(count_csv_values "${spec}")"

    if [[ "${count}" -eq 1 ]]; then
        printf '%s\n' "$(build_threads_spec "${spec//[[:space:]]/}" "${stages}")"
        return
    fi

    if [[ "${count}" -ne "${stages}" ]]; then
        echo "[error] invalid MANUAL_THREADS_SPEC='${spec}', expected 1 or ${stages} values" >&2
        exit 1
    fi

    printf '%s\n' "${spec}"
}

NPROC_THREADS="${NPROC_THREADS:-$(get_cpu_threads)}"
THREAD_STAGE_COUNT="${THREAD_STAGE_COUNT:-2}"
MANUAL_THREADS_SPEC="${MANUAL_THREADS_SPEC:-$(build_threads_spec "${NPROC_THREADS}" "${THREAD_STAGE_COUNT}")}"
MANUAL_THREADS_SPEC="$(normalize_threads_spec "${MANUAL_THREADS_SPEC}" "${THREAD_STAGE_COUNT}")"
MANUAL_RUN_TAG="manual_threads_nproc${NPROC_THREADS}"
AUTOTUNE_RUN_TAG="autotune"

CSV_FILE="${CSV_FILE:-${SCRIPT_DIR}/${ALGO}_omp_tuning_ablation_results.csv}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/log}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${ALGO}_omp_tuning_ablation.log}"

mkdir -p "${LOG_DIR}"
: > "${LOG_FILE}"

printf 'dataset_folder,dataset_name,file_size_bytes,algo,input_type,ratio,comp_mbps,decomp_mbps,error\n' > "${CSV_FILE}"

dataset_files=()
dataset_types=()

add_dataset_files() {
    local root="$1"
    local input_type="$2"

    if [[ ! -d "${root}" ]]; then
        echo "[warn] dataset root not found, skip: ${root}" | tee -a "${LOG_FILE}"
        return
    fi

    local files=()
    while IFS= read -r -d '' file; do
        files+=("${file}")
    done < <(find "${root}" -type f -print0 | sort -z)

    if [[ ${#files[@]} -eq 0 ]]; then
        echo "[warn] no files found under, skip: ${root}" | tee -a "${LOG_FILE}"
        return
    fi

    for file in "${files[@]}"; do
        dataset_files+=("${file}")
        dataset_types+=("${input_type}")
    done
}

run_bench_case() {
    local dataset_file="$1"
    local input_type="$2"
    local run_tag="$3"
    local threads_spec="${4:-}"

    local dataset_folder
    local dataset_name
    local file_size_bytes
    local tmp_csv
    local tmp_log
    local algo_tag

    dataset_folder="$(basename -- "$(dirname -- "${dataset_file}")")"
    dataset_name="$(basename -- "${dataset_file}")"
    file_size_bytes="$(stat -c%s "${dataset_file}")"
    tmp_csv="$(mktemp /tmp/cpu_mans_bench_uall.XXXXXX.csv)"
    tmp_log="$(mktemp /tmp/cpu_mans_bench_uall.XXXXXX.log)"
    algo_tag="${ALGO}_${run_tag}"

    local -a cmd=("${BENCH_BIN}" "${input_type}" "${dataset_file}" --mode "${MODE}" --chunks "${CHUNKS_MB}" --csv "${tmp_csv}")
    if [[ -n "${threads_spec}" ]]; then
        cmd+=(--threads "${threads_spec}")
    fi

    echo "[run] (${run_tag}) (${input_type}) ${dataset_file}" | tee -a "${LOG_FILE}"
    if (cd "${WORKDIR}" && "${cmd[@]}") > "${tmp_log}" 2>&1; then
        cat "${tmp_log}" >> "${LOG_FILE}"

        local wrote_rows=0
        while IFS=, read -r chunk_label chunk_bytes ratio_pct comp_mbps decomp_mbps; do
            if [[ -z "${chunk_label}" ]]; then
                continue
            fi
            local ratio
            ratio="$(awk -v pct="${ratio_pct}" 'BEGIN { if (pct + 0 > 0) printf "%.2f", 100.0 / pct; else print "" }')"
            printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
                "${dataset_folder}" \
                "${dataset_name}" \
                "${file_size_bytes}" \
                "${algo_tag}" \
                "${input_type}" \
                "${ratio}" \
                "${comp_mbps}" \
                "${decomp_mbps}" \
                "" >> "${CSV_FILE}"
            wrote_rows=1
        done < <(awk 'NR > 1' "${tmp_csv}")

        if [[ "${wrote_rows}" -eq 0 ]]; then
            local err="empty_result_csv"
            printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
                "${dataset_folder}" \
                "${dataset_name}" \
                "${file_size_bytes}" \
                "${algo_tag}" \
                "${input_type}" \
                "" "" "" \
                "${err}" >> "${CSV_FILE}"
            echo "[warn] empty result rows: ${dataset_file} (${run_tag})" | tee -a "${LOG_FILE}"
        fi
    else
        cat "${tmp_log}" >> "${LOG_FILE}"
        local err
        err="$(awk '
            /Decompression mismatch|Compression failed|Invalid|Out of memory|Segmentation fault|core dumped|^\[Error\]/ { msg=$0 }
            END { if (msg != "") print msg; else print "bench_command_failed" }
        ' "${tmp_log}" | tail -n 1)"
        err="${err//,/;}"
        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
            "${dataset_folder}" \
            "${dataset_name}" \
            "${file_size_bytes}" \
            "${algo_tag}" \
            "${input_type}" \
            "" "" "" \
            "${err}" >> "${CSV_FILE}"
        echo "[warn] bench failed: ${dataset_file} (${run_tag})" | tee -a "${LOG_FILE}"
    fi

    rm -f "${tmp_csv}" "${tmp_log}" "${tmp_csv}".*.timing.csv
}

add_dataset_files "${U2_DATASET_ROOT}" "-u2"
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
echo "[info] u4_root: ${U4_DATASET_ROOT}" | tee -a "${LOG_FILE}"
echo "[info] chunks_mb: ${CHUNKS_MB}" | tee -a "${LOG_FILE}"
echo "[info] manual_threads_spec: ${MANUAL_THREADS_SPEC}" | tee -a "${LOG_FILE}"
echo "[info] csv: ${CSV_FILE}" | tee -a "${LOG_FILE}"
echo "[info] log: ${LOG_FILE}" | tee -a "${LOG_FILE}"

for i in "${!dataset_files[@]}"; do
    dataset_file="${dataset_files[$i]}"
    input_type="${dataset_types[$i]}"

    run_bench_case "${dataset_file}" "${input_type}" "${MANUAL_RUN_TAG}" "${MANUAL_THREADS_SPEC}"
    run_bench_case "${dataset_file}" "${input_type}" "${AUTOTUNE_RUN_TAG}"
done

echo "[done] finished. csv=${CSV_FILE}" | tee -a "${LOG_FILE}"
