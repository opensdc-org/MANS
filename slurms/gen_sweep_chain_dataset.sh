#!/usr/bin/env bash
set -euo pipefail

###### ***** Editable Constants (NO ARGS) ***** ######
# SLURM_NODES_LIST=(1)
SLURM_NODES_LIST=(1 10 20 25 30 35 40)
# SLURM_NODES_LIST=(1)
# SLURM_NODES_LIST=(16)
NPROC_PER_NODE=64
CPU_LIMIT=1
CHUNK_SIZES_MB=(4.0)
DATASET_DIMS=(758 852 841)
ENABLE_READ_EXCLUDE=0
MAX_ACTIVE_JOBS=10
BEST_THREAD_DIR="/public/share/acnnprvuzd/MANS/slurms/runs_autotune"
READ_BIN_DIR="/public/share/acnnprvuzd/MANS/build/bin/h5z-mans"

LIST=(mans fse_ans fse_huffman mans_original none)
# LIST=(fse_ans fse_huffman mans_original mans none)
CACHE_BUST_FILTERS=(mans mans_original)

SOURCE_DATASET_BIN="/public/share/acnnprvuzd/MANS/datasets/sfc-gi/2-sfc-1_758x842x841_1048576kB.u2"
RUNS_DIR_PREFIX="runs_sweep_dataset"

HDF5_PLUGIN_PATH_VALUE="/public/share/acnnprvuzd/MANS/build/bin/plugins"
HDF5_PLUGIN_PRELOAD_VALUE="libH5Z-MANS.so"

MODULE_SETUP_LINES=(
  "module purge"
  "module load compiler/cmake/3.20.4"
  "module load hdf5-1.14.3-intelmpi2021_p"
  "module unload compiler/devtoolset/7.3.1"
  "module load compiler/gcc/9.3.0"
)

SLURM_PARTITION="sdicnormal"
SLURM_TIME="06:00:00"
JOB_NAME_PREFIX="mans-chunk-sweep"
###### ***** End Constants ***** ######

if (( CPU_LIMIT != 1 )); then
  echo "[ERROR] this script expects CPU_LIMIT=1 (rank=cores per node)." >&2
  exit 1
fi
if (( NPROC_PER_NODE % CPU_LIMIT != 0 )); then
  echo "[ERROR] CPU_LIMIT(${CPU_LIMIT}) does not divide NPROC_PER_NODE(${NPROC_PER_NODE})" >&2
  exit 1
fi

if (( MAX_ACTIVE_JOBS < 1 )); then
  echo "[ERROR] MAX_ACTIVE_JOBS must be >= 1" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_SETUP_SCRIPT="$(printf '%s\n' "${MODULE_SETUP_LINES[@]}")"
SUBMIT_USER="${USER:-$(id -un)}"

if [[ ! -f "${SOURCE_DATASET_BIN}" ]]; then
  echo "[ERROR] source dataset not found: ${SOURCE_DATASET_BIN}" >&2
  exit 1
fi

source_base="$(basename "${SOURCE_DATASET_BIN}")"
DATASET_NAME="${source_base%.*}"
if [[ -z "${DATASET_NAME}" ]]; then
  echo "[ERROR] failed to derive dataset name from SOURCE_DATASET_BIN=${SOURCE_DATASET_BIN}" >&2
  exit 1
fi
DATASET_COPIES_DIR="${SCRIPT_DIR}/dataset_copies"
DATASET_BIN_DIR="${DATASET_COPIES_DIR}/${DATASET_NAME}"
mkdir -p "${DATASET_BIN_DIR}"

for bin_prefix in "${LIST[@]}"; do
  write_bin="${READ_BIN_DIR}/${bin_prefix}_write"
  read_bin="${READ_BIN_DIR}/${bin_prefix}_read"
  for bin in "${write_bin}" "${read_bin}"; do
    if [[ ! -x "${bin}" ]]; then
      echo "[ERROR] binary not found or not executable: ${bin}" >&2
      exit 1
    fi
  done
done

for bust_prefix in "${CACHE_BUST_FILTERS[@]}"; do
  bust_write_bin="${READ_BIN_DIR}/${bust_prefix}_write"
  if [[ ! -x "${bust_write_bin}" ]]; then
    echo "[ERROR] cache-bust write binary not found or not executable: ${bust_write_bin}" >&2
    exit 1
  fi
done

###### ***** Helpers ***** ######
wait_for_submit_slot() {
  local active
  while true; do
    active="$(squeue -u "${SUBMIT_USER}" -h 2>/dev/null | wc -l | tr -d ' ')"
    if [[ -z "${active}" ]]; then
      active=0
    fi
    if (( active < MAX_ACTIVE_JOBS )); then
      return 0
    fi
    echo "[WAIT] active jobs for ${SUBMIT_USER}: ${active} (limit=${MAX_ACTIVE_JOBS}), waiting 100s..." >&2
    sleep 100
  done
}

wait_job_running_and_get_nodelist() {
  local job_id="$1"
  local st nodelist sacct_state

  echo "[WAIT] job ${job_id} -> RUNNING to get nodelist ..." >&2
  while true; do
    st="$(squeue -j "${job_id}" -h -o "%t" 2>/dev/null | head -n 1 || true)"
    nodelist="$(squeue -j "${job_id}" -h -o "%N" 2>/dev/null | head -n 1 || true)"

    if [[ -z "${st}" ]]; then
      if command -v sacct >/dev/null 2>&1; then
        sacct_state="$(sacct -n -X -P -j "${job_id}" -o State 2>/dev/null | head -n 1 | cut -d'|' -f1 | tr -d '[:space:]' || true)"
        if [[ -z "${sacct_state}" || "${sacct_state}" == PENDING* || "${sacct_state}" == CONFIGURING* || "${sacct_state}" == RUNNING* || "${sacct_state}" == COMPLETING* ]]; then
          sleep 2
          continue
        fi
        echo "[ERROR] job ${job_id} not found in squeue, sacct state=${sacct_state}" >&2
        return 1
      fi
      echo "[ERROR] job ${job_id} not found in squeue" >&2
      return 1
    fi

    if [[ "${st}" == "R" && -n "${nodelist}" && "${nodelist}" != "(null)" ]]; then
      echo "[OK] job ${job_id} RUNNING on nodes: ${nodelist}" >&2
      printf '%s\n' "${nodelist}"
      return 0
    fi

    sleep 2
  done
}

pick_cache_bust_prefix() {
  local current="$1"
  local cand
  for cand in "${CACHE_BUST_FILTERS[@]}"; do
    if [[ "${cand}" == "${current}" ]]; then
      continue
    fi
    if [[ -x "${READ_BIN_DIR}/${cand}_write" ]]; then
      printf '%s\n' "${cand}"
      return 0
    fi
  done
  return 1
}

echo "###### ***** Generate Slurm Files (single write/read templates per prefix) ***** ######"

echo "[INFO] dataset bins dir: ${DATASET_BIN_DIR}"

prev_read_job_id=""
ntasks_per_node=$((NPROC_PER_NODE / CPU_LIMIT))

for bin_prefix in "${LIST[@]}"; do
  write_bin="${READ_BIN_DIR}/${bin_prefix}_write"
  read_bin="${READ_BIN_DIR}/${bin_prefix}_read"

  cache_bust_prefix="$(pick_cache_bust_prefix "${bin_prefix}" || true)"
  if [[ -z "${cache_bust_prefix}" ]]; then
    echo "[ERROR] no cache-bust candidate available for ${bin_prefix} from CACHE_BUST_FILTERS=(${CACHE_BUST_FILTERS[*]})" >&2
    exit 1
  fi

  RUNS_DIR="${SCRIPT_DIR}/${RUNS_DIR_PREFIX}_${bin_prefix}"
  LOGS_DIR="${RUNS_DIR}/logs"
  DATASETS_DIR="${RUNS_DIR}/datasets"
  CSV_PATH="${RUNS_DIR}/chunksize_thread_sweep.csv"

  mkdir -p "${RUNS_DIR}" "${LOGS_DIR}" "${DATASETS_DIR}"

  cat > "${CSV_PATH}" <<'CSV_EOF'
chunk_size_mb,slurm_nodes,rank,cpu_limit,mode,throughput_mibps,time_s,best_thread_csv
CSV_EOF

  write_slurm="${RUNS_DIR}/${bin_prefix}_write_cpu${CPU_LIMIT}.slurm"
  read_slurm="${RUNS_DIR}/${bin_prefix}_read_cpu${CPU_LIMIT}.slurm"

  cat > "${write_slurm}" <<SLURM_WRITE_EOF
#!/usr/bin/env bash
#SBATCH --time=${SLURM_TIME}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --exclusive

set -euo pipefail

${MODULE_SETUP_SCRIPT}

RUNS_DIR="${RUNS_DIR}"
LOGS_DIR="${LOGS_DIR}"
DATASETS_DIR="${DATASETS_DIR}"
DATASET_BIN_DIR="${DATASET_BIN_DIR}"
CSV_PATH="${CSV_PATH}"
BEST_THREAD_DIR="${BEST_THREAD_DIR}"
SOURCE_DATASET_BIN="${SOURCE_DATASET_BIN}"
READ_BIN_DIR="${READ_BIN_DIR}"
WRITE_BIN="${write_bin}"
BIN_PREFIX="${bin_prefix}"
DATASET_DIMS=(${DATASET_DIMS[*]})

CHUNK_MB="\${SWEEP_CHUNK_MB:?SWEEP_CHUNK_MB is required}"
SLURM_NODES_COUNT="\${SWEEP_SLURM_NODES:?SWEEP_SLURM_NODES is required}"
RANK="\${SWEEP_RANK:?SWEEP_RANK is required}"
NTASKS_PER_NODE="\${SWEEP_NTASKS_PER_NODE:?SWEEP_NTASKS_PER_NODE is required}"
CPU_LIMIT="\${SWEEP_CPU_LIMIT:?SWEEP_CPU_LIMIT is required}"
CACHE_BUST_PREFIX="\${SWEEP_CACHE_BUST_PREFIX:?SWEEP_CACHE_BUST_PREFIX is required}"

export OMP_NUM_THREADS="\${CPU_LIMIT}"
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export HDF5_PLUGIN_PATH="${HDF5_PLUGIN_PATH_VALUE}"
export HDF5_PLUGIN_PRELOAD="${HDF5_PLUGIN_PRELOAD_VALUE}"

BEST_THREAD_CSV="\${BEST_THREAD_DIR}/best_thread_cpu\${CPU_LIMIT}.csv"
if [[ -f "\${BEST_THREAD_CSV}" ]]; then
  export MANS_THREAD_CSV="\${BEST_THREAD_CSV}"
else
  BEST_THREAD_CSV="NA"
  unset MANS_THREAD_CSV || true
fi

parse_metric_from_text() {
  local text="\$1"
  local label="\$2"
  local line
  line="\$(printf '%s\n' "\${text}" | awk '/ranks=.*time=.*s.*throughput=.*MiB\\/s/ {last=\$0} END{print last}')"
  if [[ -z "\${line}" ]]; then
    echo "[ERROR] summary line not found for \${label}" >&2
    return 1
  fi
  local t thr
  t="\$(echo "\${line}" | sed -E 's/.* time=([0-9.]+) s.*/\\1/')"
  thr="\$(echo "\${line}" | sed -E 's/.* throughput=([0-9.]+) MiB\\/s.*/\\1/')"
  if [[ -z "\${t}" || -z "\${thr}" ]]; then
    echo "[ERROR] failed to parse metrics from: \${line}" >&2
    return 1
  fi
  echo "\${thr},\${t}"
}

cd "\${RUNS_DIR}"
mkdir -p "\${DATASETS_DIR}" "\${LOGS_DIR}" "\${DATASET_BIN_DIR}"

if [[ ! -f "\${SOURCE_DATASET_BIN}" ]]; then
  echo "[ERROR] source dataset missing: \${SOURCE_DATASET_BIN}" >&2
  exit 1
fi

if [[ "\${CACHE_BUST_PREFIX}" == "\${BIN_PREFIX}" ]]; then
  echo "[ERROR] CACHE_BUST_PREFIX(\${CACHE_BUST_PREFIX}) must differ from BIN_PREFIX(\${BIN_PREFIX})" >&2
  exit 1
fi
CACHE_BUST_WRITE_BIN="\${READ_BIN_DIR}/\${CACHE_BUST_PREFIX}_write"
if [[ ! -x "\${CACHE_BUST_WRITE_BIN}" ]]; then
  echo "[ERROR] cache-bust write binary not executable: \${CACHE_BUST_WRITE_BIN}" >&2
  exit 1
fi

source_sig="\$(stat -Lc '%s:%Y' "\${SOURCE_DATASET_BIN}")"
source_stamp="\${SOURCE_DATASET_BIN}|\${source_sig}"
stamp_file="\${DATASET_BIN_DIR}/.source_dataset_stamp"
existing_stamp=""
if [[ -f "\${stamp_file}" ]]; then
  existing_stamp="\$(cat "\${stamp_file}")"
fi

source_real="\$(readlink -f "\${SOURCE_DATASET_BIN}" 2>/dev/null || printf '%s\n' "\${SOURCE_DATASET_BIN}")"
dataset_bin_dir_real="\$(readlink -f "\${DATASET_BIN_DIR}" 2>/dev/null || printf '%s\n' "\${DATASET_BIN_DIR}")"
if [[ "\${source_real}" == \${dataset_bin_dir_real}/rank*.bin ]]; then
  echo "[ERROR] SOURCE_DATASET_BIN points to a managed rank copy: \${SOURCE_DATASET_BIN}" >&2
  exit 1
fi

need_refresh_all=0
if [[ "\${existing_stamp}" != "\${source_stamp}" ]]; then
  need_refresh_all=1
fi

copy_required=0
for ((r = 0; r < RANK; ++r)); do
  dst="\${DATASET_BIN_DIR}/rank\${r}.bin"
  if (( need_refresh_all != 0 )) || [[ ! -f "\${dst}" ]]; then
    copy_required=\$((copy_required + 1))
  fi
done

if (( copy_required > 0 )); then
  if (( need_refresh_all != 0 )); then
    echo "[INFO] source changed, refresh required rank bins without deleting existing files: need=\${copy_required} rank=\${RANK}"
  else
    echo "[INFO] generate missing rank bins: need=\${copy_required} rank=\${RANK}"
  fi
  export SOURCE_DATASET_BIN DATASET_BIN_DIR RANK NTASKS_PER_NODE NEED_REFRESH_ALL
  srun --nodes="\${SLURM_NODES_COUNT}" \
       --ntasks="\${SLURM_NODES_COUNT}" \
       --ntasks-per-node=1 \
       --cpus-per-task=1 \
       bash -c '
set -euo pipefail
node_id="\${SLURM_NODEID:?SLURM_NODEID is required}"
start=\$((node_id * NTASKS_PER_NODE))
end=\$((start + NTASKS_PER_NODE))
for ((r = start; r < end && r < RANK; ++r)); do
  dst="\${DATASET_BIN_DIR}/rank\${r}.bin"
  if (( NEED_REFRESH_ALL != 0 )) || [[ ! -f "\${dst}" ]]; then
    cp -f --reflink=auto "\${SOURCE_DATASET_BIN}" "\${dst}"
  fi
done
'
fi

missing_required_after=0
for ((r = 0; r < RANK; ++r)); do
  if [[ ! -f "\${DATASET_BIN_DIR}/rank\${r}.bin" ]]; then
    missing_required_after=\$((missing_required_after + 1))
  fi
done
if (( missing_required_after != 0 )); then
  echo "[ERROR] dataset bin setup incomplete: missing=\${missing_required_after} required_rank=\${RANK}" >&2
  exit 1
fi
printf '%s\n' "\${source_stamp}" > "\${stamp_file}"

BIN_TEMPLATE="\${DATASET_BIN_DIR}/rank%d.bin"
TEST_H5_TEMPLATE="\${DATASETS_DIR}/rank%d.h5"
CACHE_H5_TEMPLATE="\${DATASETS_DIR}/cache_\${CACHE_BUST_PREFIX}_rank%d.h5"

WRITE_DIMS_ARGS=()
if [[ "\${BIN_PREFIX}" == "mans" ]]; then
  if (( \${#DATASET_DIMS[@]} == 0 )); then
    echo "[ERROR] DATASET_DIMS is empty but BIN_PREFIX=mans" >&2
    exit 1
  fi
  WRITE_DIMS_ARGS=(--dims "\${DATASET_DIMS[@]}")
fi

CACHE_DIMS_ARGS=()
if [[ "\${CACHE_BUST_PREFIX}" == "mans" ]]; then
  if (( \${#DATASET_DIMS[@]} == 0 )); then
    echo "[ERROR] DATASET_DIMS is empty but CACHE_BUST_PREFIX=mans" >&2
    exit 1
  fi
  CACHE_DIMS_ARGS=(--dims "\${DATASET_DIMS[@]}")
fi

chunk_tag="\${CHUNK_MB//./p}"
export MANS_TIMING_ITER=1
export MANS_TIMING_DUMP_PATH="\${RUNS_DIR}/plugin_timing_\${BIN_PREFIX}_r\${RANK}_cpu\${CPU_LIMIT}_c\${chunk_tag}_write.csv"

set +e
write_cmd=(mpirun -np "\${RANK}" --bind-to core --map-by ppr:\${NTASKS_PER_NODE}:node:pe=\${CPU_LIMIT} "\${WRITE_BIN}" --chunk-size-mb "\${CHUNK_MB}" --bin-template "\${BIN_TEMPLATE}" --h5-template "\${TEST_H5_TEMPLATE}")
if (( \${#WRITE_DIMS_ARGS[@]} > 0 )); then
  write_cmd+=("\${WRITE_DIMS_ARGS[@]}")
fi
write_output="\$("\${write_cmd[@]}" 2>&1)"
mpirun_rc=\$?
set -e
printf '%s\n' "\${write_output}"

set +e
metric="\$(parse_metric_from_text "\${write_output}" "write")"
metric_rc=\$?
set -e

if (( mpirun_rc != 0 )); then
  if (( metric_rc == 0 )); then
    echo "[WARN] mpirun failed rc=\${mpirun_rc} but summary found; continuing." >&2
  else
    echo "[ERROR] mpirun failed rc=\${mpirun_rc} and no summary." >&2
    exit "\${mpirun_rc}"
  fi
fi

if [[ -f "\${RUNS_DIR}/plugin_timing.csv" ]]; then
  mv -f "\${RUNS_DIR}/plugin_timing.csv" "\${RUNS_DIR}/plugin_timing_\${BIN_PREFIX}_r\${RANK}_cpu\${CPU_LIMIT}_c\${chunk_tag}_write_fallback.csv"
fi

if (( metric_rc != 0 )); then
  echo "[ERROR] summary line not found for write." >&2
  exit 1
fi
thr="\${metric%%,*}"
t="\${metric##*,}"

echo "\${CHUNK_MB},\${SLURM_NODES_COUNT},\${RANK},\${CPU_LIMIT},compress,\${thr},\${t},\${BEST_THREAD_CSV}" >> "\${CSV_PATH}"

# cache-busting write: run a different filter write, then remove its output
export MANS_TIMING_DUMP_PATH="\${RUNS_DIR}/plugin_timing_\${BIN_PREFIX}_r\${RANK}_cpu\${CPU_LIMIT}_c\${chunk_tag}_cache.csv"
set +e
cache_cmd=(mpirun -np "\${RANK}" --bind-to core --map-by ppr:\${NTASKS_PER_NODE}:node:pe=\${CPU_LIMIT} "\${CACHE_BUST_WRITE_BIN}" --chunk-size-mb "\${CHUNK_MB}" --bin-template "\${BIN_TEMPLATE}" --h5-template "\${CACHE_H5_TEMPLATE}")
if (( \${#CACHE_DIMS_ARGS[@]} > 0 )); then
  cache_cmd+=("\${CACHE_DIMS_ARGS[@]}")
fi
cache_output="\$("\${cache_cmd[@]}" 2>&1)"
cache_rc=\$?
set -e
printf '%s\n' "\${cache_output}"

if (( cache_rc != 0 )); then
  echo "[ERROR] cache-busting write failed rc=\${cache_rc} filter=\${CACHE_BUST_PREFIX}" >&2
  exit "\${cache_rc}"
fi

rm -f "\${DATASETS_DIR}"/cache_*.h5
SLURM_WRITE_EOF

  cat > "${read_slurm}" <<SLURM_READ_EOF
#!/usr/bin/env bash
#SBATCH --time=${SLURM_TIME}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --exclusive

set -euo pipefail

${MODULE_SETUP_SCRIPT}

RUNS_DIR="${RUNS_DIR}"
LOGS_DIR="${LOGS_DIR}"
DATASETS_DIR="${DATASETS_DIR}"
CSV_PATH="${CSV_PATH}"
BEST_THREAD_DIR="${BEST_THREAD_DIR}"
READ_BIN="${read_bin}"
BIN_PREFIX="${bin_prefix}"

CHUNK_MB="\${SWEEP_CHUNK_MB:?SWEEP_CHUNK_MB is required}"
SLURM_NODES_COUNT="\${SWEEP_SLURM_NODES:?SWEEP_SLURM_NODES is required}"
RANK="\${SWEEP_RANK:?SWEEP_RANK is required}"
NTASKS_PER_NODE="\${SWEEP_NTASKS_PER_NODE:?SWEEP_NTASKS_PER_NODE is required}"
CPU_LIMIT="\${SWEEP_CPU_LIMIT:?SWEEP_CPU_LIMIT is required}"

export OMP_NUM_THREADS="\${CPU_LIMIT}"
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export HDF5_PLUGIN_PATH="${HDF5_PLUGIN_PATH_VALUE}"
export HDF5_PLUGIN_PRELOAD="${HDF5_PLUGIN_PRELOAD_VALUE}"

BEST_THREAD_CSV="\${BEST_THREAD_DIR}/best_thread_cpu\${CPU_LIMIT}.csv"
if [[ -f "\${BEST_THREAD_CSV}" ]]; then
  export MANS_THREAD_CSV="\${BEST_THREAD_CSV}"
else
  BEST_THREAD_CSV="NA"
  unset MANS_THREAD_CSV || true
fi

parse_metric_from_text() {
  local text="\$1"
  local label="\$2"
  local line
  line="\$(printf '%s\n' "\${text}" | awk '/ranks=.*time=.*s.*throughput=.*MiB\\/s/ {last=\$0} END{print last}')"
  if [[ -z "\${line}" ]]; then
    echo "[ERROR] summary line not found for \${label}" >&2
    return 1
  fi
  local t thr
  t="\$(echo "\${line}" | sed -E 's/.* time=([0-9.]+) s.*/\\1/')"
  thr="\$(echo "\${line}" | sed -E 's/.* throughput=([0-9.]+) MiB\\/s.*/\\1/')"
  if [[ -z "\${t}" || -z "\${thr}" ]]; then
    echo "[ERROR] failed to parse metrics from: \${line}" >&2
    return 1
  fi
  echo "\${thr},\${t}"
}

cd "\${RUNS_DIR}"
mkdir -p "\${DATASETS_DIR}" "\${LOGS_DIR}"

TEST_H5_TEMPLATE="\${DATASETS_DIR}/rank%d.h5"

chunk_tag="\${CHUNK_MB//./p}"
export MANS_TIMING_ITER=1
export MANS_TIMING_DUMP_PATH="\${RUNS_DIR}/plugin_timing_\${BIN_PREFIX}_r\${RANK}_cpu\${CPU_LIMIT}_c\${chunk_tag}_read.csv"

set +e
read_cmd=(mpirun -np "\${RANK}" --bind-to core --map-by ppr:\${NTASKS_PER_NODE}:node:pe=\${CPU_LIMIT} "\${READ_BIN}" --h5-template "\${TEST_H5_TEMPLATE}")
read_output="\$("\${read_cmd[@]}" 2>&1)"
mpirun_rc=\$?
set -e
printf '%s\n' "\${read_output}"

set +e
metric="\$(parse_metric_from_text "\${read_output}" "read")"
metric_rc=\$?
set -e

if (( mpirun_rc != 0 )); then
  if (( metric_rc == 0 )); then
    echo "[WARN] mpirun failed rc=\${mpirun_rc} but summary found; continuing." >&2
  else
    echo "[ERROR] mpirun failed rc=\${mpirun_rc} and no summary." >&2
    exit "\${mpirun_rc}"
  fi
fi

if [[ -f "\${RUNS_DIR}/plugin_timing.csv" ]]; then
  mv -f "\${RUNS_DIR}/plugin_timing.csv" "\${RUNS_DIR}/plugin_timing_\${BIN_PREFIX}_r\${RANK}_cpu\${CPU_LIMIT}_c\${chunk_tag}_read_fallback.csv"
fi

if (( metric_rc != 0 )); then
  echo "[ERROR] summary line not found for read." >&2
  exit 1
fi
thr="\${metric%%,*}"
t="\${metric##*,}"

echo "\${CHUNK_MB},\${SLURM_NODES_COUNT},\${RANK},\${CPU_LIMIT},decompress,\${thr},\${t},\${BEST_THREAD_CSV}" >> "\${CSV_PATH}"
SLURM_READ_EOF

  chmod +x "${write_slurm}" "${read_slurm}"
  echo "generated: ${write_slurm}"
  echo "generated: ${read_slurm}"

  echo
  echo "###### ***** Submit Chain (${bin_prefix}): node -> chunk(write->read->cleanup) ***** ######"

  for slurm_nodes in "${SLURM_NODES_LIST[@]}"; do
    if (( slurm_nodes < 1 )); then
      echo "[ERROR] slurm_nodes(${slurm_nodes}) must be >= 1" >&2
      exit 1
    fi

    NPROC_TOTAL=$((NPROC_PER_NODE * slurm_nodes))
    rank=${NPROC_TOTAL}

    echo
    echo "====== prefix=${bin_prefix} nodes=${slurm_nodes} rank=${rank} cpu=${CPU_LIMIT} cache_bust=${cache_bust_prefix} ======"

    for chunk_mb in "${CHUNK_SIZES_MB[@]}"; do
      chunk_tag="${chunk_mb//./p}"
      common_export="ALL,SWEEP_CHUNK_MB=${chunk_mb},SWEEP_SLURM_NODES=${slurm_nodes},SWEEP_RANK=${rank},SWEEP_NTASKS_PER_NODE=${ntasks_per_node},SWEEP_CPU_LIMIT=${CPU_LIMIT},SWEEP_CACHE_BUST_PREFIX=${cache_bust_prefix}"

      wait_for_submit_slot
      if [[ -z "${prev_read_job_id}" ]]; then
        write_job_id="$(sbatch --parsable \
          --job-name="${JOB_NAME_PREFIX}-${bin_prefix}-w-n${slurm_nodes}-c${chunk_tag}" \
          --nodes="${slurm_nodes}" \
          --ntasks="${rank}" \
          --ntasks-per-node="${ntasks_per_node}" \
          --cpus-per-task="${CPU_LIMIT}" \
          --output="${LOGS_DIR}/${bin_prefix}-write-n${slurm_nodes}-r${rank}-cpu${CPU_LIMIT}-c${chunk_tag}.out" \
          --error="${LOGS_DIR}/${bin_prefix}-write-n${slurm_nodes}-r${rank}-cpu${CPU_LIMIT}-c${chunk_tag}.err" \
          --export="${common_export}" \
          "${write_slurm}")"
      else
        write_job_id="$(sbatch --parsable \
          --dependency=afterok:${prev_read_job_id} \
          --job-name="${JOB_NAME_PREFIX}-${bin_prefix}-w-n${slurm_nodes}-c${chunk_tag}" \
          --nodes="${slurm_nodes}" \
          --ntasks="${rank}" \
          --ntasks-per-node="${ntasks_per_node}" \
          --cpus-per-task="${CPU_LIMIT}" \
          --output="${LOGS_DIR}/${bin_prefix}-write-n${slurm_nodes}-r${rank}-cpu${CPU_LIMIT}-c${chunk_tag}.out" \
          --error="${LOGS_DIR}/${bin_prefix}-write-n${slurm_nodes}-r${rank}-cpu${CPU_LIMIT}-c${chunk_tag}.err" \
          --export="${common_export}" \
          "${write_slurm}")"
      fi
      write_job_id="${write_job_id%%;*}"
      echo "[SUBMIT] write job_id=${write_job_id} prefix=${bin_prefix} nodes=${slurm_nodes} chunk=${chunk_mb}"

      wait_for_submit_slot
      if (( ENABLE_READ_EXCLUDE != 0 )); then
        write_nodes="$(wait_job_running_and_get_nodelist "${write_job_id}")"
        read_job_id="$(sbatch --parsable \
          --dependency=afterok:${write_job_id} \
          --exclude="${write_nodes}" \
          --job-name="${JOB_NAME_PREFIX}-${bin_prefix}-r-n${slurm_nodes}-c${chunk_tag}" \
          --nodes="${slurm_nodes}" \
          --ntasks="${rank}" \
          --ntasks-per-node="${ntasks_per_node}" \
          --cpus-per-task="${CPU_LIMIT}" \
          --output="${LOGS_DIR}/${bin_prefix}-read-n${slurm_nodes}-r${rank}-cpu${CPU_LIMIT}-c${chunk_tag}.out" \
          --error="${LOGS_DIR}/${bin_prefix}-read-n${slurm_nodes}-r${rank}-cpu${CPU_LIMIT}-c${chunk_tag}.err" \
          --export="${common_export}" \
          "${read_slurm}")"
        echo "[SUBMIT] read  (afterok:${write_job_id}, exclude=${write_nodes})"
      else
        read_job_id="$(sbatch --parsable \
          --dependency=afterok:${write_job_id} \
          --job-name="${JOB_NAME_PREFIX}-${bin_prefix}-r-n${slurm_nodes}-c${chunk_tag}" \
          --nodes="${slurm_nodes}" \
          --ntasks="${rank}" \
          --ntasks-per-node="${ntasks_per_node}" \
          --cpus-per-task="${CPU_LIMIT}" \
          --output="${LOGS_DIR}/${bin_prefix}-read-n${slurm_nodes}-r${rank}-cpu${CPU_LIMIT}-c${chunk_tag}.out" \
          --error="${LOGS_DIR}/${bin_prefix}-read-n${slurm_nodes}-r${rank}-cpu${CPU_LIMIT}-c${chunk_tag}.err" \
          --export="${common_export}" \
          "${read_slurm}")"
        echo "[SUBMIT] read  (afterok:${write_job_id}, exclude=disabled)"
      fi

      read_job_id="${read_job_id%%;*}"
      echo "[SUBMIT] read  job_id=${read_job_id} prefix=${bin_prefix} nodes=${slurm_nodes} chunk=${chunk_mb}"

      prev_read_job_id="${read_job_id}"
    done
  done

  if [[ -n "${prev_read_job_id}" ]]; then
    wait_for_submit_slot
    cleanup_job_id="$(sbatch --parsable \
      --dependency=afterok:${prev_read_job_id} \
      --job-name="${JOB_NAME_PREFIX}-${bin_prefix}-cleanup" \
      --nodes=1 \
      --ntasks=1 \
      --cpus-per-task=1 \
      --partition="${SLURM_PARTITION}" \
      --output="${LOGS_DIR}/${bin_prefix}-cleanup.out" \
      --error="${LOGS_DIR}/${bin_prefix}-cleanup.err" \
      --wrap="rm -f \"${DATASETS_DIR}\"/*.h5")"
    cleanup_job_id="${cleanup_job_id%%;*}"
    echo "[SUBMIT] cleanup job_id=${cleanup_job_id} prefix=${bin_prefix} datasets=${DATASETS_DIR}"
    prev_read_job_id="${cleanup_job_id}"
  fi

done
echo
echo "All jobs submitted in strict sequence: prefix outer, then node, then chunk(write->read->cleanup)."
echo "run dirs:"
for bin_prefix in "${LIST[@]}"; do
  echo "  ${SCRIPT_DIR}/${RUNS_DIR_PREFIX}_${bin_prefix}"
done
echo "dataset bins dir: ${DATASET_BIN_DIR}"
