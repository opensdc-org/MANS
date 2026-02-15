#!/usr/bin/env bash
set -euo pipefail

###### ***** Editable Constants (NO ARGS) ***** ######
SLURM_NODES=12
NPROC_PER_NODE=96
NPROC_TOTAL=$((NPROC_PER_NODE * SLURM_NODES))
TOTAL_DATA_GB=$((96 * SLURM_NODES))

CPU_LIMIT_LIST=(1 2 3 4 6 8 16 32)
CHUNK_SIZES_MB=(8)

BEST_THREAD_DIR="/public3/home/t6s010699/MANS-hdf5-filter/slurms/runs_autotune"
DATA_GEN_BIN="/public3/home/t6s010699/MANS-hdf5-filter/build/bin/h5z-mans/mans_data_gen"
MANS_WRITE_BIN="/public3/home/t6s010699/MANS-hdf5-filter/build/bin/h5z-mans/none_write"
MANS_READ_BIN="/public3/home/t6s010699/MANS-hdf5-filter/build/bin/h5z-mans/none_read"

HDF5_PLUGIN_PATH_VALUE="/public3/home/t6s010699/MANS-hdf5-filter/build/bin/plugins"
HDF5_PLUGIN_PRELOAD_VALUE="libH5Z-MANS.so"

SLURM_PARTITION="v6_384"
SLURM_TIME="06:00:00"
JOB_NAME_PREFIX="mans-chunk-sweep"

DTYPE="u16"
ADM_THRESHOLD=4000
DATA_GEN_PROCS_PER_NODE=32
###### ***** End Constants ***** ######

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS_DIR="${SCRIPT_DIR}/runs_sweep"
LOGS_DIR="${RUNS_DIR}/logs"
DATASETS_DIR="${RUNS_DIR}/datasets"
CSV_PATH="${RUNS_DIR}/chunksize_thread_sweep.csv"

mkdir -p "${RUNS_DIR}" "${LOGS_DIR}" "${DATASETS_DIR}"

for bin in "${DATA_GEN_BIN}" "${MANS_WRITE_BIN}" "${MANS_READ_BIN}"; do
  if [[ ! -x "${bin}" ]]; then
    echo "[ERROR] binary not found or not executable: ${bin}" >&2
    exit 1
  fi
done

###### ***** Init Summary CSV (overwrite) ***** ######
cat > "${CSV_PATH}" <<'EOF'
chunk_size_mb,rank,cpu_limit,mode,throughput_mibps,time_s,best_thread_csv
EOF

###### ***** Helpers ***** ######
wait_job_running_and_get_nodelist() {
  local job_id="$1"
  local st nodelist

  echo "[WAIT] job ${job_id} -> RUNNING to get nodelist ..." >&2
  while true; do
    st="$(squeue -j "${job_id}" -h -o "%t" 2>/dev/null | head -n 1 || true)"
    nodelist="$(squeue -j "${job_id}" -h -o "%N" 2>/dev/null | head -n 1 || true)"

    if [[ -z "${st}" ]]; then
      echo "[ERROR] job ${job_id} not found in squeue" >&2
      return 1
    fi

    if [[ "${st}" == "R" && -n "${nodelist}" && "${nodelist}" != "(null)" ]]; then
      echo "[OK] job ${job_id} RUNNING on nodes: ${nodelist}" >&2
      printf '%s\n' "${nodelist}"   # 只有这一行走 stdout
      return 0
    fi

    sleep 2
  done
}

###### ***** Generate Slurm Files ***** ######
echo "###### ***** Generate Slurm Files (write/read) ***** ######"

for cpu_limit in "${CPU_LIMIT_LIST[@]}"; do
  if (( NPROC_PER_NODE % cpu_limit != 0 )); then
    echo "[ERROR] cpu_limit(${cpu_limit}) does not divide NPROC_PER_NODE(${NPROC_PER_NODE})" >&2
    exit 1
  fi

  ntasks_per_node=$((NPROC_PER_NODE / cpu_limit))
  rank=$((ntasks_per_node * SLURM_NODES))
  per_rank_mb=$(( TOTAL_DATA_GB * 1024 / rank ))

  write_slurm="${RUNS_DIR}/write_r${rank}.slurm"
  read_slurm="${RUNS_DIR}/read_r${rank}.slurm"

  #### WRITE SLURM ####
  cat > "${write_slurm}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${JOB_NAME_PREFIX}-write-r${rank}
#SBATCH --nodes=${SLURM_NODES}
#SBATCH --ntasks=${rank}
#SBATCH --ntasks-per-node=${ntasks_per_node}
#SBATCH --cpus-per-task=${cpu_limit}
#SBATCH --time=${SLURM_TIME}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --exclusive
#SBATCH --output=${LOGS_DIR}/write-r${rank}-cpu${cpu_limit}.out
#SBATCH --error=${LOGS_DIR}/write-r${rank}-cpu${cpu_limit}.err

set -euo pipefail

module purge
module load cmake/3.21.2
module load hdf5/1.10.6-openmpi-3.1.6-gcc-9.3.0

export OMP_NUM_THREADS=${cpu_limit}
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export HDF5_PLUGIN_PATH="${HDF5_PLUGIN_PATH_VALUE}"
export HDF5_PLUGIN_PRELOAD="${HDF5_PLUGIN_PRELOAD_VALUE}"

RUNS_DIR="${RUNS_DIR}"
LOGS_DIR="${LOGS_DIR}"
DATASETS_DIR="${DATASETS_DIR}"
CSV_PATH="${CSV_PATH}"
BEST_THREAD_DIR="${BEST_THREAD_DIR}"
DATA_GEN_BIN="${DATA_GEN_BIN}"
MANS_WRITE_BIN="${MANS_WRITE_BIN}"
RANK=${rank}
NTASKS_PER_NODE=${ntasks_per_node}
CPU_LIMIT=${cpu_limit}
PER_RANK_MB=${per_rank_mb}
DTYPE="${DTYPE}"
ADM_THRESHOLD=${ADM_THRESHOLD}
DATA_GEN_PROCS_PER_NODE=${DATA_GEN_PROCS_PER_NODE}

BEST_THREAD_CSV="\${BEST_THREAD_DIR}/best_thread_cpu\${CPU_LIMIT}.csv"
if [[ ! -f "\${BEST_THREAD_CSV}" ]]; then
  echo "[ERROR] best thread csv not found: \${BEST_THREAD_CSV}" >&2
  exit 1
fi
export MANS_THREAD_CSV="\${BEST_THREAD_CSV}"

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

echo "###### ***** [WRITE] Generate Dataset ***** ######"
gen_start_ts=\$(date +%s)

gen_procs_per_node=\${DATA_GEN_PROCS_PER_NODE}
if (( gen_procs_per_node < 1 )); then gen_procs_per_node=1; fi
if (( gen_procs_per_node > NTASKS_PER_NODE )); then gen_procs_per_node=\${NTASKS_PER_NODE}; fi

gen_ntasks=\$((SLURM_NNODES * gen_procs_per_node))
if (( gen_ntasks > RANK )); then gen_ntasks=\${RANK}; fi

gen_cpus_per_task=\${CPU_LIMIT}
if (( gen_cpus_per_task < 1 )); then gen_cpus_per_task=1; fi

echo "[INFO] data_gen ntasks=\${gen_ntasks} (nodes=\${SLURM_NNODES}, procs_per_node=\${gen_procs_per_node}, cpus_per_task=\${gen_cpus_per_task}, rank=\${RANK}, per_rank_mb=\${PER_RANK_MB})"

export DATA_GEN_BIN DATASETS_DIR RANK PER_RANK_MB DTYPE ADM_THRESHOLD
srun --nodes "\${SLURM_NNODES}" \\
     --ntasks "\${gen_ntasks}" \\
     --ntasks-per-node "\${gen_procs_per_node}" \\
     --cpus-per-task "\${gen_cpus_per_task}" \\
     /bin/bash -lc '
       set -euo pipefail
       for ((r = SLURM_PROCID; r < RANK; r += SLURM_NTASKS)); do
         "\${DATA_GEN_BIN}" \
           --output-name "\${DATASETS_DIR}/rank\${r}.bin" \
           --size-per-rank-mb "\${PER_RANK_MB}" \
           --dtype "\${DTYPE}" \
           --adm-threshold "\${ADM_THRESHOLD}"
       done
     '

gen_end_ts=\$(date +%s)
generated_count=\$(find "\${DATASETS_DIR}" -maxdepth 1 -type f -name 'rank*.bin' | wc -l | tr -d ' ')
echo "[INFO] dataset generation done: \${generated_count}/\${RANK} files, elapsed=\$((gen_end_ts - gen_start_ts))s"
if (( generated_count != RANK )); then
  echo "[ERROR] dataset generation incomplete: expected \${RANK}, got \${generated_count}" >&2
  exit 1
fi

echo "###### ***** [WRITE] Compress + Write (chunk sweep) ***** ######"
timing_iter=0
for chunk_mb in ${CHUNK_SIZES_MB[*]}; do
  chunk_tag="\${chunk_mb//./p}"

  timing_iter=\$((timing_iter + 1))
  export MANS_TIMING_ITER="\${timing_iter}"
  export MANS_TIMING_DUMP_PATH="\${RUNS_DIR}/plugin_timing_r\${RANK}_cpu\${CPU_LIMIT}_c\${chunk_tag}_write.csv"

  write_output="\$(mpirun -np "\${RANK}" --bind-to core --map-by ppr:\${NTASKS_PER_NODE}:node:pe=\${CPU_LIMIT} "\${MANS_WRITE_BIN}" --chunk-size-mb "\${chunk_mb}" 2>&1)"
  printf '%s\n' "\${write_output}"

  if [[ -f "\${RUNS_DIR}/plugin_timing.csv" ]]; then
    mv -f "\${RUNS_DIR}/plugin_timing.csv" "\${RUNS_DIR}/plugin_timing_r\${RANK}_cpu\${CPU_LIMIT}_c\${chunk_tag}_write_fallback.csv"
  fi

  metric="\$(parse_metric_from_text "\${write_output}" "write")"
  thr="\${metric%%,*}"
  t="\${metric##*,}"

  echo "\${chunk_mb},\${RANK},\${CPU_LIMIT},compress,\${thr},\${t},\${BEST_THREAD_CSV}" >> "\${CSV_PATH}"
  echo "[APPEND] mode=compress chunk=\${chunk_mb} rank=\${RANK} cpu=\${CPU_LIMIT} thr=\${thr}"
done

echo "###### ***** [WRITE] Done ***** ######"
EOF

  #### READ SLURM ####
  cat > "${read_slurm}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${JOB_NAME_PREFIX}-read-r${rank}
#SBATCH --nodes=${SLURM_NODES}
#SBATCH --ntasks=${rank}
#SBATCH --ntasks-per-node=${ntasks_per_node}
#SBATCH --cpus-per-task=${cpu_limit}
#SBATCH --time=${SLURM_TIME}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --exclusive
#SBATCH --output=${LOGS_DIR}/read-r${rank}-cpu${cpu_limit}.out
#SBATCH --error=${LOGS_DIR}/read-r${rank}-cpu${cpu_limit}.err

set -euo pipefail

module purge
module load cmake/3.21.2
module load hdf5/1.10.6-openmpi-3.1.6-gcc-9.3.0

export OMP_NUM_THREADS=${cpu_limit}
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export HDF5_PLUGIN_PATH="${HDF5_PLUGIN_PATH_VALUE}"
export HDF5_PLUGIN_PRELOAD="${HDF5_PLUGIN_PRELOAD_VALUE}"

RUNS_DIR="${RUNS_DIR}"
LOGS_DIR="${LOGS_DIR}"
DATASETS_DIR="${DATASETS_DIR}"
CSV_PATH="${CSV_PATH}"
BEST_THREAD_DIR="${BEST_THREAD_DIR}"
MANS_READ_BIN="${MANS_READ_BIN}"
RANK=${rank}
NTASKS_PER_NODE=${ntasks_per_node}
CPU_LIMIT=${cpu_limit}

BEST_THREAD_CSV="\${BEST_THREAD_DIR}/best_thread_cpu\${CPU_LIMIT}.csv"
if [[ ! -f "\${BEST_THREAD_CSV}" ]]; then
  echo "[ERROR] best thread csv not found: \${BEST_THREAD_CSV}" >&2
  exit 1
fi
export MANS_THREAD_CSV="\${BEST_THREAD_CSV}"

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

echo "###### ***** [READ] Read + Decompress (chunk sweep) ***** ######"
timing_iter=0
for chunk_mb in ${CHUNK_SIZES_MB[*]}; do
  chunk_tag="\${chunk_mb//./p}"

  timing_iter=\$((timing_iter + 1))
  export MANS_TIMING_ITER="\${timing_iter}"
  export MANS_TIMING_DUMP_PATH="\${RUNS_DIR}/plugin_timing_r\${RANK}_cpu\${CPU_LIMIT}_c\${chunk_tag}_read.csv"

  read_output="\$(mpirun -np "\${RANK}" --bind-to core --map-by ppr:\${NTASKS_PER_NODE}:node:pe=\${CPU_LIMIT} "\${MANS_READ_BIN}" 2>&1)"
  printf '%s\n' "\${read_output}"

  if [[ -f "\${RUNS_DIR}/plugin_timing.csv" ]]; then
    mv -f "\${RUNS_DIR}/plugin_timing.csv" "\${RUNS_DIR}/plugin_timing_r\${RANK}_cpu\${CPU_LIMIT}_c\${chunk_tag}_read_fallback.csv"
  fi

  metric="\$(parse_metric_from_text "\${read_output}" "read")"
  thr="\${metric%%,*}"
  t="\${metric##*,}"

  echo "\${chunk_mb},\${RANK},\${CPU_LIMIT},decompress,\${thr},\${t},\${BEST_THREAD_CSV}" >> "\${CSV_PATH}"
  echo "[APPEND] mode=decompress chunk=\${chunk_mb} rank=\${RANK} cpu=\${CPU_LIMIT} thr=\${thr}"
done

echo "###### ***** [READ] Cleanup Dataset Files ***** ######"
rm -f "\${DATASETS_DIR}"/rank*.bin
rm -f "\${DATASETS_DIR}"/rank*.h5

echo "###### ***** [READ] Done ***** ######"
EOF

  chmod +x "${write_slurm}" "${read_slurm}"
  echo "generated: ${write_slurm}"
  echo "generated: ${read_slurm}"
done

echo
echo "###### ***** Blocking Submit: write -> (wait RUN get nodes) -> read(exclude) -> next ***** ######"

prev_read_job_id=""

for cpu_limit in "${CPU_LIMIT_LIST[@]}"; do
  ntasks_per_node=$((NPROC_PER_NODE / cpu_limit))
  rank=$((ntasks_per_node * SLURM_NODES))

  write_slurm="${RUNS_DIR}/write_r${rank}.slurm"
  read_slurm="${RUNS_DIR}/read_r${rank}.slurm"

  echo
  echo "====== cpu_limit=${cpu_limit} rank=${rank} ======"

  # submit write (dependent on previous read if any)
  if [[ -z "${prev_read_job_id}" ]]; then
    write_job_id="$(sbatch --parsable "${write_slurm}")"
  else
    write_job_id="$(sbatch --parsable --dependency=afterok:${prev_read_job_id} "${write_slurm}")"
  fi
  write_job_id="${write_job_id%%;*}"
  echo "[SUBMIT] write job_id=${write_job_id}"

  # wait write running to get its nodes, then submit read with dependency+exclude
  write_nodes="$(wait_job_running_and_get_nodelist "${write_job_id}")"

  read_job_id="$(sbatch --parsable \
    --dependency=afterok:${write_job_id} \
    --exclude="${write_nodes}" \
    "${read_slurm}")"
  read_job_id="${read_job_id%%;*}"
  echo "[SUBMIT] read  job_id=${read_job_id} (afterok:${write_job_id}, exclude=${write_nodes})"

  prev_read_job_id="${read_job_id}"
done

echo
echo "All jobs submitted in sequence (blocking orchestration)."
echo "summary csv: ${CSV_PATH}"
echo "logs dir:    ${LOGS_DIR}"
echo "datasets:    ${DATASETS_DIR}"
