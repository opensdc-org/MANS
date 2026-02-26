#!/usr/bin/env bash
set -euo pipefail

###### ***** Editable Constants (NO ARGS) ***** ######
# SLURM_NODES_LIST=(20)
# SLURM_NODES_LIST=(1 2 4 6 8 10 12 14 16 18 20)
SLURM_NODES_LIST=(1 4 8 12 16 20 24 28 32 36 40)
DATA_GB_PER_NODE=32
NPROC_PER_NODE=64
CPU_LIMIT=1
CHUNK_SIZES_MB=(1.0)
ENABLE_READ_EXCLUDE=0

BEST_THREAD_DIR="/public/share/acnnprvuzd/MANS/slurms/runs_autotune"
DATA_GEN_BIN="/public/share/acnnprvuzd/MANS/build/bin/h5z-mans/mans_data_gen"
READ_BIN_DIR="/public/share/acnnprvuzd/MANS/build/bin/h5z-mans"
# LIST=(sz3)
LIST=(mans mans_original none sz3 fse)

HDF5_PLUGIN_PATH_VALUE="/public/share/acnnprvuzd/MANS/build/bin/plugins"
HDF5_PLUGIN_PRELOAD_VALUE="libH5Z-MANS.so"

SLURM_PARTITION="sdicnormal"
SLURM_TIME="06:00:00"
JOB_NAME_PREFIX="mans-chunk-sweep"

DTYPE="u16"
ADM_THRESHOLD=4000
DATA_GEN_PROCS_PER_NODE=32
###### ***** End Constants ***** ######

if (( CPU_LIMIT != 1 )); then
  echo "[ERROR] this script expects CPU_LIMIT=1 (rank=cores per node)." >&2
  exit 1
fi
if (( NPROC_PER_NODE % CPU_LIMIT != 0 )); then
  echo "[ERROR] CPU_LIMIT(${CPU_LIMIT}) does not divide NPROC_PER_NODE(${NPROC_PER_NODE})" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -x "${DATA_GEN_BIN}" ]]; then
  echo "[ERROR] binary not found or not executable: ${DATA_GEN_BIN}" >&2
  exit 1
fi

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
      printf '%s\n' "${nodelist}"
      return 0
    fi

    sleep 2
  done
}

echo "###### ***** Generate Slurm Files (single write/read templates per prefix) ***** ######"

prev_read_job_id=""
ntasks_per_node=$((NPROC_PER_NODE / CPU_LIMIT))

for bin_prefix in "${LIST[@]}"; do
  write_bin="${READ_BIN_DIR}/${bin_prefix}_write"
  read_bin="${READ_BIN_DIR}/${bin_prefix}_read"

  RUNS_DIR="${SCRIPT_DIR}/runs_sweep_${bin_prefix}"
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

module purge
module load compiler/cmake/3.20.4
module load hdf5-1.14.3-intelmpi2021_p
module unload compiler/devtoolset/7.3.1
module load compiler/gcc/9.3.0

RUNS_DIR="${RUNS_DIR}"
LOGS_DIR="${LOGS_DIR}"
DATASETS_DIR="${DATASETS_DIR}"
CSV_PATH="${CSV_PATH}"
BEST_THREAD_DIR="${BEST_THREAD_DIR}"
DATA_GEN_BIN="${DATA_GEN_BIN}"
WRITE_BIN="${write_bin}"
DTYPE="${DTYPE}"
ADM_THRESHOLD=${ADM_THRESHOLD}
DATA_GEN_PROCS_PER_NODE=${DATA_GEN_PROCS_PER_NODE}
BIN_PREFIX="${bin_prefix}"

CHUNK_MB="\${SWEEP_CHUNK_MB:?SWEEP_CHUNK_MB is required}"
SLURM_NODES_COUNT="\${SWEEP_SLURM_NODES:?SWEEP_SLURM_NODES is required}"
RANK="\${SWEEP_RANK:?SWEEP_RANK is required}"
NTASKS_PER_NODE="\${SWEEP_NTASKS_PER_NODE:?SWEEP_NTASKS_PER_NODE is required}"
CPU_LIMIT="\${SWEEP_CPU_LIMIT:?SWEEP_CPU_LIMIT is required}"
PER_RANK_MB="\${SWEEP_PER_RANK_MB:?SWEEP_PER_RANK_MB is required}"

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

generated_count=\$(find "\${DATASETS_DIR}" -maxdepth 1 -type f -name 'rank*.bin' | wc -l | tr -d ' ')
if (( generated_count != RANK )); then
  echo "[INFO] regenerate datasets for rank=\${RANK} (existing=\${generated_count})"
  rm -f "\${DATASETS_DIR}"/rank*.bin
  rm -f "\${DATASETS_DIR}"/rank*.h5

  gen_procs_per_node=\${DATA_GEN_PROCS_PER_NODE}
  if (( gen_procs_per_node < 1 )); then gen_procs_per_node=1; fi
  if (( gen_procs_per_node > NTASKS_PER_NODE )); then gen_procs_per_node=\${NTASKS_PER_NODE}; fi

  gen_ntasks=\$((SLURM_NNODES * gen_procs_per_node))
  if (( gen_ntasks > RANK )); then gen_ntasks=\${RANK}; fi

  gen_cpus_per_task=\${CPU_LIMIT}
  if (( gen_cpus_per_task < 1 )); then gen_cpus_per_task=1; fi

  export DATA_GEN_BIN DATASETS_DIR RANK PER_RANK_MB DTYPE ADM_THRESHOLD
  srun --nodes "\${SLURM_NNODES}" \\
       --ntasks "\${gen_ntasks}" \\
       --ntasks-per-node "\${gen_procs_per_node}" \\
       --cpus-per-task "\${gen_cpus_per_task}" \\
       /bin/bash -c '
         set -euo pipefail
         for ((r = SLURM_PROCID; r < RANK; r += SLURM_NTASKS)); do
           "\${DATA_GEN_BIN}" \
             --output-name "\${DATASETS_DIR}/rank\${r}.bin" \
             --size-per-rank-mb "\${PER_RANK_MB}" \
             --dtype "\${DTYPE}" \
             --adm-threshold "\${ADM_THRESHOLD}" \
             --ratio-constant 0.75
         done
       '

  generated_count=\$(find "\${DATASETS_DIR}" -maxdepth 1 -type f -name 'rank*.bin' | wc -l | tr -d ' ')
  if (( generated_count != RANK )); then
    echo "[ERROR] dataset generation incomplete: expected \${RANK}, got \${generated_count}" >&2
    exit 1
  fi
fi

chunk_tag="\${CHUNK_MB//./p}"
export MANS_TIMING_ITER=1
export MANS_TIMING_DUMP_PATH="\${RUNS_DIR}/plugin_timing_\${BIN_PREFIX}_r\${RANK}_cpu\${CPU_LIMIT}_c\${chunk_tag}_write.csv"

set +e
write_output="\$(mpirun -np "\${RANK}" --bind-to core --map-by ppr:\${NTASKS_PER_NODE}:node:pe=\${CPU_LIMIT} "\${WRITE_BIN}" --chunk-size-mb "\${CHUNK_MB}" 2>&1)"
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
SLURM_WRITE_EOF

  cat > "${read_slurm}" <<SLURM_READ_EOF
#!/usr/bin/env bash
#SBATCH --time=${SLURM_TIME}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --exclusive

set -euo pipefail

module purge
module load compiler/cmake/3.20.4
module load hdf5-1.14.3-intelmpi2021_p
module unload compiler/devtoolset/7.3.1
module load compiler/gcc/9.3.0

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

chunk_tag="\${CHUNK_MB//./p}"
export MANS_TIMING_ITER=1
export MANS_TIMING_DUMP_PATH="\${RUNS_DIR}/plugin_timing_\${BIN_PREFIX}_r\${RANK}_cpu\${CPU_LIMIT}_c\${chunk_tag}_read.csv"

set +e
read_output="\$(mpirun -np "\${RANK}" --bind-to core --map-by ppr:\${NTASKS_PER_NODE}:node:pe=\${CPU_LIMIT} "\${READ_BIN}" 2>&1)"
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

# hard constraint: keep only one active rank*.h5 set
rm -f "\${DATASETS_DIR}"/rank*.h5
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
    TOTAL_DATA_GB=$((DATA_GB_PER_NODE * slurm_nodes))
    rank=${NPROC_TOTAL}
    per_rank_mb=$(( TOTAL_DATA_GB * 1024 / rank ))

    echo
    echo "====== prefix=${bin_prefix} nodes=${slurm_nodes} rank=${rank} cpu=${CPU_LIMIT} ======"

    for chunk_mb in "${CHUNK_SIZES_MB[@]}"; do
      chunk_tag="${chunk_mb//./p}"
      common_export="ALL,SWEEP_CHUNK_MB=${chunk_mb},SWEEP_SLURM_NODES=${slurm_nodes},SWEEP_RANK=${rank},SWEEP_NTASKS_PER_NODE=${ntasks_per_node},SWEEP_CPU_LIMIT=${CPU_LIMIT},SWEEP_PER_RANK_MB=${per_rank_mb}"

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

  # cleanup this prefix's datasets after its full chain finishes
  if [[ -n "${prev_read_job_id}" ]]; then
    cleanup_job_id="$(sbatch --parsable \
      --dependency=afterok:${prev_read_job_id} \
      --job-name="${JOB_NAME_PREFIX}-${bin_prefix}-cleanup" \
      --nodes=1 \
      --ntasks=1 \
      --cpus-per-task=1 \
      --partition="${SLURM_PARTITION}" \
      --output="${LOGS_DIR}/${bin_prefix}-cleanup.out" \
      --error="${LOGS_DIR}/${bin_prefix}-cleanup.err" \
      --wrap="rm -f \"${DATASETS_DIR}\"/rank*.bin \"${DATASETS_DIR}\"/rank*.h5")"
    cleanup_job_id="${cleanup_job_id%%;*}"
    echo "[SUBMIT] cleanup job_id=${cleanup_job_id} prefix=${bin_prefix} datasets=${DATASETS_DIR}"
    prev_read_job_id="${cleanup_job_id}"
  fi

done

echo
echo "All jobs submitted in strict sequence: prefix outer, then node, then chunk(write->read->cleanup)."
echo "run dirs:"
for bin_prefix in "${LIST[@]}"; do
  echo "  ${SCRIPT_DIR}/runs_sweep_${bin_prefix}"
done
