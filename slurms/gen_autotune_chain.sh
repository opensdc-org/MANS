#!/usr/bin/env bash
set -euo pipefail

###### ***** Editable Constants ***** ######
CPU_LIMITS=(1 2 3 4 6 8 16 32)
THREADS_MIN=1
STRIDE=2
CHUNK_MB_MIN=8
CHUNK_MB_MAX=8

BIN_PATH="/public3/home/t6s010699/MANS-hdf5-filter/build/bin/cpu/cpu_mans_autotune"

SLURM_PARTITION="v6_384"
SLURM_TIME="02:40:00"
SLURM_NODES=1
SLURM_NTASKS=1
JOB_NAME_PREFIX="auto-tune-mans"
###### ***** End Constants ***** ######

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS_DIR="${SCRIPT_DIR}/runs_autotune"
LOGS_DIR="${RUNS_DIR}/logs"

mkdir -p "${RUNS_DIR}" "${LOGS_DIR}"

if [[ ! -x "${BIN_PATH}" ]]; then
  echo "[ERROR] binary not found or not executable: ${BIN_PATH}"
  exit 1
fi

echo "###### ***** Generate Slurm Files ***** ######"
for cpu_limit in "${CPU_LIMITS[@]}"; do
  slurm_file="${RUNS_DIR}/autotune_cpu${cpu_limit}.slurm"
  cat > "${slurm_file}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${JOB_NAME_PREFIX}-cpu${cpu_limit}
#SBATCH --nodes=${SLURM_NODES}
#SBATCH --ntasks=${SLURM_NTASKS}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${cpu_limit}
#SBATCH --time=${SLURM_TIME}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --output=${LOGS_DIR}/auto-tune-cpu${cpu_limit}.out
#SBATCH --error=${LOGS_DIR}/auto-tune-cpu${cpu_limit}.err

set -euo pipefail

module purge
module load cmake/3.21.2
module load hdf5/1.10.6-openmpi-3.1.6-gcc-9.3.0

export OMP_NUM_THREADS=${cpu_limit}
export OMP_PLACES=cores
export OMP_PROC_BIND=close

RUNS_DIR="${RUNS_DIR}"
BIN_PATH="${BIN_PATH}"

"\${BIN_PATH}" \\
  --stride ${STRIDE} \\
  --threads-min ${THREADS_MIN} \\
  --threads-max ${cpu_limit} \\
  --chunk-mb-min ${CHUNK_MB_MIN} \\
  --chunk-mb-max ${CHUNK_MB_MAX} \\
  --csv "\${RUNS_DIR}/thread_sweep_cpu${cpu_limit}.csv" \\
  --out "\${RUNS_DIR}/best_thread.csv"

awk -v cpu_limit="${cpu_limit}" 'BEGIN{FS=OFS=","} NR==1{\$0=\$0",cpu_limit"} NR>1{\$0=\$0","cpu_limit} {print}' \\
  "\${RUNS_DIR}/best_thread.csv" > "\${RUNS_DIR}/best_thread_cpu${cpu_limit}.csv"

rm -f "\${RUNS_DIR}/best_thread.csv"
EOF
  chmod +x "${slurm_file}"
  echo "generated: ${slurm_file}"
done

echo
echo "###### ***** Submit Chain (afterok) ***** ######"
prev_job_id=""
for cpu_limit in "${CPU_LIMITS[@]}"; do
  slurm_file="${RUNS_DIR}/autotune_cpu${cpu_limit}.slurm"
  if [[ -z "${prev_job_id}" ]]; then
    job_raw="$(sbatch --parsable "${slurm_file}")"
  else
    job_raw="$(sbatch --parsable --dependency=afterok:${prev_job_id} "${slurm_file}")"
  fi
  job_id="${job_raw%%;*}"
  echo "submitted cpu=${cpu_limit} job_id=${job_id}"
  prev_job_id="${job_id}"
done

echo
echo "All jobs submitted in sequence."
echo "runs dir: ${RUNS_DIR}"
echo "logs dir: ${LOGS_DIR}"
