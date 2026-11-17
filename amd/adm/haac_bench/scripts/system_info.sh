#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bench_dir=$(cd -- "$script_dir/.." && pwd)
backend=${BACKEND:-rocm}
out_dir=${1:-"$bench_dir/results/system_$(date +%Y%m%d_%H%M%S)"}
mkdir -p "$out_dir"

run_capture() {
    local output=$1
    shift
    if command -v "$1" >/dev/null 2>&1; then
        "$@" >"$out_dir/$output" 2>&1 || printf 'command_exit_status=%s\n' "$?" >>"$out_dir/$output"
    else
        printf 'unavailable: %s\n' "$*" >"$out_dir/$output"
    fi
}

{
    printf 'timestamp_utc=%s\n' "$(date -u +%FT%TZ)"
    printf 'hostname=%s\n' "$(hostname)"
    printf 'backend=%s\n' "$backend"
    printf 'uname=%s\n' "$(uname -a)"
    printf 'kernel=%s\n' "$(uname -r)"
    printf 'os_release='; . /etc/os-release 2>/dev/null && printf '%s %s\n' "${ID:-unknown}" "${VERSION_ID:-unknown}"
    printf 'git_revision='; git -C "$bench_dir" rev-parse HEAD 2>/dev/null || printf 'unknown\n'
    printf 'git_worktree_dirty='; if [[ -z "$(git -C "$bench_dir" status --porcelain --untracked-files=all)" ]]; then printf 'no\n'; else printf 'yes\n'; fi
    printf 'gpu_id=%s\n' "${GPU_ID:-0}"
    printf 'gpu_arch=%s\n' "${GPU_ARCH:-${CUDA_ARCH:-unknown}}"
} >"$out_dir/metadata.txt"

run_capture lscpu.txt lscpu
run_capture lspci_gpu.txt bash -c 'lspci -nn | grep -Ei "VGA|3D|Display"'
run_capture nvidia_smi.txt nvidia-smi -q
run_capture rocm_smi.txt rocm-smi --showproductname --showserial --showvbios --showmeminfo vram --showclocks --showuse --showpower
run_capture rocminfo.txt rocminfo
run_capture nvidia_driver.txt bash -c 'cat /proc/driver/nvidia/version'
run_capture hipcc_version.txt "${HIPCC:-hipcc}" --version
run_capture nvcc_version.txt "${NVCC:-nvcc}" --version
run_capture hipify_version.txt "${HIPIFY_PERL:-hipify-perl}" --help
run_capture rocprofv3_version.txt "${ROCPROFV3:-rocprofv3}" --version
run_capture ncu_version.txt "${NCU:-ncu}" --version

{
    env | LC_ALL=C sort | grep -E '^(BACKEND|GPU_|CUDA_|HIP_|ROCR_|ROCM_|HSA_|OMP_|LD_LIBRARY_PATH|PATH)=' || true
} >"$out_dir/relevant_environment.txt"

printf 'System snapshot: %s\n' "$out_dir"
