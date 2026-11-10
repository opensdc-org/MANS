#!/bin/bash
# One-shot ncu profiling of the 4 ADM kernels on the A100, mirroring the AMD
# rocprof counters (TCC_HIT/MISS -> lts sector hit rate; VGPR -> registers).
set -euo pipefail
cd /home/bingxing2/home/scx6546/hwj/MANS/nv/adm/haac_bench
DATA=/home/bingxing2/home/scx6546/hwj/testdata/u2/haac_vx
NCU=/home/bingxing2/apps/cuda/11.7.0/bin/ncu
SIZE=${1:-vx_1073726487_65536kB.u2}
INPUT=$DATA/$SIZE
OUT=${OUTDIR:-/tmp/ncu_prof_$$}
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=0

METRICS="lts__t_sector_hit_rate.pct,launch__registers_per_thread,dram__bytes_read.sum,dram__bytes_write.sum,sm__warps_active.avg.pct_of_peak_sustained_active"

echo "### compress_fused -> map_values_kernel_decoupled"
./build/compress_fused "$INPUT" "$OUT/f.adm" >/dev/null 2>&1
$NCU --csv --metrics "$METRICS" --kernel-name "regex:map_values_kernel_decoupled" \
    ./build/compress_fused "$INPUT" "$OUT/f.adm" 2>/dev/null

echo "### compress_unfused -> map_values_kernel_thrust"
./build/compress_unfused "$INPUT" "$OUT/u.adm" >/dev/null 2>&1
$NCU --csv --metrics "$METRICS" --kernel-name "regex:map_values_kernel_thrust" \
    ./build/compress_unfused "$INPUT" "$OUT/u.adm" 2>/dev/null

echo "### decompress_fused -> decompress"
$NCU --csv --metrics "$METRICS" --kernel-name "regex:^decompress" \
    ./build/decompress_fused "$OUT/f.adm" "$INPUT" 2>/dev/null

echo "### decompress_unfused -> decode_signals_uint16|restore_values_uint16"
$NCU --csv --metrics "$METRICS" --kernel-name "regex:decode_signals_uint16|restore_values_uint16" \
    ./build/decompress_unfused "$OUT/u.adm" "$INPUT" 2>/dev/null
