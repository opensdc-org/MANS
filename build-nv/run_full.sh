#!/bin/bash
set -e
export PATH=/home/bingxing2/apps/cuda/11.7.0/bin:$PATH
export LD_LIBRARY_PATH=/home/bingxing2/apps/cuda/11.7.0/lib64:$LD_LIBRARY_PATH
cd /home/bingxing2/home/scx6546/hwj/MANS

MAP=$(find \
  /home/bingxing2/home/scx6546/hwj/testdata/u2/haac \
  /home/bingxing2/home/scx6546/hwj/testdata/u2/exafel \
  /home/bingxing2/home/scx6546/hwj/testdata/u2/modis \
  /home/bingxing2/home/scx6546/hwj/testdata/u2/tomobank \
  /home/bingxing2/home/scx6546/hwj/testdata/u4/big-tpch \
  /home/bingxing2/home/scx6546/hwj/testdata/u4/suitesparse \
  -type f \( -name '*.u2' -o -name '*.u4' \) | sort | tr '\n' ' ')

echo "files: $(echo $MAP | wc -w)"

echo "=== RUN version A (HIP port, threshold 1009/3025) ==="
./build-nv/nv_mans_bench      mans_bench_1d_results_interleave_nv_v2.csv $MAP --warmup 2 --iters 3

echo "=== RUN version B (sc-ae native + raw fallback + fix) ==="
./build-nv/nv_mans_bench_scae mans_bench_1d_results_interleave_scae.csv  $MAP --warmup 2 --iters 3

echo "ALL_DONE"
