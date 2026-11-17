#!/usr/bin/env python3
"""Sweep the ADM-vs-raw threshold on the CPU backend for p-mode and r-mode.

For every dataset (immediate child of testdata/u2 and testdata/u4), pick the
largest file, then run cpu_mans_bench across {p,r} x {threshold list} and
report CR, comp/decomp throughput, and the ADM block ratio.
"""
import csv
import os
import re
import subprocess
import sys

BIN = "/workspace/MANS/build-amd-633/bin/cpu/cpu_mans_bench"
DATA = "/workspace/tpds/testdata"
OUT = "/workspace/MANS/tpds_sweep_results.csv"

THRESHOLDS = {
    "u2": [0, 1009, 3500, 4033],
    "u4": [0, 3025, 3500, 4033],
}
MODES = ["p", "r"]

WHOLE_RE = re.compile(r"^\s*whole\s*\|")

MAX_U32 = (1 << 32) - 1


def adm_max_compressed_size(num_elements, elem_size):
    """Mirror adm_max_compressed_size<T>() so we can skip files that exceed the
    32-bit compressed-size bound before even running the benchmark."""
    block = 32 * 16  # adm::cmp_tblock_size * adm::cmp_chunk
    gsize = (num_elements + block - 1) // block
    len1 = (gsize + 1) * 4        # sizeof(int)
    len2 = gsize * elem_size      # sizeof(T)
    len3 = (gsize + 7) // 8       # flags bitset
    len4 = num_elements           # codes (uint8_t)
    max_lane_elements = (block + 32 - 1) // 32
    max_len5 = gsize * 32 * max_lane_elements * 4  # 4 = max_bytes_signal_per_ele_{16,32}b
    return len1 + len2 + len3 + len4 + max_len5


def fits_32bit(num_elements, elem_size):
    if num_elements > MAX_U32 // elem_size:
        return False
    if num_elements * elem_size > MAX_U32:
        return False
    if adm_max_compressed_size(num_elements, elem_size) > MAX_U32:
        return False
    return True


def run_bench(dtype_flag, path, num_elements, mode, threshold):
    cmd = [
        BIN, dtype_flag, path,
        "--mode", mode,
        "--dims", "1", str(num_elements),
        "--threshold", str(threshold),
        "--warmup", "2", "--runs", "3",
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    if p.returncode != 0:
        tail = p.stderr.strip().splitlines()[-1] if p.stderr.strip() else "nonzero exit"
        return {"error": tail[:120]}

    for line in p.stdout.splitlines():
        if WHOLE_RE.match(line):
            parts = [x.strip() for x in line.split("|")]
            if len(parts) < 5:
                return {"error": "bad whole line: " + line.strip()}
            m = re.search(r"\((\d+)/(\d+)\)", parts[4])
            adm_blocks = int(m.group(1)) if m else -1
            total_blocks = int(m.group(2)) if m else -1
            return {
                "ratio": float(parts[1]),
                "comp_mbps": float(parts[2]),
                "decomp_mbps": float(parts[3]),
                "adm_ratio": float(parts[4].split()[0]),
                "adm_blocks": adm_blocks,
                "total_blocks": total_blocks,
            }
    return {"error": "no whole line"}


def main():
    if not os.path.isdir(DATA):
        print(f"data dir not found: {DATA}", file=sys.stderr)
        return 2

    rows = []
    for dtype_dir in ["u2", "u4"]:
        ddir = os.path.join(DATA, dtype_dir)
        if not os.path.isdir(ddir):
            continue
        elem_size = 2 if dtype_dir == "u2" else 4
        dtype_flag = "-u2" if dtype_dir == "u2" else "-u4"
        for ds in sorted(os.listdir(ddir)):
            dsdir = os.path.join(ddir, ds)
            if not os.path.isdir(dsdir):
                continue
            # The dtype is encoded by the parent dir (u2/u4), not the file
            # extension: some u2 datasets use ".uint16". Pick the largest regular
            # file (regardless of extension) that still fits the 32-bit
            # compressed-size bound; full unsliced files can exceed it.
            files = sorted(
                [f for f in os.listdir(dsdir)
                 if os.path.isfile(os.path.join(dsdir, f))],
                key=lambda f: os.path.getsize(os.path.join(dsdir, f)),
                reverse=True,
            )
            candidates = [
                f for f in files
                if fits_32bit(os.path.getsize(os.path.join(dsdir, f)) // elem_size,
                              elem_size)
            ]
            if not candidates:
                continue
            largest = candidates[0]
            path = os.path.join(dsdir, largest)
            size = os.path.getsize(path)
            num_elements = size // elem_size

            for mode in MODES:
                for thr in THRESHOLDS[dtype_dir]:
                    res = run_bench(dtype_flag, path, num_elements, mode, thr)
                    row = {
                        "dataset": ds,
                        "dtype": dtype_dir,
                        "file": largest,
                        "num_elements": num_elements,
                        "raw_bytes": size,
                        "mode": mode,
                        "threshold": thr,
                    }
                    if "error" in res:
                        row.update(res)
                    else:
                        row.update(res)
                    rows.append(row)
                    print(f"[done] {ds}/{largest} mode={mode} thr={thr} -> {res}",
                          flush=True)

    cols = ["dataset", "dtype", "file", "num_elements", "raw_bytes",
            "mode", "threshold", "ratio", "comp_mbps", "decomp_mbps",
            "adm_blocks", "total_blocks", "adm_ratio", "error"]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    sys.exit(main())
