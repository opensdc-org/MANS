#!/usr/bin/env python3
"""Pivot the raw tpds_sweep_results.csv into a wide, readable summary CSV.

One row per (dataset, mode); threshold columns for CR, comp/decomp MB/s and
ADM block ratio. The "tie" threshold is type-dependent (1009 for u2, 3025 for u4)
and is recorded in its own column.
"""
import csv
import os

RAW = "/workspace/MANS/tpds_sweep_results.csv"
OUT = "/workspace/MANS/tpds_sweep_summary.csv"

TIE = {"u2": 1009, "u4": 3025}


def main():
    with open(RAW, newline="") as f:
        rows = list(csv.DictReader(f))

    groups = {}  # (dataset, mode) -> {threshold: row}
    meta = {}    # (dataset, mode) -> meta dict
    for r in rows:
        key = (r["dataset"], r["mode"])
        meta[key] = {
            "dtype": r["dtype"],
            "file": r["file"],
            "num_elements": r["num_elements"],
            "raw_bytes": r["raw_bytes"],
        }
        groups.setdefault(key, {})[int(r["threshold"])] = r

    out_cols = [
        "dataset", "dtype", "mode", "file", "num_elements", "raw_bytes",
        "tie_threshold",
        "cr_t0", "cr_tie", "cr_t3500", "cr_t4033",
        "comp_mbps_t0", "comp_mbps_tie", "comp_mbps_t3500", "comp_mbps_t4033",
        "decomp_mbps_t0", "decomp_mbps_tie", "decomp_mbps_t3500", "decomp_mbps_t4033",
        "adm_ratio_t0", "adm_ratio_tie", "adm_ratio_t3500", "adm_ratio_t4033",
    ]

    def val(row, thr, field):
        r = row.get(thr)
        return r[field] if r else ""

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_cols, extrasaction="ignore")
        w.writeheader()
        for (ds, mode) in sorted(groups.keys(), key=lambda k: (k[0], k[1])):
            m = meta[(ds, mode)]
            tie = TIE[m["dtype"]]
            row = groups[(ds, mode)]
            w.writerow({
                "dataset": ds,
                "dtype": m["dtype"],
                "mode": mode,
                "file": m["file"],
                "num_elements": m["num_elements"],
                "raw_bytes": m["raw_bytes"],
                "tie_threshold": tie,
                "cr_t0": val(row, 0, "ratio"),
                "cr_tie": val(row, tie, "ratio"),
                "cr_t3500": val(row, 3500, "ratio"),
                "cr_t4033": val(row, 4033, "ratio"),
                "comp_mbps_t0": val(row, 0, "comp_mbps"),
                "comp_mbps_tie": val(row, tie, "comp_mbps"),
                "comp_mbps_t3500": val(row, 3500, "comp_mbps"),
                "comp_mbps_t4033": val(row, 4033, "comp_mbps"),
                "decomp_mbps_t0": val(row, 0, "decomp_mbps"),
                "decomp_mbps_tie": val(row, tie, "decomp_mbps"),
                "decomp_mbps_t3500": val(row, 3500, "decomp_mbps"),
                "decomp_mbps_t4033": val(row, 4033, "decomp_mbps"),
                "adm_ratio_t0": val(row, 0, "adm_ratio"),
                "adm_ratio_tie": val(row, tie, "adm_ratio"),
                "adm_ratio_t3500": val(row, 3500, "adm_ratio"),
                "adm_ratio_t4033": val(row, 4033, "adm_ratio"),
            })

    print(f"wrote {OUT} ({len(groups)} rows)")


if __name__ == "__main__":
    main()
