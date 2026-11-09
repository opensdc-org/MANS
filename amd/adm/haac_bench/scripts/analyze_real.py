#!/usr/bin/env python3
"""Run AMD ADM stages on real .u2 inputs and collect stage timings."""

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


CSV_FIELDS = [
    "input", "stage", "input_bytes", "output_bytes", "adm_bytes",
    "file_read_ms", "h2d_ms", "mapping_ms", "prefix_sum_ms", "concat_ms",
    "decode_ms", "restore_ms", "fused_kernel_ms", "d2h_ms",
    "metadata_ms", "file_write_ms", "cpu_verify_ms", "total_gpu_ms",
    "total_wall_ms", "kernel_throughput_gbps", "lossless_pass", "log",
]


def number(pattern, text, group=1):
    match = re.search(pattern, text)
    return match.group(group) if match else "NA"


def parse_log(stage, text):
    row = {key: "NA" for key in CSV_FIELDS}
    row["stage"] = stage

    if stage.startswith("compress_"):
        row["file_read_ms"] = number(r"File read cost ([\d.]+) ms", text)
        row["h2d_ms"] = number(r"H2D cost ([\d.]+) ms", text)
        row["d2h_ms"] = number(r"D2H cost ([\d.]+) ms", text)
        row["file_write_ms"] = number(r"file write cost ([\d.]+) ms", text)
        throughput = number(r"Total Cmp throughput: ([\d.]+) GB/s", text)
        if stage == "compress_fused":
            row["fused_kernel_ms"] = number(r"ADM cost ([\d.]+) ms", text)
            row["total_gpu_ms"] = row["fused_kernel_ms"]
        else:
            row["mapping_ms"] = number(r"ADM cost ([\d.]+) ms", text)
            row["prefix_sum_ms"] = number(r"prefix-sum cost ([\d.]+) ms", text)
            row["concat_ms"] = number(r"concat cost ([\d.]+) ms", text)
            vals = [row[k] for k in ("mapping_ms", "prefix_sum_ms", "concat_ms")]
            row["total_gpu_ms"] = (
                f"{sum(float(v) for v in vals):.6f}"
                if all(v != "NA" for v in vals) else "NA"
            )
    else:
        row["file_read_ms"] = number(r"File read cost ([\d.]+) ms", text)
        row["h2d_ms"] = number(r"H2D cost ([\d.]+) ms", text)
        row["d2h_ms"] = number(r"D2H cost ([\d.]+) ms", text)
        row["cpu_verify_ms"] = number(r"CPU verify cost ([\d.]+) ms", text)
        if stage == "decompress_fused":
            row["fused_kernel_ms"] = number(r"Total cost ([\d.]+) ms", text)
            row["total_gpu_ms"] = row["fused_kernel_ms"]
        else:
            row["decode_ms"] = number(r"Decode signals cost ([\d.]+) ms", text)
            row["restore_ms"] = number(r"restore values cost ([\d.]+) ms", text)
            vals = (row["decode_ms"], row["restore_ms"])
            row["total_gpu_ms"] = (
                f"{sum(float(v) for v in vals):.6f}"
                if all(v != "NA" for v in vals) else "NA"
            )
        row["lossless_pass"] = "yes" if "Pass error check!" in text else "no"

    row["total_wall_ms"] = number(r"Total wall cost ([\d.]+) ms", text)
    row["kernel_throughput_gbps"] = number(
        r"(?:Total Cmp|Decmp) throughput: ([\d.]+) GB/s", text
    )
    return row


def main():
    parser = argparse.ArgumentParser(
        description="Profile AMD ADM stages on the same real .u2 inputs."
    )
    parser.add_argument("input_dir", type=Path, help="Directory of .u2 files")
    parser.add_argument("result_dir", type=Path, help="New output directory (must not exist)")
    parser.add_argument("--build-dir", type=Path, help="Binary directory; defaults to ../build")
    parser.add_argument("--gpu", default="0", help="GPU index (default: 0)")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    result_dir = args.result_dir.resolve()
    bench_dir = Path(__file__).resolve().parent.parent
    build_dir = (args.build_dir or bench_dir / "build").resolve()
    if not input_dir.is_dir():
        parser.error(f"input directory does not exist: {input_dir}")
    if result_dir.exists():
        parser.error(f"result directory already exists; choose a new path: {result_dir}")

    inputs = sorted(input_dir.glob("*.u2"))
    if not inputs:
        parser.error(f"no .u2 files found directly under: {input_dir}")
    for path in inputs:
        if path.stat().st_size == 0 or path.stat().st_size % 2:
            parser.error(f"input must be non-empty with an even byte count: {path}")

    stages = (
        "compress_fused", "compress_unfused",
        "decompress_fused", "decompress_unfused",
    )
    missing = [str(build_dir / stage) for stage in stages
               if not os.access(build_dir / stage, os.X_OK)]
    if missing:
        parser.error("missing executable(s): " + ", ".join(missing))

    result_dir.mkdir(parents=True)
    logs_dir = result_dir / "logs"
    logs_dir.mkdir()
    temp_adm_dir = tempfile.TemporaryDirectory(prefix="haac-adm-")
    summary_path = result_dir / "stage_timings.csv"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["HIP_VISIBLE_DEVICES"] = args.gpu
    env["ROCR_VISIBLE_DEVICES"] = args.gpu

    with summary_path.open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        summary_file.flush()
        for input_path in inputs:
            name = input_path.name
            base = input_path.stem
            original_size = input_path.stat().st_size
            adm_outputs = {}
            for stage in stages:
                log_path = logs_dir / f"{base}_{stage}.log"
                comp_kind = stage.removeprefix("compress_")
                decmp_kind = stage.removeprefix("decompress_")
                adm_path = Path(temp_adm_dir.name) / f"{base}_{comp_kind}.adm"
                command = (
                    [str(build_dir / stage), str(input_path), str(adm_path)]
                    if stage.startswith("compress_") else
                    [str(build_dir / stage), str(adm_outputs[decmp_kind]), str(input_path)]
                )
                print(f"[{stage}] {name}", flush=True)
                result = subprocess.run(command, env=env, text=True,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                log_path.write_text(result.stdout, encoding="utf-8")
                if result.returncode:
                    print(f"stage failed ({result.returncode}); see {log_path}", file=sys.stderr)
                    return result.returncode or 1

                row = parse_log(stage, result.stdout)
                row.update(input=name, input_bytes=original_size, log=str(log_path))
                if stage.startswith("compress_"):
                    adm_outputs[comp_kind] = adm_path
                    row["output_bytes"] = adm_path.stat().st_size
                    row["adm_bytes"] = row["output_bytes"]
                else:
                    row["output_bytes"] = original_size
                    row["adm_bytes"] = adm_outputs[decmp_kind].stat().st_size
                writer.writerow(row)
                summary_file.flush()
                if stage.startswith("decompress_") and row["lossless_pass"] != "yes":
                    print(f"lossless verification failed; see {log_path}", file=sys.stderr)
                    return 1

    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
