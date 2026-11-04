#!/usr/bin/env python3
"""Sum L2 cache counters for a target kernel from a rocprofv3 counter CSV.

Usage:
    parse_pmc.py <counter_collection.csv> <kernel_regex> <hit_counter> <miss_counter>

Prints "<hit_sum> <miss_sum> <hit_rate_pct>" where hit_rate_pct is "NA" when
there is no counter activity (e.g. the GL2C counters read zero on gfx1100).
"""
import csv
import re
import sys


def main():
    csv_path, kernel_regex, hit_counter, miss_counter = sys.argv[1:5]
    pattern = re.compile(kernel_regex)

    hit = 0.0
    miss = 0.0
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if not pattern.search(row.get("Kernel_Name", "")):
                continue
            name = row.get("Counter_Name", "")
            try:
                value = float(row.get("Counter_Value", "0"))
            except (ValueError, TypeError):
                continue
            if name == hit_counter:
                hit += value
            elif name == miss_counter:
                miss += value

    total = hit + miss
    rate = f"{100.0 * hit / total:.2f}" if total > 0 else "NA"
    print(f"{hit:.0f} {miss:.0f} {rate}")


if __name__ == "__main__":
    main()
