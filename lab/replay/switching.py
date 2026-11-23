#!/usr/bin/env python3
"""Summarise the CSV `regime_switching` writes.

For each order of the two regimes, one row per controller: mean objective
(p95 completion + p95 TTFT over the whole trace) in seconds, the paired
per-seed difference against `fixed-1024/100` with a 95% bootstrap
interval, the per-phase objectives, and the optimizer's calls, switches
and charged stall.

    python3 lab/replay/switching.py out.csv
"""

import csv
import random
import sys
from collections import defaultdict
from statistics import mean

BASELINE = "fixed-1024/100"


def bootstrap(diffs, draws=4000, seed=0):
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(mean(rng.choice(diffs) for _ in range(n)) for _ in range(draws))
    return means[int(0.025 * draws)], means[int(0.975 * draws) - 1]


def main():
    rows = list(csv.DictReader(open(sys.argv[1])))
    incomplete = [r for r in rows if r["completed"] != rows[0]["completed"]]
    if incomplete:
        print(f"warning: {len(incomplete)} runs left work unfinished\n")
    runs = defaultdict(dict)
    controllers, orders = [], []
    for row in rows:
        runs[(row["order"], row["controller"])][int(row["seed"])] = row
        if row["controller"] not in controllers:
            controllers.append(row["controller"])
        if row["order"] not in orders:
            orders.append(row["order"])
    for order in orders:
        print(f"### {order}\n")
        print("| controller | objective (s) | Δ vs fixed-1024/100 (s) [95% CI] | phase 1 | phase 2 | calls | switches | stall (s) |")
        print("|---|---:|---|---:|---:|---:|---:|---:|")
        base = runs[(order, BASELINE)]
        for controller in controllers:
            run = runs[(order, controller)]
            seeds = sorted(run)
            diffs = [(int(run[s]["objective_us"]) - int(base[s]["objective_us"])) / 1e6 for s in seeds]
            lo, hi = bootstrap(diffs)
            avg = lambda key: mean(int(run[s][key]) for s in seeds)
            print(
                f"| {controller} | {avg('objective_us') / 1e6:.1f} | "
                f"{mean(diffs):+.2f} [{lo:+.2f}, {hi:+.2f}] | "
                f"{avg('phase1_objective_us') / 1e6:.1f} | {avg('phase2_objective_us') / 1e6:.1f} | "
                f"{avg('calls'):.0f} | {avg('switches'):.1f} | {avg('stall_us') / 1e6:.2f} |"
            )
        print()


if __name__ == "__main__":
    main()
