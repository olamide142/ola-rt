#!/usr/bin/env python3
"""Summarizes `ola serve --adaptive-log` files (E4).

For each log: optimizer calls, calls with enough history to replay,
switches and the budgets switched between, replays run, and the host
stall per call (P50, P95, max, and total), split into calls that replayed
and calls that returned at once.

    adaptive_log_summary.py LOG [LOG ...]

Each LOG is labelled by its parent directory. Standard library only.
"""

import json
import os
import sys


def percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    rank = q * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def vector(theta):
    return ",".join(f"{name}={value}" for name, value in sorted(theta.items()))


def main(paths):
    print(
        "| run | calls | replayed | switches | replays | stall p50 / p95 / max (ms), replayed "
        "| stall total (ms) | errors | switched to |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for path in paths:
        with open(path) as file:
            records = [json.loads(line) for line in file if line.strip()]
        replayed = [r for r in records if r["replayed"]]
        stalls = [r["host_micros"] / 1000 for r in replayed]
        switches = [r for r in records if r["switched"]]
        targets = "; ".join(vector(r["to"]) for r in switches) or "none"
        label = os.path.basename(os.path.dirname(os.path.abspath(path)))
        print(
            f"| {label} | {len(records)} | {len(replayed)} | {len(switches)} "
            f"| {sum(r['replays'] for r in records)} "
            f"| {percentile(stalls, 0.5):.1f} / {percentile(stalls, 0.95):.1f} / "
            f"{max(stalls, default=float('nan')):.1f} "
            f"| {sum(r['host_micros'] for r in records) / 1000:.1f} "
            f"| {sum(1 for r in records if r.get('error'))} | {targets} |"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
