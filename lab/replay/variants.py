#!/usr/bin/env python3
"""Compares two serving variants on the same replayed traces (E4).

Runs are directories named `<trace>-<load>-<seed>-<variant>` holding the
`observed.jsonl` that `client.py` wrote. For each trace, load, and seed it
pairs the two variants' requests by `index` (the traces are identical) and
reports P50 and P95 TTFT and completion for both, and the ratio
`candidate / baseline` with a 95% paired bootstrap interval over requests.

    variants.py --baseline fixed --candidate adaptive --out e4.csv RUN_DIR...

Standard library only.
"""

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict

RESAMPLES = 2000


def percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    rank = q * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def latencies(run_dir):
    result = {}
    with open(os.path.join(run_dir, "observed.jsonl")) as file:
        for row in map(json.loads, file):
            if not row or "error" in row or row.get("completed_us") is None:
                continue
            result[row["index"]] = {
                "ttft": row["first_token_us"] - row["queued_us"],
                "completion": row["completed_us"] - row["queued_us"],
            }
    return result


def ratio(pairs, q):
    base = percentile([b for b, _ in pairs], q)
    return percentile([c for _, c in pairs], q) / base if base else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()

    groups = defaultdict(dict)
    for run_dir in args.runs:
        name = os.path.basename(os.path.normpath(run_dir))
        key, _, variant = name.rpartition("-")
        groups[key][variant] = latencies(run_dir)

    rows = []
    for key in sorted(groups):
        base = groups[key].get(args.baseline)
        cand = groups[key].get(args.candidate)
        if base is None or cand is None:
            print(f"warning: {key} lacks a variant; skipped", file=sys.stderr)
            continue
        shared = sorted(set(base) & set(cand))
        for metric in ("ttft", "completion"):
            pairs = [(base[i][metric], cand[i][metric]) for i in shared]
            for name, q in (("p50", 0.5), ("p95", 0.95)):
                generator = random.Random(len(rows))
                samples = sorted(
                    ratio([generator.choice(pairs) for _ in pairs], q) for _ in range(RESAMPLES)
                )
                rows.append(
                    {
                        "run": key,
                        "metric": metric,
                        "statistic": name,
                        "requests": len(pairs),
                        "baseline_ms": percentile([b for b, _ in pairs], q) / 1000,
                        "candidate_ms": percentile([c for _, c in pairs], q) / 1000,
                        "ratio": ratio(pairs, q),
                        "ratio_low": samples[int(0.025 * RESAMPLES)],
                        "ratio_high": samples[int(0.975 * RESAMPLES) - 1],
                    }
                )

    with open(args.out, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"| run | metric | stat | n | {args.baseline} (ms) | {args.candidate} (ms) "
        f"| {args.candidate} / {args.baseline} [95% CI] |"
    )
    print("|---|---|---|---|---|---|---|")
    for row in rows:
        print(
            f"| {row['run']} | {row['metric']} | {row['statistic']} | {row['requests']} "
            f"| {row['baseline_ms']:.1f} | {row['candidate_ms']:.1f} "
            f"| {row['ratio']:.3f} [{row['ratio_low']:.3f}, {row['ratio_high']:.3f}] |"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
