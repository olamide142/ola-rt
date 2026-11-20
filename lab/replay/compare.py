#!/usr/bin/env python3
"""Compares real `ola serve` latencies with the simulator's predictions.

Each run is a pair of files: `observed.jsonl` from `client.py` and
`simulated.jsonl` from `replay_trace simulate`, joined on `index`. Real
times come from the server's clock: TTFT is `first_token_us - queued_us`
and completion is `completed_us - queued_us`. Simulated times are measured
from the same arrival.

For each run and metric it reports real and simulated P50 and P95, the
relative error `(sim - real) / real` with a 95% paired bootstrap interval
over requests, and the per-request absolute relative error (median and
P90). Writes a CSV and prints a Markdown table, then the mean absolute
error across seeds for each trace, load, and statistic.

    compare.py --out summary.csv RUN_DIR [RUN_DIR ...]

where each RUN_DIR holds `observed.jsonl` and `simulated.jsonl`.
Standard library only.
"""

import argparse
import csv
import json
import os
import random
import sys

RESAMPLES = 2000


def percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    rank = q * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def load(path):
    with open(path) as file:
        return {row["index"]: row for row in map(json.loads, file) if row}


def paired(run_dir):
    observed = load(os.path.join(run_dir, "observed.jsonl"))
    simulated = load(os.path.join(run_dir, "simulated.jsonl"))
    pairs = {"ttft": [], "completion": []}
    skipped = 0
    for index, real in observed.items():
        sim = simulated.get(index)
        if (
            "error" in real
            or sim is None
            or sim["outcome"] != "completed"
            or real.get("first_token_us") is None
            or real.get("completed_us") is None
        ):
            skipped += 1
            continue
        pairs["ttft"].append((real["first_token_us"] - real["queued_us"], sim["sim_ttft_us"]))
        pairs["completion"].append(
            (real["completed_us"] - real["queued_us"], sim["sim_completion_us"])
        )
    first = next(iter(observed.values()))
    label = {key: first.get(key) for key in ("trace", "load", "seed")}
    return label, pairs, skipped


def relative_error(pairs, q):
    real = percentile([r for r, _ in pairs], q)
    sim = percentile([s for _, s in pairs], q)
    return (sim - real) / real if real else float("nan")


def bootstrap(pairs, q, seed):
    generator = random.Random(seed)
    errors = sorted(
        relative_error([generator.choice(pairs) for _ in pairs], q) for _ in range(RESAMPLES)
    )
    return errors[int(0.025 * RESAMPLES)], errors[int(0.975 * RESAMPLES) - 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()

    rows = []
    for run_dir in args.runs:
        label, pairs, skipped = paired(run_dir)
        for metric, values in pairs.items():
            if not values:
                continue
            per_request = [abs(s - r) / r for r, s in values if r > 0]
            for name, q in (("p50", 0.5), ("p95", 0.95)):
                low, high = bootstrap(values, q, seed=len(rows))
                rows.append(
                    {
                        "run": os.path.basename(os.path.normpath(run_dir)),
                        **label,
                        "metric": metric,
                        "statistic": name,
                        "requests": len(values),
                        "skipped": skipped,
                        "real_ms": percentile([r for r, _ in values], q) / 1000,
                        "sim_ms": percentile([s for _, s in values], q) / 1000,
                        "rel_error": relative_error(values, q),
                        "rel_error_low": low,
                        "rel_error_high": high,
                        "per_request_abs_rel_p50": percentile(per_request, 0.5),
                        "per_request_abs_rel_p90": percentile(per_request, 0.9),
                    }
                )

    with open(args.out, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(
        "| run | metric | stat | n | real (ms) | sim (ms) | rel. error [95% CI] "
        "| per-request |err| p50 / p90 |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for row in rows:
        print(
            f"| {row['run']} | {row['metric']} | {row['statistic']} | {row['requests']} "
            f"| {row['real_ms']:.1f} | {row['sim_ms']:.1f} "
            f"| {row['rel_error']:+.1%} [{row['rel_error_low']:+.1%}, {row['rel_error_high']:+.1%}] "
            f"| {row['per_request_abs_rel_p50']:.1%} / {row['per_request_abs_rel_p90']:.1%} |"
        )

    # Mean absolute error across seeds, per trace, load, and statistic.
    groups = {}
    for row in rows:
        key = (row["trace"], row["load"], row["metric"], row["statistic"])
        groups.setdefault(key, []).append((row["seed"], row["rel_error"]))
    print("\n| trace | load | metric | stat | seeds | mean abs. error | per-seed error |")
    print("|---|---|---|---|---|---|---|")
    for (trace, load, metric, statistic), errors in groups.items():
        mean = sum(abs(error) for _, error in errors) / len(errors)
        seeds = ", ".join(f"{seed}: {error:+.1%}" for seed, error in sorted(errors))
        print(f"| {trace} | {load} | {metric} | {statistic} | {len(errors)} | {mean:.1%} | {seeds} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
