#!/usr/bin/env python3
"""Summarizes a soak test: one `ola serve` process, the same trace replayed
back to back (R1 in the second GPU round).

For each repetition it reports makespan, P50/P95 TTFT and completion, and,
from the server's `--trace` output, how iteration time splits into model
calls and host time between calls:

* decode-only iterations: a least-squares fit of call time against rows
  (`intercept + per_row * rows`), so growth in the per-row or the fixed
  part shows separately;
* prefill invocations: mean call time per input position;
* host gap: time from one iteration's last call finishing to the next
  iteration's first call starting, while the server had work.

With `--smi` (an `nvidia-smi --query-gpu=... --format=csv -l 1` log) and
`--reps` (lines of `repetition start_epoch_s end_epoch_s`), it adds mean and
minimum SM clock, maximum temperature, mean power, and memory used at the
end of each repetition.

    soak_summary.py --trace server-trace.jsonl [--smi smi.csv --reps reps.txt] \\
        --out soak.csv REP_DIR [REP_DIR ...]

Each REP_DIR holds the `observed.jsonl` that `client.py` wrote; requests
are matched to repetitions by their server-clock times. Standard library
only.
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime


def percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    rank = q * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def fit_line(points):
    """Least-squares (intercept, slope) of y on x, or None if x is constant."""
    n = len(points)
    if n < 2:
        return None
    mean_x = sum(x for x, _ in points) / n
    mean_y = sum(y for _, y in points) / n
    sxx = sum((x - mean_x) ** 2 for x, _ in points)
    if sxx == 0:
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / sxx
    return mean_y - slope * mean_x, slope


def invocations(path):
    """One entry per (iteration, batch): start, finish, phase, rows, tokens."""
    calls = {}
    with open(path) as file:
        for line in file:
            record = json.loads(line)
            if record.get("event") != "invocation":
                continue
            key = (record["iteration"], record["batch"])
            calls.setdefault(
                key,
                {
                    "iteration": record["iteration"],
                    "start": record["at_us"],
                    "finish": record["finished_us"],
                    "phase": record["phase"],
                    "rows": record["rows"],
                    "input_tokens": record["input_tokens"],
                    "padding_tokens": record["padding_tokens"],
                },
            )
    return sorted(calls.values(), key=lambda call: (call["start"], call["iteration"]))


def iterations(calls):
    grouped = defaultdict(list)
    for call in calls:
        grouped[call["iteration"]].append(call)
    result = []
    for number in sorted(grouped):
        members = grouped[number]
        result.append(
            {
                "start": min(call["start"] for call in members),
                "finish": max(call["finish"] for call in members),
                "calls": members,
            }
        )
    return result


def load_observed(rep_dir):
    with open(os.path.join(rep_dir, "observed.jsonl")) as file:
        return [row for row in map(json.loads, file) if row and "error" not in row]


def load_smi(path):
    rows = []
    with open(path) as file:
        reader = csv.reader(file)
        header = [name.strip() for name in next(reader)]
        for values in reader:
            if len(values) != len(header):
                continue
            row = dict(zip(header, (value.strip() for value in values)))
            try:
                stamp = datetime.strptime(row["timestamp"], "%Y/%m/%d %H:%M:%S.%f").timestamp()
            except ValueError:
                continue
            number = lambda name: float(row[name].split()[0])  # noqa: E731
            rows.append(
                {
                    "t": stamp,
                    "sm_mhz": number("clocks.current.sm [MHz]"),
                    "temp_c": number("temperature.gpu"),
                    "power_w": number("power.draw [W]"),
                    "mem_mib": number("memory.used [MiB]"),
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trace", required=True)
    parser.add_argument("--smi")
    parser.add_argument("--reps")
    parser.add_argument("--out", required=True)
    parser.add_argument("rep_dirs", nargs="+")
    args = parser.parse_args()

    all_iterations = iterations(invocations(args.trace))
    smi = load_smi(args.smi) if args.smi else []
    windows = {}
    if args.reps:
        with open(args.reps) as file:
            for line in file:
                name, start, end = line.split()
                windows[name] = (float(start), float(end))

    rows = []
    for rep_dir in args.rep_dirs:
        name = os.path.basename(os.path.normpath(rep_dir))
        observed = load_observed(rep_dir)
        begin = min(row["queued_us"] for row in observed)
        end = max(row["completed_us"] for row in observed)
        mine = [it for it in all_iterations if begin <= it["start"] <= end]

        decode_points, prefill_rates, gaps = [], [], []
        for previous, current in zip(mine, mine[1:]):
            gaps.append(current["start"] - previous["finish"])
        for it in mine:
            if all(call["phase"] == "decode" for call in it["calls"]):
                rows_in = sum(call["rows"] for call in it["calls"])
                decode_points.append((rows_in, it["finish"] - it["start"]))
            for call in it["calls"]:
                if call["phase"] == "prefill" and call["input_tokens"]:
                    prefill_rates.append(
                        (call["finish"] - call["start"])
                        / (call["input_tokens"] + call["padding_tokens"])
                    )
        line = fit_line(decode_points)
        eight = [y for x, y in decode_points if x == 8]
        ttft = [r["first_token_us"] - r["queued_us"] for r in observed]
        completion = [r["completed_us"] - r["queued_us"] for r in observed]
        row = {
            "repetition": name,
            "requests": len(observed),
            "makespan_s": (end - begin) / 1e6,
            "ttft_p50_ms": percentile(ttft, 0.5) / 1e3,
            "ttft_p95_ms": percentile(ttft, 0.95) / 1e3,
            "completion_p50_ms": percentile(completion, 0.5) / 1e3,
            "completion_p95_ms": percentile(completion, 0.95) / 1e3,
            "iterations": len(mine),
            "decode_iterations": len(decode_points),
            "decode_intercept_ms": line[0] / 1e3 if line else None,
            "decode_per_row_ms": line[1] / 1e3 if line else None,
            "decode_8_rows_ms": sum(eight) / len(eight) / 1e3 if eight else None,
            "prefill_us_per_position": (
                sum(prefill_rates) / len(prefill_rates) if prefill_rates else None
            ),
            "host_gap_mean_ms": sum(gaps) / len(gaps) / 1e3 if gaps else None,
            "host_gap_p95_ms": percentile(gaps, 0.95) / 1e3 if gaps else None,
            "model_share": (
                sum(it["finish"] - it["start"] for it in mine) / (end - begin) if mine else None
            ),
        }
        if name in windows and smi:
            start, stop = windows[name]
            samples = [s for s in smi if start <= s["t"] <= stop]
            if samples:
                row.update(
                    {
                        "sm_mhz_mean": sum(s["sm_mhz"] for s in samples) / len(samples),
                        "sm_mhz_min": min(s["sm_mhz"] for s in samples),
                        "temp_c_max": max(s["temp_c"] for s in samples),
                        "power_w_mean": sum(s["power_w"] for s in samples) / len(samples),
                        "mem_mib_end": samples[-1]["mem_mib"],
                    }
                )
        rows.append(row)

    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(args.out, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    def cell(value, spec):
        return "" if value is None else format(value, spec)

    print(
        "| rep | makespan (s) | TTFT p50/p95 (ms) | completion p50/p95 (ms) "
        "| decode a + b·rows (ms) | 8-row decode (ms) | prefill µs/pos | host gap mean/p95 (ms) "
        "| SM MHz mean/min | max °C | mem end (MiB) |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        print(
            f"| {row['repetition']} | {row['makespan_s']:.2f} "
            f"| {row['ttft_p50_ms']:.0f} / {row['ttft_p95_ms']:.0f} "
            f"| {row['completion_p50_ms']:.0f} / {row['completion_p95_ms']:.0f} "
            f"| {cell(row['decode_intercept_ms'], '.1f')} + {cell(row['decode_per_row_ms'], '.2f')} "
            f"| {cell(row['decode_8_rows_ms'], '.1f')} | {cell(row['prefill_us_per_position'], '.0f')} "
            f"| {cell(row['host_gap_mean_ms'], '.2f')} / {cell(row['host_gap_p95_ms'], '.2f')} "
            f"| {cell(row.get('sm_mhz_mean'), '.0f')} / {cell(row.get('sm_mhz_min'), '.0f')} "
            f"| {cell(row.get('temp_c_max'), '.0f')} | {cell(row.get('mem_mib_end'), '.0f')} |"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
