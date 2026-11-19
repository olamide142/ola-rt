#!/usr/bin/env python3
"""Summarizes repeated real-model prefill grouping runs (E3).

Reads `llama-prefill-strategies` reports and, for each prompt-length set,
reports the mean prefill iteration of unbucketed, power-of-two, and
cost-aware grouping with 95% intervals, which fixed strategy was faster and
whether the intervals separate, and whether cost-aware grouping chose the
same grouping (invocations and padding) as the faster fixed strategy.

With `--simulated bucketing.csv` (from `reviewer_eval bucketing`), it also
prints the simulated rate at which power-of-two beats unbucketed for each
distribution and set size, next to the measured outcome.

    prefill_summary.py --sets sets.csv --out e3.csv [--simulated bucketing.csv] REPORT...

`sets.csv` is `reviewer_eval prefill_sets` output; reports are matched to
sets by their `prompt_lengths` parameter. Standard library only.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict


def estimate(report, variant, metric):
    summary = next(s for s in report["summaries"] if s["variant"] == variant)
    value = summary["estimates"][metric]
    interval = value.get("ci95") or {}
    return value["mean"], interval.get("low"), interval.get("high")


def separated(a, b):
    """Whether two (mean, low, high) intervals do not overlap."""
    return None not in (a[1], a[2], b[1], b[2]) and (a[2] < b[1] or b[2] < a[1])


def simulated_win_rates(path):
    trials = defaultdict(dict)
    with open(path) as file:
        for row in csv.DictReader(file):
            key = (row["distribution"], int(row["set_size"]), int(row["trial"]))
            trials[key][row["strategy"]] = float(row["modelled_us"])
    wins = defaultdict(lambda: [0, 0])
    for (distribution, size, _), strategies in trials.items():
        wins[(distribution, size)][0] += strategies["power_of_two"] < strategies["unbucketed"]
        wins[(distribution, size)][1] += 1
    return {key: won / total for key, (won, total) in wins.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sets", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--simulated")
    parser.add_argument("reports", nargs="+")
    args = parser.parse_args()

    with open(args.sets) as file:
        sets = {row["lengths"]: row for row in csv.DictReader(file)}
    simulated = simulated_win_rates(args.simulated) if args.simulated else {}

    rows = []
    for path in args.reports:
        with open(path) as file:
            report = json.load(file)
        lengths = report["workload"]["parameters"]["prompt_lengths"]
        source = sets.get(lengths, {"distribution": "?", "set_size": "?"})
        times = {
            variant: estimate(report, variant, "prefill_iteration")
            for variant in ("unbucketed", "power-of-two", "cost-aware")
        }
        shape = {
            variant: (
                round(estimate(report, variant, "invocations")[0]),
                round(estimate(report, variant, "padding_tokens")[0]),
            )
            for variant in times
        }
        faster = min(("unbucketed", "power-of-two"), key=lambda variant: times[variant][0])
        chosen = next(
            (v for v in ("unbucketed", "power-of-two") if shape["cost-aware"] == shape[v]),
            "other",
        )
        best = min(times[v][0] for v in times)
        size = source["set_size"]
        rows.append(
            {
                "distribution": source["distribution"],
                "set_size": size,
                "lengths": lengths,
                "revision": report["metadata"]["build"]["revision"],
                "unbucketed_ms": times["unbucketed"][0] / 1000,
                "power_of_two_ms": times["power-of-two"][0] / 1000,
                "cost_aware_ms": times["cost-aware"][0] / 1000,
                "unbucketed_calls_padding": "{}/{}".format(*shape["unbucketed"]),
                "power_of_two_calls_padding": "{}/{}".format(*shape["power-of-two"]),
                "cost_aware_calls_padding": "{}/{}".format(*shape["cost-aware"]),
                "faster_fixed": faster,
                "intervals_separate": separated(times["unbucketed"], times["power-of-two"]),
                "power_of_two_speedup": times["unbucketed"][0] / times["power-of-two"][0],
                "cost_aware_chose": chosen,
                "cost_aware_picked_faster": chosen == faster,
                "cost_aware_vs_best": times["cost-aware"][0] / best,
                "simulated_p2_win_rate": simulated.get(
                    (source["distribution"], int(size) if str(size).isdigit() else size)
                ),
            }
        )

    with open(args.out, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(
        "| distribution | n | U (ms) | P2 (ms) | CA (ms) | faster | separate | CA chose "
        "| CA / best | sim. P2 win rate |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        rate = row["simulated_p2_win_rate"]
        print(
            f"| {row['distribution']} | {row['set_size']} | {row['unbucketed_ms']:.1f} "
            f"| {row['power_of_two_ms']:.1f} | {row['cost_aware_ms']:.1f} | {row['faster_fixed']} "
            f"| {'yes' if row['intervals_separate'] else 'no'} | {row['cost_aware_chose']} "
            f"| {row['cost_aware_vs_best']:.3f} | {'' if rate is None else f'{rate:.0%}'} |"
        )
    p2_wins = sum(row["faster_fixed"] == "power-of-two" for row in rows)
    picked = sum(row["cost_aware_picked_faster"] for row in rows)
    print(
        f"\npower-of-two faster in {p2_wins}/{len(rows)} sets; "
        f"cost-aware chose the faster fixed grouping in {picked}/{len(rows)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
