#!/usr/bin/env python3
"""Which workload regimes prefer different fixed token budgets?

Reads the CSV `budget_regimes` writes. For every regime (trace, load) it
reports the budget with the lowest mean objective (p95 completion + p95
TTFT). For every pair of regimes A, B whose best budgets a*, b* differ, it
asks whether each regime's best beats the other's *in that regime* on
paired per-seed differences whose 95% bootstrap interval excludes zero:
only then does the preferred action change between them.

For such a pair it also reports the headroom an adaptive controller could
win: the mean objective of the best single budget for the A-then-B pair
(the static oracle) minus the per-regime optima. That headroom, per
request window, is what switching costs must stay below.

    python3 lab/replay/regimes.py out.csv [--loads light,moderate,...]
"""

import argparse
import csv
import itertools
import random
from collections import defaultdict
from statistics import mean


def load(path):
    rows = defaultdict(dict)  # (trace, load) -> {(budget): {seed: objective}}
    for row in csv.DictReader(open(path)):
        regime = (row["trace"], row["load"])
        budget = (int(row["log2_max_tokens"]), int(row["prefill_share_percent"]))
        rows[regime].setdefault(budget, {})[int(row["seed"])] = int(row["objective_us"])
    return rows


def label(budget):
    return f"{1 << budget[0]}t/{budget[1]}%"


def bootstrap(diffs, draws=4000, seed=0):
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(mean(rng.choice(diffs) for _ in range(n)) for _ in range(draws))
    return means[int(0.025 * draws)], means[int(0.975 * draws) - 1]


def paired(regime_rows, better, worse):
    """Per-seed objective(worse) - objective(better); positive favours `better`."""
    seeds = sorted(set(regime_rows[better]) & set(regime_rows[worse]))
    diffs = [regime_rows[worse][s] - regime_rows[better][s] for s in seeds]
    lo, hi = bootstrap(diffs)
    return mean(diffs), lo, hi


def best(regime_rows):
    return min(regime_rows, key=lambda b: mean(regime_rows[b].values()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--loads", help="comma-separated loads to keep")
    args = parser.parse_args()
    rows = load(args.csv)
    if args.loads:
        keep = set(args.loads.split(","))
        rows = {r: v for r, v in rows.items() if r[1] in keep}

    print("## Best fixed budget per regime\n")
    print("| trace | load | best | mean objective (ms) | largest (1024t/100%) | Δ largest (ms) [95% CI] |")
    print("|---|---|---|---:|---:|---|")
    optimum = {}
    for regime in sorted(rows):
        r = rows[regime]
        b = best(r)
        optimum[regime] = b
        largest = (10, 100)
        d, lo, hi = paired(r, b, largest) if b != largest else (0, 0, 0)
        print(
            f"| {regime[0]} | {regime[1]} | {label(b)} | {mean(r[b].values()) / 1e3:.0f} | "
            f"{mean(r[largest].values()) / 1e3:.0f} | {d / 1e3:+.0f} [{lo / 1e3:+.0f}, {hi / 1e3:+.0f}] |"
        )

    print("\n## Regime pairs whose preferred budget changes\n")
    print(
        "Both cells must exclude zero: a* beats b* in A, and b* beats a* in B. "
        "Headroom is the static oracle's mean objective over A and B minus the "
        "per-regime optima's, in ms per regime run.\n"
    )
    print("| A | B | a* | b* | a* over b* in A (ms) | b* over a* in B (ms) | static oracle | headroom (ms) |")
    print("|---|---|---|---|---|---|---|---:|")
    found = 0
    for a, b in itertools.combinations(sorted(rows), 2):
        if a[0] != b[0]:
            continue  # a regime change is a load change within one trace family
        sa, sb = optimum[a], optimum[b]
        if sa == sb:
            continue
        da = paired(rows[a], sa, sb)
        db = paired(rows[b], sb, sa)
        if da[1] <= 0 or db[1] <= 0:
            continue
        found += 1
        common = set(rows[a]) & set(rows[b])
        static = min(
            common, key=lambda x: mean(rows[a][x].values()) + mean(rows[b][x].values())
        )
        static_total = mean(rows[a][static].values()) + mean(rows[b][static].values())
        oracle_total = mean(rows[a][sa].values()) + mean(rows[b][sb].values())
        print(
            f"| {a[0]} {a[1]} | {b[1]} | {label(sa)} | {label(sb)} | "
            f"{da[0] / 1e3:+.0f} [{da[1] / 1e3:+.0f}, {da[2] / 1e3:+.0f}] | "
            f"{db[0] / 1e3:+.0f} [{db[1] / 1e3:+.0f}, {db[2] / 1e3:+.0f}] | "
            f"{label(static)} | {(static_total - oracle_total) / 2e3:.0f} |"
        )
    if not found:
        print("| none | | | | | | | |")


if __name__ == "__main__":
    main()
