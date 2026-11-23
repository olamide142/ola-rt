#!/usr/bin/env python3
"""Build an ola-rt trace from ServeGen production workload data, fetched on
the fly.

ServeGen (Xiang et al., NSDI '26, https://github.com/alibaba/ServeGen)
publishes per-client workload fits from Alibaba Cloud Model Studio. For each
client (`chunk-N`) and each 600 s window there is an arrival rate, an
inter-arrival coefficient of variation with a fitted Gamma or Weibull
distribution (`chunk-N-trace.csv`), and input and output length
distributions (`chunk-N-dataset.json`). Those windows carry real regime
changes: diurnal rate swings, bursty clients, and length mixes that shift
over the day.

Nothing is cloned or cached. Files are read over HTTPS from a pinned commit
(`--ref`), so a trace is reproducible from its command line:

1. every client's trace CSV for the model (small), to rank clients by
   requests in the requested span;
2. the dataset JSON of the `--clients` busiest clients only.

Length distributions are published more coarsely than rates (every 6 h for
`m-large`), so each 600 s window uses the latest at or before it.

Generation follows ServeGen's `generate_workload`: per client and window,
draw the window's request count from its rate, draw inter-arrival gaps from
its fitted distribution, rescale them to fill the window, and sample lengths
from the window's distributions. Two knobs fit production rates to one GPU
without changing the shape of the workload: `--rate-scale` thins every
client's rate, and `--time-scale` compresses time (a 600 s window lasts
600 / time-scale seconds, so rates rise by that factor).

Output is JSON Lines in the `replay_trace export` format (`arrival_us`,
`prompt_tokens`, `max_new_tokens`, `priority`, `deadline_us`), plus `window`
and `client`. The first line records provenance and has no `arrival_us`.
Each request runs to its sampled output length.

    python3 lab/replay/servegen.py --model m-large --start 36000 --hours 4 \\
        --clients 8 --rate-scale 0.05 --time-scale 10 --seeds 1,2,3 \\
        --out servegen-{seed}.jsonl

Standard library only.
"""

import argparse
import ast
import bisect
import csv
import http.client
import io
import json
import math
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

REPO = "alibaba/ServeGen"
# The commit the study was built against; pass --ref to use another.
DEFAULT_REF = "d70c8b2d5bc45f4b60a146fbc43dc5523f871c8b"
WINDOW_SECONDS = 600


def fetch(url, attempts=4):
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                return response.read().decode()
        except (OSError, http.client.HTTPException) as error:
            if attempt + 1 == attempts:
                raise
            print(f"retrying {url}: {error}", file=sys.stderr)
            time.sleep(2 ** attempt)


def raw(ref, path):
    return fetch(f"https://raw.githubusercontent.com/{REPO}/{ref}/{path}")


def client_ids(ref, category, model):
    listing = json.loads(
        fetch(f"https://api.github.com/repos/{REPO}/contents/data/{category}/{model}?ref={ref}")
    )
    ids = sorted(
        int(entry["name"].split("-")[1])
        for entry in listing
        if entry["name"].endswith("-trace.csv")
    )
    return ids


def parse_trace(text):
    """window start (s) -> (rate per s, cv, distribution, shape, scale)."""
    windows = {}
    for row in csv.reader(io.StringIO(text)):
        if len(row) < 6 or not row[3]:
            continue
        rate = float(row[1])
        if rate <= 0:
            continue
        windows[int(row[0])] = (rate, float(row[2]), row[3], float(row[4]), float(row[5]))
    return windows


def parse_dataset(text):
    """window start (s) -> {field: (values, cumulative probabilities)}."""
    out = {}
    for start, fields in json.loads(text).items():
        parsed = {}
        for field in ("input_tokens", "output_tokens"):
            pmf = ast.literal_eval(fields.get(field, "{}"))
            if not pmf:
                break
            values = sorted(pmf)
            cumulative, total = [], 0.0
            for value in values:
                total += pmf[value]
                cumulative.append(total)
            parsed[field] = (values, [c / total for c in cumulative])
        else:
            out[int(start)] = parsed
    return out


def sample(distribution, rng):
    values, cumulative = distribution
    index = bisect.bisect_left(cumulative, rng.random())
    return values[min(index, len(values) - 1)]


def gaps(name, shape, scale, n, rng):
    if name == "Gamma":
        return [rng.gammavariate(shape, scale) for _ in range(n)]
    if name == "Weibull":
        return [rng.weibullvariate(scale, shape) for _ in range(n)]
    raise ValueError(f"unknown arrival pattern {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--category", default="language")
    parser.add_argument("--model", default="m-large")
    parser.add_argument("--ref", default=DEFAULT_REF)
    parser.add_argument("--start", type=int, default=0, help="span start, seconds into the trace")
    parser.add_argument("--hours", type=float, default=24.0, help="span length in trace hours")
    parser.add_argument("--clients", type=int, default=8, help="busiest clients to include")
    parser.add_argument("--rate-scale", type=float, default=1.0)
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--max-input", type=int, default=4096)
    parser.add_argument("--max-output", type=int, default=1024)
    parser.add_argument("--seeds", default="1", help="comma-separated; one trace per seed from one fetch")
    parser.add_argument("--out", required=True, help="output path; {seed} is replaced by the seed")
    args = parser.parse_args()

    end = args.start + int(args.hours * 3600)
    ids = client_ids(args.ref, args.category, args.model)
    print(f"{len(ids)} clients in {args.category}/{args.model} @ {args.ref[:7]}", file=sys.stderr)
    def trace_of(client):
        windows = parse_trace(raw(args.ref, f"data/{args.category}/{args.model}/chunk-{client}-trace.csv"))
        return client, {t: w for t, w in windows.items() if args.start <= t < end}

    with ThreadPoolExecutor(max_workers=16) as pool:
        traces = dict(pool.map(trace_of, ids))
    volume = {c: sum(w[0] for w in ws.values()) for c, ws in traces.items()}
    chosen = sorted(volume, key=volume.get, reverse=True)[: args.clients]
    chosen = [c for c in chosen if volume[c] > 0]
    print(f"busiest {len(chosen)}: {chosen}", file=sys.stderr)

    def dataset_of(client):
        text = raw(args.ref, f"data/{args.category}/{args.model}/chunk-{client}-dataset.json")
        return client, parse_dataset(text)

    with ThreadPoolExecutor(max_workers=8) as pool:
        datasets = dict(pool.map(dataset_of, chosen))
    seeds = [int(seed) for seed in args.seeds.split(",")]
    for seed in seeds:
        out = args.out.replace("{seed}", str(seed))
        requests = generate(args, chosen, traces, datasets, random.Random(seed))
        write(out, args, seed, chosen, requests)


def generate(args, chosen, traces, datasets, rng):
    """One sample of the workload, ServeGen's way."""
    requests = []
    for client in chosen:
        dataset = datasets[client]
        # Length distributions are published at a coarser step than rates
        # (every 6 h for m-large); each window uses the latest at or before it.
        published = sorted(dataset)
        for start, (rate, _cv, name, shape, scale) in sorted(traces[client].items()):
            index = bisect.bisect_right(published, start) - 1
            if index < 0:
                continue
            lengths = dataset[published[index]]
            expected = rate * args.rate_scale * WINDOW_SECONDS
            count = int(expected) + (rng.random() < expected - int(expected))
            if count == 0:
                continue
            drawn = gaps(name, shape, scale, count, rng)
            stretch = WINDOW_SECONDS / sum(drawn)
            at = float(start)
            for gap in drawn:
                at += gap * stretch
                seconds = args.start + (at - args.start) / args.time_scale
                requests.append(
                    {
                        "arrival_us": round(seconds * 1e6),
                        "prompt_tokens": max(1, min(sample(lengths["input_tokens"], rng), args.max_input)),
                        "max_new_tokens": max(1, min(sample(lengths["output_tokens"], rng), args.max_output)),
                        "priority": 0,
                        "deadline_us": None,
                        "window": start,
                        "client": client,
                    }
                )
    requests.sort(key=lambda r: r["arrival_us"])
    return requests


def write(path, args, seed, chosen, requests):
    origin = requests[0]["arrival_us"] if requests else 0
    with open(path, "w") as out:
        provenance = {key: getattr(args, key) for key in vars(args) if key not in ("out", "seeds")}
        provenance.update(
            seed=seed,
            source=f"https://github.com/{REPO}",
            chosen_clients=chosen,
            requests=len(requests),
        )
        out.write(json.dumps({"servegen": provenance}) + "\n")
        for request in requests:
            request["arrival_us"] -= origin
            out.write(json.dumps(request) + "\n")
    span = (requests[-1]["arrival_us"] / 1e6) if requests else 0
    print(f"wrote {len(requests)} requests over {span:.0f} s to {path}", file=sys.stderr)

if __name__ == "__main__":
    main()
