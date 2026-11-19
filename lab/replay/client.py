#!/usr/bin/env python3
"""Open-loop replay of an exported trace against `ola serve`.

Reads a trace written by `replay_trace export`, sends each request to
`POST /v1/generate` at its arrival time without waiting on earlier
responses, and writes one JSON line per request with the server's own
lifecycle timestamps (microseconds since the server started).

Prompts are fixed token IDs of the traced length, so tokenization never
changes a prompt's length. Before the measured run, a few sequential
warm-up requests cover the trace's prompt lengths, because CUDA kernels
compile on first use of each shape.

    client.py --url http://127.0.0.1:8080 --model m \\
        --trace trace.jsonl --out observed.jsonl [--warmups 8]

Standard library only.
"""

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request

# Ordinary vocabulary IDs, clear of the special tokens at the start of the
# SmolLM2 and Llama vocabularies.
FIRST_TOKEN_ID = 1000
TOKEN_ID_SPAN = 5000


def prompt_tokens(index, length):
    """Deterministic, request-specific prompt of `length` token IDs."""
    return [FIRST_TOKEN_ID + (index * 7919 + offset) % TOKEN_ID_SPAN for offset in range(length)]


def generate(url, body, timeout):
    request = urllib.request.Request(
        url + "/v1/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def warm_up(url, model, trace, count, timeout):
    lengths = sorted({row["prompt_tokens"] for row in trace})
    if count < len(lengths):
        # Spread the warm-ups across the range of prompt lengths.
        step = (len(lengths) - 1) / max(count - 1, 1)
        lengths = sorted({lengths[round(i * step)] for i in range(count)})
    for length in lengths:
        generate(
            url,
            {"model": model, "tokens": prompt_tokens(0, length), "max_new_tokens": 2},
            timeout,
        )
    return lengths


def send(url, model, row, scheduled, results, timeout):
    body = {
        "model": model,
        "tokens": prompt_tokens(row["index"], row["prompt_tokens"]),
        "max_new_tokens": row["max_new_tokens"],
        "priority": row["priority"],
    }
    if row.get("deadline_us") is not None:
        body["deadline_ms"] = max(1, round(row["deadline_us"] / 1000))
    sent = time.monotonic()
    record = dict(row)
    record["send_lag_us"] = round((sent - scheduled) * 1e6)
    try:
        response = generate(url, body, timeout)
        timing = response["timing"]
        record.update(
            {
                "id": response["id"],
                "finish_reason": response["finish_reason"],
                "generated_tokens": response["usage"]["generated_tokens"],
                "queued_us": timing["queued_us"],
                "started_us": timing["started_us"],
                "first_token_us": timing["first_token_us"],
                "completed_us": timing["completed_us"],
                "client_latency_us": round((time.monotonic() - sent) * 1e6),
            }
        )
    except (urllib.error.URLError, OSError, KeyError, ValueError) as error:
        record["error"] = str(error)
    results[row["index"]] = record


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--warmups", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    with open(args.trace) as file:
        trace = [json.loads(line) for line in file if line.strip()]
    warmed = warm_up(args.url, args.model, trace, args.warmups, args.timeout)
    print(f"warmed prompt lengths {warmed}", file=sys.stderr)

    results = {}
    threads = []
    start = time.monotonic() + 0.5
    for row in trace:
        scheduled = start + row["arrival_us"] / 1e6
        delay = scheduled - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        thread = threading.Thread(
            target=send, args=(args.url, args.model, row, scheduled, results, args.timeout)
        )
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()

    failed = 0
    with open(args.out, "w") as file:
        for row in trace:
            record = results[row["index"]]
            failed += "error" in record
            file.write(json.dumps(record) + "\n")
    lags = sorted(results[row["index"]]["send_lag_us"] for row in trace)
    print(
        f"{len(trace)} requests, {failed} failed · send lag p50 {lags[len(lags) // 2]} µs, "
        f"max {lags[-1]} µs",
        file=sys.stderr,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
