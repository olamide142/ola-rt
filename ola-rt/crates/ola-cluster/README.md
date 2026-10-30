# ola-cluster

`ola-cluster` coordinates several OlaRT executors as one service. It decides
which worker runs each request and keeps a record of who owns what. It is
synchronous and sans-IO: the [`Coordinator`] consumes submissions and worker
messages and returns commands, client events, and trace records, but never
opens a socket or reads a clock. Simulations, tests, and (from P9 D2) the
networked cluster drive the same decision code.

The design is in `ola-rt/crates/DISTRIBUTED.md`.

## Authority

- Each worker's `Executor` stays authoritative for request lifecycle on that
  worker. The coordinator can only ask a worker to admit or cancel, and a
  worker may refuse.
- The coordinator is authoritative for ownership: every request is pending,
  owned by exactly one worker, or terminal.

## Planning

Each [`Coordinator::plan`] call builds an immutable [`ClusterSnapshot`] of
pending requests and worker summaries, asks the [`ClusterPolicy`] for a
[`ClusterPlan`], validates it, and only then commits. The plan holds exactly
one decision per pending request: assign it to a worker, defer it, or reject
it. A rejected plan changes nothing. Policies take `&self` in `plan`, so
identical snapshots give identical decisions; they learn only in `observe`.

Worker summaries can be stale. A snapshot counts assignments a worker has not
yet reflected in its summary as in-flight load, so a burst of arrivals does
not all land on the worker that last looked idle.

Two baselines are provided: [`RoundRobinRouting`] and
[`LeastLoadedRouting`]. Both only consider ready workers that host the
request's model; ties break by worker ID.

## Local clusters

[`WorkerHost`] wraps an `Executor` and speaks the coordinator's message
vocabulary. [`LocalCluster`] connects a coordinator and several hosts in one
process and delivers messages synchronously and in worker-ID order, so runs
are deterministic. It is the in-process transport for tests and simulation.

## Traces

Every coordinator decision and ownership change produces a
[`ClusterTraceRecord`] keyed by cluster request ID. `Accepted` records carry
the worker and its local request ID, which join with `ola-observe` traces
for that worker.
