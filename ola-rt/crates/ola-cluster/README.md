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

Routing baselines are [`RoundRobinRouting`], [`LeastLoadedRouting`],
[`PowerOfTwoRouting`], [`KvHeadroomRouting`], and [`PrefixAffinityRouting`].
[`AdaptiveClusterPolicy`] picks the least-loaded or KV-headroom baseline by
its weights, then reassigns each model's batch with a MILP
([`optimize_batch_routing`]) under worker request and KV capacity. Its
optimizer minimizes a weighted cost for current load, KV headroom pressure,
and calibrated network cost. Its weights are fixed between
[`AdaptiveClusterPolicy::set_weights`] calls; it does not learn online. Every
policy only considers ready workers that host the request's model; ties break
by worker ID.

## Distributed policy feedback

The coordinator remains the single routing authority; workers report state and
execute coordinator commands rather than independently reaching consensus on
routes. It retains the latest 256 observations in each feedback category:
terminal requests, worker state, migration outcomes, and acknowledged policy
updates. `Coordinator::feedback_snapshot` and
`TcpCoordinator::feedback_snapshot` return a point-in-time copy that also
includes the active policy descriptor and configuration. These bounded
histories are best-effort recent feedback, not a durable event log; consumers
should tolerate older observations rolling out of the window.

An embedding service using `ola-serve` can read
`ClusterGateway::feedback_snapshot` and submit a validated
`ClusterGateway::update_routing_weights` proposal. The gateway serializes these
operations with coordinator work. The default adaptive policy accepts weight
updates; other policies may reject them. Weight updates affect coordinator
routing immediately, but they do not constitute consensus or replicated
failover, and the gateway does not expose these controls as HTTP endpoints.
This creates an integration seam for an external adaptation loop without
requiring a KV connector or a separate orchestrator in the runtime.

## Local clusters

[`WorkerHost`] wraps an `Executor` and speaks the coordinator's message
vocabulary. [`LocalCluster`] connects a coordinator and several hosts in one
process and delivers messages synchronously and in worker-ID order, so runs
are deterministic. It is the in-process transport for tests and simulation.

## Networked clusters

[`WorkerSession`] serves one [`WorkerHost`] over TCP: it heartbeats and
reports a summary every heartbeat interval and runs executor iterations
whenever there is work. [`TcpCoordinator`] owns a coordinator and one
[`CoordinatorWorkerLink`] per worker. Each link's frames are read by its own
task into a shared channel, so [`TcpCoordinator::next_frame`] is cancel-safe
and an event loop can `select!` on it, then call
[`TcpCoordinator::apply_frame`]. A link that fails loses its worker. The
`ola-serve` gateway is one such event loop.

## Recovery and migration

A lost worker's requests return to pending and are replanned by recompute.
The coordinator drops tokens a recovered request replays, so a client sees
each position once; that replay is faithful only for greedy or seeded
generation. If a lost worker returns still running a recovered request, the
coordinator cancels its copy and ignores its reports.

[`TcpCoordinator::migrate`] moves a running request with a two-phase
handoff: the source exports, the target admits, and only the target's
acceptance commits. A refusal, or losing the target, rolls the request back
to the source. The sans-IO steps are [`Coordinator::stage_migration`],
[`Coordinator::finish_migration`], and [`Coordinator::rollback_migration`];
commands produced during recovery are drained with
[`Coordinator::take_commands`]. Details and known limitations are in
`DISTRIBUTED.md`.

## Traces

Every coordinator decision and ownership change produces a
[`ClusterTraceRecord`] keyed by cluster request ID. `Accepted` records carry
the worker and its local request ID, which join with `ola-observe` traces
for that worker. Migrations record `Migrated` with the handoff mode and
bytes, and a cancelled stale copy records `Superseded`.
