# Distributed execution design

Status: **implemented through the D5 foundation; backend-specific D4 extensions
remain** (2026-10-31). Covers `todo.md` P9, which replaces the
"multi-GPU placement", "distributed request routing", "distributed KV-cache
management", and "tensor and pipeline parallelism" bullets under later work.

## Goal

Run one OlaRT service across several executors on several machines. A
cluster-level policy decides where work goes, components coordinate through
explicit messages, and the same optimization machinery that tunes one
executor (`AdaptivePolicy`, `AdaptiveOptimizer`, MILP) also tunes the cluster.

Done when a seeded workload served by a coordinator and at least two worker
processes produces the same per-request outputs as one executor, every
routing, migration, and recovery decision is traced with reasons, and an
adaptive cluster policy beats the best fixed routing baseline on held-out
traces in both simulation and a live loopback cluster.

## What exists

Everything below assumes one process.

- **One owner.** `ola-serve`'s driver thread owns one `Executor` and talks
  to clients over in-process channels.
- **Plans have no location.** `ExecutionPlan`, `PlannedBatch`, and
  `PlannedRow` say what runs and when, never where. `PlanningSnapshot`
  carries at most one `DeviceSnapshot`.
- **KV and prefix state is process-local.** `KvBlockManager` hands out
  `usize` block IDs; prefix matches can only be found locally.
- **Sessions can't leave the process.** `GenerationSession` holds tensors,
  cursor, and sampler RNG in memory, with no export or import.
- **Placement is a single label.** `ModelPlacement` is `Host` or
  `Device(String)`.
- **One clock domain.** `Timestamp` is microseconds of one monotonic epoch.
- **No partial failure.** Errors separate transport, runtime, model, and
  admission failures, but none means "a node disappeared".

Two pieces already do most of the work distribution needs:

- **Recompute preemption.** P7 resumes a preempted request from its tokens
  with identical output. Moving a request to another worker by recompute is
  the same operation done on another executor.
- **Adaptive optimization.** Factory + optimizer + validation + fallback +
  observe-only updates is exactly the loop a cluster policy needs, one level up.

## Shape of the system

```text
                 clients (native HTTP / OpenAI)
                              │
                ┌─────────────▼──────────────┐
                │  gateway  (ola-serve)      │  HTTP front end, token streams
                │  ┌──────────────────────┐  │
                │  │ Coordinator          │  │  ola-cluster, sans-IO
                │  │  ClusterPolicy ──────┼──┼── medium loop: route, migrate
                │  │  PlacementPolicy ────┼──┼── slow loop: replicas, roles, θ
                │  │  ownership + leases  │  │
                │  └──────────┬───────────┘  │
                └─────────────┼──────────────┘
                              │ ola-net (WireMessage over a Transport)
          ┌───────────────────┼────────────────────┐
          ▼                   ▼                    ▼
   ┌─────────────┐     ┌─────────────┐      ┌─────────────┐
   │ worker A    │     │ worker B    │      │ worker C    │  ola-serve worker
   │ Executor    │◄───►│ Executor    │      │ Executor    │  role, wraps the
   │ ExecPolicy  │ KV  │ ExecPolicy  │      │ ExecPolicy  │  existing driver
   │ (fast loop) │     │ (fast loop) │      │ (fast loop) │
   └─────────────┘     └─────────────┘      └─────────────┘
```

Three control loops, as in `notes/fomalism.md` §4:

| Loop   | Owner               | Period            | Decides                                                       |
|--------|---------------------|-------------------|---------------------------------------------------------------|
| fast   | `ExecutionPolicy`   | each iteration    | scheduling, admission, batching, kernels (unchanged)          |
| medium | `ClusterPolicy`     | each arrival/tick | which worker, migrate or not, KV transfer vs recompute        |
| slow   | `PlacementPolicy`   | seconds–minutes   | model residency, replicas, prefill/decode roles, worker θ     |

## Decisions

1. **Two new crates.**
   - `ola-cluster`: synchronous and sans-IO. Cluster vocabulary, snapshots,
     plans, validation, the `ClusterPolicy`/`PlacementPolicy` traits, and the
     `Coordinator` state machine. It consumes messages and returns messages
     and actions; it never opens a socket or reads a clock. Depends on
     `ola-core`, `ola-runtime`, `ola-kernel`; no tokio.
   - `ola-net`: the network layer. Versioned `WireMessage`, a `Transport`
     trait, and three transports (below). Depends on `ola-cluster` types,
     serde, and tokio `net`.

   `ola-serve` gains `worker` and `gateway` roles; `ola-cli` exposes them.
   Keeping the coordinator sans-IO means the simulator, tests, and the live
   cluster run the same decision code.

2. **Workers stay authoritative for request lifecycle; the coordinator is
   authoritative for ownership.** This extends the existing rule ("keep
   authoritative lifecycle mutation inside the runtime"). The coordinator
   records which worker owns each request and can only *ask* a worker to
   admit, cancel, export, or import. A worker may refuse (`AssignmentRejected
   { reason }`); local admission still runs its own policy and memory checks.
   A request has exactly one owner per epoch.

3. **Snapshots are explicitly stale.** `ClusterSnapshot` holds one
   `WorkerSummary` per worker, each with the worker's epoch and the
   coordinator time it was received. Policies treat summaries as estimates,
   and the coordinator adds its own not-yet-acknowledged assignments to them
   (optimistic accounting) so a burst of arrivals doesn't all land on the
   same "idle" worker.

4. **`ExecutionPlan` stays local.** Cluster decisions go in a separate
   `ClusterPlan` (assign, migrate, replicate, unload, set-role, push-θ,
   defer) with its own validation: no unknown or dead worker, no double
   ownership, no migration of a request that isn't live, no placement over a
   worker's reported memory. A rejected plan changes nothing, as with P1.

5. **Policy traits mirror `ExecutionPolicy`.**

   ```rust
   pub trait ClusterPolicy: Send {
       fn descriptor(&self) -> PolicyDescriptor;
       fn plan(&self, snapshot: &ClusterSnapshot) -> Result<ClusterDecision, PolicyError>;
       fn observe(&mut self, iteration: &ClusterIteration) {}
   }
   ```

   `plan` takes `&self`, so it is deterministic and learns only in `observe`.
   `ClusterDecision` carries a `ClusterDecisionRecord` (inputs, alternatives,
   estimates, reasons), like `DecisionRecord`. `PlacementPolicy` has the same
   shape and runs on a slower tick.

6. **Identity and time.**
   - `ola-core` gains `WorkerId` and `ClusterRequestId`. The gateway issues
     cluster IDs; each worker keeps its local `RequestId`, and the
     coordinator maps between them. Traces join on the cluster ID.
   - Each worker reports in its own clock domain. The coordinator estimates
     per-worker offsets from heartbeat round trips and converts to
     coordinator time; cluster latency metrics (TTFT, completion) use
     coordinator time only. Simulation uses one virtual clock and skips this.

7. **Migration has two levels, both behind one `Executor` API.**
   - `Executor::export_request(id) -> ResumableRequest` removes a live
     request the same way preemption does, but returns its state instead of
     requeueing it. `Executor::import_request(ResumableRequest)` admits it
     through the existing `Resumed` path.
   - Level 1, **token recompute**: `ResumableRequest` holds the prompt,
     generated tokens, generation config, and sampler RNG state. This needs
     no model changes beyond exposing the RNG state.
   - Level 2, **KV transfer**: an optional `SessionState` payload with the
     KV blocks (geometry, dtype, per-layer K/V bytes) and cursor, from new
     `GenerationSession::export_state` / `CausalModel::import_session` hooks.
     The default returns `None`, which falls back to level 1. The same hooks
     provide KV swapping to host memory (already a later-work item).
   - Handoff is two-phase: source suspends and exports → target imports and
     acks → coordinator commits the ownership change → source frees. Until
     the commit the source keeps its state, so a failed transfer just resumes
     at the source.

8. **Recovery reuses the token log.** Every streamed token passes through
   the gateway, so the coordinator already has each request's tokens. When a
   worker's lease expires, its requests are re-imported elsewhere by
   recompute (level 1), using the RNG state from the worker's most recent
   checkpoint, or fail with `WorkerLost` if policy says so. Workers include
   RNG state with token batches so the checkpoint is never more than one
   batch old.

9. **Transports.**
   - `InProcTransport`: channels, deterministic order. Unit tests and
     single-process multi-executor runs.
   - `SimTransport`: virtual time with a per-link latency, bandwidth, and
     drop/partition model. Used by `ola-bench`.
   - `TcpTransport`: tokio, length-prefixed frames. Control messages are
     serde JSON (debuggable, matches existing reports); KV payloads travel as
     separate binary frames that a JSON header references. `ola-cluster`
     provides a typed `SessionState` adapter for those frames, validating the
     media type and protocol version before a backend imports the opaque
     payload. No gRPC/tonic
     dependency until a reason appears.
   Every message carries a protocol version, sender epoch, and correlation
   ID. The gateway logs messages to JSON Lines so a live run can be replayed
   through the sans-IO coordinator.

10. **The HTTP layer becomes backend-agnostic.** Add an `InferenceBackend`
    trait (submit, stream, cancel, result, health, readiness, models) with
    two implementations: the existing local `RuntimeHandle` and a cluster
    handle backed by the coordinator. The native and OpenAI routes don't
    change.

## Integration by layer

| Crate         | Change                                                                                                   |
|---------------|----------------------------------------------------------------------------------------------------------|
| `ola-core`    | `WorkerId`, `ClusterRequestId`, `ClockDomain`; serde on the IDs that cross the wire.                      |
| `ola-kernel`  | Kernel capabilities and device descriptors become part of `WorkerSummary`, so routing only picks workers that can run the planned kernels. |
| `ola-model`   | Optional `export_state` / `import_session` hooks; sampler RNG state becomes exportable.                   |
| `ola-burn`    | Llama and Mistral implement the hooks (level 2) after level 1 works.                                      |
| `ola-runtime` | `export_request` / `import_request`; `ReleaseReason::Migrated`; read-only `WorkerSummary` built from `PlanningSnapshot` + KV pool status + prefix fingerprints (hashes of full immutable blocks, which the prefix cache already verifies). `ExecutionPlan` unchanged. |
| `ola-cluster` | New. Vocabulary, `ClusterSnapshot`, `ClusterPlan` + validation, policies, `Coordinator`, adaptive wrapper. |
| `ola-net`     | New. `WireMessage`, `Transport`, in-proc / sim / TCP transports, message log.                             |
| `ola-observe` | Trace records gain `worker` and cluster ID; new records `Routed`, `AssignmentRejected`, `Migrated { mode, bytes, duration }`, `WorkerJoined/Lost`, `PolicyPushed`. Observation stays read-only. |
| `ola-serve`   | `worker` role (driver + protocol server), `gateway` role (HTTP + coordinator), `InferenceBackend`.         |
| `ola-bench`   | Multi-worker virtual-time harness: N executors with the synthetic model and `CostModel`, `SimTransport`, network cost model, failure injection. Reports record topology identity. |
| `ola-cli`     | `ola serve --role standalone|worker|gateway`, cluster config (workers, roles, policies) via Serde like `ExecutorConfig`. |
| `ola-ui`      | Later: cluster view of workers, ownership, and routing decisions.                                         |

## Interweaving with the optimization layer

The cluster reuses the optimization stack rather than adding a second one.

1. **Adaptive cluster policy.** `AdaptiveClusterPolicy` wraps a
   `ClusterPolicy` exactly as `AdaptivePolicy` wraps `ExecutionPolicy`:
   factory builds a policy from θ, optimizer proposes θ in `observe`,
   candidates are validated with a known-safe fallback, installs are
   transactional and rate-limited. `AdaptiveOptimizer` is reused as is; it
   already works over named parameters. Cluster θ includes routing weights
   (load vs cache affinity vs transfer cost), migration thresholds, the
   KV-transfer-vs-recompute break-even, and the prefill/decode split.

2. **Optimizing across levels.** The slow loop can push θ to workers through
   a `PolicyUpdate` message. The worker installs it at its next observation
   boundary through the existing transactional path, so per-worker tuning
   becomes something the cluster coordinates. The coordinator also sets each
   worker's `AdaptiveConstraints` (memory budget, slot ceiling), so local
   optimizers stay inside the cluster's placement. Worker summaries report
   current θ and recent `AdaptiveRequestObservation`s, so the global
   optimizer sees what local tuning did.

3. **MILP formulations.** `good_lp` is already a dependency.
   - Placement: binary `x[m,w]` (model m resident on worker w) under
     per-worker memory, minimizing expected queueing latency from calibrated
     per-worker costs plus load cost.
   - Batch routing: arrivals within one tick to workers as a min-cost
     assignment with capacity, KV-headroom, and prefix-hit terms.
   - Disaggregation: worker roles (prefill / decode / mixed) chosen to
     balance prefill and decode token rates.
   The finite enumerator stays the exact oracle for small instances, as it
   is today.

4. **Objective.** Extend `J(θ)` from `ADAPTIVE_OPTIMIZATION_PLAN.md` with
   bytes transferred, migrations, and rejected assignments, measured in
   coordinator time.

5. **Calibration.** P4 calibration gains link probes (latency, bandwidth,
   KV export/import throughput), stored in reports so cost models and
   optimizers use measured rather than assumed network costs.

## Steps

Each step ends with tests and a `progress_report.md` entry.

- **D0 — vocabulary and in-process cluster.** `ola-cluster` types,
  `ClusterPlan` validation, sans-IO `Coordinator`, `InProcTransport`,
  round-robin and least-loaded routing over N executors in one process.
  *Done when* seeded per-request outputs match a single executor and cluster
  traces join with worker traces.
- **D1 — simulation and routing baselines.** Multi-worker virtual-time
  harness; baselines: round-robin, join-shortest-queue, power-of-two-choices,
  KV-headroom-aware, prefix-affinity. Registered experiments in the CLI and UI.
  *Done when* the baselines share one reproducible comparison protocol across
  topologies and network models (the P3 criterion, one level up).
- **D2 — network layer and roles.** `ola-net` TCP transport, worker and
  gateway roles, `InferenceBackend`, heartbeats and leases, CLI and config.
  *Done when* the OpenAI endpoint serves through a gateway and two worker
  processes on loopback, and streaming, cancellation, overload, and graceful
  shutdown tests pass over the network.
- **D3 — migration and recovery.** `export_request` / `import_request`,
  level-1 migration, worker drain, lease-expiry recovery, failure and
  partition injection in the simulator.
  *Done when* migrated and recovered requests produce outputs identical to
  uninterrupted runs, and every handoff is traced.
- **D4 — KV transfer and disaggregation.** Session export/import in Burn
  Llama/Mistral, binary KV frames, prefill/decode roles, transfer-vs-recompute
  choice in policy.
  *Done when* transferred sessions match recompute, and a study reports when
  transfer beats recompute on measured links.
- **D5 — hierarchical optimization.** `AdaptiveClusterPolicy`, θ push to
  workers, MILP placement and batch routing, link calibration.
  *Done when* the adaptive cluster policy improves the chosen reward over the
  best fixed D1 baseline on held-out seeded traces, in simulation and on a
  loopback cluster, without validation failures.
- **D6 — pipeline parallelism (research).** A model split into stages on
  different workers, activations sent over `ola-net`, placement choosing the
  pipeline degree. Starts only if the research needs models larger than one
  device.

D0–D1 need no network or extra hardware. D2–D5 work on one machine with
multiple processes; a second machine is only needed to measure real links.

## Out of scope

- **Tensor parallelism across machines.** It needs collective communication
  (NCCL/MPI) that Burn and CubeCL don't provide across hosts, and per-layer
  all-reduce over commodity networks is rarely worth it. Revisit for
  multiple GPUs in one host.
- **Coordinator high availability.** One coordinator; if it dies the cluster
  stops accepting work. Consensus/replication is a separate project.
- **Security.** Workers bind to localhost by default and use a shared token;
  TLS and real authentication come later.
- **Elastic autoscaling** of machines, as opposed to placement over a fixed
  worker set.
