//! Coordinator, routing, and local-cluster behaviour on the synthetic model.

use std::collections::{BTreeMap, BTreeSet};

use ola_bench::{KvGeometry, synthetic_artifact, synthetic_prompt};
use ola_cluster::{
    ClientEvent, ClusterAction, ClusterPlan, ClusterPlanError, ClusterPolicy, ClusterRequestStatus,
    ClusterSnapshot, ClusterTraceEvent, ClusterTraceRecord, Coordinator, CoordinatorError,
    CoordinatorInput, DeferReason, KvHeadroomRouting, LeastLoadedRouting, LocalCluster,
    PowerOfTwoRouting, PrefixAffinityRouting, RejectReason, RoundRobinRouting, RouteReason,
    WorkerCommand, WorkerHealth, WorkerHost, WorkerMessage, WorkerRole, WorkerSummary,
};
use ola_core::{
    ClusterRequestId, Input, ModelDescriptor, ModelId, RequestId, RequestSpec, Timestamp, WorkerId,
};
use ola_model::KvPoolStatus;
use ola_net::WireBody;
use ola_runtime::{Executor, MigratedRequest, Runtime};

const MODEL: &str = "synthetic";

fn at(micros: u64) -> Timestamp {
    Timestamp::from_micros(micros)
}

fn executor() -> Executor {
    let mut executor = Executor::with_max_active_requests(Runtime::default(), 2).unwrap();
    executor
        .register_model(
            ModelDescriptor::language(MODEL, "synthetic"),
            synthetic_artifact(KvGeometry::default()),
        )
        .unwrap();
    executor
}

fn spec(prompt_tokens: usize, max_new_tokens: usize, eos_after: Option<usize>) -> RequestSpec {
    let mut spec = RequestSpec::text(MODEL, "");
    spec.inputs = vec![Input::Tokens(synthetic_prompt(prompt_tokens, eos_after))];
    spec.generation.max_new_tokens = max_new_tokens;
    spec
}

#[test]
fn level_one_migration_exports_and_resumes_on_another_worker() {
    let request = ClusterRequestId::new(77);
    let mut source = WorkerHost::new("source", executor());
    let mut target = WorkerHost::new("target", executor());
    let request_spec = spec(5, 4, None);

    let accepted = source
        .handle(
            WorkerCommand::Admit {
                request,
                spec: request_spec.clone(),
                arrival_at: at(0),
            },
            at(0),
        )
        .unwrap();
    assert!(matches!(
        accepted.as_slice(),
        [WorkerMessage::Accepted { .. }]
    ));
    source.step(at(1)).unwrap();

    let exported = source
        .handle(WorkerCommand::MigrateOut { request }, at(2))
        .unwrap();
    let WorkerMessage::Migrated { request: id, state } = exported.into_iter().next().unwrap()
    else {
        panic!("migration did not produce a payload");
    };
    assert_eq!(id, request);
    assert!(!state.generated_tokens.is_empty());
    assert!(source.local_id(request).is_none());

    assert!(
        source
            .handle(WorkerCommand::CommitMigration { request }, at(2),)
            .unwrap()
            .is_empty()
    );

    let imported = target
        .handle(
            WorkerCommand::AdmitMigrated {
                request,
                state,
                arrival_at: at(0),
            },
            at(2),
        )
        .unwrap();
    assert!(matches!(
        imported.as_slice(),
        [WorkerMessage::Accepted { .. }]
    ));
    for tick in 3..20 {
        if let Some((_, messages)) = target.step(at(tick)).unwrap()
            && messages
                .iter()
                .any(|message| matches!(message, WorkerMessage::Finished { request: id, .. } if *id == request))
            {
                return;
            }
    }
    panic!("migrated request did not finish");
}

#[test]
fn aborted_migration_restores_the_source_owner() {
    let request = ClusterRequestId::new(78);
    let mut source = WorkerHost::new("source", executor());
    source
        .handle(
            WorkerCommand::Admit {
                request,
                spec: spec(4, 3, None),
                arrival_at: at(0),
            },
            at(0),
        )
        .unwrap();
    source.step(at(1)).unwrap();
    let exported = source
        .handle(WorkerCommand::MigrateOut { request }, at(2))
        .unwrap();
    assert!(matches!(
        exported.as_slice(),
        [WorkerMessage::Migrated { request: id, .. }] if *id == request
    ));
    let restored = source
        .handle(WorkerCommand::AbortMigration { request }, at(3))
        .unwrap();
    assert!(matches!(
        restored.as_slice(),
        [WorkerMessage::Accepted { request: id, .. }] if *id == request
    ));
    assert!(source.local_id(request).is_some());
}

#[test]
fn worker_accepts_and_acknowledges_adaptive_policy_updates() {
    let mut host = WorkerHost::new("worker", executor());
    let messages = host
        .handle(
            WorkerCommand::PolicyUpdate {
                load_weight: 3,
                kv_headroom_weight: 7,
                network_weight: 0,
            },
            at(0),
        )
        .unwrap();
    assert_eq!(host.policy_weights().load, 3);
    assert_eq!(host.policy_weights().kv_headroom, 7);
    assert!(matches!(
        messages.as_slice(),
        [WorkerMessage::PolicyUpdated {
            load_weight: 3,
            kv_headroom_weight: 7,
            network_weight: 0,
        }]
    ));
}

#[test]
fn worker_roles_gate_new_and_migrated_admissions() {
    let mut host = WorkerHost::new("worker", executor());
    host.set_role(WorkerRole::Decode);
    let refused = host
        .handle(
            WorkerCommand::Admit {
                request: ClusterRequestId::new(1),
                spec: spec(4, 3, None),
                arrival_at: at(0),
            },
            at(0),
        )
        .unwrap();
    assert!(matches!(
        refused.as_slice(),
        [WorkerMessage::Refused { .. }]
    ));
    let accepted_migration = host
        .handle(
            WorkerCommand::AdmitMigrated {
                request: ClusterRequestId::new(3),
                state: MigratedRequest {
                    spec: spec(4, 3, None),
                    prompt_tokens: vec![1, 2, 3, 4],
                    generated_tokens: Vec::new(),
                    token_times: Vec::new(),
                    preemptions: 0,
                    session_state: None,
                },
                arrival_at: at(0),
            },
            at(0),
        )
        .unwrap();
    assert!(matches!(
        accepted_migration.as_slice(),
        [WorkerMessage::Accepted { .. }]
    ));
    host.set_role(WorkerRole::Prefill);
    let refused_migration = host
        .handle(
            WorkerCommand::AdmitMigrated {
                request: ClusterRequestId::new(2),
                state: MigratedRequest {
                    spec: spec(4, 3, None),
                    prompt_tokens: vec![1, 2, 3, 4],
                    generated_tokens: Vec::new(),
                    token_times: Vec::new(),
                    preemptions: 0,
                    session_state: None,
                },
                arrival_at: at(0),
            },
            at(0),
        )
        .unwrap();
    assert!(matches!(
        refused_migration.as_slice(),
        [WorkerMessage::Refused { .. }]
    ));
}

/// (arrival, spec) pairs with varied lengths, early stops, and arrivals.
fn workload() -> Vec<(u64, RequestSpec)> {
    (0..12_u64)
        .map(|index| {
            let prompt = 3 + (index as usize * 5) % 17;
            let eos = (index % 3 == 0).then_some(2 + index as usize % 4);
            (index / 3 * 10, spec(prompt, 4 + index as usize % 5, eos))
        })
        .collect()
}

fn single_executor_outputs(workload: &[(u64, RequestSpec)]) -> Vec<Vec<u32>> {
    let mut executor = executor();
    let mut ids = Vec::new();
    let mut clock = 0;
    let mut next = 0;
    while next < workload.len()
        || executor.has_active_request()
        || executor.state().queued_len() > 0
    {
        while next < workload.len() && workload[next].0 <= clock {
            ids.push(
                executor
                    .submit(workload[next].1.clone(), at(clock))
                    .unwrap(),
            );
            next += 1;
        }
        executor.step_iteration(at(clock)).unwrap();
        clock += 1;
    }
    ids.into_iter()
        .map(|id| executor.take_result(id).unwrap().token_ids)
        .collect()
}

struct ClusterRun {
    outputs: BTreeMap<ClusterRequestId, Vec<u32>>,
    streamed: BTreeMap<ClusterRequestId, Vec<u32>>,
    trace: Vec<ClusterTraceRecord>,
    ids: Vec<ClusterRequestId>,
    cluster: LocalCluster,
}

fn run_cluster<P: ClusterPolicy + 'static>(
    policy: P,
    workers: &[&str],
    workload: &[(u64, RequestSpec)],
) -> ClusterRun {
    let mut cluster = LocalCluster::new(policy);
    for worker in workers {
        cluster
            .add_worker(WorkerHost::new(*worker, executor()), at(0))
            .unwrap();
    }
    let mut run = ClusterRun {
        outputs: BTreeMap::new(),
        streamed: BTreeMap::new(),
        trace: Vec::new(),
        ids: Vec::new(),
        cluster,
    };
    let mut clock = 0;
    let mut next = 0;
    while next < workload.len() || !run.cluster.is_idle() {
        while next < workload.len() && workload[next].0 <= clock {
            let id = run
                .cluster
                .submit(workload[next].1.clone(), at(clock))
                .unwrap();
            run.ids.push(id);
            next += 1;
        }
        let step = run.cluster.step(at(clock)).unwrap();
        for (id, event) in step.client {
            match event {
                ClientEvent::Token { token_id, .. } => {
                    run.streamed.entry(id).or_default().push(token_id);
                }
                ClientEvent::Finished { token_ids, .. } => {
                    run.outputs.insert(id, token_ids);
                }
                other => panic!("unexpected client event {other:?}"),
            }
        }
        run.trace.extend(step.trace);
        clock += 1;
        assert!(clock < 10_000, "cluster did not drain");
    }
    run
}

fn routed_to(trace: &[ClusterTraceRecord]) -> BTreeMap<u64, WorkerId> {
    trace
        .iter()
        .filter_map(|record| match &record.event {
            ClusterTraceEvent::Routed { worker, .. } => Some((record.request, worker.clone())),
            _ => None,
        })
        .collect()
}

#[test]
fn cluster_outputs_match_a_single_executor() {
    let workload = workload();
    let expected = single_executor_outputs(&workload);
    for workers in [&["a"][..], &["a", "b"], &["a", "b", "c"]] {
        for run in [
            run_cluster(RoundRobinRouting, workers, &workload),
            run_cluster(LeastLoadedRouting, workers, &workload),
        ] {
            let actual = run
                .ids
                .iter()
                .map(|id| run.outputs[id].clone())
                .collect::<Vec<_>>();
            assert_eq!(actual, expected, "{workers:?}");
            assert_eq!(run.streamed, run.outputs, "streamed tokens match results");
        }
    }
}

#[test]
fn cluster_runs_are_reproducible() {
    let workload = workload();
    let first = run_cluster(LeastLoadedRouting, &["a", "b", "c"], &workload);
    let second = run_cluster(LeastLoadedRouting, &["a", "b", "c"], &workload);
    assert_eq!(first.trace, second.trace);
}

#[test]
fn round_robin_spreads_simultaneous_arrivals_evenly() {
    let workload = (0..6).map(|_| (0, spec(4, 2, None))).collect::<Vec<_>>();
    let run = run_cluster(RoundRobinRouting, &["a", "b", "c"], &workload);
    let mut counts = BTreeMap::<WorkerId, usize>::new();
    for worker in routed_to(&run.trace).into_values() {
        *counts.entry(worker).or_default() += 1;
    }
    assert_eq!(counts.into_values().collect::<Vec<_>>(), [2, 2, 2]);
}

#[test]
fn every_request_is_traced_from_submission_to_completion_and_joins_worker_ids() {
    let workload = workload();
    let run = run_cluster(LeastLoadedRouting, &["a", "b"], &workload);
    for id in &run.ids {
        let kinds = run
            .trace
            .iter()
            .filter(|record| record.request == id.get())
            .map(|record| match &record.event {
                ClusterTraceEvent::Submitted { .. } => "submitted",
                ClusterTraceEvent::Routed { .. } => "routed",
                ClusterTraceEvent::Accepted { .. } => "accepted",
                ClusterTraceEvent::Completed { .. } => "completed",
                other => panic!("unexpected trace event {other:?}"),
            })
            .collect::<Vec<_>>();
        assert_eq!(kinds, ["submitted", "routed", "accepted", "completed"]);
    }
    // Accepted records name the worker-local request, whose retained record
    // is the same request.
    for record in &run.trace {
        if let ClusterTraceEvent::Accepted {
            worker,
            local_request,
        } = &record.event
        {
            let host = run.cluster.worker(worker).unwrap();
            let local = host
                .executor()
                .state()
                .request(RequestId::new(*local_request))
                .unwrap();
            let index = run
                .ids
                .iter()
                .position(|id| id.get() == record.request)
                .unwrap();
            assert_eq!(local.request.spec, workload[index].1);
        }
    }
    let json = serde_json::to_string(&run.trace).unwrap();
    let back: Vec<ClusterTraceRecord> = serde_json::from_str(&json).unwrap();
    assert_eq!(back, run.trace);
}

fn summary(worker: &str, models: &[&str], queued: usize, live: usize) -> WorkerSummary {
    WorkerSummary {
        worker: WorkerId::new(worker),
        generation: 0,
        reported_at: at(0),
        health: WorkerHealth::Ready,
        role: WorkerRole::Mixed,
        models: models.iter().map(|model| ModelId::new(*model)).collect(),
        max_active_requests: 4,
        queued_requests: queued,
        live_requests: live,
        queued_tokens: 0,
        admitted_requests: 0,
        kv: BTreeMap::new(),
    }
}

#[test]
fn policy_decision_can_be_committed_after_external_replication() {
    let mut coordinator = Coordinator::new(LeastLoadedRouting);
    coordinator.join(summary("a", &[MODEL], 0, 0)).unwrap();
    let (request, _) = coordinator.submit(spec(2, 2, None), at(1)).unwrap();
    let generation = coordinator.snapshot(at(2)).generation;

    let proposal = coordinator.plan_decision(at(2)).unwrap().unwrap();
    assert_eq!(coordinator.snapshot(at(2)).generation, generation);
    assert_eq!(
        coordinator.status(request),
        Some(&ClusterRequestStatus::Pending)
    );

    let input = CoordinatorInput::CommitPlan {
        plan: proposal.plan,
        routes: proposal.record.routes,
        at: at(2),
    };
    let encoded = serde_json::to_vec(&input).unwrap();
    let decoded: CoordinatorInput = serde_json::from_slice(&encoded).unwrap();
    let output = coordinator.apply_committed_input(decoded).unwrap().output;
    assert_eq!(output.commands.len(), 1);
    assert!(matches!(
        coordinator.status(request),
        Some(ClusterRequestStatus::Assigned { worker }) if worker == &WorkerId::new("a")
    ));
}

#[test]
fn stale_proposed_plan_is_rejected_without_mutating_requests() {
    let mut coordinator = Coordinator::new(LeastLoadedRouting);
    coordinator.join(summary("a", &[MODEL], 0, 0)).unwrap();
    let (request, _) = coordinator.submit(spec(2, 2, None), at(1)).unwrap();
    let proposal = coordinator.plan_decision(at(2)).unwrap().unwrap();
    coordinator.set_network_cost(WorkerId::new("a"), 5);

    assert!(matches!(
        coordinator.commit_plan(proposal, at(2)),
        Err(CoordinatorError::InvalidPlan(_))
    ));
    assert_eq!(
        coordinator.status(request),
        Some(&ClusterRequestStatus::Pending)
    );
}

#[test]
fn submission_can_be_validated_before_its_committed_application() {
    let mut coordinator = Coordinator::new(LeastLoadedRouting);
    let request_spec = spec(2, 2, None);
    let request = coordinator.prepare_submit(&request_spec).unwrap();

    assert_eq!(coordinator.pending_len(), 0);
    assert_eq!(coordinator.status(request), None);
    coordinator
        .commit_submit(request, request_spec.clone(), at(1))
        .unwrap();
    assert_eq!(coordinator.pending_len(), 1);
    assert_eq!(
        coordinator.commit_submit(request, request_spec, at(2)),
        Err(CoordinatorError::UnexpectedRequestId {
            expected: ClusterRequestId::new(2),
            received: request,
        })
    );
}

#[test]
fn worker_leases_expire_and_new_epochs_revive_them() {
    let mut coordinator = Coordinator::new(RoundRobinRouting);
    coordinator.join(summary("a", &[MODEL], 0, 0)).unwrap();

    assert!(
        coordinator
            .expire_workers(at(9), std::time::Duration::from_micros(10))
            .is_empty()
    );
    assert_eq!(
        coordinator.workers().next().unwrap().health,
        WorkerHealth::Ready
    );

    assert_eq!(
        coordinator.expire_workers(at(10), std::time::Duration::from_micros(10)),
        [WorkerId::new("a")]
    );
    assert_eq!(
        coordinator.workers().next().unwrap().health,
        WorkerHealth::Lost
    );

    coordinator
        .heartbeat(&WorkerId::new("a"), 3, at(11))
        .unwrap();
    assert_eq!(
        coordinator.workers().next().unwrap().health,
        WorkerHealth::Ready
    );
    assert_eq!(
        coordinator.heartbeat(&WorkerId::new("a"), 2, at(12)),
        Err(CoordinatorError::StaleWorkerEpoch {
            worker: WorkerId::new("a"),
            epoch: 2,
        })
    );
}

#[test]
fn worker_commands_round_trip_through_wire_payloads() {
    let command = WorkerCommand::Admit {
        request: ClusterRequestId::new(7),
        spec: spec(3, 2, None),
        arrival_at: at(11),
    };
    let body = WireBody::json_payload(&command).unwrap();
    let decoded: WorkerCommand = body.decode_json().unwrap();
    assert_eq!(decoded, command);

    let message = WorkerMessage::Accepted {
        request: ClusterRequestId::new(7),
        local: RequestId::new(4),
    };
    let body = WireBody::json_payload(&message).unwrap();
    let decoded: WorkerMessage = body.decode_json().unwrap();
    assert_eq!(decoded, message);
}

#[test]
fn least_loaded_counts_queued_live_in_flight_and_same_plan_assignments() {
    let mut coordinator = Coordinator::new(LeastLoadedRouting);
    coordinator.join(summary("a", &[MODEL], 1, 1)).unwrap();
    coordinator.join(summary("b", &[MODEL], 0, 0)).unwrap();
    for _ in 0..4 {
        coordinator.submit(spec(4, 2, None), at(0)).unwrap();
    }
    let (iteration, output) = coordinator.plan(at(0)).unwrap().unwrap();
    let chosen = iteration
        .plan
        .actions
        .iter()
        .map(|action| match action {
            ClusterAction::Assign { worker, .. } => worker.as_str().to_owned(),
            other => panic!("unexpected {other:?}"),
        })
        .collect::<Vec<_>>();
    // b starts empty; after two assignments both have load 2; ties go to a.
    assert_eq!(chosen, ["b", "b", "a", "b"]);
    assert_eq!(output.commands.len(), 4);

    // Summaries have not caught up, so assignments count as in flight.
    let snapshot = coordinator.snapshot(at(1));
    assert_eq!(snapshot.workers[&WorkerId::new("a")].in_flight, 1);
    assert_eq!(snapshot.workers[&WorkerId::new("b")].in_flight, 3);
    let mut caught_up = summary("b", &[MODEL], 3, 0);
    caught_up.admitted_requests = 3;
    coordinator
        .handle(
            &WorkerId::new("b"),
            WorkerMessage::Summary(caught_up),
            at(1),
        )
        .unwrap();
    let snapshot = coordinator.snapshot(at(1));
    assert_eq!(snapshot.workers[&WorkerId::new("b")].in_flight, 0);
    assert_eq!(snapshot.workers[&WorkerId::new("b")].load(), 3);
}

fn assigned_workers(coordinator: &mut Coordinator) -> Vec<String> {
    let (iteration, _) = coordinator.plan(at(0)).unwrap().unwrap();
    iteration
        .plan
        .actions
        .iter()
        .map(|action| match action {
            ClusterAction::Assign { worker, .. } => worker.as_str().to_owned(),
            other => panic!("unexpected {other:?}"),
        })
        .collect()
}

fn with_pool(
    mut summary: WorkerSummary,
    free_blocks: usize,
    cached_blocks: usize,
) -> WorkerSummary {
    summary.kv.insert(
        ModelId::new(MODEL),
        KvPoolStatus {
            total_blocks: 64,
            free_blocks,
            allocated_blocks: 64 - free_blocks - cached_blocks,
            cached_blocks,
            tokens_per_block: 4,
            live_sequences: 0,
            evictions: 0,
            prefix_hit_tokens: 0,
        },
    );
    summary
}

#[test]
fn power_of_two_is_deterministic_and_prefers_the_less_loaded_sample() {
    let workers = [("a", 9), ("b", 0), ("c", 9), ("d", 9)];
    let route = |seed| {
        let mut coordinator = Coordinator::new(PowerOfTwoRouting::new(seed));
        for (worker, queued) in workers {
            coordinator
                .join(summary(worker, &[MODEL], queued, 0))
                .unwrap();
        }
        for _ in 0..40 {
            coordinator.submit(spec(4, 2, None), at(0)).unwrap();
        }
        assigned_workers(&mut coordinator)
    };
    let first = route(11);
    assert_eq!(first, route(11));
    assert_ne!(first, route(12));
    // Every worker gets sampled, but the idle one wins whenever it is drawn
    // until the others' load catches up.
    let counts = first
        .iter()
        .fold(BTreeMap::<&str, usize>::new(), |mut counts, worker| {
            *counts.entry(worker.as_str()).or_default() += 1;
            counts
        });
    assert!(counts["b"] > counts.values().copied().max().unwrap() / 2);
    assert_eq!(counts.len(), 4);
}

#[test]
fn kv_headroom_counts_cached_queued_in_flight_and_same_plan_demand() {
    let mut coordinator = Coordinator::new(KvHeadroomRouting);
    // a: 20 free + 4 evictable blocks = 96 tokens, but 40 tokens queued.
    let mut a = with_pool(summary("a", &[MODEL], 1, 0), 20, 4);
    a.queued_tokens = 40;
    // b: 16 free blocks = 64 tokens.
    let b = with_pool(summary("b", &[MODEL], 0, 0), 16, 0);
    coordinator.join(a).unwrap();
    coordinator.join(b).unwrap();
    let snapshot = coordinator.snapshot(at(0));
    let model = ModelId::new(MODEL);
    assert_eq!(
        snapshot.workers[&WorkerId::new("a")].kv_headroom_tokens(&model),
        Some(56)
    );
    assert_eq!(
        snapshot.workers[&WorkerId::new("b")].kv_headroom_tokens(&model),
        Some(64)
    );

    // Each request needs 10 + 10 tokens: b (64), then a (56 vs 44), then
    // b (44 vs 36), then a again.
    for _ in 0..4 {
        coordinator.submit(spec(10, 10, None), at(0)).unwrap();
    }
    assert_eq!(assigned_workers(&mut coordinator), ["b", "a", "b", "a"]);
    let snapshot = coordinator.snapshot(at(0));
    assert_eq!(snapshot.workers[&WorkerId::new("a")].in_flight_tokens, 40);
    assert_eq!(
        snapshot.workers[&WorkerId::new("a")].kv_headroom_tokens(&model),
        Some(16)
    );
}

#[test]
fn prefix_affinity_routes_a_prefix_home_until_it_is_overloaded() {
    let prompt = |group: u32| {
        let mut spec = spec(8, 2, None);
        spec.inputs = vec![Input::Tokens(
            [group; 4].into_iter().chain([7, 8, 9, 10]).collect(),
        )];
        spec
    };
    let mut coordinator = Coordinator::new(PrefixAffinityRouting::new(4, 1));
    for worker in ["a", "b", "c"] {
        coordinator.join(summary(worker, &[MODEL], 0, 0)).unwrap();
    }
    // Same 4-token prefix, different suffixes: one home, then spill as the
    // home gets more than one request ahead of the least-loaded worker.
    for _ in 0..3 {
        coordinator.submit(prompt(5), at(0)).unwrap();
    }
    let (iteration, _) = coordinator.plan(at(0)).unwrap().unwrap();
    let reasons = iteration
        .decision
        .routes
        .iter()
        .map(|route| route.reason)
        .collect::<Vec<_>>();
    assert_eq!(
        reasons,
        [
            RouteReason::PrefixAffinity { rank: 0 },
            RouteReason::PrefixAffinity { rank: 0 },
            RouteReason::PrefixAffinity { rank: 1 },
        ]
    );
    let homes = iteration
        .plan
        .actions
        .iter()
        .map(|action| match action {
            ClusterAction::Assign { worker, .. } => worker.clone(),
            other => panic!("unexpected {other:?}"),
        })
        .collect::<Vec<_>>();
    assert_eq!(homes[0], homes[1]);
    assert_ne!(homes[0], homes[2]);

    // Without a token prompt there is no prefix to follow.
    let mut text = spec(4, 2, None);
    text.inputs = vec![Input::Text("hello".into())];
    coordinator.submit(text, at(0)).unwrap();
    let (iteration, _) = coordinator.plan(at(0)).unwrap().unwrap();
    assert_eq!(iteration.decision.routes[0].reason, RouteReason::NoPrefix);
}

#[test]
fn routing_skips_workers_without_the_model_and_rejects_unserved_models() {
    let mut coordinator = Coordinator::new(RoundRobinRouting);
    coordinator.join(summary("a", &["other"], 0, 0)).unwrap();
    coordinator.join(summary("b", &[MODEL], 0, 0)).unwrap();
    let (served, _) = coordinator.submit(spec(4, 2, None), at(0)).unwrap();
    let mut unserved = spec(4, 2, None);
    unserved.model = ModelId::new("missing");
    let (missing, _) = coordinator.submit(unserved, at(0)).unwrap();

    let (iteration, output) = coordinator.plan(at(0)).unwrap().unwrap();
    assert_eq!(
        iteration.plan.actions,
        [
            ClusterAction::Assign {
                request: served,
                worker: WorkerId::new("b"),
            },
            ClusterAction::Reject {
                request: missing,
                reason: RejectReason::ModelNotServed {
                    model: ModelId::new("missing"),
                },
            },
        ]
    );
    assert!(matches!(
        coordinator.status(missing),
        Some(ClusterRequestStatus::Failed(_))
    ));
    assert!(matches!(
        output.client.as_slice(),
        [(id, ClientEvent::Failed { .. })] if *id == missing
    ));
    assert_eq!(coordinator.pending_len(), 0);
}

#[test]
fn requests_wait_while_only_draining_workers_host_the_model() {
    let mut cluster = LocalCluster::new(RoundRobinRouting);
    cluster
        .add_worker(WorkerHost::new("a", executor()), at(0))
        .unwrap();
    cluster
        .worker_mut(&WorkerId::new("a"))
        .unwrap()
        .set_health(WorkerHealth::Draining);
    // The coordinator learns of the change from the next summary.
    cluster.step(at(0)).unwrap();
    let id = cluster.submit(spec(4, 2, None), at(1)).unwrap();
    let step = cluster.step(at(1)).unwrap();
    let plan = step.cluster.unwrap().plan;
    assert!(matches!(
        plan.actions.as_slice(),
        [ClusterAction::Defer {
            reason: DeferReason::NoReadyWorker { .. },
            ..
        }]
    ));
    assert_eq!(
        cluster.coordinator().status(id),
        Some(&ClusterRequestStatus::Pending)
    );

    cluster
        .worker_mut(&WorkerId::new("a"))
        .unwrap()
        .set_health(WorkerHealth::Ready);
    cluster.step(at(2)).unwrap();
    let mut clock = 3;
    let mut finished = false;
    while !finished {
        let step = cluster.step(at(clock)).unwrap();
        finished = step.client.iter().any(|(event_id, event)| {
            *event_id == id && matches!(event, ClientEvent::Finished { .. })
        });
        clock += 1;
        assert!(clock < 100);
    }
}

#[test]
fn cancellation_reaches_pending_and_owned_requests() {
    let mut cluster = LocalCluster::new(RoundRobinRouting);
    cluster
        .add_worker(WorkerHost::new("a", executor()), at(0))
        .unwrap();
    let owned = cluster.submit(spec(4, 50, None), at(0)).unwrap();
    cluster.step(at(0)).unwrap();
    let pending = cluster.submit(spec(4, 50, None), at(1)).unwrap();

    cluster.cancel(pending, at(1)).unwrap();
    cluster.cancel(owned, at(1)).unwrap();
    let step = cluster.step(at(2)).unwrap();
    let cancelled = step
        .client
        .iter()
        .filter(|(_, event)| *event == ClientEvent::Cancelled)
        .map(|(id, _)| *id)
        .collect::<BTreeSet<_>>();
    assert_eq!(cancelled, BTreeSet::from([owned, pending]));
    assert_eq!(
        cluster.coordinator().status(owned),
        Some(&ClusterRequestStatus::Cancelled)
    );
    assert!(cluster.is_idle());
    let cancelled_owner = step.trace.iter().find_map(|record| match &record.event {
        ClusterTraceEvent::Cancelled { worker } if record.request == owned.get() => worker.clone(),
        _ => None,
    });
    assert_eq!(cancelled_owner, Some(WorkerId::new("a")));
    // Cancelling again is a no-op.
    cluster.cancel(owned, at(3)).unwrap();
}

#[test]
fn worker_failures_reach_the_client() {
    // The synthetic tokenizer rejects text, so preparation fails on the worker.
    let mut cluster = LocalCluster::new(RoundRobinRouting);
    cluster
        .add_worker(WorkerHost::new("a", executor()), at(0))
        .unwrap();
    let id = cluster
        .submit(RequestSpec::text(MODEL, "not tokens"), at(0))
        .unwrap();
    let step = cluster.step(at(0)).unwrap();
    assert!(matches!(
        step.client.as_slice(),
        [(event_id, ClientEvent::Failed { .. })] if *event_id == id
    ));
    assert!(step.trace.iter().any(|record| matches!(
        &record.event,
        ClusterTraceEvent::Failed { worker, .. } if worker.as_str() == "a"
    )));
}

#[test]
fn coordinator_rejects_reports_from_non_owners_and_handles_refusals() {
    let mut coordinator = Coordinator::new(RoundRobinRouting);
    coordinator.join(summary("a", &[MODEL], 0, 0)).unwrap();
    coordinator.join(summary("b", &[MODEL], 0, 0)).unwrap();
    assert_eq!(
        coordinator.join(summary("a", &[MODEL], 0, 0)),
        Err(CoordinatorError::DuplicateWorker(WorkerId::new("a")))
    );
    let (id, _) = coordinator.submit(spec(4, 2, None), at(0)).unwrap();
    let (_, output) = coordinator.plan(at(0)).unwrap().unwrap();
    assert!(matches!(
        output.commands.as_slice(),
        [(worker, WorkerCommand::Admit { request, .. })] if worker.as_str() == "a" && *request == id
    ));

    let wrong = coordinator.handle(
        &WorkerId::new("b"),
        WorkerMessage::Accepted {
            request: id,
            local: RequestId::new(1),
        },
        at(1),
    );
    assert_eq!(
        wrong,
        Err(CoordinatorError::UnexpectedMessage {
            from: WorkerId::new("b"),
            request: id,
        })
    );
    // Tokens before acceptance are out of order.
    assert!(
        coordinator
            .handle(
                &WorkerId::new("a"),
                WorkerMessage::Finished {
                    request: id,
                    token_ids: Vec::new(),
                    text: String::new(),
                    prompt_tokens: 0,
                },
                at(1),
            )
            .is_err()
    );

    let output = coordinator
        .handle(
            &WorkerId::new("a"),
            WorkerMessage::Refused {
                request: id,
                message: "full".into(),
            },
            at(1),
        )
        .unwrap();
    assert_eq!(
        coordinator.status(id),
        Some(&ClusterRequestStatus::Failed("full".into()))
    );
    assert_eq!(
        output.client,
        [(
            id,
            ClientEvent::Failed {
                message: "full".into()
            }
        )]
    );
}

fn planning_snapshot() -> (Coordinator, ClusterSnapshot, ClusterRequestId) {
    let mut coordinator = Coordinator::new(RoundRobinRouting);
    coordinator.join(summary("a", &[MODEL], 0, 0)).unwrap();
    let mut draining = summary("b", &[MODEL], 0, 0);
    draining.health = WorkerHealth::Draining;
    coordinator.join(draining).unwrap();
    coordinator.join(summary("c", &["other"], 0, 0)).unwrap();
    let (id, _) = coordinator.submit(spec(4, 2, None), at(0)).unwrap();
    let snapshot = coordinator.snapshot(at(0));
    (coordinator, snapshot, id)
}

#[test]
fn plan_validation_rejects_every_invalid_decision() {
    let (_, snapshot, id) = planning_snapshot();
    let plan = |actions: Vec<ClusterAction>| ClusterPlan {
        generation: snapshot.generation,
        actions,
    };
    let assign = |worker: &str| ClusterAction::Assign {
        request: id,
        worker: WorkerId::new(worker),
    };
    assert_eq!(plan(vec![assign("a")]).validate(&snapshot), Ok(()));

    let stale = ClusterPlan {
        generation: snapshot.generation + 1,
        actions: vec![assign("a")],
    };
    assert!(matches!(
        stale.validate(&snapshot),
        Err(ClusterPlanError::Stale { .. })
    ));
    assert_eq!(
        plan(Vec::new()).validate(&snapshot),
        Err(ClusterPlanError::MissingDecision(id))
    );
    assert_eq!(
        plan(vec![assign("a"), assign("a")]).validate(&snapshot),
        Err(ClusterPlanError::DuplicateDecision(id))
    );
    let unknown = ClusterRequestId::new(99);
    assert_eq!(
        plan(vec![ClusterAction::Assign {
            request: unknown,
            worker: WorkerId::new("a"),
        }])
        .validate(&snapshot),
        Err(ClusterPlanError::UnknownRequest(unknown))
    );
    assert!(matches!(
        plan(vec![assign("z")]).validate(&snapshot),
        Err(ClusterPlanError::UnknownWorker { .. })
    ));
    assert!(matches!(
        plan(vec![assign("b")]).validate(&snapshot),
        Err(ClusterPlanError::WorkerNotReady { .. })
    ));
    assert!(matches!(
        plan(vec![assign("c")]).validate(&snapshot),
        Err(ClusterPlanError::ModelNotHosted { .. })
    ));
    // `a` is ready, so neither reason holds.
    assert_eq!(
        plan(vec![ClusterAction::Defer {
            request: id,
            reason: DeferReason::NoReadyWorker {
                model: ModelId::new(MODEL),
            },
        }])
        .validate(&snapshot),
        Err(ClusterPlanError::ReasonDoesNotHold(id))
    );
    assert_eq!(
        plan(vec![ClusterAction::Reject {
            request: id,
            reason: RejectReason::ModelNotServed {
                model: ModelId::new(MODEL),
            },
        }])
        .validate(&snapshot),
        Err(ClusterPlanError::ReasonDoesNotHold(id))
    );
}

#[test]
fn an_invalid_plan_changes_nothing() {
    struct Broken;
    impl ClusterPolicy for Broken {
        fn descriptor(&self) -> ola_runtime::PolicyDescriptor {
            ola_runtime::PolicyDescriptor::new("broken")
        }
        fn plan(
            &self,
            snapshot: &ClusterSnapshot,
        ) -> Result<ola_cluster::ClusterDecision, ola_cluster::ClusterPolicyError> {
            Ok(ola_cluster::ClusterDecision {
                plan: ClusterPlan {
                    generation: snapshot.generation,
                    actions: Vec::new(),
                },
                record: Default::default(),
            })
        }
    }
    let mut coordinator = Coordinator::new(Broken);
    coordinator.join(summary("a", &[MODEL], 0, 0)).unwrap();
    let (id, _) = coordinator.submit(spec(4, 2, None), at(0)).unwrap();
    let before = coordinator.snapshot(at(0));
    assert_eq!(
        coordinator.plan(at(0)),
        Err(CoordinatorError::InvalidPlan(
            ClusterPlanError::MissingDecision(id)
        ))
    );
    assert_eq!(coordinator.snapshot(at(0)), before);
    assert_eq!(coordinator.status(id), Some(&ClusterRequestStatus::Pending));
}

fn token(request: ClusterRequestId, generated_tokens: usize) -> WorkerMessage {
    WorkerMessage::Token {
        request,
        token_id: 7,
        text: String::new(),
        generated_tokens,
        phase: ola_runtime::ExecutionPhase::Decode,
        at: at(1),
    }
}

fn accepted(request: ClusterRequestId, local: u64) -> WorkerMessage {
    WorkerMessage::Accepted {
        request,
        local: RequestId::new(local),
    }
}

/// Submits one request, plans it, and has its owner accept it.
fn running(coordinator: &mut Coordinator) -> (ClusterRequestId, WorkerId) {
    let (id, _) = coordinator.submit(spec(4, 8, None), at(0)).unwrap();
    let (_, output) = coordinator.plan(at(0)).unwrap().unwrap();
    let [(owner, WorkerCommand::Admit { .. })] = output.commands.as_slice() else {
        panic!("expected one admission");
    };
    let owner = owner.clone();
    coordinator.handle(&owner, accepted(id, 1), at(0)).unwrap();
    (id, owner)
}

fn migrated(request_spec: RequestSpec) -> MigratedRequest {
    MigratedRequest {
        spec: request_spec,
        prompt_tokens: vec![1, 2, 3, 4],
        generated_tokens: vec![7],
        token_times: vec![at(1)],
        preemptions: 0,
        session_state: None,
    }
}

#[test]
fn recovered_requests_do_not_replay_tokens_and_stale_copies_are_cancelled() {
    let mut coordinator = Coordinator::new(RoundRobinRouting);
    for worker in ["a", "b", "c"] {
        coordinator.join(summary(worker, &[MODEL], 0, 0)).unwrap();
    }
    let (id, first) = running(&mut coordinator);
    for generated in 1..=2 {
        let output = coordinator
            .handle(&first, token(id, generated), at(1))
            .unwrap();
        assert_eq!(output.client.len(), 1);
    }

    assert_eq!(coordinator.recover_worker(&first, at(2)).unwrap(), [id]);
    let (_, output) = coordinator.plan(at(2)).unwrap().unwrap();
    let [(second, WorkerCommand::Admit { .. })] = output.commands.as_slice() else {
        panic!("recovery did not reassign the request");
    };
    let second = second.clone();
    assert_ne!(second, first);
    coordinator.handle(&second, accepted(id, 1), at(3)).unwrap();
    // The new owner regenerates from the prompt; replays stay hidden.
    for generated in 1..=2 {
        let output = coordinator
            .handle(&second, token(id, generated), at(3))
            .unwrap();
        assert!(output.client.is_empty());
    }
    let output = coordinator.handle(&second, token(id, 3), at(3)).unwrap();
    assert!(matches!(
        output.client.as_slice(),
        [(
            _,
            ClientEvent::Token {
                generated_tokens: 3,
                ..
            }
        )]
    ));

    // The lost worker comes back still running its copy.
    let stale = coordinator.handle(&first, token(id, 3), at(4)).unwrap();
    assert!(stale.client.is_empty());
    assert_eq!(
        stale.commands,
        [(first.clone(), WorkerCommand::Cancel { request: id })]
    );
    let stale = coordinator
        .handle(
            &first,
            WorkerMessage::Finished {
                request: id,
                token_ids: vec![7; 3],
                text: String::new(),
                prompt_tokens: 0,
            },
            at(4),
        )
        .unwrap();
    assert!(stale.client.is_empty() && stale.commands.is_empty());
    assert!(matches!(
        coordinator.status(id),
        Some(ClusterRequestStatus::Running { worker, .. }) if worker == &second
    ));

    // A worker that never owned the request is still a protocol error.
    let never = ["a", "b", "c"]
        .into_iter()
        .map(WorkerId::new)
        .find(|worker| worker != &first && worker != &second)
        .unwrap();
    assert!(matches!(
        coordinator.handle(&never, token(id, 4), at(5)),
        Err(CoordinatorError::UnexpectedMessage { .. })
    ));
}

#[test]
fn a_committed_migration_leaves_the_source_load_accounting_intact() {
    let mut coordinator = Coordinator::new(RoundRobinRouting);
    coordinator.join(summary("a", &[MODEL], 0, 0)).unwrap();
    coordinator.join(summary("b", &[MODEL], 0, 0)).unwrap();
    let (id, source) = running(&mut coordinator);
    let target = WorkerId::new(if source.as_str() == "a" { "b" } else { "a" });
    let mut reported = summary(source.as_str(), &[MODEL], 0, 1);
    reported.admitted_requests = 1;
    coordinator
        .handle(&source, WorkerMessage::Summary(reported.clone()), at(1))
        .unwrap();

    coordinator
        .stage_migration(
            id,
            &source,
            target.clone(),
            migrated(spec(4, 8, None)),
            at(2),
        )
        .unwrap();
    coordinator.handle(&target, accepted(id, 1), at(2)).unwrap();
    coordinator.finish_migration(id, at(2));
    // Only the source can take new work now.
    let mut draining = summary(target.as_str(), &[MODEL], 0, 1);
    draining.health = WorkerHealth::Draining;
    coordinator
        .handle(&target, WorkerMessage::Summary(draining), at(3))
        .unwrap();
    reported.live_requests = 0;
    coordinator
        .handle(&source, WorkerMessage::Summary(reported), at(3))
        .unwrap();

    coordinator.submit(spec(4, 8, None), at(4)).unwrap();
    let (_, output) = coordinator.plan(at(4)).unwrap().unwrap();
    assert!(matches!(
        output.commands.as_slice(),
        [(worker, WorkerCommand::Admit { .. })] if worker == &source
    ));
    // The new admission is in flight until the source's summary counts it.
    assert_eq!(coordinator.snapshot(at(4)).workers[&source].in_flight, 1);
}

#[test]
fn losing_a_migration_target_rolls_the_request_back_to_its_source() {
    let mut coordinator = Coordinator::new(RoundRobinRouting);
    coordinator.join(summary("a", &[MODEL], 0, 0)).unwrap();
    coordinator.join(summary("b", &[MODEL], 0, 0)).unwrap();
    let (id, source) = running(&mut coordinator);
    let target = WorkerId::new(if source.as_str() == "a" { "b" } else { "a" });
    coordinator
        .stage_migration(
            id,
            &source,
            target.clone(),
            migrated(spec(4, 8, None)),
            at(1),
        )
        .unwrap();

    assert!(
        coordinator
            .recover_worker(&target, at(2))
            .unwrap()
            .is_empty()
    );
    assert!(matches!(
        coordinator.status(id),
        Some(ClusterRequestStatus::Running { worker, .. }) if worker == &source
    ));
    assert_eq!(
        coordinator.take_commands(),
        [(source, WorkerCommand::AbortMigration { request: id })]
    );
    assert!(coordinator.staged_migration(id).is_none());
}

#[test]
fn imported_requests_continue_their_text_stream() {
    let mut host = WorkerHost::new("target", executor());
    let request = ClusterRequestId::new(4);
    let accepted = host
        .handle(
            WorkerCommand::AdmitMigrated {
                request,
                state: migrated(spec(4, 3, None)),
                arrival_at: at(0),
            },
            at(0),
        )
        .unwrap();
    assert!(matches!(
        accepted.as_slice(),
        [WorkerMessage::Accepted { .. }]
    ));
    let mut deltas = Vec::new();
    let mut finished = None;
    while finished.is_none() {
        let (_, messages) = host.step(at(1)).unwrap().unwrap();
        for message in messages {
            match message {
                WorkerMessage::Token { text, .. } => deltas.push(text),
                WorkerMessage::Finished { text, .. } => finished = Some(text),
                _ => {}
            }
        }
    }
    // The source already streamed "7"; the new owner picks up after it.
    assert!(deltas[0].starts_with(' '), "{deltas:?}");
    assert_eq!(format!("7{}", deltas.concat()), finished.unwrap());
}
