//! Virtual-time cluster trials over the synthetic model.

use ola_bench::{
    ClusterSimulationConfig, KvGeometry, NetworkModel, SchedulingTrace, SimulationConfig,
    TraceParameters, TrialOutcome, run_cluster_trial, run_scheduling_trial, summarize_cluster_run,
};
use ola_cluster::{
    ClusterPolicy, ClusterTraceEvent, KvHeadroomRouting, LeastLoadedRouting, PowerOfTwoRouting,
    PrefixAffinityRouting, RoundRobinRouting,
};
use ola_runtime::{ComposedPolicy, MemoryAwareAdmission};

const PARAMETERS: TraceParameters = TraceParameters {
    seed: 3,
    requests: 60,
    mean_interarrival_micros: 400,
};

fn worker() -> SimulationConfig {
    SimulationConfig {
        max_active_requests: 4,
        ..SimulationConfig::default()
    }
}

fn policies() -> Vec<Box<dyn ClusterPolicy>> {
    vec![
        Box::new(RoundRobinRouting),
        Box::new(LeastLoadedRouting),
        Box::new(PowerOfTwoRouting::new(7)),
        Box::new(KvHeadroomRouting),
        Box::new(PrefixAffinityRouting::new(32, 2)),
    ]
}

#[test]
fn every_policy_reproduces_single_executor_outputs() {
    let trace = SchedulingTrace::mixed_short_long(PARAMETERS);
    let single = run_scheduling_trial(
        &trace,
        ComposedPolicy::default().with_admission(MemoryAwareAdmission),
        &worker(),
    )
    .unwrap();
    let config = ClusterSimulationConfig::uniform(3, worker(), NetworkModel::default());
    for policy in policies() {
        let name = policy.descriptor().name;
        let run = run_cluster_trial(&trace, policy, &config).unwrap();
        for (cluster, alone) in run.requests.iter().zip(&single.requests) {
            assert_eq!(cluster.outcome, TrialOutcome::Completed, "{name}");
            assert_eq!(cluster.output_tokens, alone.output_tokens, "{name}");
        }
        assert_eq!(
            run.workers
                .iter()
                .map(|worker| worker.routed)
                .sum::<usize>(),
            trace.requests.len()
        );
    }
}

#[test]
fn cluster_trials_are_reproducible() {
    let trace = SchedulingTrace::overload(PARAMETERS);
    let config = ClusterSimulationConfig::uniform(2, worker(), NetworkModel::default());
    for (first, second) in policies().into_iter().zip(policies()) {
        assert_eq!(
            run_cluster_trial(&trace, first, &config).unwrap(),
            run_cluster_trial(&trace, second, &config).unwrap()
        );
    }
}

#[test]
fn latencies_are_measured_at_the_coordinator() {
    let trace = SchedulingTrace::mixed_short_long(PARAMETERS);
    let slow = NetworkModel {
        latency_micros: 5_000,
        bytes_per_micro: 0,
    };
    let instant = ClusterSimulationConfig::uniform(1, worker(), NetworkModel::INSTANT);
    let delayed = ClusterSimulationConfig::uniform(1, worker(), slow);
    let fast = run_cluster_trial(&trace, RoundRobinRouting, &instant).unwrap();
    let late = run_cluster_trial(&trace, RoundRobinRouting, &delayed).unwrap();
    for (fast, late) in fast.requests.iter().zip(&late.requests) {
        // The admit and the first token each cross the network once.
        let fast_ttft = fast.first_token_micros.unwrap() - fast.arrival_micros;
        let late_ttft = late.first_token_micros.unwrap() - late.arrival_micros;
        assert!(late_ttft >= 2 * slow.latency_micros, "{late_ttft}");
        assert!(late_ttft >= fast_ttft);
    }
}

#[test]
fn bandwidth_delays_large_messages() {
    let trace = SchedulingTrace::mixed_short_long(TraceParameters {
        requests: 10,
        ..PARAMETERS
    });
    let narrow = NetworkModel {
        latency_micros: 0,
        bytes_per_micro: 1,
    };
    let run = run_cluster_trial(
        &trace,
        RoundRobinRouting,
        &ClusterSimulationConfig::uniform(1, worker(), narrow),
    )
    .unwrap();
    let first = &run.requests[0];
    // The admit carries the prompt and output limit at 4 bytes per token,
    // plus a 64-byte header, at one byte per microsecond.
    let admit = 64 + 4 * (first.prompt_tokens + first.max_new_tokens) as u64;
    assert!(first.first_token_micros.unwrap() - first.arrival_micros > admit);
    assert!(run.network_bytes > 0 && run.messages > 0);
}

#[test]
fn prefix_affinity_keeps_groups_together_and_hits_the_cache() {
    let trace = SchedulingTrace::prefix_groups(
        TraceParameters {
            requests: 80,
            ..PARAMETERS
        },
        4,
        64,
    );
    let config = ClusterSimulationConfig::uniform(
        4,
        SimulationConfig {
            prefix_cache: true,
            kv: KvGeometry {
                capacity_bytes: 4_096 * KvGeometry::default().bytes_per_token,
                ..KvGeometry::default()
            },
            ..worker()
        },
        NetworkModel::default(),
    );
    let isolated = vec![None; trace.requests.len()];
    let summary = |policy: Box<dyn ClusterPolicy>| {
        let run = run_cluster_trial(&trace, policy, &config).unwrap();
        summarize_cluster_run(&run, &isolated, 4)
    };
    let affinity = summary(Box::new(PrefixAffinityRouting::new(32, usize::MAX)));
    let round_robin = summary(Box::new(RoundRobinRouting));
    assert!(
        affinity.prefix_hit_tokens > round_robin.prefix_hit_tokens,
        "affinity {} vs round-robin {}",
        affinity.prefix_hit_tokens,
        round_robin.prefix_hit_tokens
    );
}

#[test]
fn every_route_is_traced_with_a_reason() {
    let trace = SchedulingTrace::mixed_short_long(PARAMETERS);
    let config = ClusterSimulationConfig::uniform(3, worker(), NetworkModel::default());
    for policy in policies() {
        let run = run_cluster_trial(&trace, policy, &config).unwrap();
        let routed = run
            .cluster_trace
            .iter()
            .filter(|record| {
                matches!(
                    &record.event,
                    ClusterTraceEvent::Routed {
                        reason: Some(_),
                        candidates,
                        ..
                    } if candidates.len() == 3
                )
            })
            .count();
        assert_eq!(routed, trace.requests.len());
    }
}

#[test]
#[ignore = "prints a comparison table; run with --ignored --nocapture"]
fn print_baseline_comparison() {
    for (name, trace) in [
        (
            "mixed",
            SchedulingTrace::mixed_short_long(TraceParameters {
                requests: 200,
                mean_interarrival_micros: 300,
                ..PARAMETERS
            }),
        ),
        (
            "groups",
            SchedulingTrace::prefix_groups(
                TraceParameters {
                    requests: 200,
                    mean_interarrival_micros: 300,
                    ..PARAMETERS
                },
                6,
                96,
            ),
        ),
    ] {
        let config = ClusterSimulationConfig::uniform(
            4,
            SimulationConfig {
                prefix_cache: true,
                ..worker()
            },
            NetworkModel::default(),
        );
        let isolated = vec![None; trace.requests.len()];
        for policy in policies() {
            let label = policy.descriptor().to_string();
            let run = run_cluster_trial(&trace, policy, &config).unwrap();
            let summary = summarize_cluster_run(&run, &isolated, 4);
            println!(
                "{name:7} {label:48} p50 {:>7} p95 {:>7} ttft95 {:>7} hits {:>6} imb {:.2} routed {:?}",
                summary.scheduling.completion.p50_micros,
                summary.scheduling.completion.p95_micros,
                summary.scheduling.time_to_first_token.p95_micros,
                summary.prefix_hit_tokens,
                summary.busy_imbalance,
                summary.routed,
            );
        }
    }
}
