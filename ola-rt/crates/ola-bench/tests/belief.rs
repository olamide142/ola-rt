//! Remaining-output beliefs size reservations but never weaken physical
//! KV safety: every plan is still validated, an over-committed pool is
//! recovered by preemption, and outputs are unchanged.

use ola_bench::{
    KvGeometry, SchedulingTrace, SimulationConfig, TraceParameters, TrialOutcome,
    run_scheduling_trial,
};
use ola_runtime::{
    BeliefPolicy, ComposedPolicy, ExecutionPolicy, MemoryAwareAdmission, NewestFirst,
    OutputBeliefConfig, OutputLengthBelief, QuantileAdmission,
};

fn config() -> SimulationConfig {
    let geometry = KvGeometry::default();
    SimulationConfig {
        max_active_requests: 24,
        kv: KvGeometry {
            capacity_bytes: 2_048 * geometry.bytes_per_token,
            ..geometry
        },
        ..SimulationConfig::default()
    }
}

fn belief_policy(min_samples: usize) -> BeliefPolicy {
    BeliefPolicy::new(
        OutputLengthBelief::new(OutputBeliefConfig {
            window: 32,
            min_samples,
            ..OutputBeliefConfig::default()
        }),
        |belief| -> Box<dyn ExecutionPolicy> {
            Box::new(
                ComposedPolicy::default()
                    .with_admission(QuantileAdmission::new(belief, 0.5))
                    .with_preemption(NewestFirst),
            )
        },
    )
}

/// Early stoppers teach the belief that outputs are short; then every
/// request runs to its limit, so the belief under-reserves.
fn shifted() -> SchedulingTrace {
    let parameters = TraceParameters {
        seed: 11,
        requests: 40,
        mean_interarrival_micros: 1_000,
    };
    SchedulingTrace::early_stopping(parameters).followed_by(
        SchedulingTrace::run_to_limit(TraceParameters {
            seed: 12,
            ..parameters
        }),
        "shifted",
        1_000,
    )
}

#[test]
fn a_wrong_belief_is_recovered_by_preemption_without_changing_outputs() {
    let trace = shifted();
    let believed = run_scheduling_trial(&trace, belief_policy(8), &config()).unwrap();
    let worst_case = run_scheduling_trial(
        &trace,
        ComposedPolicy::default().with_admission(MemoryAwareAdmission),
        &config(),
    )
    .unwrap();

    let preemptions = believed
        .iterations
        .iter()
        .map(|iteration| iteration.preemptions)
        .sum::<usize>();
    assert!(preemptions > 0, "the belief never under-reserved");
    // Every plan passed validation (a rejected plan aborts the trial), no
    // request failed, and greedy outputs equal the worst-case run's.
    for (believed, worst) in believed.requests.iter().zip(&worst_case.requests) {
        assert_eq!(believed.outcome, TrialOutcome::Completed);
        assert_eq!(believed.output_tokens, worst.output_tokens);
    }
    // Physical KV use never exceeded the pool.
    let capacity = config().kv.capacity_bytes;
    assert!(
        believed
            .iterations
            .iter()
            .all(|iteration| iteration.kv_allocated_bytes <= capacity)
    );
}

#[test]
fn without_a_usable_belief_admission_is_the_worst_case() {
    let trace = shifted();
    // No class ever reaches this many samples.
    let cold = run_scheduling_trial(&trace, belief_policy(10_000), &config()).unwrap();
    let worst_case = run_scheduling_trial(
        &trace,
        ComposedPolicy::default()
            .with_admission(MemoryAwareAdmission)
            .with_preemption(NewestFirst),
        &config(),
    )
    .unwrap();
    assert_eq!(cold.requests, worst_case.requests);
    assert_eq!(cold.iterations, worst_case.iterations);
}
