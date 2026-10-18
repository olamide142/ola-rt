//! Workload replay is deterministic under KV pressure, preemption, eviction,
//! prefix reuse, and cancellation.

use ola_bench::{
    KvGeometry, Perturbation, SchedulingRun, SchedulingTrace, SimulationConfig, TraceParameters,
    run_perturbed_trial,
};
use ola_runtime::{ComposedPolicy, NewestFirst, WatermarkAdmission};

fn pressured(prefix_cache: bool, capacity_tokens: u64) -> SimulationConfig {
    let geometry = KvGeometry::default();
    SimulationConfig {
        max_active_requests: 24,
        // A pool small enough that preemption and eviction both happen.
        kv: KvGeometry {
            capacity_bytes: capacity_tokens * geometry.bytes_per_token,
            ..geometry
        },
        prefix_cache,
        ..SimulationConfig::default()
    }
}

fn policy(headroom_tokens: usize) -> ComposedPolicy {
    ComposedPolicy::default()
        .with_admission(WatermarkAdmission { headroom_tokens })
        .with_preemption(NewestFirst)
}

fn parameters(seed: u64) -> TraceParameters {
    TraceParameters {
        seed,
        requests: 80,
        mean_interarrival_micros: 500,
    }
}

fn run(
    trace: &SchedulingTrace,
    config: &SimulationConfig,
    headroom_tokens: usize,
    cancel: bool,
) -> SchedulingRun {
    let perturbations = if cancel {
        vec![
            Perturbation::Cancel {
                at_micros: 20_000,
                trace_index: 3,
            },
            Perturbation::Cancel {
                at_micros: 40_000,
                trace_index: 11,
            },
        ]
    } else {
        Vec::new()
    };
    run_perturbed_trial(trace, policy(headroom_tokens), config, &perturbations).unwrap()
}

#[test]
fn replaying_a_pressured_workload_reproduces_every_iteration() {
    // (trace, prefix cache, pool tokens, headroom, must preempt)
    let cases = [
        (
            SchedulingTrace::early_stopping(parameters(7)),
            false,
            768,
            16,
            true,
        ),
        // Long outputs outgrow the pool while equal-marker prompts share
        // blocks, so preemption and reuse interleave.
        (
            SchedulingTrace::early_stopping(parameters(7)),
            true,
            768,
            16,
            true,
        ),
        // Reservations count a shared prefix per request, so sharing alone
        // never over-commits this pool; replay must still match.
        (
            SchedulingTrace::shared_prefix(parameters(7), 96),
            true,
            320,
            0,
            false,
        ),
    ];
    for (trace, prefix_cache, capacity_tokens, headroom_tokens, preempts) in cases {
        for cancel in [false, true] {
            let config = pressured(prefix_cache, capacity_tokens);
            let first = run(&trace, &config, headroom_tokens, cancel);
            let second = run(&trace, &config, headroom_tokens, cancel);

            assert_eq!(first, second, "{} replay diverged", trace.name);
            let preemptions = first
                .iterations
                .iter()
                .map(|iteration| iteration.preemptions)
                .sum::<usize>();
            assert_eq!(
                preemptions > 0,
                preempts,
                "{} (cache {prefix_cache}): {preemptions} preemptions",
                trace.name
            );
            if prefix_cache {
                let reused = first.iterations.last().unwrap().prefix_hit_tokens;
                assert!(reused > 0, "{}: prefixes must be reused", trace.name);
            }
        }
    }
}
