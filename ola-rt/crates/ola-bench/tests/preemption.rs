//! Preemption by recompute on the synthetic model's real block pool.

use ola_bench::{KvGeometry, synthetic_artifact, synthetic_prompt, synthetic_token};
use ola_core::{Input, ModelDescriptor, RequestId, RequestSpec, Timestamp};
use ola_observe::{TraceEvent, Tracer};
use ola_runtime::{
    ComposedPolicy, Executor, ExecutorEvent, NewestFirst, PlanDivergence, RequestStatus, Runtime,
};

const PROMPT: usize = 4;
const OUTPUT: usize = 10;

/// Six 4-token blocks: three requests of 4 + 9 cached tokens (4 blocks each)
/// cannot all finish together.
fn executor(policy: ComposedPolicy) -> (Executor, Vec<RequestId>) {
    let mut executor = Executor::with_max_active_requests(Runtime::default(), 3)
        .unwrap()
        .with_policy(policy);
    executor
        .register_model(
            ModelDescriptor::language("synthetic", "synthetic"),
            synthetic_artifact(KvGeometry {
                bytes_per_token: 8,
                tokens_per_block: 4,
                capacity_bytes: 6 * 4 * 8,
            }),
        )
        .unwrap();
    let ids = (0..3)
        .map(|index| {
            let mut spec = RequestSpec::text("synthetic", "");
            spec.inputs = vec![Input::Tokens(synthetic_prompt(PROMPT, None))];
            spec.generation.max_new_tokens = OUTPUT;
            executor
                .submit(spec, Timestamp::from_micros(index))
                .unwrap()
        })
        .collect();
    (executor, ids)
}

/// Steps until idle, returning every iteration.
fn run(executor: &mut Executor) -> Vec<ola_runtime::ExecutorIteration> {
    let mut iterations = Vec::new();
    for step in 1..=200 {
        let iteration = executor
            .step_iteration(Timestamp::from_micros(step * 10))
            .unwrap();
        let idle = iteration.is_idle();
        iterations.push(iteration);
        if idle {
            return iterations;
        }
    }
    panic!("the workload did not finish");
}

fn expected() -> Vec<u32> {
    (1..=OUTPUT)
        .map(|step| synthetic_token(PROMPT, step))
        .collect()
}

#[test]
fn preempted_requests_resume_and_match_uninterrupted_output() {
    let (mut executor, ids) = executor(ComposedPolicy::default().with_preemption(NewestFirst));
    let iterations = run(&mut executor);
    let events = iterations.iter().flat_map(|iteration| &iteration.events);

    let preempted = events
        .clone()
        .filter(|event| matches!(event, ExecutorEvent::Preempted { .. }))
        .count();
    assert!(preempted > 0, "the pool must force at least one preemption");
    assert!(
        !events
            .clone()
            .any(|event| matches!(event, ExecutorEvent::Failed { .. }))
    );
    assert!(iterations.iter().all(|iteration| {
        !iteration
            .divergences
            .iter()
            .any(|divergence| matches!(divergence, PlanDivergence::KvCapacityExhausted { .. }))
    }));
    for id in &ids {
        let result = executor.result(*id).unwrap();
        assert_eq!(result.token_ids, expected(), "request {id}");
        assert_eq!(result.token_times.len(), OUTPUT);
        assert_eq!(result.metrics.prompt_tokens, PROMPT);
        assert_eq!(
            executor.state().request(*id).unwrap().status,
            RequestStatus::Completed
        );
    }
    let total_preemptions = ids
        .iter()
        .map(|id| executor.state().request(*id).unwrap().preemptions)
        .sum::<u32>();
    assert_eq!(total_preemptions as usize, preempted);
    // Newest first: the oldest request is never preempted.
    assert_eq!(executor.state().request(ids[0]).unwrap().preemptions, 0);
    assert_eq!(executor.state().memory.kv_cache_bytes, 0);
}

#[test]
fn without_preemption_the_same_workload_runs_out_of_blocks() {
    let (mut executor, _) = executor(ComposedPolicy::default());
    let iterations = run(&mut executor);

    assert!(iterations.iter().any(|iteration| {
        iteration
            .divergences
            .iter()
            .any(|divergence| matches!(divergence, PlanDivergence::KvCapacityExhausted { .. }))
    }));
    assert!(
        iterations
            .iter()
            .flat_map(|iteration| &iteration.events)
            .any(|event| matches!(event, ExecutorEvent::Failed { .. }))
    );
}

#[test]
fn a_request_cancelled_while_preempted_releases_everything() {
    let (mut executor, ids) = executor(ComposedPolicy::default().with_preemption(NewestFirst));
    let mut victim = None;
    for step in 1..=200 {
        let iteration = executor
            .step_iteration(Timestamp::from_micros(step * 10))
            .unwrap();
        victim = iteration.events.iter().find_map(|event| match event {
            ExecutorEvent::Preempted { request_id, .. } => Some(*request_id),
            _ => None,
        });
        if victim.is_some() {
            break;
        }
    }
    let victim = victim.expect("a preemption happens");
    assert_eq!(
        executor.state().request(victim).unwrap().status,
        RequestStatus::Queued
    );

    executor
        .cancel(victim, Timestamp::from_micros(5_000))
        .unwrap();
    let mut step = 501;
    while !executor
        .step_iteration(Timestamp::from_micros(step * 10))
        .unwrap()
        .is_idle()
    {
        step += 1;
    }

    assert_eq!(
        executor.state().request(victim).unwrap().status,
        RequestStatus::Cancelled
    );
    for id in ids.iter().filter(|id| **id != victim) {
        assert_eq!(executor.result(*id).unwrap().token_ids, expected());
    }
    assert_eq!(executor.state().memory.kv_cache_bytes, 0);
}

#[test]
fn traces_show_preemption_and_output_continuing_without_repeats() {
    let (mut executor, _) = executor(ComposedPolicy::default().with_preemption(NewestFirst));
    let mut tracer = Tracer::default();
    let mut records = Vec::new();
    for step in 1..=200 {
        let iteration = executor
            .step_iteration(Timestamp::from_micros(step * 10))
            .unwrap();
        records.extend(tracer.observe_iteration(&iteration, executor.state()));
        if iteration.is_idle() {
            break;
        }
    }
    let victim = records
        .iter()
        .find(|record| matches!(record.event, TraceEvent::Preempted { .. }))
        .expect("a preemption is traced")
        .request;
    let mine = records
        .iter()
        .filter(|record| record.request == victim)
        .collect::<Vec<_>>();
    let indices = mine
        .iter()
        .filter_map(|record| match record.event {
            TraceEvent::Token { index, .. } => Some(index),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(indices, (1..=OUTPUT).collect::<Vec<_>>());
    // Admitted twice: before and after the preemption.
    let admissions = mine
        .iter()
        .filter(|record| matches!(&record.event, TraceEvent::Admission { outcome, .. } if outcome == "admitted"))
        .count();
    assert!(admissions >= 2, "{mine:?}");
    assert!(matches!(mine.last().unwrap().event, TraceEvent::Completed));
}
