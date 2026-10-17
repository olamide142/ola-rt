//! Prefix-cache reuse on the synthetic model's block pool.

use ola_bench::{
    KvGeometry, synthetic_artifact, synthetic_artifact_with_prefix_cache, synthetic_shared_prompt,
};
use ola_core::{Input, ModelDescriptor, RequestId, RequestSpec, Timestamp};
use ola_model::CausalLanguageModel;
use ola_observe::{TraceEvent, Tracer};
use ola_runtime::{
    BatchPhase, ComposedPolicy, Executor, ExecutorIteration, MemoryAwareAdmission, NewestFirst,
    Runtime,
};

/// Sixteen 4-token blocks.
const KV: KvGeometry = KvGeometry {
    bytes_per_token: 8,
    tokens_per_block: 4,
    capacity_bytes: 16 * 4 * 8,
};

fn executor(model: CausalLanguageModel, policy: ComposedPolicy) -> Executor {
    let mut executor = Executor::with_max_active_requests(Runtime::default(), 4)
        .unwrap()
        .with_policy(policy)
        .with_prefill_chunk_size(64)
        .unwrap();
    executor
        .register_model(ModelDescriptor::language("synthetic", "synthetic"), model)
        .unwrap();
    executor
}

/// A 14-token prompt sharing its first 9 tokens (two full blocks) with
/// every other prompt built here.
fn spec(seed: u64) -> RequestSpec {
    let mut spec = RequestSpec::text("synthetic", "");
    spec.inputs = vec![Input::Tokens(synthetic_shared_prompt(14, 9, seed))];
    spec.generation.max_new_tokens = 6;
    spec
}

fn run(executor: &mut Executor, from: u64) -> Vec<ExecutorIteration> {
    let mut iterations = Vec::new();
    for step in from..from + 200 {
        let iteration = executor
            .step_iteration(Timestamp::from_micros(step))
            .unwrap();
        let idle = iteration.is_idle();
        iterations.push(iteration);
        if idle {
            return iterations;
        }
    }
    panic!("did not finish");
}

/// Input tokens of `request`'s prefill invocations.
fn prefill_tokens(iterations: &[ExecutorIteration], request: RequestId) -> usize {
    iterations
        .iter()
        .flat_map(|iteration| &iteration.execution_batches)
        .filter(|batch| batch.phase == BatchPhase::Prefill && batch.request_ids == [request])
        .map(|batch| batch.input_tokens)
        .sum()
}

fn policy() -> ComposedPolicy {
    ComposedPolicy::default().with_admission(MemoryAwareAdmission)
}

#[test]
fn a_later_request_skips_the_cached_prefix_and_matches_uncached_output() {
    let model = synthetic_artifact_with_prefix_cache(KV);
    let mut cached = executor(model.clone(), policy());
    let first = cached.submit(spec(1), Timestamp::ZERO).unwrap();
    run(&mut cached, 1);
    let second = cached.submit(spec(2), Timestamp::from_micros(300)).unwrap();
    let iterations = run(&mut cached, 301);

    // Two shared blocks (8 tokens) are reused; 6 of 14 are computed.
    assert_eq!(prefill_tokens(&iterations, second), 6);
    assert_eq!(model.kv_pool_status().unwrap().prefix_hit_tokens, 8);

    let mut plain = executor(synthetic_artifact(KV), policy());
    let plain_first = plain.submit(spec(1), Timestamp::ZERO).unwrap();
    let plain_second = plain.submit(spec(2), Timestamp::ZERO).unwrap();
    run(&mut plain, 1);
    assert_eq!(
        cached.result(first).unwrap().token_ids,
        plain.result(plain_first).unwrap().token_ids
    );
    assert_eq!(
        cached.result(second).unwrap().token_ids,
        plain.result(plain_second).unwrap().token_ids
    );
}

#[test]
fn concurrent_requests_share_live_prefix_blocks() {
    let model = synthetic_artifact_with_prefix_cache(KV);
    let mut executor = executor(model.clone(), policy());
    let first = executor.submit(spec(1), Timestamp::ZERO).unwrap();
    executor.step_iteration(Timestamp::from_micros(1)).unwrap();
    // The first request is decoding; its prompt blocks are live and shared.
    let second = executor.submit(spec(2), Timestamp::from_micros(1)).unwrap();
    let mut tracer = Tracer::default();
    let iteration = executor.step_iteration(Timestamp::from_micros(2)).unwrap();
    let records = tracer.observe_iteration(&iteration, executor.state());
    run(&mut executor, 3);

    assert_eq!(prefill_tokens(std::slice::from_ref(&iteration), second), 6);
    assert!(records.iter().any(|record| record.request == second.get()
        && matches!(
            record.event,
            TraceEvent::Admission {
                cached_prefix_tokens: Some(8),
                ..
            }
        )));
    assert_eq!(
        executor.result(first).unwrap().token_ids.len(),
        executor.result(second).unwrap().token_ids.len()
    );
    // Nothing was evicted: the shared blocks were live, not cached.
    assert_eq!(model.kv_pool_status().unwrap().evictions, 0);
}

#[test]
fn a_preempted_request_resumes_from_its_own_cached_blocks() {
    // Six blocks: three requests cannot all finish together.
    let kv = KvGeometry {
        capacity_bytes: 6 * 4 * 8,
        ..KV
    };
    let model = synthetic_artifact_with_prefix_cache(kv);
    let mut executor = executor(
        model.clone(),
        ComposedPolicy::default().with_preemption(NewestFirst),
    );
    let ids = (0..3)
        .map(|index| {
            let mut spec = spec(10 + index);
            spec.generation.max_new_tokens = 10;
            executor.submit(spec, Timestamp::ZERO).unwrap()
        })
        .collect::<Vec<_>>();
    let iterations = run(&mut executor, 1);

    let preempted = iterations
        .iter()
        .flat_map(|iteration| &iteration.events)
        .filter(|event| matches!(event, ola_runtime::ExecutorEvent::Preempted { .. }))
        .count();
    assert!(preempted > 0);
    let mut plain = self::executor(
        synthetic_artifact(KvGeometry {
            capacity_bytes: 64 * 4 * 8,
            ..KV
        }),
        policy(),
    );
    let reference = (0..3)
        .map(|index| {
            let mut spec = spec(10 + index);
            spec.generation.max_new_tokens = 10;
            plain.submit(spec, Timestamp::ZERO).unwrap()
        })
        .collect::<Vec<_>>();
    run(&mut plain, 1);
    for (id, reference) in ids.iter().zip(&reference) {
        assert_eq!(
            executor.result(*id).unwrap().token_ids,
            plain.result(*reference).unwrap().token_ids
        );
    }
    // Resumes and later admissions reused cached blocks.
    assert!(model.kv_pool_status().unwrap().prefix_hit_tokens > 0);
}
