//! Behaviour of the driver thread and its handles.

mod support;

use std::{thread, time::Duration};

use ola_core::{RequestId, RequestSpec};
use ola_runtime::{ExecutionPhase, RequestStatus, RuntimeError};
use ola_serve::{DriverConfig, RequestEvent, ServeError};
use support::{FAIL_PROMPT, Gate, REPLY, driver, request};

#[test]
fn streams_text_deltas_that_reassemble_the_final_output() {
    let gate = Gate::open();
    let (driver, _) = driver(&gate, 1, DriverConfig::default());
    let mut handle = driver.handle().submit(request("hi")).wait().unwrap();

    let mut text = String::new();
    let mut phases = Vec::new();
    let output = loop {
        match handle.blocking_next().unwrap() {
            RequestEvent::Token {
                text: delta, phase, ..
            } => {
                assert!(!delta.contains('\u{FFFD}'), "split character in {delta:?}");
                text.push_str(&delta);
                phases.push(phase);
            }
            RequestEvent::Finished(output) => break output,
            other => panic!("unexpected {other:?}"),
        }
    };

    assert_eq!(output.text, REPLY);
    assert_eq!(text, output.text);
    assert_eq!(phases[0], ExecutionPhase::Prefill);
    assert!(
        phases[1..]
            .iter()
            .all(|phase| *phase == ExecutionPhase::Decode)
    );
    assert_eq!(output.request_id, handle.id());
    assert!(output.timing.completed_at.is_some());
    assert_eq!(handle.blocking_next(), None);
    let executor = driver.shutdown().unwrap();
    assert_eq!(executor.result(output.request_id), None, "result was moved");
}

#[test]
fn async_clients_submit_stream_and_collect_results() {
    let gate = Gate::open();
    let (driver, _) = driver(&gate, 2, DriverConfig::default());
    let runtime = driver.handle();

    let outputs = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap()
        .block_on(async {
            let first = runtime.submit(request("a")).await.unwrap();
            let mut second = runtime.submit(request("b")).await.unwrap();
            let mut tokens = 0;
            let second = loop {
                match second.next().await.unwrap() {
                    RequestEvent::Token { .. } => tokens += 1,
                    RequestEvent::Finished(output) => break output,
                    other => panic!("unexpected {other:?}"),
                }
            };
            assert_eq!(tokens, second.token_ids.len());
            let snapshot = runtime.status(second.request_id).await.unwrap();
            assert_eq!(snapshot.status, RequestStatus::Completed);
            (first.result().await.unwrap(), second)
        });

    assert_eq!(outputs.0.text, REPLY);
    assert_eq!(outputs.1.text, REPLY);
    assert_eq!(runtime.in_flight(), 0);
}

#[test]
fn invalid_and_failing_requests_report_distinct_errors() {
    let gate = Gate::open();
    let (driver, _) = driver(&gate, 1, DriverConfig::default());
    let runtime = driver.handle();

    assert_eq!(
        runtime
            .submit(RequestSpec::text("missing", "hi"))
            .wait()
            .err(),
        Some(ServeError::UnknownModel("missing".into()))
    );
    let mut empty = request("hi");
    empty.inputs.clear();
    assert!(matches!(
        runtime.submit(empty).wait().err(),
        Some(ServeError::Invalid(RuntimeError::InvalidRequest(_)))
    ));
    let failed = runtime
        .submit(request(FAIL_PROMPT))
        .wait()
        .unwrap()
        .blocking_result();
    assert!(
        matches!(&failed, Err(ServeError::Execution(message)) if message.contains("scripted failure")),
        "{failed:?}"
    );
    assert_eq!(runtime.in_flight(), 0);
    assert_eq!(
        runtime.status(RequestId::new(99)).wait(),
        Err(ServeError::UnknownRequest(RequestId::new(99)))
    );
}

#[test]
fn submissions_beyond_the_limit_are_refused_immediately() {
    let gate = Gate::closed();
    let config = DriverConfig {
        max_in_flight: 2,
        ..DriverConfig::default()
    };
    let (driver, _) = driver(&gate, 1, config);
    let runtime = driver.handle();

    // The driver blocks inside the first prefill, yet the third submission
    // is refused without waiting for it.
    let first = runtime.submit(request("a"));
    let second = runtime.submit(request("b"));
    assert_eq!(
        runtime.submit(request("c")).wait().err(),
        Some(ServeError::Overloaded { limit: 2 })
    );
    assert_eq!(runtime.in_flight(), 2);

    gate.open_fully();
    first.wait().unwrap().blocking_result().unwrap();
    second.wait().unwrap().blocking_result().unwrap();
    assert_eq!(runtime.in_flight(), 0);
    runtime
        .submit(request("d"))
        .wait()
        .unwrap()
        .blocking_result()
        .unwrap();
}

#[test]
fn cancelling_ends_the_stream_and_releases_the_session() {
    let gate = Gate::closed();
    let (driver, _) = driver(&gate, 1, DriverConfig::default());
    let runtime = driver.handle();
    gate.permit(2);
    let mut handle = runtime.submit(request("hi")).wait().unwrap();
    assert!(matches!(
        handle.blocking_next(),
        Some(RequestEvent::Token { .. })
    ));
    assert!(matches!(
        handle.blocking_next(),
        Some(RequestEvent::Token { .. })
    ));

    // Once the driver is blocked in the third model call, the cancel waits
    // for that call to finish.
    gate.wait_until_blocked();
    let cancel = handle.cancel();
    gate.permit(1);
    cancel.wait().unwrap();
    let mut rest = Vec::new();
    while let Some(event) = handle.blocking_next() {
        rest.push(event);
    }
    assert!(matches!(
        rest.as_slice(),
        [RequestEvent::Token { .. }, RequestEvent::Cancelled]
    ));
    assert_eq!(
        handle.status().wait().unwrap().status,
        RequestStatus::Cancelled
    );
    assert_eq!(handle.cancel().wait(), Ok(()), "cancelling again succeeds");

    gate.open_fully();
    let executor = driver.shutdown().unwrap();
    assert!(!executor.has_active_request());
    assert_eq!(executor.state().memory.kv_cache_bytes, 0);
}

#[test]
fn dropping_a_handle_cancels_its_request() {
    let gate = Gate::closed();
    let (driver, _) = driver(&gate, 1, DriverConfig::default());
    let runtime = driver.handle();
    gate.permit(1);
    let mut handle = runtime.submit(request("hi")).wait().unwrap();
    let id = handle.id();
    assert!(matches!(
        handle.blocking_next(),
        Some(RequestEvent::Token { .. })
    ));

    drop(handle);
    gate.permit(1);

    assert_eq!(
        runtime.status(id).wait().unwrap().status,
        RequestStatus::Cancelled
    );
    assert_eq!(runtime.in_flight(), 0);
    gate.open_fully();
}

#[test]
fn an_abandoned_submission_is_cancelled_before_it_runs() {
    let gate = Gate::closed();
    let (driver, _) = driver(&gate, 1, DriverConfig::default());
    let runtime = driver.handle();
    gate.permit(1);
    let first = runtime.submit(request("a")).wait().unwrap();
    gate.wait_until_blocked();

    // Submitted while the driver is blocked, then dropped unresolved.
    drop(runtime.submit(request("b")));
    gate.open_fully();

    let abandoned = RequestId::new(first.id().get() + 1);
    first.blocking_result().unwrap();
    let snapshot = runtime.status(abandoned).wait().unwrap();
    assert_eq!(snapshot.status, RequestStatus::Cancelled);
    assert_eq!(snapshot.timing.started_at, None, "it never ran");
}

#[test]
fn a_detached_request_runs_to_completion() {
    let gate = Gate::closed();
    let (driver, _) = driver(&gate, 1, DriverConfig::default());
    let runtime = driver.handle();
    let id = runtime.submit(request("hi")).wait().unwrap().detach();
    gate.open_fully();

    let watcher = runtime.submit(request("next")).wait().unwrap();
    watcher.blocking_result().unwrap();
    assert_eq!(
        runtime.status(id).wait().unwrap().status,
        RequestStatus::Completed
    );
}

#[test]
fn shutdown_ends_unfinished_requests_and_refuses_new_ones() {
    let gate = Gate::closed();
    let (driver, _) = driver(&gate, 1, DriverConfig::default());
    let runtime = driver.handle();
    gate.permit(1);
    let mut running = runtime.submit(request("a")).wait().unwrap();
    assert!(matches!(
        running.blocking_next(),
        Some(RequestEvent::Token { .. })
    ));
    gate.wait_until_blocked();
    // The driver may or may not accept this before it stops; either way the
    // request ends with `ShuttingDown`.
    let queued = runtime.submit(request("b"));

    let stopping = thread::spawn(move || driver.shutdown());
    while runtime.is_accepting() {
        thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(
        runtime.submit(request("c")).wait().err(),
        Some(ServeError::ShuttingDown)
    );
    gate.open_fully();
    let executor = stopping.join().unwrap().unwrap();

    assert_eq!(running.blocking_result(), Err(ServeError::ShuttingDown));
    assert_eq!(
        queued.wait().and_then(|handle| handle.blocking_result()),
        Err(ServeError::ShuttingDown)
    );
    assert_eq!(
        runtime.status(RequestId::new(1)).wait(),
        Err(ServeError::ShuttingDown)
    );
    assert!(!executor.has_active_request());
    assert_eq!(executor.state().queued_len(), 0);
}

#[test]
fn tokens_are_stamped_when_their_model_call_finishes() {
    let gate = Gate::closed();
    let (driver, clock) = driver(&gate, 1, DriverConfig::default());
    let mut handle = driver.handle().submit(request("hi")).wait().unwrap();

    // Each call is released 100 µs after the previous one, so its token
    // must carry the later time rather than its iteration's start.
    // The reply's bytes, then end-of-sequence.
    let calls = REPLY.len() + 1;
    for _ in 0..calls {
        gate.wait_until_blocked();
        clock.advance(Duration::from_micros(100));
        gate.permit(1);
        assert!(matches!(
            handle.blocking_next(),
            Some(RequestEvent::Token { .. })
        ));
    }
    let Some(RequestEvent::Finished(output)) = handle.blocking_next() else {
        panic!("expected the result after the last token");
    };

    assert_eq!(output.token_times.len(), output.token_ids.len());
    let expected = (1..=output.token_times.len() as u64)
        .map(|step| ola_core::Timestamp::from_micros(step * 100))
        .collect::<Vec<_>>();
    assert_eq!(output.token_times, expected);
    assert_eq!(output.timing.first_token_at, Some(expected[0]));
    assert_eq!(output.timing.completed_at, expected.last().copied());
    assert_eq!(output.timing.started_at, Some(ola_core::Timestamp::ZERO));
}
