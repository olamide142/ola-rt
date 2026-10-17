//! Request traces kept by the driver, written to a sink, and served.

mod support;

use std::{
    io::{self, Write},
    sync::{Arc, Mutex},
};

use ola_core::{ManualClock, RequestId, Timestamp};
use ola_observe::{TraceEvent, TraceRecord, trace::read_jsonl};
use ola_serve::{Driver, DriverConfig, RequestEvent, ServeError};
use support::{Gate, REPLY, executor, request};

/// A sink tests can read back.
#[derive(Clone, Default)]
struct Shared(Arc<Mutex<Vec<u8>>>);

impl Write for Shared {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.lock().unwrap().write(bytes)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

struct Broken;

impl Write for Broken {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> {
        Err(io::Error::other("disk full"))
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn traced(config: DriverConfig) -> DriverConfig {
    DriverConfig {
        trace: true,
        ..config
    }
}

fn kinds(records: &[TraceRecord]) -> Vec<&'static str> {
    records
        .iter()
        .map(|record| match record.event {
            TraceEvent::Arrived { .. } => "arrived",
            TraceEvent::Ranked { .. } => "ranked",
            TraceEvent::Admission { .. } => "admission",
            TraceEvent::BackfillSkipped => "backfill_skipped",
            TraceEvent::Selection { .. } => "selection",
            TraceEvent::Invocation { .. } => "invocation",
            TraceEvent::PrefillProgress { .. } => "prefill_progress",
            TraceEvent::Token { .. } => "token",
            TraceEvent::Completed => "completed",
            TraceEvent::Failed { .. } => "failed",
            TraceEvent::Rejected { .. } => "rejected",
            TraceEvent::Cancelled => "cancelled",
        })
        .collect()
}

#[test]
fn a_finished_request_has_a_complete_trace_and_the_sink_matches_it() {
    let gate = Gate::open();
    let sink = Shared::default();
    let driver = Driver::spawn_with_trace_sink(
        executor(&gate, 1),
        ManualClock::new(Timestamp::ZERO),
        traced(DriverConfig::default()),
        sink.clone(),
    )
    .unwrap();
    let runtime = driver.handle();

    let output = runtime
        .submit(request("hi"))
        .wait()
        .unwrap()
        .blocking_result()
        .unwrap();
    let trace = runtime.trace(output.request_id).wait().unwrap();

    let kinds = kinds(&trace);
    assert_eq!(&kinds[..4], ["arrived", "ranked", "admission", "selection"]);
    assert_eq!(kinds.last(), Some(&"completed"));
    let tokens = kinds.iter().filter(|kind| **kind == "token").count();
    assert_eq!(tokens, output.token_ids.len());
    assert_eq!(tokens, REPLY.len() + 1);
    // Every record after arrival names the iteration that produced it, and
    // iterations never go backwards.
    let iterations = trace[1..]
        .iter()
        .map(|record| record.iteration.unwrap())
        .collect::<Vec<_>>();
    assert!(iterations.windows(2).all(|pair| pair[0] <= pair[1]));

    drop(runtime);
    driver.shutdown().unwrap();
    let written = read_jsonl(sink.0.lock().unwrap().as_slice()).unwrap();
    assert_eq!(written, trace);
}

#[test]
fn a_failing_sink_is_reported_and_does_not_disturb_requests() {
    let gate = Gate::open();
    let driver = Driver::spawn_with_trace_sink(
        executor(&gate, 1),
        ManualClock::new(Timestamp::ZERO),
        DriverConfig::default(),
        Broken,
    )
    .unwrap();
    let runtime = driver.handle();
    assert!(!runtime.health().trace_write_failed);

    let output = runtime
        .submit(request("hi"))
        .wait()
        .unwrap()
        .blocking_result()
        .unwrap();

    assert_eq!(output.text, REPLY);
    assert!(runtime.health().trace_write_failed);
    // Writing to a sink does not keep traces unless asked to.
    assert_eq!(
        runtime.trace(output.request_id).wait(),
        Err(ServeError::TracingDisabled)
    );
}

#[test]
fn cancellation_ends_the_trace_and_eviction_forgets_it() {
    let gate = Gate::closed();
    let config = DriverConfig {
        retain_finished: 1,
        ..traced(DriverConfig::default())
    };
    let driver = Driver::spawn(
        executor(&gate, 1),
        ManualClock::new(Timestamp::ZERO),
        config,
    )
    .unwrap();
    let runtime = driver.handle();

    gate.permit(1);
    let mut first = runtime.submit(request("a")).wait().unwrap();
    assert!(matches!(
        first.blocking_next(),
        Some(RequestEvent::Token { .. })
    ));
    // With the driver blocked in the second call, both commands queue and
    // are answered in order once that call finishes.
    gate.wait_until_blocked();
    let live = runtime.trace(first.id());
    let cancel = first.cancel();
    gate.permit(1);
    let live = live.wait().unwrap();
    assert_eq!(kinds(&live).last(), Some(&"token"));
    cancel.wait().unwrap();
    let cancelled = runtime.trace(first.id()).wait().unwrap();
    assert_eq!(cancelled.len(), live.len() + 1);
    let last = cancelled.last().unwrap();
    assert_eq!(last.event, TraceEvent::Cancelled);
    assert_eq!(last.iteration, None);

    gate.open_fully();
    let second = runtime
        .submit(request("b"))
        .wait()
        .unwrap()
        .blocking_result()
        .unwrap();
    // Retaining one finished request evicts the first, trace included.
    assert_eq!(
        runtime.trace(first.id()).wait(),
        Err(ServeError::UnknownRequest(first.id()))
    );
    assert!(runtime.trace(second.request_id).wait().is_ok());
    assert_eq!(
        runtime.trace(RequestId::new(99)).wait(),
        Err(ServeError::UnknownRequest(RequestId::new(99)))
    );
}
