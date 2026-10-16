//! Retention, health, readiness, and shutdown behaviour of the driver.

mod support;

use std::{thread, time::Duration};

use ola_core::{ManualClock, ModelDescriptor, Timestamp};
use ola_runtime::{Executor, RequestStatus, Runtime};
use ola_serve::{Driver, DriverConfig, RequestEvent, ServeError, ServiceState};
use support::{Gate, MODEL, PANIC_PROMPT, REPLY, driver, request};

#[test]
fn finished_requests_stay_queryable_up_to_the_retention_limit() {
    let gate = Gate::open();
    let config = DriverConfig {
        retain_finished: 2,
        ..DriverConfig::default()
    };
    let (driver, _) = driver(&gate, 1, config);
    let runtime = driver.handle();

    let outputs = (0..3)
        .map(|_| {
            runtime
                .submit(request("hi"))
                .wait()
                .unwrap()
                .blocking_result()
                .unwrap()
        })
        .collect::<Vec<_>>();

    let oldest = outputs[0].request_id;
    assert_eq!(
        runtime.status(oldest).wait(),
        Err(ServeError::UnknownRequest(oldest))
    );
    assert_eq!(
        runtime.cancel(oldest).wait(),
        Err(ServeError::UnknownRequest(oldest))
    );
    for output in &outputs[1..] {
        let snapshot = runtime.status(output.request_id).wait().unwrap();
        assert_eq!(snapshot.status, RequestStatus::Completed);
        assert_eq!(snapshot.timing, output.timing);
        assert_eq!(snapshot.generated_tokens, output.token_ids.len());
        assert_eq!(runtime.cancel(output.request_id).wait(), Ok(()));
    }

    // The executor itself retains nothing for finished requests.
    let executor = driver.shutdown().unwrap();
    for output in &outputs {
        assert!(executor.state().request(output.request_id).is_none());
        assert!(executor.result(output.request_id).is_none());
    }
}

#[test]
fn health_and_models_are_readable_while_an_iteration_runs() {
    let gate = Gate::closed();
    let (driver, _) = driver(&gate, 1, DriverConfig::default());
    let runtime = driver.handle();

    let health = runtime.health();
    assert!(health.is_ready());
    assert_eq!((health.in_flight, health.iterations), (0, 0));
    let models = runtime.models();
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].descriptor.id, MODEL.into());
    assert!(models[0].executable);

    gate.permit(1);
    let mut handle = runtime.submit(request("hi")).wait().unwrap();
    assert!(matches!(
        handle.blocking_next(),
        Some(RequestEvent::Token { .. })
    ));
    gate.wait_until_blocked();
    // The driver is stuck inside a model call, yet health answers.
    let busy = runtime.health();
    assert_eq!((busy.in_flight, busy.active, busy.queued), (1, 1, 0));
    assert!(busy.iterations >= 1);

    gate.open_fully();
    handle.blocking_result().unwrap();
    let idle = runtime.health();
    assert_eq!((idle.in_flight, idle.active), (0, 0));
    assert!(idle.iterations > busy.iterations);
}

#[test]
fn a_driver_without_executable_models_is_not_ready() {
    let mut runtime = Runtime::default();
    runtime
        .register_model(ModelDescriptor::language("metadata-only", "none"))
        .unwrap();
    let driver = Driver::spawn(
        Executor::new(runtime),
        ManualClock::new(Timestamp::ZERO),
        DriverConfig::default(),
    )
    .unwrap();
    let runtime = driver.handle();

    assert_eq!(runtime.health().state, ServiceState::NoModels);
    assert!(!runtime.models()[0].executable);
    assert_eq!(
        runtime.submit(request("hi")).wait().err(),
        Some(ServeError::NotReady)
    );
}

#[test]
fn draining_finishes_accepted_work_and_refuses_new_work() {
    let gate = Gate::closed();
    let (driver, _) = driver(&gate, 1, DriverConfig::default());
    let runtime = driver.handle();
    gate.permit(1);
    let running = runtime.submit(request("a")).wait().unwrap();
    gate.wait_until_blocked();

    let stopping = thread::spawn(move || driver.drain(Duration::from_secs(60)));
    while runtime.health().state == ServiceState::Ready {
        thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(runtime.health().state, ServiceState::Draining);
    assert_eq!(
        runtime.submit(request("b")).wait().err(),
        Some(ServeError::ShuttingDown)
    );
    gate.open_fully();

    assert_eq!(running.blocking_result().unwrap().text, REPLY);
    let executor = stopping.join().unwrap().unwrap();
    assert!(!executor.has_active_request());
    assert_eq!(runtime.health().state, ServiceState::Stopped);
}

#[test]
fn draining_ends_work_still_running_at_the_deadline() {
    let gate = Gate::closed();
    let (driver, clock) = driver(&gate, 1, DriverConfig::default());
    let runtime = driver.handle();
    gate.permit(1);
    let running = runtime.submit(request("a")).wait().unwrap();
    gate.wait_until_blocked();

    let stopping = thread::spawn(move || driver.drain(Duration::from_secs(1)));
    while runtime.health().state == ServiceState::Ready {
        thread::sleep(Duration::from_millis(1));
    }
    clock.advance(Duration::from_secs(2));
    gate.open_fully();

    assert_eq!(running.blocking_result(), Err(ServeError::ShuttingDown));
    let executor = stopping.join().unwrap().unwrap();
    assert!(!executor.has_active_request());
    assert_eq!(executor.state().memory.kv_cache_bytes, 0);
}

#[test]
fn a_panicking_driver_reports_stopped_and_ends_every_stream() {
    let gate = Gate::open();
    let (driver, _) = driver(&gate, 2, DriverConfig::default());
    let runtime = driver.handle();

    let result = runtime
        .submit(request(PANIC_PROMPT))
        .wait()
        .and_then(|handle| handle.blocking_result());

    assert_eq!(result, Err(ServeError::ShuttingDown));
    while runtime.health().state != ServiceState::Stopped {
        thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(
        runtime.submit(request("hi")).wait().err(),
        Some(ServeError::ShuttingDown)
    );
    assert!(matches!(driver.shutdown(), Err(ServeError::Internal(_))));
}
