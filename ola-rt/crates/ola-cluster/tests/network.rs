//! Loopback behaviour of `TcpCoordinator` under cancellation, worker loss,
//! and slow heartbeats.

use std::time::{Duration, Instant};

use ola_bench::{KvGeometry, synthetic_artifact};
use ola_cluster::{
    ClientEvent, ClusterRequestStatus, ClusterTraceEvent, RoundRobinRouting, TcpCoordinator,
    WorkerHost, WorkerSession,
};
use ola_core::{ClusterRequestId, Input, ModelDescriptor, RequestSpec, Timestamp};
use ola_net::TcpPeer;
use ola_runtime::{Executor, Runtime};
use tokio::{net::TcpListener, task::JoinHandle};

fn worker(id: &'static str) -> WorkerHost {
    let mut executor = Executor::with_max_active_requests(Runtime::default(), 2).unwrap();
    executor
        .register_model(
            ModelDescriptor::language("synthetic", "synthetic"),
            // Room for long requests, so a worker can die mid-stream.
            synthetic_artifact(KvGeometry {
                capacity_bytes: 64 * 32 * 1024,
                ..KvGeometry::default()
            }),
        )
        .unwrap();
    WorkerHost::new(id, executor)
}

async fn spawn_worker(id: &'static str, heartbeat: Duration) -> (String, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap().to_string();
    let task = tokio::spawn(async move {
        let (peer, _) = TcpPeer::accept(&listener).await.unwrap();
        let _ = WorkerSession::new(peer, worker(id), 1, heartbeat)
            .run()
            .await;
    });
    (address, task)
}

fn request(max_new_tokens: usize) -> RequestSpec {
    let mut spec = RequestSpec::text("synthetic", "");
    spec.inputs = vec![Input::Tokens(vec![0, 2, 3, 4])];
    spec.generation.max_new_tokens = max_new_tokens;
    spec.generation.seed = Some(7);
    spec
}

/// Polls until `request` finishes, returning its streamed and final tokens
/// and every trace event seen meanwhile.
async fn finish(
    coordinator: &mut TcpCoordinator,
    request: ClusterRequestId,
) -> (Vec<u32>, Vec<u32>, Vec<ClusterTraceEvent>) {
    let mut streamed = Vec::new();
    let mut trace = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        assert!(Instant::now() < deadline, "request did not finish");
        let (_, output) = coordinator.poll(Timestamp::ZERO).await.unwrap();
        trace.extend(output.trace.into_iter().map(|record| record.event));
        for (id, event) in output.client {
            if id != request {
                continue;
            }
            match event {
                ClientEvent::Token { token_id, .. } => streamed.push(token_id),
                ClientEvent::Finished { token_ids, .. } => return (streamed, token_ids, trace),
                other => panic!("unexpected client event {other:?}"),
            }
        }
    }
}

#[tokio::test]
async fn abandoned_receives_do_not_corrupt_the_link() {
    let (address, task) = spawn_worker("worker", Duration::from_millis(1)).await;
    let mut coordinator = TcpCoordinator::new(RoundRobinRouting);
    let worker_id = coordinator.connect_worker(address, None).await.unwrap();

    // Drop many receives part-way; heartbeats arrive every millisecond, so
    // some are abandoned while frames are in flight.
    for _ in 0..50 {
        let _ = tokio::time::timeout(
            Duration::from_micros(300),
            coordinator.receive(&worker_id, Timestamp::ZERO),
        )
        .await;
    }
    let (request, _) = coordinator
        .submit_and_plan(request(6), Timestamp::ZERO)
        .await
        .unwrap();
    let (streamed, finished, _) = finish(&mut coordinator, request).await;
    assert_eq!(streamed, finished);
    assert_eq!(finished.len(), 6);
    task.abort();
}

/// Long enough that the owner is still generating when the test kills it.
const LONG: usize = 20_000;

#[tokio::test]
async fn a_disconnected_worker_is_recovered_without_replaying_tokens() {
    let (first_address, first_task) = spawn_worker("a", Duration::from_millis(5)).await;
    let (second_address, second_task) = spawn_worker("b", Duration::from_millis(5)).await;
    let mut coordinator = TcpCoordinator::new(RoundRobinRouting);
    let first = coordinator
        .connect_worker(first_address, None)
        .await
        .unwrap();
    let second = coordinator
        .connect_worker(second_address, None)
        .await
        .unwrap();

    let (request, _) = coordinator
        .submit_and_plan(request(LONG), Timestamp::ZERO)
        .await
        .unwrap();
    assert_eq!(coordinator.status(request).unwrap().owner(), Some(&first));
    // Wait for some streamed output, then kill the owner's process.
    let mut streamed = Vec::new();
    while streamed.len() < 3 {
        let (_, output) = coordinator.poll(Timestamp::ZERO).await.unwrap();
        for (id, event) in output.client {
            if let (true, ClientEvent::Token { token_id, .. }) = (id == request, event) {
                streamed.push(token_id);
            }
        }
    }
    first_task.abort();

    let (rest, finished, trace) = finish(&mut coordinator, request).await;
    streamed.extend(rest);
    // The survivor took the request over and finished it.
    assert!(trace.iter().any(|event| matches!(
        event,
        ClusterTraceEvent::Accepted { worker, .. } if worker == &second
    )));
    assert!(trace.iter().any(|event| matches!(
        event,
        ClusterTraceEvent::Completed { worker, .. } if worker == &second
    )));
    assert!(matches!(
        coordinator.status(request),
        Some(ClusterRequestStatus::Completed)
    ));
    // Seeded recompute regenerates the same output; the client saw each
    // token exactly once.
    assert_eq!(streamed, finished);
    assert_eq!(finished.len(), LONG);
    assert_eq!(coordinator.connected().count(), 1);
    second_task.abort();
}

#[tokio::test]
async fn generation_is_not_paced_by_the_heartbeat() {
    let (address, task) = spawn_worker("worker", Duration::from_secs(10)).await;
    let mut coordinator = TcpCoordinator::new(RoundRobinRouting);
    coordinator.connect_worker(address, None).await.unwrap();
    let started = Instant::now();
    let (request, _) = coordinator
        .submit_and_plan(request(32), Timestamp::ZERO)
        .await
        .unwrap();
    let (_, finished, _) = finish(&mut coordinator, request).await;
    assert_eq!(finished.len(), 32);
    // One iteration per 10 s heartbeat would take minutes.
    assert!(started.elapsed() < Duration::from_secs(2));
    task.abort();
}
