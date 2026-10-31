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

#[tokio::test]
async fn large_session_snapshots_cross_links_as_binary_frames() {
    use ola_cluster::{CoordinatorLinkEvent, CoordinatorWorkerLink, WorkerCommand, WorkerMessage};
    use ola_model::SessionState;
    use ola_net::{WireBody, WireMessage, read_attached, write_attached};
    use ola_runtime::MigratedRequest;

    // Over 16 MiB once JSON writes each byte as a number.
    let payload = (0..4 * 1024 * 1024)
        .map(|index| (index % 251) as u8)
        .collect::<Vec<_>>();
    let state = MigratedRequest {
        spec: request(4),
        prompt_tokens: vec![0, 2, 3, 4],
        generated_tokens: vec![7],
        token_times: vec![Timestamp::ZERO],
        preemptions: 0,
        session_state: Some(SessionState {
            format: "test/v1".into(),
            cursor: 5,
            payload,
        }),
    };

    // A bare worker: handshake, take one command, echo it back as an export.
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let summary = worker("raw").summary(Timestamp::ZERO);
    let raw = tokio::spawn(async move {
        let (mut peer, _) = TcpPeer::accept(&listener).await.unwrap();
        peer.send(&WireMessage::new(
            1,
            1,
            WireBody::Hello {
                worker: summary.worker.clone(),
            },
        ))
        .await
        .unwrap();
        peer.send(&WireMessage::new(
            1,
            2,
            WireBody::json_payload(&WorkerMessage::Summary(summary)).unwrap(),
        ))
        .await
        .unwrap();
        let (mut reader, mut writer) = peer.into_split();
        let (message, frame) = read_attached(&mut reader).await.unwrap();
        let control_bytes = serde_json::to_vec(&message).unwrap().len();
        let WorkerCommand::AdmitMigrated { request, state, .. } =
            message.body.decode_json().unwrap()
        else {
            panic!("expected a migrated admission");
        };
        assert!(state.session_state.is_none(), "snapshot stayed in the JSON");
        assert!(
            control_bytes < 64 * 1024,
            "control message is {control_bytes} bytes"
        );
        let reply = WireMessage::new(
            1,
            3,
            WireBody::json_payload(&WorkerMessage::Migrated { request, state }).unwrap(),
        );
        write_attached(&mut writer, &reply, &frame.unwrap())
            .await
            .unwrap();
        (reader, writer)
    });

    let (frames, mut received) = tokio::sync::mpsc::unbounded_channel();
    let (mut link, _) = CoordinatorWorkerLink::connect(address, None, frames)
        .await
        .unwrap();
    link.send_command(&WorkerCommand::AdmitMigrated {
        request: ClusterRequestId::new(9),
        state: state.clone(),
        arrival_at: Timestamp::ZERO,
    })
    .await
    .unwrap();
    let (_, frame) = received.recv().await.unwrap();
    let CoordinatorLinkEvent::Message(WorkerMessage::Migrated {
        state: returned, ..
    }) = link.accept_frame(frame).await.unwrap()
    else {
        panic!("expected the exported request back");
    };
    assert_eq!(returned, state);
    let _connection = raw.await.unwrap();
}

#[tokio::test]
async fn a_worker_session_admits_a_large_migrated_snapshot() {
    use ola_cluster::{CoordinatorLinkEvent, CoordinatorWorkerLink, WorkerCommand, WorkerMessage};
    use ola_model::SessionState;
    use ola_runtime::MigratedRequest;

    let (address, task) = spawn_worker("target", Duration::from_millis(5)).await;
    let (frames, mut received) = tokio::sync::mpsc::unbounded_channel();
    let (mut link, _) = CoordinatorWorkerLink::connect(address, None, frames)
        .await
        .unwrap();
    let request_id = ClusterRequestId::new(5);
    link.send_command(&WorkerCommand::AdmitMigrated {
        request: request_id,
        state: MigratedRequest {
            spec: request(3),
            prompt_tokens: vec![0, 2, 3, 4],
            generated_tokens: vec![7],
            token_times: vec![Timestamp::ZERO],
            preemptions: 0,
            // The synthetic model cannot import it and recomputes instead;
            // what matters is that the snapshot arrives.
            session_state: Some(SessionState {
                format: "test/v1".into(),
                cursor: 5,
                payload: vec![3; 4 * 1024 * 1024],
            }),
        },
        arrival_at: Timestamp::ZERO,
    })
    .await
    .unwrap();
    loop {
        let (_, frame) = tokio::time::timeout(Duration::from_secs(5), received.recv())
            .await
            .expect("worker did not answer")
            .unwrap();
        if let CoordinatorLinkEvent::Message(message) = link.accept_frame(frame).await.unwrap() {
            match message {
                WorkerMessage::Accepted { request, .. } if request == request_id => break,
                WorkerMessage::Refused { message, .. } => panic!("refused: {message}"),
                _ => {}
            }
        }
    }
    task.abort();
}
