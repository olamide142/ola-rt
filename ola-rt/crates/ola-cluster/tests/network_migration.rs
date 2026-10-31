//! Coordinator-managed migration over two TCP worker sessions.

use std::time::Duration;

use ola_bench::{KvGeometry, synthetic_artifact};
use ola_cluster::{
    ClusterTraceEvent, LeastLoadedRouting, TcpCoordinator, WorkerHost, WorkerSession,
};
use ola_core::{Input, ModelDescriptor, RequestSpec, Timestamp};
use ola_net::TcpPeer;
use ola_runtime::{Executor, Runtime};
use tokio::net::TcpListener;

fn worker(id: &'static str) -> WorkerHost {
    let mut executor = Executor::with_max_active_requests(Runtime::default(), 2).unwrap();
    executor
        .register_model(
            ModelDescriptor::language("synthetic", "synthetic"),
            synthetic_artifact(KvGeometry::default()),
        )
        .unwrap();
    WorkerHost::new(id, executor)
}

fn request() -> RequestSpec {
    RequestSpec {
        model: "synthetic".into(),
        inputs: vec![Input::Tokens(vec![0, 2, 3, 4])],
        generation: ola_core::GenerationConfig {
            max_new_tokens: 100,
            temperature: 0.0,
            top_p: 1.0,
            seed: Some(7),
        },
        priority: ola_core::Priority::NORMAL,
        deadline: None,
        attributes: Default::default(),
    }
}

#[tokio::test]
async fn coordinator_migrates_between_tcp_workers_and_traces_commit() {
    let first_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let first_address = first_listener.local_addr().unwrap();
    let first_task = tokio::spawn(async move {
        let (peer, _) = TcpPeer::accept(&first_listener).await.unwrap();
        WorkerSession::new(peer, worker("source"), 1, Duration::from_millis(2))
            .run()
            .await
            .unwrap();
    });
    let second_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let second_address = second_listener.local_addr().unwrap();
    let second_task = tokio::spawn(async move {
        let (peer, _) = TcpPeer::accept(&second_listener).await.unwrap();
        WorkerSession::new(peer, worker("target"), 1, Duration::from_millis(2))
            .run()
            .await
            .unwrap();
    });

    let mut coordinator = TcpCoordinator::new(LeastLoadedRouting);
    let source = coordinator
        .connect_worker(first_address.to_string(), None)
        .await
        .unwrap();
    let target = coordinator
        .connect_worker(second_address.to_string(), None)
        .await
        .unwrap();
    let (request_id, _) = coordinator
        .submit_and_plan(request(), Timestamp::ZERO)
        .await
        .unwrap();
    for tick in 1..10 {
        let _ = tokio::time::timeout(
            Duration::from_secs(2),
            coordinator.receive(&source, Timestamp::from_micros(tick)),
        )
        .await
        .expect("source acceptance timed out")
        .unwrap();
        if matches!(
            coordinator.coordinator().status(request_id),
            Some(ola_cluster::ClusterRequestStatus::Running { worker, .. }) if worker == &source
        ) {
            break;
        }
    }
    // Let the source produce at least one token before the handoff.
    let _ = tokio::time::timeout(
        Duration::from_secs(2),
        coordinator.receive(&source, Timestamp::from_micros(2)),
    )
    .await
    .expect("source token timed out")
    .unwrap();
    let migration = tokio::time::timeout(
        Duration::from_secs(2),
        coordinator.migrate(
            request_id,
            &source,
            target.clone(),
            Timestamp::from_micros(3),
        ),
    )
    .await
    .expect("migration timed out")
    .unwrap();
    assert!(migration.trace.iter().any(|record| matches!(
        record.event,
        ClusterTraceEvent::Migrated { ref from, ref to, .. }
            if from == &source && to == &target
    )));
    assert!(matches!(
        coordinator.coordinator().status(request_id),
        Some(ola_cluster::ClusterRequestStatus::Running { worker, .. }) if worker == &target
    ));

    for tick in 4..200 {
        let output = coordinator
            .receive(&target, Timestamp::from_micros(tick))
            .await
            .unwrap();
        if output.client.iter().any(|(id, event)| {
            *id == request_id && matches!(event, ola_cluster::ClientEvent::Finished { .. })
        }) {
            first_task.abort();
            second_task.abort();
            return;
        }
    }
    first_task.abort();
    second_task.abort();
    panic!("migrated TCP request did not finish");
}
