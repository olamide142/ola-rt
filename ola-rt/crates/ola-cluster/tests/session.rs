//! Loopback coverage for the TCP worker session.

use std::time::Duration;

use ola_bench::{KvGeometry, synthetic_artifact};
use ola_cluster::{ClusterRequestStatus, RoundRobinRouting, TcpCoordinator, WorkerHealth, WorkerHost, WorkerSession};
use ola_core::{Input, ModelDescriptor, RequestSpec, Timestamp, WorkerId};
use ola_net::TcpPeer;
use ola_runtime::{Executor, Runtime};
use tokio::net::TcpListener;

fn worker() -> WorkerHost {
    let mut executor = Executor::with_max_active_requests(Runtime::default(), 1).unwrap();
    executor
        .register_model(
            ModelDescriptor::language("synthetic", "synthetic"),
            synthetic_artifact(KvGeometry::default()),
        )
        .unwrap();
    WorkerHost::new("worker-a", executor)
}

#[tokio::test]
async fn worker_session_announces_summarizes_and_closes() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (peer, _) = TcpPeer::accept(&listener).await.unwrap();
        WorkerSession::new(peer, worker(), 4, Duration::from_millis(5))
            .run()
            .await
            .unwrap()
    });

    let mut coordinator = TcpCoordinator::new(RoundRobinRouting);
    let worker_id = coordinator
        .connect_worker(address, Some(WorkerId::new("worker-a")))
        .await
        .unwrap();
    let summary = coordinator.coordinator().workers().next().unwrap();
    assert_eq!(summary.worker, WorkerId::new("worker-a"));
    assert_eq!(summary.health, WorkerHealth::Ready);

    tokio::time::sleep(Duration::from_millis(10)).await;
    coordinator
        .receive(&worker_id, Timestamp::from_micros(10))
        .await
        .unwrap();
    assert!(coordinator
        .expire_workers(Timestamp::from_micros(19), Duration::from_micros(10))
        .is_empty());

    let mut spec = RequestSpec::text("synthetic", "");
    spec.inputs = vec![Input::Tokens(vec![1, 2, 3])];
    spec.generation.max_new_tokens = 1;
    let (request, _) = coordinator
        .submit_and_plan(spec, Timestamp::ZERO)
        .await
        .unwrap();
    while !matches!(
        coordinator.coordinator().status(request),
        Some(ClusterRequestStatus::Running { .. })
    ) {
        let output = coordinator
            .receive(&worker_id, Timestamp::from_micros(1))
            .await
            .unwrap();
        assert!(output.client.is_empty());
    }
    assert!(matches!(
        coordinator.coordinator().status(request),
        Some(ClusterRequestStatus::Running { .. })
    ));
    coordinator.close_worker(&worker_id).await.unwrap();
    let returned = server.await.unwrap();
    assert_eq!(returned.id(), &WorkerId::new("worker-a"));
}
