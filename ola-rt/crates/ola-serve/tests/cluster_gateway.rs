//! Loopback gateway-to-worker serving coverage.

use std::time::Duration;

use axum::{
    body::Body,
    http::{Request, StatusCode},
    response::Response,
};
use ola_bench::{KvGeometry, synthetic_artifact};
use ola_cluster::{WorkerHost, WorkerSession};
use ola_core::ModelDescriptor;
use ola_net::TcpPeer;
use ola_runtime::{Executor, Runtime};
use ola_serve::cluster::ClusterGateway;
use tokio::net::TcpListener;
use tower::ServiceExt;

fn worker_named(id: &'static str) -> WorkerHost {
    let mut executor = Executor::with_max_active_requests(Runtime::default(), 1).unwrap();
    executor
        .register_model(
            ModelDescriptor::language("synthetic", "synthetic"),
            synthetic_artifact(KvGeometry::default()),
        )
        .unwrap();
    WorkerHost::new(id, executor)
}

fn worker() -> WorkerHost {
    worker_named("gateway-worker")
}

async fn json_response(response: Response) -> serde_json::Value {
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

#[tokio::test]
async fn gateway_serves_a_network_worker() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let worker_task = tokio::spawn(async move {
        let (peer, _) = TcpPeer::accept(&listener).await.unwrap();
        WorkerSession::new(peer, worker(), 1, Duration::from_millis(2))
            .run()
            .await
            .unwrap();
    });

    let gateway = ClusterGateway::connect([(address.to_string(), None)])
        .await
        .unwrap();
    let app = gateway.router();
    let request = Request::builder()
        .method("POST")
        .uri("/v1/generate")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"model":"synthetic","tokens":[1,2,3],"max_new_tokens":2}"#,
        ))
        .unwrap();
    let output = json_response(app.clone().oneshot(request).await.unwrap()).await;
    assert_eq!(output["id"].as_u64(), Some(1));
    assert!(!output["token_ids"].as_array().unwrap().is_empty());

    let invalid_completion = Request::builder()
        .method("POST")
        .uri("/v1/completions")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"model":"synthetic"}"#))
        .unwrap();
    let completion = app.oneshot(invalid_completion).await.unwrap();
    assert_eq!(completion.status(), StatusCode::BAD_REQUEST);

    drop(gateway);
    worker_task.abort();
}

#[tokio::test]
async fn gateway_routes_across_two_network_workers() {
    let first_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let first_address = first_listener.local_addr().unwrap();
    let first_task = tokio::spawn(async move {
        let (peer, _) = TcpPeer::accept(&first_listener).await.unwrap();
        WorkerSession::new(
            peer,
            worker_named("gateway-worker-a"),
            1,
            Duration::from_millis(2),
        )
        .run()
        .await
        .unwrap();
    });
    let second_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let second_address = second_listener.local_addr().unwrap();
    let second_task = tokio::spawn(async move {
        let (peer, _) = TcpPeer::accept(&second_listener).await.unwrap();
        WorkerSession::new(
            peer,
            worker_named("gateway-worker-b"),
            1,
            Duration::from_millis(2),
        )
        .run()
        .await
        .unwrap();
    });

    let gateway = ClusterGateway::connect([
        (first_address.to_string(), None),
        (second_address.to_string(), None),
    ])
    .await
    .unwrap();
    let health = gateway
        .router()
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let health = json_response(health).await;
    assert_eq!(health["workers"].as_u64(), Some(2));

    let app = gateway.router();
    let first_request = Request::builder()
        .method("POST")
        .uri("/v1/generate")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"model":"synthetic","tokens":[1,2,3],"max_new_tokens":2}"#,
        ))
        .unwrap();
    let second_request = Request::builder()
        .method("POST")
        .uri("/v1/generate")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"model":"synthetic","tokens":[4,5,6],"max_new_tokens":2}"#,
        ))
        .unwrap();
    let first = json_response(app.clone().oneshot(first_request).await.unwrap()).await;
    let second = json_response(app.oneshot(second_request).await.unwrap()).await;
    assert_ne!(first["id"], second["id"]);
    assert!(!first["token_ids"].as_array().unwrap().is_empty());
    assert!(!second["token_ids"].as_array().unwrap().is_empty());

    drop(gateway);
    first_task.abort();
    second_task.abort();
}
