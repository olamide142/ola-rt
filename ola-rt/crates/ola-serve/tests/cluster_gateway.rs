//! Loopback gateway-to-worker serving coverage.

use std::time::Duration;

use axum::{
    body::Body,
    http::{Request, StatusCode},
    response::Response,
};
use ola_bench::{KvGeometry, synthetic_artifact};
use ola_cluster::{LeastLoadedRouting, WorkerHost, WorkerSession};
use ola_core::ModelDescriptor;
use ola_net::TcpPeer;
use ola_runtime::{Executor, Runtime};
use ola_serve::cluster::{ClusterGateway, ClusterGatewayConfig};
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

fn long_worker(id: &'static str) -> WorkerHost {
    let mut executor = Executor::with_max_active_requests(Runtime::default(), 1).unwrap();
    executor
        .register_model(
            ModelDescriptor::language("synthetic", "synthetic"),
            synthetic_artifact(KvGeometry {
                capacity_bytes: 64 * 32 * 1024,
                ..KvGeometry::default()
            }),
        )
        .unwrap();
    WorkerHost::new(id, executor)
}

async fn spawn_worker(
    host: WorkerHost,
    heartbeat: Duration,
) -> (String, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap().to_string();
    let task = tokio::spawn(async move {
        let (peer, _) = TcpPeer::accept(&listener).await.unwrap();
        let _ = WorkerSession::new(peer, host, 1, heartbeat).run().await;
    });
    (address, task)
}

fn generate(body: &str) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/generate")
        .header("content-type", "application/json")
        .body(Body::from(body.to_owned()))
        .unwrap()
}

async fn ready_workers(app: &axum::Router) -> u64 {
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/ready")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .unwrap();
    let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    body["ready_workers"].as_u64().unwrap()
}

fn fast_supervision() -> ClusterGatewayConfig {
    ClusterGatewayConfig {
        lease_timeout: Duration::from_millis(100),
        tick: Duration::from_millis(10),
    }
}

#[tokio::test]
async fn an_idle_gateway_keeps_leases_and_drops_a_dead_worker() {
    let (address, task) = spawn_worker(worker(), Duration::from_millis(5)).await;
    let gateway =
        ClusterGateway::connect_with([(address, None)], LeastLoadedRouting, fast_supervision())
            .await
            .unwrap();
    let app = gateway.router();
    // Several leases pass with no request in flight; heartbeats still count.
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(ready_workers(&app).await, 1);

    task.abort();
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(ready_workers(&app).await, 0);
}

#[tokio::test]
async fn cancelling_an_unknown_request_is_not_found() {
    let (address, task) = spawn_worker(worker(), Duration::from_millis(5)).await;
    let gateway = ClusterGateway::connect([(address, None)]).await.unwrap();
    let response = gateway
        .router()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri("/v1/requests/999")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    task.abort();
}

#[tokio::test]
async fn an_abandoned_request_frees_its_worker_and_health_stays_responsive() {
    // One slot: the next request can only run if the abandoned one is gone.
    let (address, task) = spawn_worker(long_worker("worker"), Duration::from_millis(5)).await;
    let gateway = ClusterGateway::connect([(address, None)]).await.unwrap();
    let app = gateway.router();

    // A leading 0 tells the synthetic model never to stop early.
    let long = r#"{"model":"synthetic","tokens":[0,2,3,4],"max_new_tokens":30000}"#;
    let abandoned = tokio::time::timeout(
        Duration::from_millis(100),
        app.clone().oneshot(generate(long)),
    )
    .await;
    assert!(
        abandoned.is_err(),
        "the long request should still be running"
    );

    // Health answers while the worker is busy streaming.
    let started = std::time::Instant::now();
    assert_eq!(ready_workers(&app).await, 1);
    assert!(started.elapsed() < Duration::from_millis(500));

    let short = r#"{"model":"synthetic","tokens":[0,5,6],"max_new_tokens":2}"#;
    let started = std::time::Instant::now();
    let output = json_response(app.oneshot(generate(short)).await.unwrap()).await;
    assert_eq!(output["token_ids"].as_array().unwrap().len(), 2);
    // Finishing 30 000 abandoned tokens first would take far longer.
    assert!(
        started.elapsed() < Duration::from_secs(1),
        "short request waited {:?}",
        started.elapsed()
    );
    task.abort();
}

/// The `data:` payloads of an SSE body.
async fn sse_data(response: Response) -> Vec<String> {
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .unwrap();
    String::from_utf8(bytes.to_vec())
        .unwrap()
        .lines()
        .filter_map(|line| line.strip_prefix("data: ").map(str::to_owned))
        .collect()
}

#[tokio::test]
async fn streamed_completions_carry_text_deltas_and_usage() {
    let (address, task) = spawn_worker(worker(), Duration::from_millis(5)).await;
    let gateway = ClusterGateway::connect([(address, None)]).await.unwrap();
    let app = gateway.router();
    let completion = |stream: bool| {
        Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(format!(
                r#"{{"model":"synthetic","prompt":[0,2,3,4],"max_tokens":4,"stream":{stream}}}"#
            )))
            .unwrap()
    };

    let whole = json_response(app.clone().oneshot(completion(false)).await.unwrap()).await;
    assert_eq!(whole["usage"]["prompt_tokens"].as_u64(), Some(4));
    assert_eq!(whole["usage"]["completion_tokens"].as_u64(), Some(4));
    assert_eq!(whole["usage"]["total_tokens"].as_u64(), Some(8));
    assert_eq!(whole["choices"][0]["finish_reason"], "length");

    let data = sse_data(app.oneshot(completion(true)).await.unwrap()).await;
    assert_eq!(data.last().map(String::as_str), Some("[DONE]"));
    let chunks = data[..data.len() - 1]
        .iter()
        .map(|chunk| serde_json::from_str::<serde_json::Value>(chunk).unwrap())
        .collect::<Vec<_>>();
    // Several text chunks, then one final chunk with the finish reason.
    assert!(chunks.len() > 2, "{chunks:?}");
    let streamed = chunks
        .iter()
        .map(|chunk| chunk["choices"][0]["text"].as_str().unwrap())
        .collect::<String>();
    assert_eq!(streamed, whole["choices"][0]["text"].as_str().unwrap());
    let last = chunks.last().unwrap();
    assert_eq!(last["choices"][0]["finish_reason"], "length");
    assert_eq!(last["usage"]["prompt_tokens"].as_u64(), Some(4));
    task.abort();
}

#[tokio::test]
async fn streamed_generation_tokens_carry_text() {
    let (address, task) = spawn_worker(worker(), Duration::from_millis(5)).await;
    let gateway = ClusterGateway::connect([(address, None)]).await.unwrap();
    let app = gateway.router();
    let response = app
        .oneshot(generate(
            r#"{"model":"synthetic","tokens":[0,2,3,4],"max_new_tokens":3,"stream":true}"#,
        ))
        .await
        .unwrap();
    let events = sse_data(response)
        .await
        .into_iter()
        .map(|data| serde_json::from_str::<serde_json::Value>(&data).unwrap())
        .collect::<Vec<_>>();
    let (done, tokens) = events.split_last().unwrap();
    let streamed = tokens
        .iter()
        .map(|token| token["text"].as_str().unwrap())
        .collect::<String>();
    assert_eq!(tokens.len(), 3);
    assert_eq!(streamed, done["text"].as_str().unwrap());
    task.abort();
}
