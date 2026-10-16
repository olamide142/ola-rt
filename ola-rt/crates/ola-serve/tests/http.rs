//! The native HTTP API, exercised over real TCP connections.

mod support;

use std::{
    io::{BufRead, BufReader, Read, Write},
    net::{SocketAddr, TcpStream},
    thread,
    time::Duration,
};

use ola_core::{ManualClock, ModelDescriptor, Timestamp};
use ola_runtime::{Executor, Runtime};
use ola_serve::{Driver, DriverConfig, RuntimeHandle, http};
use serde_json::{Value, json};
use support::{FAIL_PROMPT, Gate, REPLY, driver};
use tokio::sync::oneshot;

/// An HTTP server on an ephemeral port, stopped when dropped.
struct Server {
    address: SocketAddr,
    stop: Option<oneshot::Sender<()>>,
    thread: Option<thread::JoinHandle<()>>,
}

impl Server {
    fn start(runtime: RuntimeHandle) -> Self {
        let (stop, stopped) = oneshot::channel::<()>();
        let (bound, address) = std::sync::mpsc::channel();
        let thread = thread::spawn(move || {
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .unwrap()
                .block_on(async move {
                    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
                    bound.send(listener.local_addr().unwrap()).unwrap();
                    http::serve(listener, runtime, async {
                        let _ = stopped.await;
                    })
                    .await
                    .unwrap();
                });
        });
        Self {
            address: address.recv().unwrap(),
            stop: Some(stop),
            thread: Some(thread),
        }
    }

    fn connect(&self, method: &str, path: &str, body: Option<&Value>) -> TcpStream {
        let mut stream = TcpStream::connect(self.address).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(20)))
            .unwrap();
        let body = body.map(Value::to_string).unwrap_or_default();
        write!(
            stream,
            "{method} {path} HTTP/1.1\r\nHost: test\r\nConnection: close\r\n\
             Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        )
        .unwrap();
        stream
    }

    /// Sends one request and returns the status, lower-cased headers, and
    /// decoded body.
    fn call(&self, method: &str, path: &str, body: Option<Value>) -> Response {
        let mut raw = Vec::new();
        self.connect(method, path, body.as_ref())
            .read_to_end(&mut raw)
            .unwrap();
        Response::parse(&raw)
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.stop.take().unwrap().send(());
        let _ = self.thread.take().unwrap().join();
    }
}

struct Response {
    status: u16,
    headers: String,
    body: String,
}

impl Response {
    fn parse(raw: &[u8]) -> Self {
        let text = String::from_utf8(raw.to_vec()).unwrap();
        let (head, body) = text.split_once("\r\n\r\n").unwrap();
        let headers = head.to_ascii_lowercase();
        let status = head[9..12].parse().unwrap();
        let body = if headers.contains("transfer-encoding: chunked") {
            dechunk(body)
        } else {
            body.to_owned()
        };
        Self {
            status,
            headers,
            body,
        }
    }

    fn json(&self) -> Value {
        serde_json::from_str(&self.body).unwrap_or_else(|_| panic!("not JSON: {}", self.body))
    }

    fn error_kind(&self) -> String {
        self.json()["error"]["kind"].as_str().unwrap().to_owned()
    }

    /// Server-sent events as `(event, data)` pairs, skipping keep-alives.
    fn events(&self) -> Vec<(String, Value)> {
        self.body
            .split("\n\n")
            .filter_map(|block| {
                let event = block
                    .lines()
                    .find_map(|line| line.strip_prefix("event: "))?;
                let data = block.lines().find_map(|line| line.strip_prefix("data: "))?;
                Some((event.to_owned(), serde_json::from_str(data).unwrap()))
            })
            .collect()
    }
}

fn dechunk(mut body: &str) -> String {
    let mut decoded = String::new();
    while let Some((size, rest)) = body.split_once("\r\n") {
        let size = usize::from_str_radix(size.trim(), 16).unwrap();
        if size == 0 {
            break;
        }
        decoded.push_str(&rest[..size]);
        body = &rest[size + 2..];
    }
    decoded
}

fn generate(prompt: &str, stream: bool) -> Value {
    json!({ "model": support::MODEL, "prompt": prompt, "max_new_tokens": 64, "stream": stream })
}

#[test]
fn health_readiness_and_models_describe_the_service() {
    let gate = Gate::open();
    let (driver, _) = driver(&gate, 1, DriverConfig::default());
    let server = Server::start(driver.handle());

    let health = server.call("GET", "/health", None);
    assert_eq!(health.status, 200);
    assert_eq!(health.json()["state"], "ready");
    assert_eq!(health.json()["max_in_flight"], 256);
    assert_eq!(server.call("GET", "/ready", None).status, 200);
    let models = server.call("GET", "/v1/models", None).json();
    assert_eq!(
        models,
        json!({ "models": [{ "id": "script", "architecture": "script", "executable": true }] })
    );
}

#[test]
fn generation_returns_the_result_and_the_request_stays_queryable() {
    let gate = Gate::open();
    let (driver, _) = driver(&gate, 1, DriverConfig::default());
    let server = Server::start(driver.handle());

    let response = server.call("POST", "/v1/generate", Some(generate("hi", false)));
    assert_eq!(response.status, 200, "{}", response.body);
    let result = response.json();
    assert_eq!(result["text"], REPLY);
    assert_eq!(result["usage"]["prompt_tokens"], 2);
    let id = result["id"].as_u64().unwrap();
    assert!(result["timing"]["completed_us"].is_u64());

    let status = server
        .call("GET", &format!("/v1/requests/{id}"), None)
        .json();
    assert_eq!(status["status"], "completed");
    assert_eq!(
        status["generated_tokens"].as_u64(),
        Some(result["token_ids"].as_array().unwrap().len() as u64)
    );
    assert_eq!(
        server
            .call("DELETE", &format!("/v1/requests/{id}"), None)
            .status,
        204
    );
    let missing = server.call("GET", "/v1/requests/999", None);
    assert_eq!(
        (missing.status, missing.error_kind().as_str()),
        (404, "unknown_request")
    );
}

#[test]
fn each_failure_has_its_own_status_and_kind() {
    let gate = Gate::open();
    let (driver, _) = driver(&gate, 1, DriverConfig::default());
    let server = Server::start(driver.handle());
    let post = |body: Value| server.call("POST", "/v1/generate", Some(body));

    let cases = [
        (json!({ "model": "script" }), 400, "bad_request"),
        (
            json!({ "model": "script", "prompt": "a", "tokens": [1] }),
            400,
            "bad_request",
        ),
        (
            json!({ "model": "script", "prompt": "a", "temperature": 0.7 }),
            400,
            "bad_request",
        ),
        (
            json!({ "model": "script", "prompt": "a", "max_new_tokens": 0 }),
            400,
            "invalid_request",
        ),
        (
            json!({ "model": "missing", "prompt": "a" }),
            404,
            "unknown_model",
        ),
        (generate(FAIL_PROMPT, false), 500, "execution_failed"),
    ];
    for (body, status, kind) in cases {
        let response = post(body.clone());
        assert_eq!(
            (response.status, response.error_kind().as_str()),
            (status, kind),
            "{body}: {}",
            response.body
        );
    }
    let malformed = server.call("POST", "/v1/generate", Some(json!("not an object")));
    assert_eq!(malformed.status, 400);
}

#[test]
fn streaming_sends_start_tokens_and_done() {
    let gate = Gate::open();
    let (driver, _) = driver(&gate, 1, DriverConfig::default());
    let server = Server::start(driver.handle());

    let response = server.call("POST", "/v1/generate", Some(generate("hi", true)));
    assert_eq!(response.status, 200);
    assert!(response.headers.contains("content-type: text/event-stream"));
    let events = response.events();

    assert_eq!(events[0].0, "start");
    let (done, result) = events.last().unwrap();
    assert_eq!(done, "done");
    assert_eq!(result["text"], REPLY);
    let tokens = &events[1..events.len() - 1];
    assert!(tokens.iter().all(|(event, _)| event == "token"));
    assert_eq!(tokens[0].1["phase"], "prefill");
    let text = tokens
        .iter()
        .map(|(_, data)| data["text"].as_str().unwrap())
        .collect::<String>();
    assert_eq!(text, REPLY);
    assert_eq!(result["id"], events[0].1["id"]);
}

#[test]
fn a_full_server_answers_429_with_retry_after() {
    let gate = Gate::closed();
    let config = DriverConfig {
        max_in_flight: 1,
        ..DriverConfig::default()
    };
    let (driver, _) = driver(&gate, 1, config);
    let server = Server::start(driver.handle());

    let mut first = server.connect("POST", "/v1/generate", Some(&generate("a", true)));
    let mut line = String::new();
    let mut reader = BufReader::new(&mut first);
    while !line.starts_with("event: start") {
        line.clear();
        reader.read_line(&mut line).unwrap();
    }
    let refused = server.call("POST", "/v1/generate", Some(generate("b", false)));
    assert_eq!(
        (refused.status, refused.error_kind().as_str()),
        (429, "overloaded")
    );
    assert!(refused.headers.contains("retry-after: 1"));

    gate.open_fully();
    let mut rest = String::new();
    reader.read_to_string(&mut rest).unwrap();
    assert!(rest.contains("event: done"));
}

#[test]
fn closing_a_stream_cancels_its_request() {
    let gate = Gate::closed();
    let (driver, _) = driver(&gate, 1, DriverConfig::default());
    let runtime = driver.handle();
    let server = Server::start(runtime.clone());

    gate.permit(1);
    let mut client = server.connect("POST", "/v1/generate", Some(&generate("hi", true)));
    let mut reader = BufReader::new(&mut client);
    let mut line = String::new();
    while !line.starts_with("event: token") {
        line.clear();
        reader.read_line(&mut line).unwrap();
    }
    drop(reader);
    drop(client);

    // The server notices the closed socket on its next writes; release one
    // model call at a time so the request cannot finish first.
    for _ in 0..8 {
        if runtime.in_flight() == 0 {
            break;
        }
        gate.permit(1);
        thread::sleep(Duration::from_millis(50));
    }
    gate.open_fully();
    let status = server.call("GET", "/v1/requests/1", None).json();
    assert_eq!(status["status"], "cancelled", "{status}");
}

#[test]
fn a_service_without_executable_models_is_not_ready() {
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
    let server = Server::start(driver.handle());

    let ready = server.call("GET", "/ready", None);
    assert_eq!(ready.status, 503);
    assert_eq!(ready.json()["state"], "no_models");
    let refused = server.call("POST", "/v1/generate", Some(generate("a", false)));
    assert_eq!(
        (refused.status, refused.error_kind().as_str()),
        (503, "not_ready")
    );
}

#[test]
fn disconnecting_while_waiting_for_a_json_result_cancels_the_request() {
    let gate = Gate::closed();
    let (driver, _) = driver(&gate, 1, DriverConfig::default());
    let runtime = driver.handle();
    let server = Server::start(runtime.clone());

    gate.permit(1);
    let client = server.connect("POST", "/v1/generate", Some(&generate("hi", false)));
    gate.wait_until_blocked();
    drop(client);

    // Nothing is written until the result is ready; hyper notices the
    // closed socket on its read side and drops the handler.
    for _ in 0..8 {
        if runtime.in_flight() == 0 {
            break;
        }
        gate.permit(1);
        thread::sleep(Duration::from_millis(50));
    }
    gate.open_fully();
    let status = server.call("GET", "/v1/requests/1", None).json();
    assert_eq!(status["status"], "cancelled", "{status}");
}
