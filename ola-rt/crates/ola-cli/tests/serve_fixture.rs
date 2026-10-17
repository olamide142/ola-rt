//! End-to-end serving against the pinned tiny-Llama fixture.
//!
//! Ignored by default; `make test-fixture MODEL=…` runs them with
//! `OLA_LLAMA_CONFORMANCE_DIR` set. Every output served over HTTP is
//! compared with the same request executed directly on a fresh executor.

use std::{
    io::{BufRead, BufReader, Read, Write},
    net::{SocketAddr, TcpStream},
    path::PathBuf,
    sync::mpsc,
    thread,
    time::Duration,
};

use burn::tensor::Device;
use ola_burn::llama::LlamaFactory;
use ola_core::{GenerationConfig, ModelDescriptor, MonotonicClock, RequestSpec, Timestamp};
use ola_model::CausalLanguageModel;
use ola_runtime::{ComposedPolicy, Executor, ExecutorEvent, MemoryAwareAdmission, Runtime};
use ola_serve::{Driver, DriverConfig, http};
use serde_json::{Value, json};

const MODEL: &str = "tiny";

fn fixture() -> CausalLanguageModel {
    let directory = PathBuf::from(
        std::env::var_os("OLA_LLAMA_CONFORMANCE_DIR")
            .expect("set OLA_LLAMA_CONFORMANCE_DIR to the tiny-Llama fixture"),
    );
    LlamaFactory::new(Device::flex())
        .load_local(&directory)
        .unwrap()
}

fn executor(model: &CausalLanguageModel, max_active: usize) -> Executor {
    let mut executor = Executor::with_max_active_requests(Runtime::default(), max_active)
        .unwrap()
        .with_policy(ComposedPolicy::default().with_admission(MemoryAwareAdmission));
    executor
        .register_model(ModelDescriptor::language(MODEL, "llama"), model.clone())
        .unwrap();
    executor
}

/// The same request run alone on a fresh executor, without the driver.
fn oracle(model: &CausalLanguageModel, prompt: &str, max_new_tokens: usize) -> String {
    let mut executor = executor(model, 1);
    let mut spec = RequestSpec::text(MODEL, prompt);
    spec.generation = GenerationConfig {
        max_new_tokens,
        temperature: 0.0,
        top_p: 1.0,
        seed: None,
    };
    let id = executor.submit(spec, Timestamp::ZERO).unwrap();
    for step in 1.. {
        let iteration = executor
            .step_iteration(Timestamp::from_micros(step))
            .unwrap();
        if iteration
            .events
            .iter()
            .any(|event| matches!(event, ExecutorEvent::Token { finished: true, .. }))
        {
            break;
        }
    }
    executor.take_result(id).unwrap().text
}

struct Server {
    address: SocketAddr,
    driver: Option<Driver>,
    stop: Option<tokio::sync::oneshot::Sender<()>>,
    thread: Option<thread::JoinHandle<()>>,
}

impl Server {
    fn start(model: &CausalLanguageModel, max_active: usize, config: DriverConfig) -> Self {
        let driver =
            Driver::spawn(executor(model, max_active), MonotonicClock::start(), config).unwrap();
        let runtime = driver.handle();
        let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
        let (bound, address) = mpsc::channel();
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
            driver: Some(driver),
            stop: Some(stop),
            thread: Some(thread),
        }
    }

    fn connect(&self, method: &str, path: &str, body: Option<&Value>) -> TcpStream {
        let mut stream = TcpStream::connect(self.address).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(120)))
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

    /// Returns the status and the decoded body.
    fn call(&self, method: &str, path: &str, body: Option<Value>) -> (u16, String) {
        let mut raw = String::new();
        self.connect(method, path, body.as_ref())
            .read_to_string(&mut raw)
            .unwrap();
        let (head, body) = raw.split_once("\r\n\r\n").unwrap();
        let status = head[9..12].parse().unwrap();
        let body = if head
            .to_ascii_lowercase()
            .contains("transfer-encoding: chunked")
        {
            dechunk(body)
        } else {
            body.to_owned()
        };
        (status, body)
    }

    fn json(&self, method: &str, path: &str, body: Option<Value>) -> (u16, Value) {
        let (status, body) = self.call(method, path, body);
        (status, serde_json::from_str(&body).unwrap())
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        if let Some(driver) = self.driver.take() {
            let _ = driver.shutdown();
        }
        let _ = self.stop.take().unwrap().send(());
        let _ = self.thread.take().unwrap().join();
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

/// Server-sent events as `(event, data)` pairs.
fn events(body: &str) -> Vec<(String, Value)> {
    body.split("\n\n")
        .filter_map(|block| {
            let event = block
                .lines()
                .find_map(|line| line.strip_prefix("event: "))?;
            let data = block.lines().find_map(|line| line.strip_prefix("data: "))?;
            Some((event.to_owned(), serde_json::from_str(data).unwrap()))
        })
        .collect()
}

fn generate(prompt: &str, max_new_tokens: usize, stream: bool) -> Value {
    json!({ "model": MODEL, "prompt": prompt, "max_new_tokens": max_new_tokens, "stream": stream })
}

/// Reads a streaming response until its first `token` event.
fn read_first_token(stream: &mut TcpStream) -> BufReader<&mut TcpStream> {
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    while !line.starts_with("event: token") {
        line.clear();
        assert!(
            reader.read_line(&mut line).unwrap() > 0,
            "stream ended early"
        );
    }
    reader
}

#[test]
#[ignore = "requires OLA_LLAMA_CONFORMANCE_DIR"]
fn json_streamed_and_openai_outputs_match_direct_execution() {
    let model = fixture();
    let expected = oracle(&model, "Hello", 8);
    let server = Server::start(&model, 4, DriverConfig::default());

    let (status, result) = server.json("POST", "/v1/generate", Some(generate("Hello", 8, false)));
    assert_eq!(status, 200);
    assert_eq!(result["text"], expected);

    let (status, body) = server.call("POST", "/v1/generate", Some(generate("Hello", 8, true)));
    assert_eq!(status, 200);
    let events = events(&body);
    let streamed = events
        .iter()
        .filter(|(event, _)| event == "token")
        .map(|(_, data)| data["text"].as_str().unwrap())
        .collect::<String>();
    assert_eq!(streamed, expected);
    assert_eq!(events.last().unwrap().1["text"], expected);

    let (status, completion) = server.json(
        "POST",
        "/v1/completions",
        Some(json!({ "model": MODEL, "prompt": "Hello", "max_tokens": 8 })),
    );
    assert_eq!(status, 200);
    assert_eq!(completion["choices"][0]["text"], expected);
}

#[test]
#[ignore = "requires OLA_LLAMA_CONFORMANCE_DIR"]
fn concurrent_clients_each_get_their_single_request_output() {
    let model = fixture();
    let prompts = ["Hello", "The quick brown fox", "Once upon a time", "A"];
    let expected = prompts.map(|prompt| oracle(&model, prompt, 6));
    let server = Server::start(&model, 4, DriverConfig::default());

    let outputs = thread::scope(|scope| {
        let clients = prompts.map(|prompt| {
            let server = &server;
            scope.spawn(move || {
                server.json("POST", "/v1/generate", Some(generate(prompt, 6, false)))
            })
        });
        clients.map(|client| client.join().unwrap())
    });

    for ((status, result), expected) in outputs.iter().zip(&expected) {
        assert_eq!(*status, 200);
        assert_eq!(result["text"], *expected);
    }
}

#[test]
#[ignore = "requires OLA_LLAMA_CONFORMANCE_DIR"]
fn cancelling_a_stream_over_http_releases_its_kv_cache() {
    let model = fixture();
    let server = Server::start(&model, 1, DriverConfig::default());

    let mut client = server.connect("POST", "/v1/generate", Some(&generate("Hello", 200, true)));
    let mut reader = read_first_token(&mut client);
    let (status, _) = server.call("DELETE", "/v1/requests/1", None);
    assert_eq!(status, 204);
    let mut rest = String::new();
    reader.read_to_string(&mut rest).unwrap();
    let ended = events(&dechunk_tail(&rest));
    let (event, data) = ended.last().unwrap();
    assert_eq!(event, "error");
    assert_eq!(data["error"]["kind"], "cancelled");

    let (_, health) = server.json("GET", "/health", None);
    assert_eq!(health["in_flight"], 0);
    assert_eq!(health["kv_cache_bytes"], 0);
    let (_, status) = server.json("GET", "/v1/requests/1", None);
    assert_eq!(status["status"], "cancelled");
}

#[test]
#[ignore = "requires OLA_LLAMA_CONFORMANCE_DIR"]
fn overload_and_failure_are_reported_without_disturbing_running_work() {
    let model = fixture();
    let config = DriverConfig {
        max_in_flight: 1,
        ..DriverConfig::default()
    };
    let server = Server::start(&model, 1, config);
    let expected = oracle(&model, "Hello", 64);

    let mut client = server.connect("POST", "/v1/generate", Some(&generate("Hello", 64, true)));
    let mut reader = read_first_token(&mut client);
    let (status, refused) = server.json("POST", "/v1/generate", Some(generate("Hi", 4, false)));
    assert_eq!(
        (status, refused["error"]["kind"].as_str()),
        (429, Some("overloaded"))
    );
    let mut rest = String::new();
    reader.read_to_string(&mut rest).unwrap();
    let (event, done) = events(&dechunk_tail(&rest)).pop().unwrap();
    assert_eq!(event, "done");
    assert_eq!(done["text"], expected);

    // A token outside the vocabulary fails that request alone.
    let (status, failed) = server.json(
        "POST",
        "/v1/generate",
        Some(json!({ "model": MODEL, "tokens": [u32::MAX], "max_new_tokens": 2 })),
    );
    assert_eq!(status, 500, "{failed}");
    assert_eq!(failed["error"]["kind"], "execution_failed");
    let (status, after) = server.json("POST", "/v1/generate", Some(generate("Hello", 64, false)));
    assert_eq!(status, 200);
    assert_eq!(after["text"], expected);
}

#[test]
#[ignore = "requires OLA_LLAMA_CONFORMANCE_DIR"]
fn draining_lets_a_running_stream_finish_and_refuses_new_requests() {
    let model = fixture();
    let mut server = Server::start(&model, 1, DriverConfig::default());
    let expected = oracle(&model, "Hello", 24);

    let mut client = server.connect("POST", "/v1/generate", Some(&generate("Hello", 24, true)));
    let mut reader = read_first_token(&mut client);
    let driver = server.driver.take().unwrap();
    let draining = thread::spawn(move || driver.drain(Duration::from_secs(60)));
    loop {
        let (status, _) = server.call("GET", "/ready", None);
        if status == 503 {
            break;
        }
        thread::sleep(Duration::from_millis(5));
    }
    let (status, refused) = server.json("POST", "/v1/generate", Some(generate("Hi", 4, false)));
    assert_eq!(
        (status, refused["error"]["kind"].as_str()),
        (503, Some("shutting_down"))
    );

    let mut rest = String::new();
    reader.read_to_string(&mut rest).unwrap();
    let (event, done) = events(&dechunk_tail(&rest)).pop().unwrap();
    assert_eq!(event, "done");
    assert_eq!(done["text"], expected);
    let executor = draining.join().unwrap().unwrap();
    assert!(!executor.has_active_request());
}

/// Decodes the remainder of a chunked body read after its headers and some
/// chunks were already consumed line by line.
fn dechunk_tail(rest: &str) -> String {
    // Chunk-size lines are bare hexadecimal; dropping them leaves the
    // event-stream text, whose own lines never consist of hex digits alone.
    rest.split("\r\n")
        .filter(|line| line.is_empty() || !line.chars().all(|c| c.is_ascii_hexdigit()))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
#[ignore = "requires OLA_LLAMA_CONFORMANCE_DIR"]
fn a_served_request_is_traced_from_arrival_to_completion() {
    let model = fixture();
    let config = DriverConfig {
        trace: true,
        ..DriverConfig::default()
    };
    let server = Server::start(&model, 2, config);

    let (status, result) = server.json("POST", "/v1/generate", Some(generate("Hello", 6, false)));
    assert_eq!(status, 200);
    let id = result["id"].as_u64().unwrap();
    let (status, trace) = server.json("GET", &format!("/v1/requests/{id}/trace"), None);
    assert_eq!(status, 200);
    let records = trace["records"].as_array().unwrap();
    let events = records
        .iter()
        .map(|record| record["event"].as_str().unwrap())
        .collect::<Vec<_>>();

    assert_eq!(events.first(), Some(&"arrived"));
    assert_eq!(events.last(), Some(&"completed"));
    assert!(events.contains(&"admission"));
    // Each token follows the invocation that produced it and carries its
    // finish time; together they are the response's token times.
    let mut token_times = Vec::new();
    for (index, record) in records.iter().enumerate() {
        if record["event"] == "token" {
            let invocation = &records[index - 1];
            assert_eq!(invocation["event"], "invocation", "{records:?}");
            assert_eq!(record["at_us"], invocation["finished_us"]);
            assert!(invocation["at_us"].as_u64() < invocation["finished_us"].as_u64());
            assert_eq!(record["iteration"], invocation["iteration"]);
            token_times.push(record["at_us"].as_u64().unwrap());
        }
    }
    assert_eq!(json!(token_times), result["token_us"]);
    assert!(token_times.windows(2).all(|pair| pair[0] < pair[1]));
    let (_, health) = server.json("GET", "/health", None);
    assert!(health["busy_us"].as_u64().unwrap() > 0);
}

#[test]
#[ignore = "requires OLA_LLAMA_CONFORMANCE_DIR"]
fn preempted_requests_on_the_fixture_match_their_solo_output() {
    let directory = PathBuf::from(std::env::var_os("OLA_LLAMA_CONFORMANCE_DIR").unwrap());
    let tight = LlamaFactory::new(Device::flex())
        .with_kv_cache(ola_burn::llama::LlamaKvCacheConfig::new(32, 4))
        .load_local(&directory)
        .unwrap();
    let model = fixture();
    let prompts = ["Hello", "The quick brown fox", "Once upon a time"];
    let reference = prompts.map(|prompt| oracle(&model, prompt, 16));

    let mut executor = Executor::with_max_active_requests(Runtime::default(), 3)
        .unwrap()
        .with_policy(ComposedPolicy::default().with_preemption(ola_runtime::NewestFirst));
    executor
        .register_model(ModelDescriptor::language(MODEL, "llama"), tight)
        .unwrap();
    let ids = prompts.map(|prompt| {
        let mut spec = RequestSpec::text(MODEL, prompt);
        spec.generation = GenerationConfig {
            max_new_tokens: 16,
            temperature: 0.0,
            top_p: 1.0,
            seed: None,
        };
        executor.submit(spec, Timestamp::ZERO).unwrap()
    });
    let mut preemptions = 0;
    for step in 1..=500 {
        let iteration = executor
            .step_iteration(Timestamp::from_micros(step))
            .unwrap();
        for event in &iteration.events {
            match event {
                ExecutorEvent::Preempted { .. } => preemptions += 1,
                ExecutorEvent::Failed { message, .. } => panic!("{message}"),
                _ => {}
            }
        }
        if iteration.is_idle() {
            break;
        }
    }

    assert!(preemptions > 0, "the pool must force preemption");
    for (id, expected) in ids.iter().zip(&reference) {
        assert_eq!(&executor.take_result(*id).unwrap().text, expected);
    }
}

#[test]
#[ignore = "requires OLA_LLAMA_CONFORMANCE_DIR"]
fn prefix_cache_reuse_on_the_fixture_matches_recomputation() {
    let directory = PathBuf::from(std::env::var_os("OLA_LLAMA_CONFORMANCE_DIR").unwrap());
    let cached = LlamaFactory::new(Device::flex())
        .with_kv_cache(ola_burn::llama::LlamaKvCacheConfig::new(512, 4).with_prefix_cache())
        .load_local(&directory)
        .unwrap();
    let model = fixture();
    let system = "You are a careful assistant. Answer briefly and precisely. Question: ";
    let prompts = [
        format!("{system}what is two plus two?"),
        format!("{system}name a primary colour."),
    ];
    let reference = prompts.clone().map(|prompt| oracle(&model, &prompt, 12));

    let mut executor = Executor::default().with_prefill_chunk_size(64).unwrap();
    executor
        .register_model(ModelDescriptor::language(MODEL, "llama"), cached.clone())
        .unwrap();
    let mut step = 0;
    for (prompt, expected) in prompts.iter().zip(&reference) {
        let mut spec = RequestSpec::text(MODEL, prompt.as_str());
        spec.generation = GenerationConfig {
            max_new_tokens: 12,
            temperature: 0.0,
            top_p: 1.0,
            seed: None,
        };
        let id = executor.submit(spec, Timestamp::from_micros(step)).unwrap();
        loop {
            step += 1;
            if executor
                .step_iteration(Timestamp::from_micros(step))
                .unwrap()
                .is_idle()
            {
                break;
            }
        }
        assert_eq!(&executor.take_result(id).unwrap().text, expected);
    }
    let reused = cached.kv_pool_status().unwrap().prefix_hit_tokens;
    assert!(
        reused >= 8,
        "the second prompt reused the shared prefix ({reused} tokens)"
    );
}
