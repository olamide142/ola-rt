//! Request traces from a real executor run.

use std::any::Any;

use ola_core::{GenerationConfig, ModelDescriptor, RequestId, RequestSpec, Timestamp};
use ola_model::{
    CausalLanguageModel, CausalModel, GeneratedToken, GenerationSession, ModelExecutionError,
    SessionMetrics, Tokenizer,
};
use ola_observe::{
    TraceEvent, TraceRecord, Tracer,
    trace::{read_jsonl, write_jsonl},
};
use ola_runtime::{Executor, Runtime};

struct Numbers;

impl Tokenizer for Numbers {
    fn encode(&self, text: &str, _: bool) -> Result<Vec<u32>, ModelExecutionError> {
        Ok(text
            .split_whitespace()
            .map(|word| word.parse().unwrap())
            .collect())
    }

    fn decode(&self, tokens: &[u32], _: bool) -> Result<String, ModelExecutionError> {
        Ok(tokens
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(" "))
    }
}

/// Answers each prompt with consecutive numbers after its last token.
struct Counting;

impl CausalModel for Counting {
    fn create_session(
        &self,
        _: GenerationConfig,
    ) -> Result<Box<dyn GenerationSession>, ModelExecutionError> {
        Ok(Box::new(CountingSession { next: 0 }))
    }
}

struct CountingSession {
    next: u32,
}

impl GenerationSession for CountingSession {
    fn prefill(&mut self, prompt: &[u32]) -> Result<GeneratedToken, ModelExecutionError> {
        self.next = prompt.last().copied().unwrap_or(0) + 1;
        self.decode(0)
    }

    fn decode(&mut self, _: u32) -> Result<GeneratedToken, ModelExecutionError> {
        self.next += 1;
        Ok(GeneratedToken {
            id: self.next - 1,
            is_end_of_sequence: false,
        })
    }

    fn metrics(&self) -> SessionMetrics {
        SessionMetrics::default()
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

fn request(prompt: &str, max_new_tokens: usize) -> RequestSpec {
    let mut spec = RequestSpec::text("counting", prompt);
    spec.generation = GenerationConfig {
        max_new_tokens,
        temperature: 0.0,
        top_p: 1.0,
        seed: None,
    };
    spec
}

/// The event names of one request's records, in order.
fn kinds(records: &[TraceRecord], request: RequestId) -> Vec<String> {
    records
        .iter()
        .filter(|record| record.request == request.get())
        .map(|record| {
            let json = serde_json::to_value(&record.event).unwrap();
            let mut kind = json["event"].as_str().unwrap().to_owned();
            if let Some(outcome) = json.get("outcome").and_then(|value| value.as_str()) {
                kind = format!("{kind}:{outcome}");
            }
            if let Some(selected) = json.get("selected").and_then(serde_json::Value::as_bool) {
                kind = format!("{kind}:{}", if selected { "selected" } else { "deferred" });
            }
            kind
        })
        .collect()
}

#[test]
fn traces_follow_each_request_from_arrival_to_its_outcome() {
    let mut executor = Executor::with_max_active_requests(Runtime::default(), 1).unwrap();
    executor
        .register_model(
            ModelDescriptor::language("counting", "counting"),
            CausalLanguageModel::new(Numbers, Counting),
        )
        .unwrap();
    let first = executor
        .submit(request("1 2", 2), Timestamp::from_micros(1))
        .unwrap();
    let second = executor
        .submit(request("7", 1), Timestamp::from_micros(2))
        .unwrap();
    let cancelled = executor
        .submit(request("9", 1), Timestamp::from_micros(3))
        .unwrap();
    let mut tracer = Tracer::default();
    let mut records = Vec::new();

    executor
        .cancel(cancelled, Timestamp::from_micros(4))
        .unwrap();
    records.extend(tracer.observe_cancellation(cancelled, executor.state()));
    for step in [10, 20, 30] {
        let iteration = executor
            .step_iteration(Timestamp::from_micros(step))
            .unwrap();
        records.extend(tracer.observe_iteration(&iteration, executor.state()));
    }

    assert_eq!(
        kinds(&records, first),
        [
            "arrived",
            "ranked",
            "admission:admitted",
            "selection:selected",
            "invocation",
            "token",
            // Iteration 2: its decode reaches the limit, freeing its slot.
            "selection:selected",
            "invocation",
            "token",
            "completed",
        ]
    );
    assert_eq!(
        kinds(&records, second),
        [
            "arrived",
            "ranked",
            "admission:no_slot",
            // Iteration 2 reuses the predicted release.
            "ranked",
            "admission:admitted",
            "selection:selected",
            "invocation",
            "token",
            "completed",
        ]
    );
    assert_eq!(kinds(&records, cancelled), ["arrived", "cancelled"]);

    // Records carry the iteration and the time each fact became true.
    let tokens = records
        .iter()
        .filter(|record| matches!(record.event, TraceEvent::Token { .. }))
        .map(|record| (record.request, record.iteration, record.at_us))
        .collect::<Vec<_>>();
    assert_eq!(
        tokens,
        [
            (first.get(), Some(1), 10),
            (first.get(), Some(2), 20),
            (second.get(), Some(2), 20),
        ]
    );
    let arrival = &records
        .iter()
        .find(|record| record.request == second.get())
        .unwrap();
    assert_eq!(arrival.at_us, 2);
    assert_eq!(arrival.iteration, None);
    assert!(
        records
            .iter()
            .filter(|record| record.event.is_terminal())
            .count()
            == 3
    );

    // Nothing more is recorded once every request has ended.
    let idle = executor.step_iteration(Timestamp::from_micros(40)).unwrap();
    assert!(tracer.observe_iteration(&idle, executor.state()).is_empty());

    let mut jsonl = Vec::new();
    write_jsonl(&records, &mut jsonl).unwrap();
    assert_eq!(read_jsonl(jsonl.as_slice()).unwrap(), records);
    let first_line = String::from_utf8(jsonl).unwrap();
    let first_line = first_line.lines().next().unwrap();
    assert!(first_line.starts_with(r#"{"request":3,"at_us":3,"event":"arrived""#));
}
