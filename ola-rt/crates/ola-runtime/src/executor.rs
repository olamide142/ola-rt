use std::{collections::BTreeMap, error::Error, fmt};

use ola_core::{Input, ModelDescriptor, ModelId, Request, RequestId, RequestSpec, Timestamp};
use ola_model::{
    CausalLanguageModel, GeneratedToken, GenerationSession, ModelExecutionError, SessionMetrics,
};

use crate::{FifoScheduler, Runtime, RuntimeError, RuntimeState, Scheduler};

/// Final output retained after successful execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionResult {
    /// Request that produced this output.
    pub request_id: RequestId,
    /// Generated token IDs in emission order.
    pub token_ids: Vec<u32>,
    /// Artifact-tokenizer decoding of `token_ids`.
    pub text: String,
    /// Final model-session measurements captured before cache release.
    pub metrics: SessionMetrics,
}

/// One observable outcome from a synchronous executor step.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExecutorEvent {
    /// Neither queued nor active work was available.
    Idle,
    /// One token was emitted by prefill or decode.
    Token {
        /// Request that produced the token.
        request_id: RequestId,
        /// Selected token and EOS indication.
        token: GeneratedToken,
        /// Number of output tokens emitted for this request so far.
        generated_tokens: usize,
        /// Physical KV bytes currently attributed to the request.
        kv_cache_bytes: u64,
        /// Whether the request completed with this token.
        finished: bool,
    },
    /// Model preparation or execution failed and the request became terminal.
    Failed {
        /// Request that failed.
        request_id: RequestId,
        /// Stable human-readable diagnostic retained in runtime state.
        message: String,
    },
}

/// Failure to configure or control the executor itself.
#[derive(Debug, Eq, PartialEq)]
pub enum ExecutorError {
    /// Runtime admission or lifecycle operation failed.
    Runtime(RuntimeError),
    /// Executable model attachment referenced absent metadata.
    ModelNotRegistered(ModelId),
    /// An executable was attached more than once under one model ID.
    DuplicateExecutableModel(ModelId),
}

impl fmt::Display for ExecutorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Runtime(error) => error.fmt(formatter),
            Self::ModelNotRegistered(model) => {
                write!(
                    formatter,
                    "model `{model}` has no registered runtime metadata"
                )
            }
            Self::DuplicateExecutableModel(model) => {
                write!(
                    formatter,
                    "model `{model}` already has an executable attached"
                )
            }
        }
    }
}

impl Error for ExecutorError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Runtime(error) => Some(error),
            _ => None,
        }
    }
}

impl From<RuntimeError> for ExecutorError {
    fn from(value: RuntimeError) -> Self {
        Self::Runtime(value)
    }
}

struct ActiveGeneration {
    request_id: RequestId,
    model: CausalLanguageModel,
    session: Box<dyn GenerationSession>,
    generated: Vec<u32>,
    last_token: GeneratedToken,
    max_new_tokens: usize,
    last_step_at: Timestamp,
}

/// Deterministic single-request synchronous model executor.
///
/// Each [`step`](Self::step) performs at most one model operation: a queued
/// request's prefill or an active request's single-token decode. This makes the
/// execution boundary observable and testable before asynchronous execution
/// and continuous batching are introduced.
pub struct Executor<S = FifoScheduler> {
    runtime: Runtime<S>,
    models: BTreeMap<ModelId, CausalLanguageModel>,
    active: Option<ActiveGeneration>,
    results: BTreeMap<RequestId, ExecutionResult>,
}

impl Default for Executor<FifoScheduler> {
    fn default() -> Self {
        Self::new(Runtime::default())
    }
}

impl<S: Scheduler> Executor<S> {
    /// Wraps a runtime with no attached executable models.
    pub fn new(runtime: Runtime<S>) -> Self {
        Self {
            runtime,
            models: BTreeMap::new(),
            active: None,
            results: BTreeMap::new(),
        }
    }

    /// Registers runtime metadata and its corresponding executable atomically.
    pub fn register_model(
        &mut self,
        descriptor: ModelDescriptor,
        model: CausalLanguageModel,
    ) -> Result<(), ExecutorError> {
        let id = descriptor.id.clone();
        if self.models.contains_key(&id) {
            return Err(ExecutorError::DuplicateExecutableModel(id));
        }
        self.runtime.register_model(descriptor)?;
        self.models.insert(id, model);
        Ok(())
    }

    /// Attaches an executable to metadata already registered on the runtime.
    pub fn attach_model(
        &mut self,
        id: ModelId,
        model: CausalLanguageModel,
    ) -> Result<(), ExecutorError> {
        if !self.runtime.state().models().contains(&id) {
            return Err(ExecutorError::ModelNotRegistered(id));
        }
        if self.models.contains_key(&id) {
            return Err(ExecutorError::DuplicateExecutableModel(id));
        }
        self.models.insert(id, model);
        Ok(())
    }

    /// Validates and queues a request on the owned runtime.
    pub fn submit(
        &mut self,
        spec: RequestSpec,
        arrival_at: Timestamp,
    ) -> Result<RequestId, ExecutorError> {
        Ok(self.runtime.submit(spec, arrival_at)?)
    }

    /// Cancels queued or active work and releases request-local model state.
    pub fn cancel(
        &mut self,
        request_id: RequestId,
        cancelled_at: Timestamp,
    ) -> Result<(), ExecutorError> {
        self.runtime.cancel(request_id, cancelled_at)?;
        if self
            .active
            .as_ref()
            .is_some_and(|active| active.request_id == request_id)
        {
            self.active = None;
            self.refresh_kv_memory();
        }
        Ok(())
    }

    /// Performs one prefill or decode operation at the supplied timestamp.
    pub fn step(&mut self, at: Timestamp) -> Result<ExecutorEvent, ExecutorError> {
        match self.active.take() {
            Some(active) if at < active.last_step_at => {
                let request = active.request_id;
                let previous = active.last_step_at;
                self.active = Some(active);
                Err(RuntimeError::TimestampRegression {
                    request,
                    event: "executor step",
                    previous,
                    observed: at,
                }
                .into())
            }
            Some(active) => self.decode(active, at),
            None => self.start_next(at),
        }
    }

    /// Returns runtime lifecycle and memory state.
    pub fn state(&self) -> &RuntimeState {
        self.runtime.state()
    }

    /// Returns a completed result retained by request ID.
    pub fn result(&self, request_id: RequestId) -> Option<&ExecutionResult> {
        self.results.get(&request_id)
    }

    /// Returns whether one request currently owns a generation session.
    pub const fn has_active_request(&self) -> bool {
        self.active.is_some()
    }

    /// Borrows the underlying runtime for read-only inspection.
    pub const fn runtime(&self) -> &Runtime<S> {
        &self.runtime
    }

    fn start_next(&mut self, at: Timestamp) -> Result<ExecutorEvent, ExecutorError> {
        let Some(lease) = self.runtime.schedule_next(at)? else {
            return Ok(ExecutorEvent::Idle);
        };
        let request = self
            .runtime
            .state()
            .request(lease.request_id)
            .expect("a scheduled request retains its runtime record")
            .request
            .clone();
        let Some(model) = self.models.get(&lease.model_id).cloned() else {
            return self.fail_started(
                lease.request_id,
                format!("model `{}` has no executable attached", lease.model_id),
                at,
            );
        };

        let prompt_tokens = match prepare_prompt(&model, &request) {
            Ok(tokens) => tokens,
            Err(error) => return self.fail_started(lease.request_id, error.to_string(), at),
        };
        let mut session = match model.create_session(request.spec.generation.clone()) {
            Ok(session) => session,
            Err(error) => return self.fail_started(lease.request_id, error.to_string(), at),
        };
        let token = match session.prefill(&prompt_tokens) {
            Ok(token) => token,
            Err(error) => return self.fail_started(lease.request_id, error.to_string(), at),
        };
        self.runtime.mark_first_token(lease.request_id, at)?;
        let active = ActiveGeneration {
            request_id: lease.request_id,
            model,
            session,
            generated: vec![token.id],
            last_token: token,
            max_new_tokens: request.spec.generation.max_new_tokens,
            last_step_at: at,
        };
        self.observe_token(active, at)
    }

    fn decode(
        &mut self,
        mut active: ActiveGeneration,
        at: Timestamp,
    ) -> Result<ExecutorEvent, ExecutorError> {
        let token = match active.session.decode(active.last_token.id) {
            Ok(token) => token,
            Err(error) => {
                return self.fail_started(active.request_id, error.to_string(), at);
            }
        };
        active.generated.push(token.id);
        active.last_token = token;
        active.last_step_at = at;
        self.observe_token(active, at)
    }

    fn observe_token(
        &mut self,
        active: ActiveGeneration,
        at: Timestamp,
    ) -> Result<ExecutorEvent, ExecutorError> {
        let metrics = active.session.metrics();
        let finished =
            active.last_token.is_end_of_sequence || active.generated.len() >= active.max_new_tokens;
        let event = ExecutorEvent::Token {
            request_id: active.request_id,
            token: active.last_token,
            generated_tokens: active.generated.len(),
            kv_cache_bytes: metrics.kv_cache_bytes,
            finished,
        };

        if finished {
            let text = match active.model.tokenizer().decode(&active.generated, true) {
                Ok(text) => text,
                Err(error) => return self.fail_started(active.request_id, error.to_string(), at),
            };
            self.runtime.complete(active.request_id, at)?;
            self.results.insert(
                active.request_id,
                ExecutionResult {
                    request_id: active.request_id,
                    token_ids: active.generated,
                    text,
                    metrics,
                },
            );
            drop(active.session);
            self.refresh_kv_memory();
        } else {
            self.active = Some(active);
            self.refresh_kv_memory();
        }
        Ok(event)
    }

    fn fail_started(
        &mut self,
        request_id: RequestId,
        message: String,
        at: Timestamp,
    ) -> Result<ExecutorEvent, ExecutorError> {
        self.runtime.fail(request_id, message.clone(), at)?;
        self.refresh_kv_memory();
        Ok(ExecutorEvent::Failed {
            request_id,
            message,
        })
    }

    fn refresh_kv_memory(&mut self) {
        self.runtime.state_mut().memory.kv_cache_bytes = self
            .active
            .as_ref()
            .map_or(0, |active| active.session.metrics().kv_cache_bytes);
    }
}

fn prepare_prompt(
    model: &CausalLanguageModel,
    request: &Request,
) -> Result<Vec<u32>, ModelExecutionError> {
    match request.spec.inputs.as_slice() {
        [Input::Text(text)] => model.tokenizer().encode(text, true),
        [Input::Tokens(tokens)] if !tokens.is_empty() => Ok(tokens.clone()),
        [Input::Tokens(_)] => Err(ModelExecutionError::new(
            ola_model::ExecutionStage::Tokenize,
            "token input cannot be empty",
        )),
        [_] => Err(ModelExecutionError::new(
            ola_model::ExecutionStage::Tokenize,
            "the initial executor supports text or token input only",
        )),
        _ => Err(ModelExecutionError::new(
            ola_model::ExecutionStage::Tokenize,
            "the initial executor requires exactly one input",
        )),
    }
}

#[cfg(test)]
mod tests {
    use ola_core::GenerationConfig;
    use ola_model::{
        CausalModel, ExecutionStage, GeneratedToken, GenerationSession, SessionMetrics, Tokenizer,
    };

    use super::*;
    use crate::RequestStatus;

    #[derive(Clone, Copy)]
    struct NumericTokenizer;

    impl Tokenizer for NumericTokenizer {
        fn encode(
            &self,
            text: &str,
            _add_special_tokens: bool,
        ) -> Result<Vec<u32>, ModelExecutionError> {
            text.split_whitespace()
                .map(|piece| {
                    piece.parse().map_err(|_| {
                        ModelExecutionError::new(ExecutionStage::Tokenize, "expected integers")
                    })
                })
                .collect()
        }

        fn decode(
            &self,
            token_ids: &[u32],
            _skip_special_tokens: bool,
        ) -> Result<String, ModelExecutionError> {
            Ok(token_ids
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(" "))
        }
    }

    struct IncrementModel;

    impl CausalModel for IncrementModel {
        fn create_session(
            &self,
            _generation: GenerationConfig,
        ) -> Result<Box<dyn GenerationSession>, ModelExecutionError> {
            Ok(Box::new(IncrementSession::default()))
        }
    }

    #[derive(Default)]
    struct IncrementSession {
        metrics: SessionMetrics,
        ready: bool,
    }

    impl GenerationSession for IncrementSession {
        fn prefill(
            &mut self,
            prompt_tokens: &[u32],
        ) -> Result<GeneratedToken, ModelExecutionError> {
            let last = prompt_tokens.last().copied().ok_or_else(|| {
                ModelExecutionError::new(ExecutionStage::Prefill, "prompt cannot be empty")
            })?;
            self.ready = true;
            self.metrics.prompt_tokens = prompt_tokens.len();
            self.metrics.generated_tokens = 1;
            self.metrics.kv_cache_bytes = prompt_tokens.len() as u64 * 8;
            Ok(GeneratedToken {
                id: last + 1,
                is_end_of_sequence: false,
            })
        }

        fn decode(&mut self, input_token: u32) -> Result<GeneratedToken, ModelExecutionError> {
            if !self.ready {
                return Err(ModelExecutionError::new(
                    ExecutionStage::Decode,
                    "decode requires prefill",
                ));
            }
            self.metrics.generated_tokens += 1;
            self.metrics.kv_cache_bytes += 8;
            Ok(GeneratedToken {
                id: input_token + 1,
                is_end_of_sequence: false,
            })
        }

        fn metrics(&self) -> SessionMetrics {
            self.metrics
        }
    }

    fn artifact() -> CausalLanguageModel {
        CausalLanguageModel::new(NumericTokenizer, IncrementModel)
    }

    fn request(prompt: &str, max_new_tokens: usize) -> RequestSpec {
        let mut request = RequestSpec::text("tiny", prompt);
        request.generation = GenerationConfig {
            max_new_tokens,
            temperature: 0.0,
            top_p: 1.0,
            seed: Some(0),
        };
        request
    }

    #[test]
    fn drives_prefill_and_decode_to_a_retained_result() {
        let mut executor = Executor::default();
        executor
            .register_model(ModelDescriptor::language("tiny", "increment"), artifact())
            .unwrap();
        let id = executor
            .submit(request("1 2", 3), Timestamp::from_micros(10))
            .unwrap();

        let first = executor.step(Timestamp::from_micros(20)).unwrap();
        assert!(matches!(
            first,
            ExecutorEvent::Token {
                token: GeneratedToken { id: 3, .. },
                generated_tokens: 1,
                kv_cache_bytes: 16,
                finished: false,
                ..
            }
        ));
        assert_eq!(executor.state().memory.kv_cache_bytes, 16);
        assert_eq!(
            executor.state().request(id).unwrap().status,
            RequestStatus::Running
        );

        executor.step(Timestamp::from_micros(30)).unwrap();
        let last = executor.step(Timestamp::from_micros(40)).unwrap();
        assert!(matches!(
            last,
            ExecutorEvent::Token {
                token: GeneratedToken { id: 5, .. },
                generated_tokens: 3,
                kv_cache_bytes: 32,
                finished: true,
                ..
            }
        ));
        assert_eq!(executor.state().memory.kv_cache_bytes, 0);
        assert_eq!(
            executor.state().request(id).unwrap().status,
            RequestStatus::Completed
        );
        assert_eq!(
            executor.result(id),
            Some(&ExecutionResult {
                request_id: id,
                token_ids: vec![3, 4, 5],
                text: "3 4 5".into(),
                metrics: SessionMetrics {
                    prompt_tokens: 2,
                    generated_tokens: 3,
                    kv_cache_bytes: 32,
                },
            })
        );
    }

    #[test]
    fn execution_failure_becomes_terminal_without_leaking_cache_accounting() {
        let mut executor = Executor::default();
        executor
            .register_model(ModelDescriptor::language("tiny", "increment"), artifact())
            .unwrap();
        let id = executor
            .submit(request("not-a-number", 2), Timestamp::ZERO)
            .unwrap();

        let event = executor.step(Timestamp::from_micros(1)).unwrap();

        assert!(matches!(event, ExecutorEvent::Failed { request_id, .. } if request_id == id));
        assert!(matches!(
            &executor.state().request(id).unwrap().status,
            RequestStatus::Failed(message) if message.contains("expected integers")
        ));
        assert!(!executor.has_active_request());
        assert_eq!(executor.state().memory.kv_cache_bytes, 0);
    }

    #[test]
    fn cancelling_active_work_drops_session_memory() {
        let mut executor = Executor::default();
        executor
            .register_model(ModelDescriptor::language("tiny", "increment"), artifact())
            .unwrap();
        let id = executor.submit(request("1 2", 3), Timestamp::ZERO).unwrap();
        executor.step(Timestamp::from_micros(1)).unwrap();

        executor.cancel(id, Timestamp::from_micros(2)).unwrap();

        assert_eq!(
            executor.state().request(id).unwrap().status,
            RequestStatus::Cancelled
        );
        assert!(!executor.has_active_request());
        assert_eq!(executor.state().memory.kv_cache_bytes, 0);
    }

    #[test]
    fn rejects_a_regressive_step_without_dropping_active_state() {
        let mut executor = Executor::default();
        executor
            .register_model(ModelDescriptor::language("tiny", "increment"), artifact())
            .unwrap();
        let id = executor
            .submit(request("1 2", 2), Timestamp::from_micros(10))
            .unwrap();
        executor.step(Timestamp::from_micros(20)).unwrap();

        let error = executor.step(Timestamp::from_micros(19)).unwrap_err();

        assert_eq!(
            error,
            ExecutorError::Runtime(RuntimeError::TimestampRegression {
                request: id,
                event: "executor step",
                previous: Timestamp::from_micros(20),
                observed: Timestamp::from_micros(19),
            })
        );
        assert!(executor.has_active_request());
        assert_eq!(executor.state().memory.kv_cache_bytes, 16);
        assert!(matches!(
            executor.step(Timestamp::from_micros(30)).unwrap(),
            ExecutorEvent::Token { finished: true, .. }
        ));
    }
}
