//! A scripted model whose every model call waits on a test-controlled gate.

#![allow(dead_code)]

use std::{
    any::Any,
    sync::{Arc, Condvar, Mutex},
    time::Duration,
};

use ola_core::{GenerationConfig, ManualClock, ModelDescriptor, RequestSpec, Timestamp};
use ola_model::{
    CausalLanguageModel, CausalModel, ExecutionStage, GeneratedToken, GenerationSession,
    ModelExecutionError, SessionMetrics, Tokenizer,
};
use ola_runtime::{Executor, Runtime};
use ola_serve::{Driver, DriverConfig};

/// Every request answers with this text, one UTF-8 byte per token.
pub const REPLY: &str = "héllo, 世界";

/// Prompts starting with this fail during prefill.
pub const FAIL_PROMPT: &str = "fail";

pub const MODEL: &str = "script";

/// Counts model calls the driver may make; each prefill or decode takes one.
#[derive(Clone, Default)]
pub struct Gate(Arc<(Mutex<GateState>, Condvar)>);

#[derive(Default)]
struct GateState {
    /// Remaining calls, or `None` when every call passes.
    permits: Option<usize>,
    /// Calls currently blocked waiting for a permit.
    waiting: usize,
}

impl Gate {
    /// A gate that admits no calls until permits are added.
    pub fn closed() -> Self {
        Self::with(Some(0))
    }

    /// A gate that admits every call.
    pub fn open() -> Self {
        Self::with(None)
    }

    fn with(permits: Option<usize>) -> Self {
        Self(Arc::new((
            Mutex::new(GateState {
                permits,
                waiting: 0,
            }),
            Condvar::new(),
        )))
    }

    pub fn permit(&self, calls: usize) {
        let (state, wake) = &*self.0;
        if let Some(remaining) = state.lock().unwrap().permits.as_mut() {
            *remaining += calls;
        }
        wake.notify_all();
    }

    pub fn open_fully(&self) {
        let (state, wake) = &*self.0;
        state.lock().unwrap().permits = None;
        wake.notify_all();
    }

    /// Waits until the driver is blocked inside a model call.
    pub fn wait_until_blocked(&self) {
        let (state, wake) = &*self.0;
        let mut state = state.lock().unwrap();
        while state.waiting == 0 {
            state = wake
                .wait_timeout(state, Duration::from_millis(5))
                .unwrap()
                .0;
        }
    }

    fn pass(&self) {
        let (state, wake) = &*self.0;
        let mut state = state.lock().unwrap();
        loop {
            match state.permits {
                None => return,
                Some(remaining) if remaining > 0 => {
                    state.permits = Some(remaining - 1);
                    return;
                }
                Some(_) => {
                    state.waiting += 1;
                    wake.notify_all();
                    state = wake.wait(state).unwrap();
                    state.waiting -= 1;
                }
            }
        }
    }
}

/// One token per byte, offset by one; token 0 is end-of-sequence.
pub struct ByteTokenizer;

impl Tokenizer for ByteTokenizer {
    fn encode(&self, text: &str, _: bool) -> Result<Vec<u32>, ModelExecutionError> {
        Ok(text.bytes().map(|byte| u32::from(byte) + 1).collect())
    }

    fn decode(&self, tokens: &[u32], skip_special: bool) -> Result<String, ModelExecutionError> {
        let bytes = tokens
            .iter()
            .filter(|token| !(skip_special && **token == 0))
            .map(|token| u8::try_from(token.saturating_sub(1)).unwrap())
            .collect::<Vec<_>>();
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
}

struct ScriptModel {
    gate: Gate,
}

impl CausalModel for ScriptModel {
    fn create_session(
        &self,
        _: GenerationConfig,
    ) -> Result<Box<dyn GenerationSession>, ModelExecutionError> {
        Ok(Box::new(ScriptSession {
            gate: self.gate.clone(),
            prompt_tokens: 0,
            generated: 0,
        }))
    }
}

struct ScriptSession {
    gate: Gate,
    prompt_tokens: usize,
    generated: usize,
}

impl ScriptSession {
    fn emit(&mut self) -> GeneratedToken {
        let reply = REPLY.as_bytes();
        let token = reply
            .get(self.generated)
            .map_or(0, |byte| u32::from(*byte) + 1);
        self.generated += 1;
        GeneratedToken {
            id: token,
            is_end_of_sequence: token == 0,
        }
    }
}

impl GenerationSession for ScriptSession {
    fn prefill(&mut self, prompt: &[u32]) -> Result<GeneratedToken, ModelExecutionError> {
        self.gate.pass();
        if ByteTokenizer.decode(prompt, true)?.starts_with(FAIL_PROMPT) {
            return Err(ModelExecutionError::new(
                ExecutionStage::Prefill,
                "scripted failure",
            ));
        }
        self.prompt_tokens = prompt.len();
        Ok(self.emit())
    }

    fn decode(&mut self, _: u32) -> Result<GeneratedToken, ModelExecutionError> {
        self.gate.pass();
        Ok(self.emit())
    }

    fn metrics(&self) -> SessionMetrics {
        SessionMetrics {
            prompt_tokens: self.prompt_tokens,
            generated_tokens: self.generated,
            kv_cache_bytes: 0,
        }
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

pub fn request(prompt: &str) -> RequestSpec {
    let mut spec = RequestSpec::text(MODEL, prompt);
    spec.generation.max_new_tokens = 64;
    spec
}

pub fn executor(gate: &Gate, max_active: usize) -> Executor {
    let mut executor = Executor::with_max_active_requests(Runtime::default(), max_active).unwrap();
    executor
        .register_model(
            ModelDescriptor::language(MODEL, "script"),
            CausalLanguageModel::new(ByteTokenizer, ScriptModel { gate: gate.clone() }),
        )
        .unwrap();
    executor
}

pub fn driver(gate: &Gate, max_active: usize, config: DriverConfig) -> (Driver, ManualClock) {
    let clock = ManualClock::new(Timestamp::ZERO);
    let driver = Driver::spawn(executor(gate, max_active), clock.clone(), config).unwrap();
    (driver, clock)
}
