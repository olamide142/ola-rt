//! Executable documentation for model, architecture, and execution selection.

use ola_core::{ArchitectureId, GenerationConfig, ModelDescriptor};
use ola_model::{
    ArchitectureFactory, ArchitectureLoadError, ArchitectureRegistry, CausalLanguageModel,
    CausalModel, ExecutionStage, GeneratedToken, GenerationSession, ModelCatalog, ModelDefinition,
    ModelExecutionError, SessionMetrics, Tokenizer,
};

struct StubArchitecture {
    id: ArchitectureId,
}

impl ArchitectureFactory<CausalLanguageModel> for StubArchitecture {
    fn architecture(&self) -> &ArchitectureId {
        &self.id
    }

    fn load(
        &self,
        _definition: &ModelDefinition,
    ) -> Result<CausalLanguageModel, ArchitectureLoadError> {
        Ok(CausalLanguageModel::new(NumericTokenizer, IncrementModel))
    }
}

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
                    ModelExecutionError::new(
                        ExecutionStage::Tokenize,
                        "example accepts integer tokens",
                    )
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
    prefill_complete: bool,
}

impl GenerationSession for IncrementSession {
    fn prefill(&mut self, prompt_tokens: &[u32]) -> Result<GeneratedToken, ModelExecutionError> {
        if self.prefill_complete {
            return Err(ModelExecutionError::new(
                ExecutionStage::Prefill,
                "prefill may only run once",
            ));
        }
        let last = prompt_tokens.last().copied().ok_or_else(|| {
            ModelExecutionError::new(ExecutionStage::Prefill, "prompt cannot be empty")
        })?;
        self.prefill_complete = true;
        self.metrics.prompt_tokens = prompt_tokens.len();
        self.metrics.generated_tokens = 1;
        Ok(GeneratedToken {
            id: last + 1,
            is_end_of_sequence: false,
        })
    }

    fn decode(&mut self, input_token: u32) -> Result<GeneratedToken, ModelExecutionError> {
        if !self.prefill_complete {
            return Err(ModelExecutionError::new(
                ExecutionStage::Decode,
                "decode requires prefill",
            ));
        }
        self.metrics.generated_tokens += 1;
        Ok(GeneratedToken {
            id: input_token + 1,
            is_end_of_sequence: input_token >= 4,
        })
    }

    fn metrics(&self) -> SessionMetrics {
        self.metrics
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut catalog = ModelCatalog::default();
    catalog.register(ModelDefinition::local(
        ModelDescriptor::language("tiny-random-llama", "llama"),
        "models/tiny-random-llama",
    ))?;
    catalog.register(ModelDefinition::local(
        ModelDescriptor::language("tinyllama-chat", "llama"),
        "models/tinyllama-chat",
    ))?;
    catalog.register(ModelDefinition::local(
        ModelDescriptor::language("small-qwen", "qwen2"),
        "models/small-qwen",
    ))?;

    let mut architectures = ArchitectureRegistry::default();
    architectures.register(StubArchitecture { id: "llama".into() })?;
    architectures.register(StubArchitecture { id: "qwen2".into() })?;

    let selected = catalog.get(&"tinyllama-chat".into())?;
    let loaded = architectures.load(selected)?;
    let prompt = loaded.inner().tokenizer().encode("1 2", true)?;
    let mut session = loaded.inner().create_session(GenerationConfig::default())?;
    let first = session.prefill(&prompt)?;
    let second = session.decode(first.id)?;

    println!(
        "model={} architecture={} generated=[{}, {}] metrics={:?}",
        loaded.descriptor().id,
        loaded.descriptor().architecture,
        first.id,
        second.id,
        session.metrics(),
    );
    Ok(())
}
