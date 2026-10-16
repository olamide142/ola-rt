use std::{
    path::{Path, PathBuf},
    sync::mpsc::{self, Receiver, Sender, TryRecvError},
    thread,
    time::{Duration, Instant},
};

use burn::tensor::Device;
use ola_bench::{
    ExperimentCategory, ExperimentContext, ExperimentDescriptor, ExperimentKind,
    ExperimentRegistry, Requirement, TimeDomain,
};
use ola_burn::llama::LlamaFactory;
use ola_core::{GenerationConfig, ModelDescriptor, RequestSpec, Timestamp};
use ola_model::{CausalLanguageModel, SessionMetrics};
use ola_runtime::{ComposedPolicy, Executor, ExecutorEvent, MemoryAwareAdmission};

use crate::{
    ChatMessage, ChatRole, ConnectionState, RunState, RuntimeSummary, UiBackend, UiBackendError,
    WorkItem, WorkspaceSnapshot,
};

const MAX_ACTIVITY: usize = 100;

/// Compute backend selected for local model execution.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum LocalDevice {
    /// Portable CPU execution through Burn NdArray.
    #[default]
    Cpu,
    /// GPU execution through Burn WGPU and the platform graphics API.
    Wgpu,
    /// Native CubeCL CUDA execution; requires a local CUDA toolkit.
    Cuda,
}

impl LocalDevice {
    const fn label(self) -> &'static str {
        match self {
            Self::Cpu => "local / Burn Flex<f32> / CPU",
            Self::Wgpu => "local / Burn WGPU<f32> / GPU",
            Self::Cuda => "local / Burn CUDA<f32> / GPU",
        }
    }
}

/// In-process adapter connecting the terminal shell to OlaRT foundation code.
///
/// Model loading and all launched work execute on one background worker. The
/// UI thread only sends commands and drains small result events, keeping input
/// and rendering responsive even when CPU generation is slow.
pub struct LocalBackend {
    commands: Sender<WorkerCommand>,
    events: Receiver<WorkerEvent>,
    snapshot: WorkspaceSnapshot,
    descriptors: Vec<ExperimentDescriptor>,
}

impl LocalBackend {
    /// Starts a local adapter, optionally loading one Hugging Face Llama
    /// artifact directory on the CPU in the background.
    pub fn new(model_directory: Option<PathBuf>) -> Self {
        Self::with_device(model_directory, LocalDevice::Cpu)
    }

    /// Starts a local adapter on an explicitly selected compute backend.
    pub fn with_device(model_directory: Option<PathBuf>, device: LocalDevice) -> Self {
        Self::with_options(model_directory, device, None)
    }

    /// Starts a local adapter with optional benchmark-report persistence.
    ///
    /// When `results_directory` is present, completed loaded-model benchmarks
    /// create versioned JSON reports there. Existing files are never replaced.
    pub fn with_options(
        model_directory: Option<PathBuf>,
        device: LocalDevice,
        results_directory: Option<PathBuf>,
    ) -> Self {
        let (command_sender, command_receiver) = mpsc::channel();
        let (event_sender, event_receiver) = mpsc::channel();
        let loading_model = model_directory.is_some();
        let descriptors = ExperimentRegistry::builtin().descriptors();
        thread::Builder::new()
            .name("ola-ui-worker".into())
            .spawn(move || {
                worker_loop(
                    model_directory,
                    device,
                    results_directory,
                    command_receiver,
                    event_sender,
                )
            })
            .expect("failed to spawn the OlaRT UI worker");

        let snapshot = WorkspaceSnapshot {
            runtime: RuntimeSummary {
                connection: if loading_model {
                    ConnectionState::Busy
                } else {
                    ConnectionState::Connected
                },
                backend: device.label().into(),
                ..RuntimeSummary::default()
            },
            benchmarks: work_items(&descriptors, ExperimentCategory::Benchmark),
            experiments: work_items(&descriptors, ExperimentCategory::Experiment),
            activity: vec![if loading_model {
                "background model load started".into()
            } else {
                "local runtime connected; no Llama artifact selected".into()
            }],
            notice: Some(if loading_model {
                format!("Loading the selected Llama artifact on {}", device.label())
            } else {
                "Pass --model <directory> or set OLA_MODEL_DIR to enable chat".into()
            }),
            ..WorkspaceSnapshot::default()
        };
        Self {
            commands: command_sender,
            events: event_receiver,
            snapshot,
            descriptors,
        }
    }

    fn drain_events(&mut self) -> Result<(), UiBackendError> {
        loop {
            match self.events.try_recv() {
                Ok(event) => self.apply_event(event),
                Err(TryRecvError::Empty) => return Ok(()),
                Err(TryRecvError::Disconnected) => {
                    self.snapshot.runtime.connection = ConnectionState::Failed(
                        "background worker disconnected unexpectedly".into(),
                    );
                    return Err(UiBackendError::new(
                        "local UI background worker disconnected",
                    ));
                }
            }
        }
    }

    fn apply_event(&mut self, event: WorkerEvent) {
        match event {
            WorkerEvent::ModelReady { name } => {
                self.snapshot.selected_model = Some(name.clone());
                self.snapshot.runtime.connection = ConnectionState::Connected;
                self.snapshot.runtime.resident_models = 1;
                self.snapshot.notice = Some("Local Llama generation is ready".into());
                for descriptor in &self.descriptors {
                    if descriptor.requirements.contains(&Requirement::LoadedModel) {
                        let items = match descriptor.category {
                            ExperimentCategory::Benchmark => &mut self.snapshot.benchmarks,
                            ExperimentCategory::Experiment => &mut self.snapshot.experiments,
                        };
                        set_work_state(items, descriptor.id, RunState::Ready, None);
                    }
                }
                self.record_activity(format!("model loaded: {name}"));
            }
            WorkerEvent::ModelFailed(message) => {
                self.snapshot.runtime.connection = ConnectionState::Failed(message.clone());
                self.snapshot.notice = Some(message.clone());
                self.record_activity(format!("model load failed: {message}"));
            }
            WorkerEvent::ChatComplete {
                text,
                metrics,
                elapsed,
            } => {
                self.snapshot.chat.push(ChatMessage {
                    role: ChatRole::Assistant,
                    content: text,
                });
                self.snapshot.runtime.connection = ConnectionState::Connected;
                self.record_activity(format!(
                    "generated {} tokens in {} · {} terminal KV",
                    metrics.generated_tokens,
                    format_duration(elapsed),
                    format_bytes(metrics.kv_cache_bytes)
                ));
            }
            WorkerEvent::ChatFailed(message) => {
                self.snapshot.runtime.connection = ConnectionState::Connected;
                self.snapshot.chat.push(ChatMessage {
                    role: ChatRole::System,
                    content: format!("generation failed: {message}"),
                });
                self.record_activity(format!("generation failed: {message}"));
            }
            WorkerEvent::WorkComplete { kind, id, result } => {
                let needs_model = self.needs_model(&id);
                let items = match kind {
                    WorkKind::Benchmark => &mut self.snapshot.benchmarks,
                    WorkKind::Experiment => &mut self.snapshot.experiments,
                };
                set_work_state(items, &id, RunState::Succeeded, Some(result.clone()));
                if needs_model {
                    self.snapshot.runtime.connection = ConnectionState::Connected;
                }
                self.record_activity(format!("{id} completed: {result}"));
            }
            WorkerEvent::WorkFailed { kind, id, message } => {
                let needs_model = self.needs_model(&id);
                let items = match kind {
                    WorkKind::Benchmark => &mut self.snapshot.benchmarks,
                    WorkKind::Experiment => &mut self.snapshot.experiments,
                };
                set_work_state(items, &id, RunState::Failed, Some(message.clone()));
                if needs_model {
                    self.snapshot.runtime.connection = ConnectionState::Connected;
                }
                self.record_activity(format!("{id} failed: {message}"));
            }
        }
    }

    fn needs_model(&self, id: &str) -> bool {
        self.descriptors.iter().any(|descriptor| {
            descriptor.id == id && descriptor.requirements.contains(&Requirement::LoadedModel)
        })
    }

    fn record_activity(&mut self, message: String) {
        self.snapshot.activity.push(message);
        if self.snapshot.activity.len() > MAX_ACTIVITY {
            self.snapshot
                .activity
                .drain(0..self.snapshot.activity.len() - MAX_ACTIVITY);
        }
    }

    fn start_work(&mut self, kind: WorkKind, id: &str) -> Result<(), UiBackendError> {
        self.drain_events()?;
        let noun = match kind {
            WorkKind::Benchmark => "benchmark",
            WorkKind::Experiment => "experiment",
        };
        let items = match kind {
            WorkKind::Benchmark => &self.snapshot.benchmarks,
            WorkKind::Experiment => &self.snapshot.experiments,
        };
        let state = find_work(items, id)?;
        if state == RunState::Unavailable {
            return Err(UiBackendError::new(format!(
                "{noun} `{id}` is unavailable with the current connections"
            )));
        }
        if state == RunState::Running {
            return Err(UiBackendError::new(format!(
                "{noun} `{id}` is already running"
            )));
        }
        let needs_model = self.needs_model(id);
        if needs_model && self.snapshot.runtime.connection == ConnectionState::Busy {
            return Err(UiBackendError::new("the local model is already busy"));
        }
        self.send(WorkerCommand::Run {
            kind,
            id: id.to_owned(),
        })?;
        let items = match kind {
            WorkKind::Benchmark => &mut self.snapshot.benchmarks,
            WorkKind::Experiment => &mut self.snapshot.experiments,
        };
        set_work_state(items, id, RunState::Running, None);
        if needs_model {
            self.snapshot.runtime.connection = ConnectionState::Busy;
        }
        self.record_activity(format!("{noun} started: {id}"));
        Ok(())
    }

    fn send(&self, command: WorkerCommand) -> Result<(), UiBackendError> {
        self.commands
            .send(command)
            .map_err(|_| UiBackendError::new("local UI background worker is unavailable"))
    }
}

impl UiBackend for LocalBackend {
    fn snapshot(&mut self) -> Result<WorkspaceSnapshot, UiBackendError> {
        self.drain_events()?;
        Ok(self.snapshot.clone())
    }

    fn submit_prompt(&mut self, prompt: &str) -> Result<(), UiBackendError> {
        self.drain_events()?;
        if self.snapshot.selected_model.is_none() {
            return Err(UiBackendError::new(
                "no local Llama model is loaded; start with --model <directory>",
            ));
        }
        if self.snapshot.runtime.connection == ConnectionState::Busy {
            return Err(UiBackendError::new("the local model is already busy"));
        }
        self.send(WorkerCommand::Chat(prompt.to_owned()))?;
        self.snapshot.chat.push(ChatMessage {
            role: ChatRole::User,
            content: prompt.to_owned(),
        });
        self.snapshot.runtime.connection = ConnectionState::Busy;
        self.record_activity("generation started".into());
        Ok(())
    }

    fn start_benchmark(&mut self, id: &str) -> Result<(), UiBackendError> {
        self.start_work(WorkKind::Benchmark, id)
    }

    fn start_experiment(&mut self, id: &str) -> Result<(), UiBackendError> {
        self.start_work(WorkKind::Experiment, id)
    }
}

enum WorkerCommand {
    Chat(String),
    Run { kind: WorkKind, id: String },
}

#[derive(Clone, Copy)]
enum WorkKind {
    Benchmark,
    Experiment,
}

enum WorkerEvent {
    ModelReady {
        name: String,
    },
    ModelFailed(String),
    ChatComplete {
        text: String,
        metrics: SessionMetrics,
        elapsed: Duration,
    },
    ChatFailed(String),
    WorkComplete {
        kind: WorkKind,
        id: String,
        result: String,
    },
    WorkFailed {
        kind: WorkKind,
        id: String,
        message: String,
    },
}

fn worker_loop(
    model_directory: Option<PathBuf>,
    device: LocalDevice,
    results_directory: Option<PathBuf>,
    commands: Receiver<WorkerCommand>,
    events: Sender<WorkerEvent>,
) {
    let model_name = model_directory.as_ref().map(|directory| {
        directory
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("local-llama")
            .to_owned()
    });
    let model =
        model_directory
            .as_ref()
            .and_then(|directory| match load_model(directory, device) {
                Ok(model) => {
                    let name = model_name
                        .clone()
                        .unwrap_or_else(|| "local-llama".to_owned());
                    let _ = events.send(WorkerEvent::ModelReady { name });
                    Some(model)
                }
                Err(message) => {
                    let _ = events.send(WorkerEvent::ModelFailed(message));
                    None
                }
            });

    let registry = ExperimentRegistry::builtin();
    let context = ExperimentContext {
        model: model.clone(),
        model_label: model.as_ref().and(model_name.clone()),
        // The device describes the loaded model's backend; model-free
        // experiments record none, matching the CLI.
        device: model.as_ref().map(|_| {
            std::env::var("OLA_BENCH_DEVICE_LABEL").unwrap_or_else(|_| device.label().to_owned())
        }),
    };
    while let Ok(command) = commands.recv() {
        let event = match command {
            WorkerCommand::Chat(prompt) => match model.as_ref() {
                Some(model) => match generate(model, &prompt, 32) {
                    Ok(result) => WorkerEvent::ChatComplete {
                        text: result.text,
                        metrics: result.metrics,
                        elapsed: result.elapsed,
                    },
                    Err(message) => WorkerEvent::ChatFailed(message),
                },
                None => WorkerEvent::ChatFailed("no local Llama model is loaded".into()),
            },
            WorkerCommand::Run { kind, id } => {
                run_work(&registry, &context, results_directory.as_deref(), kind, &id)
            }
        };
        if events.send(event).is_err() {
            break;
        }
    }
}

/// Runs one registry experiment, saving its report when a results directory
/// is configured.
fn run_work(
    registry: &ExperimentRegistry,
    context: &ExperimentContext,
    results_directory: Option<&Path>,
    kind: WorkKind,
    id: &str,
) -> WorkerEvent {
    let outcome = match registry.run(id, context, &[]) {
        Ok(outcome) => outcome,
        Err(error) => return work_failed(kind, id, error.to_string()),
    };
    let mut result = format!(
        "{} · {}",
        outcome.summary,
        kind_label(
            outcome.report.experiment.kind,
            outcome.report.experiment.domain,
        )
    );
    if let Some(directory) = results_directory {
        match outcome.report.write_new(directory) {
            Ok(path) => result.push_str(&format!(" · saved {}", path.display())),
            Err(error) => return work_failed(kind, id, format!("could not save report: {error}")),
        }
    }
    WorkerEvent::WorkComplete {
        kind,
        id: id.to_owned(),
        result,
    }
}

/// Labels whether a result may support a claim, and which clock it uses.
fn kind_label(kind: ExperimentKind, domain: TimeDomain) -> String {
    let kind = match kind {
        ExperimentKind::Probe => "probe, not for publication",
        ExperimentKind::Study => "study",
    };
    let domain = match domain {
        TimeDomain::Virtual => "virtual time",
        TimeDomain::Host => "host time",
        TimeDomain::Untimed => "untimed",
    };
    format!("{kind}, {domain}")
}

/// Builds one workbench list from registry descriptors.
fn work_items(descriptors: &[ExperimentDescriptor], category: ExperimentCategory) -> Vec<WorkItem> {
    descriptors
        .iter()
        .filter(|descriptor| descriptor.category == category)
        .map(|descriptor| WorkItem {
            id: descriptor.id.to_owned(),
            name: descriptor.name.to_owned(),
            description: format!(
                "{} ({})",
                descriptor.description,
                kind_label(descriptor.kind, descriptor.domain)
            ),
            state: if descriptor.requirements.contains(&Requirement::LoadedModel) {
                RunState::Unavailable
            } else {
                RunState::Ready
            },
            last_result: None,
        })
        .collect()
}

fn load_model(
    directory: &std::path::Path,
    device: LocalDevice,
) -> Result<CausalLanguageModel, String> {
    match device {
        LocalDevice::Cpu => LlamaFactory::new(Device::flex())
            .load_local(directory)
            .map_err(|error| error.to_string()),
        #[cfg(feature = "wgpu")]
        LocalDevice::Wgpu => LlamaFactory::new(Device::wgpu(Default::default()))
            .load_local(directory)
            .map_err(|error| error.to_string()),
        #[cfg(not(feature = "wgpu"))]
        LocalDevice::Wgpu => {
            Err("WGPU support is not compiled; rerun ola-ui with --features wgpu".into())
        }
        #[cfg(feature = "cuda")]
        LocalDevice::Cuda => LlamaFactory::new(Device::cuda(0))
            .load_local(directory)
            .map_err(|error| error.to_string()),
        #[cfg(not(feature = "cuda"))]
        LocalDevice::Cuda => {
            Err("CUDA support is not compiled; rerun ola-ui with --features cuda".into())
        }
    }
}

struct GenerationResult {
    text: String,
    metrics: SessionMetrics,
    elapsed: Duration,
}

fn generate(
    model: &CausalLanguageModel,
    prompt: &str,
    max_new_tokens: usize,
) -> Result<GenerationResult, String> {
    const MODEL_ID: &str = "local-llama";

    let mut executor = Executor::default()
        .with_policy(ComposedPolicy::default().with_admission(MemoryAwareAdmission));
    executor
        .register_model(ModelDescriptor::language(MODEL_ID, "llama"), model.clone())
        .map_err(|error| error.to_string())?;
    let mut request = RequestSpec::text(MODEL_ID, prompt);
    request.generation = GenerationConfig {
        max_new_tokens,
        temperature: 0.0,
        top_p: 1.0,
        seed: Some(0),
    };
    let request_id = executor
        .submit(request, Timestamp::ZERO)
        .map_err(|error| error.to_string())?;
    let started = Instant::now();
    let mut step = 1_u64;
    'generation: loop {
        let iteration = executor
            .step_iteration(Timestamp::from_micros(step))
            .map_err(|error| error.to_string())?;
        for event in iteration.events {
            match event {
                ExecutorEvent::Token { finished: true, .. } => break 'generation,
                ExecutorEvent::Token { .. } | ExecutorEvent::PrefillProgress { .. } => {}
                ExecutorEvent::Failed { message, .. } => return Err(message),
                ExecutorEvent::Deferred { reason, .. } => {
                    return Err(format!("generation admission deferred: {reason}"));
                }
                ExecutorEvent::BatchDeferred {
                    phase, token_cost, ..
                } => {
                    return Err(format!(
                        "generation deferred by batch policy during {phase:?} (cost {token_cost} tokens)"
                    ));
                }
                ExecutorEvent::Rejected { reason, .. } => {
                    return Err(format!("generation admission rejected: {reason}"));
                }
                ExecutorEvent::Idle => {
                    return Err("executor became idle before generation completed".into());
                }
            }
        }
        step += 1;
    }
    let elapsed = started.elapsed();
    let result = executor
        .result(request_id)
        .expect("a finished token event retains an execution result");
    Ok(GenerationResult {
        text: result.text.clone(),
        metrics: result.metrics,
        elapsed,
    })
}

fn work_failed(kind: WorkKind, id: &str, message: impl Into<String>) -> WorkerEvent {
    WorkerEvent::WorkFailed {
        kind,
        id: id.into(),
        message: message.into(),
    }
}

fn find_work(items: &[WorkItem], id: &str) -> Result<RunState, UiBackendError> {
    items
        .iter()
        .find(|item| item.id == id)
        .map(|item| item.state.clone())
        .ok_or_else(|| UiBackendError::new(format!("unknown work item `{id}`")))
}

fn set_work_state(items: &mut [WorkItem], id: &str, state: RunState, result: Option<String>) {
    if let Some(item) = items.iter_mut().find(|item| item.id == id) {
        item.state = state;
        if result.is_some() {
            item.last_result = result;
        }
    }
}

fn format_duration(duration: Duration) -> String {
    if duration.as_secs_f64() >= 1.0 {
        format!("{:.2} s", duration.as_secs_f64())
    } else if duration.as_millis() >= 1 {
        format!("{:.1} ms", duration.as_secs_f64() * 1_000.0)
    } else {
        format!("{} µs", duration.as_micros())
    }
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1024 * 1024 {
        format!("{:.1} MiB", bytes as f64 / (1024 * 1024) as f64)
    } else if bytes >= 1024 {
        format!("{:.1} KiB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::*;

    #[test]
    fn local_backend_exposes_non_model_work_without_an_artifact() {
        let mut backend = LocalBackend::new(None);
        let snapshot = backend.snapshot().unwrap();

        assert_eq!(snapshot.runtime.connection, ConnectionState::Connected);
        assert!(
            snapshot
                .benchmarks
                .iter()
                .find(|item| item.id == "kv-block-pressure")
                .is_some_and(|item| item.state == RunState::Ready)
        );
        assert!(
            snapshot
                .benchmarks
                .iter()
                .find(|item| item.id == "llama-prefill-strategies")
                .is_some_and(|item| item.state == RunState::Unavailable)
        );
        assert!(backend.submit_prompt("hello").is_err());
    }

    #[test]
    fn local_backend_runs_the_prefill_bucketing_experiment() {
        let mut backend = LocalBackend::new(None);

        backend.start_experiment("prefill-bucketing").unwrap();
        let snapshot = wait_until(&mut backend, |snapshot| {
            snapshot
                .experiments
                .iter()
                .find(|item| item.id == "prefill-bucketing")
                .is_some_and(|item| item.state == RunState::Succeeded)
        });
        let experiment = snapshot
            .experiments
            .iter()
            .find(|item| item.id == "prefill-bucketing")
            .unwrap();

        assert!(
            experiment
                .last_result
                .as_deref()
                .is_some_and(|result| result.contains("padding 324 → 52"))
        );
    }

    #[test]
    fn local_backend_runs_the_scheduler_comparison_experiment() {
        let mut backend = LocalBackend::new(None);

        backend.start_experiment("scheduler-comparison").unwrap();
        let snapshot = wait_until_for(&mut backend, Duration::from_secs(120), |snapshot| {
            snapshot
                .experiments
                .iter()
                .find(|item| item.id == "scheduler-comparison")
                .is_some_and(|item| item.state == RunState::Succeeded)
        });
        let result = snapshot
            .experiments
            .iter()
            .find(|item| item.id == "scheduler-comparison")
            .and_then(|item| item.last_result.as_deref())
            .unwrap();

        assert!(result.starts_with("overload, 5 seeds × 120 requests, virtual time"));
        assert!(result.contains("sjf"));
        assert!(result.ends_with("study, virtual time"));
    }

    #[test]
    fn workbench_lists_come_from_the_experiment_registry() {
        let mut backend = LocalBackend::new(None);
        let snapshot = backend.snapshot().unwrap();
        let descriptors = ExperimentRegistry::builtin().descriptors();
        let ids = |category| {
            descriptors
                .iter()
                .filter(|descriptor| descriptor.category == category)
                .map(|descriptor| descriptor.id)
                .collect::<Vec<_>>()
        };

        assert_eq!(
            snapshot
                .benchmarks
                .iter()
                .map(|item| item.id.as_str())
                .collect::<Vec<_>>(),
            ids(ExperimentCategory::Benchmark)
        );
        assert_eq!(
            snapshot
                .experiments
                .iter()
                .map(|item| item.id.as_str())
                .collect::<Vec<_>>(),
            ids(ExperimentCategory::Experiment)
        );
        let fifo = snapshot
            .experiments
            .iter()
            .find(|item| item.id == "fifo-workload")
            .unwrap();
        assert!(
            fifo.description
                .ends_with("(probe, not for publication, host time)")
        );
        assert!(
            snapshot
                .benchmarks
                .iter()
                .filter(|item| item.id.starts_with("llama-") || item.id == "cost-calibration")
                .all(|item| item.state == RunState::Unavailable)
        );
    }

    #[test]
    fn local_backend_runs_the_adaptive_prefill_experiment() {
        let mut backend = LocalBackend::new(None);

        backend.start_experiment("adaptive-prefill-cost").unwrap();
        let snapshot = wait_until(&mut backend, |snapshot| {
            snapshot
                .experiments
                .iter()
                .find(|item| item.id == "adaptive-prefill-cost")
                .is_some_and(|item| item.state == RunState::Succeeded)
        });
        let result = snapshot
            .experiments
            .iter()
            .find(|item| item.id == "adaptive-prefill-cost")
            .and_then(|item| item.last_result.as_deref())
            .unwrap();

        assert!(result.contains("diverse PowerOfTwo"));
        assert!(result.contains("clustered Unbucketed"));
    }

    #[test]
    #[ignore = "requires OLA_LLAMA_CONFORMANCE_DIR and its pinned external artifact"]
    fn experiment_registry_runs_model_experiments_on_the_fixture() {
        use ola_bench::{ExperimentContext, ExperimentKind, ExperimentRegistry};

        let directory = std::env::var_os("OLA_LLAMA_CONFORMANCE_DIR")
            .map(std::path::PathBuf::from)
            .expect("set OLA_LLAMA_CONFORMANCE_DIR as documented in ola-burn/CONFORMANCE.md");
        let context = ExperimentContext {
            model: Some(
                LlamaFactory::new(Device::flex())
                    .load_local(directory)
                    .unwrap(),
            ),
            model_label: Some("ola-rt-tiny-random-llama".into()),
            device: Some("cpu".into()),
        };
        let registry = ExperimentRegistry::builtin();

        let generation = registry.run("llama-generation", &context, &[]).unwrap();
        assert!(generation.summary.contains("tok/s"));
        assert_eq!(
            generation.report.metadata.model.as_deref(),
            Some("ola-rt-tiny-random-llama")
        );

        let strategies = registry
            .run("llama-prefill-strategies", &context, &[])
            .unwrap();
        assert_eq!(strategies.report.experiment.kind, ExperimentKind::Study);
        assert_eq!(strategies.report.protocol.repetitions, 5);
        assert!(strategies.summary.contains("5 rotating trials + 1 warmup"));
        assert!(strategies.summary.contains("padding U/P/A 11 / 2 /"));
        assert!(
            strategies
                .report
                .summaries
                .iter()
                .all(|summary| summary.estimates["prefill_iteration"].ci95.is_some())
        );

        let chunked = registry
            .run("llama-chunked-prefill", &context, &[])
            .unwrap();
        assert!(chunked.summary.contains("prefill calls 1 → 3"));
        assert!(chunked.summary.contains("tensor calls 1 → 2"));
        assert!(chunked.summary.contains("output preserved"));
        assert!(chunked.report.details.is_some());

        let calibration = registry
            .run(ola_bench::COST_CALIBRATION, &context, &[])
            .unwrap();
        assert_eq!(calibration.report.experiment.kind, ExperimentKind::Study);
        assert_eq!(calibration.report.protocol.repetitions, 5);
        let fit = ola_bench::CostFit::from_report(&calibration.report).unwrap();
        assert!(fit.residual_rms_micros.is_finite());
        // A measured cost model drives the virtual-time harness.
        let trace = ola_bench::SchedulingTrace::mixed_short_long(ola_bench::TraceParameters {
            seed: 1,
            requests: 10,
            mean_interarrival_micros: 1_000,
        });
        let run = ola_bench::run_scheduling_trial(
            &trace,
            ComposedPolicy::default().with_admission(MemoryAwareAdmission),
            &ola_bench::SimulationConfig {
                cost: fit.cost_model(),
                ..ola_bench::SimulationConfig::default()
            },
        )
        .unwrap();
        assert_eq!(run.requests.len(), 10);
        println!("{}", calibration.summary);
    }

    #[test]
    #[ignore = "requires OLA_LLAMA_CONFORMANCE_DIR and its pinned external artifact"]
    fn local_backend_loads_and_generates_without_blocking_the_ui_contract() {
        let directory = std::env::var_os("OLA_LLAMA_CONFORMANCE_DIR")
            .map(PathBuf::from)
            .expect("set OLA_LLAMA_CONFORMANCE_DIR as documented in ola-burn/CONFORMANCE.md");
        let results_directory = std::env::temp_dir().join(format!(
            "ola-ui-prefill-report-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut backend = LocalBackend::with_options(
            Some(directory),
            LocalDevice::Cpu,
            Some(results_directory.clone()),
        );

        wait_until(&mut backend, |snapshot| snapshot.selected_model.is_some());
        backend.submit_prompt("Hello").unwrap();
        let snapshot = wait_until(&mut backend, |snapshot| snapshot.chat.len() == 2);

        assert_eq!(snapshot.chat[0].role, ChatRole::User);
        assert_eq!(snapshot.chat[1].role, ChatRole::Assistant);
        assert!(!snapshot.chat[1].content.is_empty());
        assert!(
            snapshot
                .activity
                .last()
                .is_some_and(|entry| entry.contains("terminal KV"))
        );
        assert_eq!(snapshot.runtime.kv_cache_bytes, 0);
        assert_eq!(snapshot.runtime.connection, ConnectionState::Connected);

        backend.start_benchmark("llama-prefill-strategies").unwrap();
        let snapshot = wait_until_for(&mut backend, Duration::from_secs(180), |snapshot| {
            snapshot
                .benchmarks
                .iter()
                .find(|item| item.id == "llama-prefill-strategies")
                .is_some_and(|item| item.state == RunState::Succeeded)
        });
        let result = snapshot
            .benchmarks
            .iter()
            .find(|item| item.id == "llama-prefill-strategies")
            .and_then(|item| item.last_result.as_deref())
            .unwrap();
        assert!(result.contains("calibrated"));
        assert!(result.contains("5 rotating trials + 1 warmup"));
        assert!(result.contains("prefill p50 U/P/A"));
        assert!(result.contains("padding U/P/A 11 / 2 /"));
        assert!(result.contains("saved"));
        assert_eq!(snapshot.runtime.connection, ConnectionState::Connected);

        backend.start_benchmark("llama-chunked-prefill").unwrap();
        let snapshot = wait_until_for(&mut backend, Duration::from_secs(180), |snapshot| {
            snapshot
                .benchmarks
                .iter()
                .find(|item| item.id == "llama-chunked-prefill")
                .is_some_and(|item| item.state == RunState::Succeeded)
        });
        let chunked = snapshot
            .benchmarks
            .iter()
            .find(|item| item.id == "llama-chunked-prefill")
            .and_then(|item| item.last_result.as_deref())
            .unwrap();
        assert!(chunked.contains("chunk size 3"));
        assert!(chunked.contains("prefill calls 1 → 3"));
        assert!(chunked.contains("tensor calls 1 → 2"));
        assert!(chunked.contains("output preserved"));
        assert!(chunked.contains("fairness (1 warm-up, 3 measured, two active)"));
        assert!(chunked.contains("p50/p95 ms"));

        // Every run with a results directory saves a versioned report.
        let paths = fs::read_dir(&results_directory)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect::<Vec<_>>();
        assert_eq!(paths.len(), 2);
        let reports = paths
            .iter()
            .map(|path| ola_bench::ExperimentReport::read(path).unwrap())
            .collect::<Vec<_>>();
        let strategies = reports
            .iter()
            .find(|report| report.experiment.id == "llama-prefill-strategies")
            .unwrap();
        assert_eq!(strategies.protocol.warmups, 1);
        assert_eq!(strategies.protocol.repetitions, 5);
        assert_eq!(strategies.variants.len(), 3);
        assert_eq!(
            strategies.metadata.model.as_deref(),
            Some("ola-rt-tiny-random-llama")
        );
        assert!(
            reports
                .iter()
                .any(|report| report.experiment.id == "llama-chunked-prefill")
        );
        for path in paths {
            fs::remove_file(path).unwrap();
        }
        fs::remove_dir(results_directory).unwrap();
    }

    #[cfg(feature = "wgpu")]
    #[test]
    #[ignore = "requires OLA_LLAMA_CONFORMANCE_DIR and a working WGPU adapter"]
    fn local_wgpu_backend_loads_and_generates_through_the_executor() {
        let directory = std::env::var_os("OLA_LLAMA_CONFORMANCE_DIR")
            .map(PathBuf::from)
            .expect("set OLA_LLAMA_CONFORMANCE_DIR as documented in ola-burn/CONFORMANCE.md");
        let mut backend = LocalBackend::with_device(Some(directory), LocalDevice::Wgpu);

        let loaded = wait_until_for(&mut backend, Duration::from_secs(180), |snapshot| {
            snapshot.selected_model.is_some()
        });
        assert!(loaded.runtime.backend.contains("WGPU"));

        backend.submit_prompt("Hello").unwrap();
        let snapshot = wait_until_for(&mut backend, Duration::from_secs(180), |snapshot| {
            snapshot.chat.len() == 2
        });

        assert_eq!(snapshot.chat[0].role, ChatRole::User);
        assert_eq!(snapshot.chat[1].role, ChatRole::Assistant);
        assert!(!snapshot.chat[1].content.is_empty());
        assert_eq!(snapshot.runtime.kv_cache_bytes, 0);
        assert_eq!(snapshot.runtime.connection, ConnectionState::Connected);
    }

    fn wait_until(
        backend: &mut LocalBackend,
        condition: impl Fn(&WorkspaceSnapshot) -> bool,
    ) -> WorkspaceSnapshot {
        wait_until_for(backend, Duration::from_secs(10), condition)
    }

    fn wait_until_for(
        backend: &mut LocalBackend,
        timeout: Duration,
        condition: impl Fn(&WorkspaceSnapshot) -> bool,
    ) -> WorkspaceSnapshot {
        let deadline = Instant::now() + timeout;
        loop {
            let snapshot = backend.snapshot().unwrap();
            if condition(&snapshot) {
                return snapshot;
            }
            assert!(
                Instant::now() < deadline,
                "local backend operation timed out: connection={:?}, notice={:?}, activity={:?}",
                snapshot.runtime.connection,
                snapshot.notice,
                snapshot.activity
            );
            thread::sleep(Duration::from_millis(10));
        }
    }
}
