use std::{
    path::PathBuf,
    sync::mpsc::{self, Receiver, Sender, TryRecvError},
    thread,
    time::{Duration, Instant},
};

#[cfg(feature = "cuda")]
use burn::backend::Cuda;
use burn::backend::NdArray;
#[cfg(feature = "wgpu")]
use burn::backend::Wgpu;
use ola_bench::FixedWorkload;
use ola_burn::llama::LlamaFactory;
use ola_core::{GenerationConfig, ModelDescriptor, RequestSpec, Timestamp};
use ola_model::{CausalLanguageModel, SessionMetrics};
use ola_runtime::{Executor, ExecutorEvent, KvBlockManager, KvBlockMetrics, Runtime};

use crate::{
    ChatMessage, ChatRole, ConnectionState, RunState, RuntimeSummary, UiBackend, UiBackendError,
    WorkItem, WorkspaceSnapshot,
};

type CpuBackend = NdArray<f32>;
#[cfg(feature = "cuda")]
type CudaBackend = Cuda<f32, i32>;
#[cfg(feature = "wgpu")]
type WgpuBackend = Wgpu<f32, i32>;

const LLAMA_BENCHMARK: &str = "llama-generation";
const KV_BENCHMARK: &str = "kv-block-pressure";
const FIFO_EXPERIMENT: &str = "fifo-workload";
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
            Self::Cpu => "local / Burn NdArray<f32> / CPU",
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
}

impl LocalBackend {
    /// Starts a local adapter, optionally loading one Hugging Face Llama
    /// artifact directory on the CPU in the background.
    pub fn new(model_directory: Option<PathBuf>) -> Self {
        Self::with_device(model_directory, LocalDevice::Cpu)
    }

    /// Starts a local adapter on an explicitly selected compute backend.
    pub fn with_device(model_directory: Option<PathBuf>, device: LocalDevice) -> Self {
        let (command_sender, command_receiver) = mpsc::channel();
        let (event_sender, event_receiver) = mpsc::channel();
        let loading_model = model_directory.is_some();
        thread::Builder::new()
            .name("ola-ui-worker".into())
            .spawn(move || worker_loop(model_directory, device, command_receiver, event_sender))
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
            benchmarks: vec![
                WorkItem {
                    id: LLAMA_BENCHMARK.into(),
                    name: "Llama generation".into(),
                    description: "Cached greedy generation latency and throughput".into(),
                    state: RunState::Unavailable,
                    last_result: None,
                },
                WorkItem {
                    id: KV_BENCHMARK.into(),
                    name: "KV block pressure".into(),
                    description: "Allocator throughput, occupancy, and fragmentation".into(),
                    state: RunState::Ready,
                    last_result: None,
                },
            ],
            experiments: vec![WorkItem {
                id: FIFO_EXPERIMENT.into(),
                name: "FIFO workload".into(),
                description: "Replay and complete a deterministic runtime trace".into(),
                state: RunState::Ready,
                last_result: None,
            }],
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
                set_work_state(
                    &mut self.snapshot.benchmarks,
                    LLAMA_BENCHMARK,
                    RunState::Ready,
                    None,
                );
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
                let items = match kind {
                    WorkKind::Benchmark => &mut self.snapshot.benchmarks,
                    WorkKind::Experiment => &mut self.snapshot.experiments,
                };
                set_work_state(items, &id, RunState::Succeeded, Some(result.clone()));
                if id == LLAMA_BENCHMARK {
                    self.snapshot.runtime.connection = ConnectionState::Connected;
                }
                self.record_activity(format!("{id} completed: {result}"));
            }
            WorkerEvent::WorkFailed { kind, id, message } => {
                let items = match kind {
                    WorkKind::Benchmark => &mut self.snapshot.benchmarks,
                    WorkKind::Experiment => &mut self.snapshot.experiments,
                };
                set_work_state(items, &id, RunState::Failed, Some(message.clone()));
                if id == LLAMA_BENCHMARK {
                    self.snapshot.runtime.connection = ConnectionState::Connected;
                }
                self.record_activity(format!("{id} failed: {message}"));
            }
        }
    }

    fn record_activity(&mut self, message: String) {
        self.snapshot.activity.push(message);
        if self.snapshot.activity.len() > MAX_ACTIVITY {
            self.snapshot
                .activity
                .drain(0..self.snapshot.activity.len() - MAX_ACTIVITY);
        }
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
        self.drain_events()?;
        let state = find_work(&self.snapshot.benchmarks, id)?;
        if state == RunState::Unavailable {
            return Err(UiBackendError::new(format!(
                "benchmark `{id}` is unavailable with the current connections"
            )));
        }
        if state == RunState::Running {
            return Err(UiBackendError::new(format!(
                "benchmark `{id}` is already running"
            )));
        }
        self.send(WorkerCommand::Benchmark(id.to_owned()))?;
        set_work_state(&mut self.snapshot.benchmarks, id, RunState::Running, None);
        if id == LLAMA_BENCHMARK {
            self.snapshot.runtime.connection = ConnectionState::Busy;
        }
        self.record_activity(format!("benchmark started: {id}"));
        Ok(())
    }

    fn start_experiment(&mut self, id: &str) -> Result<(), UiBackendError> {
        self.drain_events()?;
        let state = find_work(&self.snapshot.experiments, id)?;
        if state == RunState::Running {
            return Err(UiBackendError::new(format!(
                "experiment `{id}` is already running"
            )));
        }
        self.send(WorkerCommand::Experiment(id.to_owned()))?;
        set_work_state(&mut self.snapshot.experiments, id, RunState::Running, None);
        self.record_activity(format!("experiment started: {id}"));
        Ok(())
    }
}

enum WorkerCommand {
    Chat(String),
    Benchmark(String),
    Experiment(String),
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
    commands: Receiver<WorkerCommand>,
    events: Sender<WorkerEvent>,
) {
    let model =
        model_directory
            .as_ref()
            .and_then(|directory| match load_model(directory, device) {
                Ok(model) => {
                    let name = directory
                        .file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or("local-llama")
                        .to_owned();
                    let _ = events.send(WorkerEvent::ModelReady { name });
                    Some(model)
                }
                Err(message) => {
                    let _ = events.send(WorkerEvent::ModelFailed(message));
                    None
                }
            });

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
            WorkerCommand::Benchmark(id) => run_benchmark(&id, model.as_ref()),
            WorkerCommand::Experiment(id) => run_experiment(&id),
        };
        if events.send(event).is_err() {
            break;
        }
    }
}

fn load_model(
    directory: &std::path::Path,
    device: LocalDevice,
) -> Result<CausalLanguageModel, String> {
    match device {
        LocalDevice::Cpu => LlamaFactory::<CpuBackend>::new(Default::default())
            .load_local(directory)
            .map_err(|error| error.to_string()),
        #[cfg(feature = "wgpu")]
        LocalDevice::Wgpu => LlamaFactory::<WgpuBackend>::new(Default::default())
            .load_local(directory)
            .map_err(|error| error.to_string()),
        #[cfg(not(feature = "wgpu"))]
        LocalDevice::Wgpu => {
            Err("WGPU support is not compiled; rerun ola-ui with --features wgpu".into())
        }
        #[cfg(feature = "cuda")]
        LocalDevice::Cuda => LlamaFactory::<CudaBackend>::new(Default::default())
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

    let mut executor = Executor::default();
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
    loop {
        match executor
            .step(Timestamp::from_micros(step))
            .map_err(|error| error.to_string())?
        {
            ExecutorEvent::Token { finished: true, .. } => break,
            ExecutorEvent::Token { .. } => step += 1,
            ExecutorEvent::Failed { message, .. } => return Err(message),
            ExecutorEvent::Idle => {
                return Err("executor became idle before generation completed".into());
            }
        }
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

fn run_benchmark(id: &str, model: Option<&CausalLanguageModel>) -> WorkerEvent {
    match id {
        KV_BENCHMARK => match kv_block_pressure() {
            Ok((result, _)) => WorkerEvent::WorkComplete {
                kind: WorkKind::Benchmark,
                id: id.into(),
                result,
            },
            Err(message) => work_failed(WorkKind::Benchmark, id, message),
        },
        LLAMA_BENCHMARK => match model {
            Some(model) => match generate(model, "Hello", 8) {
                Ok(generation) => {
                    let seconds = generation.elapsed.as_secs_f64();
                    let throughput = if seconds == 0.0 {
                        0.0
                    } else {
                        generation.metrics.generated_tokens as f64 / seconds
                    };
                    WorkerEvent::WorkComplete {
                        kind: WorkKind::Benchmark,
                        id: id.into(),
                        result: format!(
                            "{} tokens · {:.2} tok/s · {} KV",
                            generation.metrics.generated_tokens,
                            throughput,
                            format_bytes(generation.metrics.kv_cache_bytes)
                        ),
                    }
                }
                Err(message) => work_failed(WorkKind::Benchmark, id, message),
            },
            None => work_failed(WorkKind::Benchmark, id, "no local Llama model is loaded"),
        },
        _ => work_failed(WorkKind::Benchmark, id, "unknown benchmark ID"),
    }
}

fn run_experiment(id: &str) -> WorkerEvent {
    match id {
        FIFO_EXPERIMENT => match fifo_workload_probe() {
            Ok(result) => WorkerEvent::WorkComplete {
                kind: WorkKind::Experiment,
                id: id.into(),
                result,
            },
            Err(message) => work_failed(WorkKind::Experiment, id, message),
        },
        _ => work_failed(WorkKind::Experiment, id, "unknown experiment ID"),
    }
}

fn kv_block_pressure() -> Result<(String, KvBlockMetrics), String> {
    const CAPACITY: u64 = 64 * 1024 * 1024;
    const TOKENS_PER_BLOCK: usize = 16;
    const BYTES_PER_TOKEN: u64 = 4 * 1024;
    const SEQUENCES: usize = 128;

    let started = Instant::now();
    let mut manager = KvBlockManager::new(CAPACITY, TOKENS_PER_BLOCK, BYTES_PER_TOKEN)
        .map_err(|error| error.to_string())?;
    let mut sequences = Vec::with_capacity(SEQUENCES);
    for index in 0..SEQUENCES {
        let sequence = manager.allocate().map_err(|error| error.to_string())?;
        manager
            .append(sequence, 1 + index % 96)
            .map_err(|error| error.to_string())?;
        sequences.push(sequence);
    }
    for (index, sequence) in sequences.into_iter().enumerate() {
        if index.is_multiple_of(3) {
            manager.free(sequence).map_err(|error| error.to_string())?;
        }
    }
    let elapsed = started.elapsed();
    let metrics = manager.metrics();
    let fragmentation = if metrics.allocated_bytes == 0 {
        0.0
    } else {
        metrics.internal_fragmentation_bytes as f64 / metrics.allocated_bytes as f64 * 100.0
    };
    Ok((
        format!(
            "{} live · {} blocks / {} · {} · {:.1}% fragment · {}",
            metrics.live_sequences,
            metrics.allocated_blocks,
            metrics.total_blocks,
            format_bytes(metrics.allocated_bytes),
            fragmentation,
            format_duration(elapsed)
        ),
        metrics,
    ))
}

fn fifo_workload_probe() -> Result<String, String> {
    const REQUESTS: usize = 100;
    let workload = FixedWorkload {
        model: "ui-probe".into(),
        requests: REQUESTS,
        prompt_tokens: 32,
        output_tokens: 16,
        interarrival: Duration::from_micros(100),
    }
    .build()
    .map_err(|error| error.to_string())?;
    let mut runtime = Runtime::default();
    runtime
        .register_model(ModelDescriptor::language("ui-probe", "probe"))
        .map_err(|error| error.to_string())?;
    let started = Instant::now();
    let submitted = workload
        .replay(&mut runtime)
        .map_err(|error| error.to_string())?;
    let mut observed = Vec::with_capacity(REQUESTS);
    for index in 0..REQUESTS {
        let base = 20_000 + index as u64 * 3;
        let session = runtime
            .schedule_next(Timestamp::from_micros(base))
            .map_err(|error| error.to_string())?
            .ok_or_else(|| "FIFO returned no request while work remained".to_owned())?;
        observed.push(session.request_id);
        runtime
            .mark_first_token(session.request_id, Timestamp::from_micros(base + 1))
            .map_err(|error| error.to_string())?;
        runtime
            .complete(session.request_id, Timestamp::from_micros(base + 2))
            .map_err(|error| error.to_string())?;
    }
    if observed != submitted {
        return Err("FIFO execution order differed from arrival order".into());
    }
    let elapsed = started.elapsed();
    Ok(format!(
        "{REQUESTS} requests · FIFO order verified · {}",
        format_duration(elapsed)
    ))
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
    use super::*;

    #[test]
    fn kv_probe_exercises_real_block_accounting() {
        let (summary, metrics) = kv_block_pressure().unwrap();

        assert_eq!(metrics.live_sequences, 85);
        assert!(metrics.allocated_blocks > 0);
        assert!(metrics.internal_fragmentation_bytes > 0);
        assert!(summary.contains("fragment"));
    }

    #[test]
    fn fifo_probe_replays_and_completes_the_real_runtime_workload() {
        let summary = fifo_workload_probe().unwrap();

        assert!(summary.contains("100 requests"));
        assert!(summary.contains("verified"));
    }

    #[test]
    fn local_backend_exposes_non_model_work_without_an_artifact() {
        let mut backend = LocalBackend::new(None);
        let snapshot = backend.snapshot().unwrap();

        assert_eq!(snapshot.runtime.connection, ConnectionState::Connected);
        assert_eq!(snapshot.benchmarks[1].state, RunState::Ready);
        assert!(backend.submit_prompt("hello").is_err());
    }

    #[test]
    #[ignore = "requires OLA_LLAMA_CONFORMANCE_DIR and its pinned external artifact"]
    fn local_backend_loads_and_generates_without_blocking_the_ui_contract() {
        let directory = std::env::var_os("OLA_LLAMA_CONFORMANCE_DIR")
            .map(PathBuf::from)
            .expect("set OLA_LLAMA_CONFORMANCE_DIR as documented in ola-burn/CONFORMANCE.md");
        let mut backend = LocalBackend::new(Some(directory));

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
    }

    #[cfg(feature = "wgpu")]
    #[test]
    #[ignore = "requires OLA_LLAMA_CONFORMANCE_DIR and a working WGPU adapter"]
    fn local_wgpu_backend_loads_and_generates_through_the_executor() {
        let directory = std::env::var_os("OLA_LLAMA_CONFORMANCE_DIR")
            .map(PathBuf::from)
            .expect("set OLA_LLAMA_CONFORMANCE_DIR as documented in ola-burn/CONFORMANCE.md");
        let mut backend = LocalBackend::with_device(Some(directory), LocalDevice::Wgpu);

        let loaded = wait_until_for(&mut backend, Duration::from_secs(60), |snapshot| {
            snapshot.selected_model.is_some()
        });
        assert!(loaded.runtime.backend.contains("WGPU"));

        backend.submit_prompt("Hello").unwrap();
        let snapshot = wait_until_for(&mut backend, Duration::from_secs(60), |snapshot| {
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
                "local backend operation timed out"
            );
            thread::sleep(Duration::from_millis(10));
        }
    }
}
