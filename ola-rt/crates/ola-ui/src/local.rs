use std::{
    fs, io,
    path::{Path, PathBuf},
    sync::mpsc::{self, Receiver, Sender, TryRecvError},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use burn::tensor::Device;
use ola_bench::{
    DEFAULT_STARVATION_FACTOR, FixedWorkload, PrefillBenchmarkMetadata, PrefillBenchmarkReport,
    PrefillExecutionConfig, PrefillReportError, PrefillThreeWayTrialConfig, PrefillTrialConfig,
    SchedulingSummary, SchedulingTrace, SimulationConfig, TraceParameters,
    analyze_cost_aware_prefill, benchmark_chunked_prefill_execution,
    benchmark_prefill_fairness_trials, benchmark_prefill_trials,
    benchmark_three_way_prefill_trials, calibrate_prefill_invocation_overhead,
    compare_prefill_batching, compare_schedulers,
};
use ola_burn::llama::LlamaFactory;
use ola_core::{GenerationConfig, ModelDescriptor, RequestSpec, Timestamp};
use ola_model::{CausalLanguageModel, SessionMetrics};
use ola_runtime::{
    ComposedPolicy, ExecutionPolicy, Executor, ExecutorEvent, JobCost, KvBlockManager,
    KvBlockMetrics, MemoryAwareAdmission, PriorityScheduler, Runtime, ShortestJobFirst,
};

use crate::{
    ChatMessage, ChatRole, ConnectionState, RunState, RuntimeSummary, UiBackend, UiBackendError,
    WorkItem, WorkspaceSnapshot,
};

const LLAMA_BENCHMARK: &str = "llama-generation";
const PREFILL_EXECUTION_BENCHMARK: &str = "llama-prefill-strategies";
const CHUNKED_PREFILL_BENCHMARK: &str = "llama-chunked-prefill";
const KV_BENCHMARK: &str = "kv-block-pressure";
const FIFO_EXPERIMENT: &str = "fifo-workload";
const PREFILL_BUCKET_EXPERIMENT: &str = "prefill-bucketing";
const ADAPTIVE_PREFILL_EXPERIMENT: &str = "adaptive-prefill-cost";
const SCHEDULER_EXPERIMENT: &str = "scheduler-comparison";
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
            benchmarks: vec![
                WorkItem {
                    id: LLAMA_BENCHMARK.into(),
                    name: "Llama generation".into(),
                    description: "Cached greedy generation latency and throughput".into(),
                    state: RunState::Unavailable,
                    last_result: None,
                },
                WorkItem {
                    id: PREFILL_EXECUTION_BENCHMARK.into(),
                    name: "Llama prefill strategies".into(),
                    description: "Calibrate and compare three prefill groupers".into(),
                    state: RunState::Unavailable,
                    last_result: None,
                },
                WorkItem {
                    id: CHUNKED_PREFILL_BENCHMARK.into(),
                    name: "Llama chunked prefill".into(),
                    description: "Compare complete prompts with resumable prompt segments".into(),
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
            experiments: vec![
                WorkItem {
                    id: FIFO_EXPERIMENT.into(),
                    name: "FIFO workload".into(),
                    description: "Replay and complete a deterministic runtime trace".into(),
                    state: RunState::Ready,
                    last_result: None,
                },
                WorkItem {
                    id: PREFILL_BUCKET_EXPERIMENT.into(),
                    name: "Prefill bucketing".into(),
                    description: "Compare unbucketed and power-of-two padding work".into(),
                    state: RunState::Ready,
                    last_result: None,
                },
                WorkItem {
                    id: ADAPTIVE_PREFILL_EXPERIMENT.into(),
                    name: "Adaptive prefill cost".into(),
                    description: "Balance dense padding against invocation overhead".into(),
                    state: RunState::Ready,
                    last_result: None,
                },
                WorkItem {
                    id: SCHEDULER_EXPERIMENT.into(),
                    name: "Scheduler comparison".into(),
                    description: "FIFO, SJF, and priority on seeded traces in virtual time".into(),
                    state: RunState::Ready,
                    last_result: None,
                },
            ],
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
                set_work_state(
                    &mut self.snapshot.benchmarks,
                    PREFILL_EXECUTION_BENCHMARK,
                    RunState::Ready,
                    None,
                );
                set_work_state(
                    &mut self.snapshot.benchmarks,
                    CHUNKED_PREFILL_BENCHMARK,
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
                if is_model_benchmark(&id) {
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
                if is_model_benchmark(&id) {
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
        if is_model_benchmark(id) && self.snapshot.runtime.connection == ConnectionState::Busy {
            return Err(UiBackendError::new("the local model is already busy"));
        }
        self.send(WorkerCommand::Benchmark(id.to_owned()))?;
        set_work_state(&mut self.snapshot.benchmarks, id, RunState::Running, None);
        if is_model_benchmark(id) {
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
            WorkerCommand::Benchmark(id) => run_benchmark(
                &id,
                model.as_ref(),
                model_name.as_deref(),
                device,
                results_directory.as_deref(),
            ),
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

fn run_benchmark(
    id: &str,
    model: Option<&CausalLanguageModel>,
    model_name: Option<&str>,
    device: LocalDevice,
    results_directory: Option<&Path>,
) -> WorkerEvent {
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
        PREFILL_EXECUTION_BENCHMARK => match model {
            Some(model) => match prefill_execution_probe(
                model,
                model_name.unwrap_or("local-llama"),
                device,
                results_directory,
            ) {
                Ok(result) => WorkerEvent::WorkComplete {
                    kind: WorkKind::Benchmark,
                    id: id.into(),
                    result,
                },
                Err(message) => work_failed(WorkKind::Benchmark, id, message),
            },
            None => work_failed(WorkKind::Benchmark, id, "no local Llama model is loaded"),
        },
        CHUNKED_PREFILL_BENCHMARK => match model {
            Some(model) => match chunked_prefill_probe(model) {
                Ok(result) => WorkerEvent::WorkComplete {
                    kind: WorkKind::Benchmark,
                    id: id.into(),
                    result,
                },
                Err(message) => work_failed(WorkKind::Benchmark, id, message),
            },
            None => work_failed(WorkKind::Benchmark, id, "no local Llama model is loaded"),
        },
        _ => work_failed(WorkKind::Benchmark, id, "unknown benchmark ID"),
    }
}

fn prefill_execution_probe(
    model: &CausalLanguageModel,
    model_name: &str,
    device: LocalDevice,
    results_directory: Option<&Path>,
) -> Result<String, String> {
    const PROMPT_LENGTHS: &[usize] = &[2, 3, 5, 7];
    const CALIBRATION_WARMUP_TRIALS: usize = 1;
    const CALIBRATION_TRIALS: usize = 4;
    const COMPARISON_WARMUP_TRIALS: usize = 1;
    const COMPARISON_TRIALS: usize = 3;
    let execution = PrefillExecutionConfig {
        prompts: PROMPT_LENGTHS
            .iter()
            .map(|length| vec![1; *length])
            .collect(),
        max_new_tokens: 2,
    };
    let calibration_config = PrefillTrialConfig {
        execution: execution.clone(),
        warmup_trials: CALIBRATION_WARMUP_TRIALS,
        measured_trials: CALIBRATION_TRIALS,
    };
    let calibration_comparison =
        benchmark_prefill_trials(model, &calibration_config).map_err(|error| error.to_string())?;
    let calibration = calibrate_prefill_invocation_overhead(&calibration_comparison)
        .map_err(|error| error.to_string())?;
    let config = PrefillThreeWayTrialConfig {
        execution: PrefillExecutionConfig {
            prompts: execution.prompts,
            max_new_tokens: execution.max_new_tokens,
        },
        warmup_trials: COMPARISON_WARMUP_TRIALS,
        measured_trials: COMPARISON_TRIALS,
        adaptive_invocation_overhead_tokens: calibration.invocation_overhead_tokens,
    };
    let comparison =
        benchmark_three_way_prefill_trials(model, &config).map_err(|error| error.to_string())?;
    let baseline = &comparison.unbucketed;
    let bucketed = &comparison.power_of_two;
    let adaptive = &comparison.cost_aware;
    let baseline_shape = &baseline.samples[0];
    let bucketed_shape = &bucketed.samples[0];
    let adaptive_shape = &adaptive.samples[0];
    let mut summary = format!(
        "{} prompts · calibrated {} token-equivalent call cost from {} → {} p50 · {COMPARISON_TRIALS} rotating trials + {COMPARISON_WARMUP_TRIALS} warmup · prefill p50 U/P/A {} / {} / {} · padding U/P/A {} / {} / {} · calls U/P/A {} / {} / {} · throughput U/P/A {:.2} / {:.2} / {:.2} tok/s · peak KV U/P/A {} / {} / {}",
        PROMPT_LENGTHS.len(),
        calibration.invocation_overhead_tokens,
        format_duration(Duration::from_micros(calibration.unbucketed_p50_micros)),
        format_duration(Duration::from_micros(calibration.power_of_two_p50_micros)),
        format_duration(Duration::from_micros(
            baseline.prefill_iteration_latency.p50_micros
        )),
        format_duration(Duration::from_micros(
            bucketed.prefill_iteration_latency.p50_micros
        )),
        format_duration(Duration::from_micros(
            adaptive.prefill_iteration_latency.p50_micros
        )),
        baseline_shape.prefill_padding_tokens,
        bucketed_shape.prefill_padding_tokens,
        adaptive_shape.prefill_padding_tokens,
        baseline_shape.prefill_invocations,
        bucketed_shape.prefill_invocations,
        adaptive_shape.prefill_invocations,
        baseline.output_tokens_per_second().unwrap_or(0.0),
        bucketed.output_tokens_per_second().unwrap_or(0.0),
        adaptive.output_tokens_per_second().unwrap_or(0.0),
        format_bytes(baseline.peak_kv_cache_bytes()),
        format_bytes(bucketed.peak_kv_cache_bytes()),
        format_bytes(adaptive.peak_kv_cache_bytes()),
    );
    if let Some(directory) = results_directory {
        let recorded_at_unix_micros = unix_time_micros()?;
        let report = PrefillBenchmarkReport::new_three_way(
            PrefillBenchmarkMetadata {
                recorded_at_unix_micros,
                model: model_name.to_owned(),
                device: std::env::var("OLA_BENCH_DEVICE_LABEL")
                    .unwrap_or_else(|_| device.label().to_owned()),
                software_revision: std::env::var("OLA_SOFTWARE_REVISION")
                    .unwrap_or_else(|_| format!("ola-ui {}", env!("CARGO_PKG_VERSION"))),
            },
            &config,
            &comparison,
        )
        .map_err(|error| error.to_string())?;
        let path = persist_prefill_report(directory, recorded_at_unix_micros, &report)
            .map_err(|error| error.to_string())?;
        summary.push_str(&format!(" · saved {}", path.display()));
    }
    Ok(summary)
}

fn chunked_prefill_probe(model: &CausalLanguageModel) -> Result<String, String> {
    const CHUNK_SIZE: usize = 3;
    const FAIRNESS_CHUNK_SIZES: &[usize] = &[2, 4, 8];
    let comparison = benchmark_chunked_prefill_execution(
        model,
        &PrefillExecutionConfig {
            prompts: vec![vec![1; 7], vec![1; 5]],
            max_new_tokens: 2,
        },
        CHUNK_SIZE,
    )
    .map_err(|error| error.to_string())?;
    let fairness = benchmark_prefill_fairness_trials(
        model,
        &PrefillExecutionConfig {
            prompts: vec![vec![1; 2], vec![1; 15], vec![1; 4], vec![1; 7]],
            max_new_tokens: 3,
        },
        2,
        FAIRNESS_CHUNK_SIZES,
        1,
        3,
    )
    .map_err(|error| error.to_string())?;
    let request_iterations = |run: &ola_bench::PrefillFairnessRun| {
        run.requests
            .iter()
            .map(|request| {
                format!(
                    "{}/{}",
                    request.first_token_iteration, request.completion_iteration
                )
            })
            .collect::<Vec<_>>()
            .join(",")
    };
    let request_latency = |latencies: &[ola_bench::PrefillFairnessLatency]| {
        latencies
            .iter()
            .map(|latency| {
                format!(
                    "{}:{}/{}:{}/{}",
                    latency.request_index,
                    latency.ttft.p50_micros / 1_000,
                    latency.ttft.p95_micros / 1_000,
                    latency.completion.p50_micros / 1_000,
                    latency.completion.p95_micros / 1_000
                )
            })
            .collect::<Vec<_>>()
            .join(",")
    };
    let chunked_latencies = fairness.chunked_latency();
    let fairness_summary = fairness
        .chunked
        .iter()
        .enumerate()
        .map(|(index, runs)| {
            let run = &runs[0];
            format!(
                "{}tok [{}] / {} iterations p50/p95 TTFT/completion ms [{}]",
                run.chunk_size.unwrap_or_default(),
                request_iterations(run),
                run.iterations,
                request_latency(&chunked_latencies[index])
            )
        })
        .collect::<Vec<_>>()
        .join("; ");
    Ok(format!(
        "2 prompts (7/5 tokens) · chunk size {CHUNK_SIZE} · prefill calls {} → {} · tensor calls {} → {} · iterations {} → {} · prefill {} → {} · output preserved · peak KV {} → {} · fairness (1 warm-up, 3 measured, two active) request TTFT/completion iterations baseline [{}], p50/p95 ms [{}], chunks {fairness_summary}",
        comparison.unchunked.prefill_invocations,
        comparison.chunked.prefill_invocations,
        comparison.unchunked.tensor_prefill_invocations,
        comparison.chunked.tensor_prefill_invocations,
        comparison.unchunked.iterations,
        comparison.chunked.iterations,
        format_duration(comparison.unchunked.prefill_iteration_elapsed),
        format_duration(comparison.chunked.prefill_iteration_elapsed),
        format_bytes(comparison.unchunked.peak_kv_cache_bytes),
        format_bytes(comparison.chunked.peak_kv_cache_bytes),
        request_iterations(&fairness.unchunked[0]),
        request_latency(&fairness.unchunked_latency()),
    ))
}

fn unix_time_micros() -> Result<u64, String> {
    let elapsed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| format!("system clock precedes Unix epoch: {error}"))?;
    u64::try_from(elapsed.as_micros())
        .map_err(|_| "current Unix timestamp exceeds the report format".to_owned())
}

fn persist_prefill_report(
    directory: &Path,
    recorded_at_unix_micros: u64,
    report: &PrefillBenchmarkReport,
) -> Result<PathBuf, PrefillReportError> {
    fs::create_dir_all(directory)?;
    for suffix in 0..1_000_u16 {
        let filename = if suffix == 0 {
            format!("prefill-strategies-{recorded_at_unix_micros}.json")
        } else {
            format!("prefill-strategies-{recorded_at_unix_micros}-{suffix}.json")
        };
        let path = directory.join(filename);
        match report.write_json_new(&path) {
            Ok(()) => return Ok(path),
            Err(PrefillReportError::Io(error)) if error.kind() == io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error),
        }
    }
    Err(PrefillReportError::Io(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "could not reserve a unique prefill report filename",
    )))
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
        PREFILL_BUCKET_EXPERIMENT => match prefill_bucketing_probe() {
            Ok(result) => WorkerEvent::WorkComplete {
                kind: WorkKind::Experiment,
                id: id.into(),
                result,
            },
            Err(message) => work_failed(WorkKind::Experiment, id, message),
        },
        ADAPTIVE_PREFILL_EXPERIMENT => match adaptive_prefill_probe() {
            Ok(result) => WorkerEvent::WorkComplete {
                kind: WorkKind::Experiment,
                id: id.into(),
                result,
            },
            Err(message) => work_failed(WorkKind::Experiment, id, message),
        },
        SCHEDULER_EXPERIMENT => match scheduler_comparison_probe() {
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
            .start_queue_head(Timestamp::from_micros(base))
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

fn prefill_bucketing_probe() -> Result<String, String> {
    const PROMPT_LENGTHS: &[usize] = &[5, 7, 9, 15, 17, 31, 33, 63];
    let comparison = compare_prefill_batching(PROMPT_LENGTHS).map_err(|error| error.to_string())?;
    let baseline_efficiency = comparison
        .unbucketed
        .input_token_efficiency()
        .unwrap_or(0.0)
        * 100.0;
    let bucketed_efficiency = comparison
        .power_of_two
        .input_token_efficiency()
        .unwrap_or(0.0)
        * 100.0;
    let reduction = comparison.padding_reduction().unwrap_or(0.0) * 100.0;
    Ok(format!(
        "{} prompts · padding {} → {} ({reduction:.1}% less) · invocations {} → {} · efficiency {baseline_efficiency:.1}% → {bucketed_efficiency:.1}%",
        PROMPT_LENGTHS.len(),
        comparison.unbucketed.padding_tokens,
        comparison.power_of_two.padding_tokens,
        comparison.unbucketed.invocations,
        comparison.power_of_two.invocations,
    ))
}

/// Compares scheduling baselines on seeded traces with the synthetic model
/// and virtual time, so the result is identical on every machine.
fn scheduler_comparison_probe() -> Result<String, String> {
    const REQUESTS: usize = 120;
    let millis = |micros: u64| micros as f64 / 1_000.0;
    let fifo = || ComposedPolicy::default().with_admission(MemoryAwareAdmission);
    let config = SimulationConfig::default();

    let overload = SchedulingTrace::overload(TraceParameters {
        seed: 3,
        requests: REQUESTS,
        mean_interarrival_micros: 2_000,
    });
    let sizes = compare_schedulers(
        &overload,
        &config,
        vec![
            Box::new(fifo()) as Box<dyn ExecutionPolicy>,
            Box::new(fifo().with_scheduler(ShortestJobFirst::new(JobCost::Prompt))),
        ],
        DEFAULT_STARVATION_FACTOR,
    )
    .map_err(|error| error.to_string())?;
    let line = |label: &str, summary: &SchedulingSummary| {
        format!(
            "{label} p50/p95 {:.1}/{:.1} ms, fairness {:.2}, starved {}",
            millis(summary.completion.p50_micros),
            millis(summary.completion.p95_micros),
            summary.slowdown.jain_fairness,
            summary.starved,
        )
    };

    let priorities = SchedulingTrace::priority_classes(TraceParameters {
        seed: 3,
        requests: REQUESTS,
        mean_interarrival_micros: 300,
    });
    let urgency = compare_schedulers(
        &priorities,
        &config,
        vec![
            Box::new(fifo()) as Box<dyn ExecutionPolicy>,
            Box::new(fifo().with_scheduler(PriorityScheduler::new())),
        ],
        DEFAULT_STARVATION_FACTOR,
    )
    .map_err(|error| error.to_string())?;
    let urgent_p95 = |summary: &SchedulingSummary| {
        summary
            .classes
            .iter()
            .find(|class| class.class == "priority=5")
            .map_or(0.0, |class| millis(class.completion.p95_micros))
    };

    Ok(format!(
        "{REQUESTS} requests, virtual time · overload: {} · {} · priority classes: urgent p95 fifo {:.1} ms → priority {:.1} ms",
        line("fifo", &sizes.summaries[0]),
        line("shortest-job-first", &sizes.summaries[1]),
        urgent_p95(&urgency.summaries[0]),
        urgent_p95(&urgency.summaries[1]),
    ))
}

fn adaptive_prefill_probe() -> Result<String, String> {
    const INVOCATION_OVERHEAD_TOKENS: usize = 32;
    const DIVERSE: &[usize] = &[5, 7, 9, 63];
    const CLUSTERED: &[usize] = &[5, 6, 7, 8];
    let diverse = analyze_cost_aware_prefill(DIVERSE, INVOCATION_OVERHEAD_TOKENS)
        .map_err(|error| error.to_string())?;
    let clustered = analyze_cost_aware_prefill(CLUSTERED, INVOCATION_OVERHEAD_TOKENS)
        .map_err(|error| error.to_string())?;
    Ok(format!(
        "{INVOCATION_OVERHEAD_TOKENS} token-equivalent call cost · diverse {:?}: cost {} → {}, {} padding / {} calls · clustered {:?}: cost {} → {}, {} padding / {} calls",
        diverse.decision.selected,
        diverse.decision.unbucketed.estimated_cost,
        diverse.decision.power_of_two.estimated_cost,
        diverse.selected.padding_tokens,
        diverse.selected.invocations,
        clustered.decision.selected,
        clustered.decision.unbucketed.estimated_cost,
        clustered.decision.power_of_two.estimated_cost,
        clustered.selected.padding_tokens,
        clustered.selected.invocations,
    ))
}

fn work_failed(kind: WorkKind, id: &str, message: impl Into<String>) -> WorkerEvent {
    WorkerEvent::WorkFailed {
        kind,
        id: id.into(),
        message: message.into(),
    }
}

fn is_model_benchmark(id: &str) -> bool {
    matches!(
        id,
        LLAMA_BENCHMARK | PREFILL_EXECUTION_BENCHMARK | CHUNKED_PREFILL_BENCHMARK
    )
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
    fn prefill_bucketing_probe_reports_the_compute_shape_tradeoff() {
        let summary = prefill_bucketing_probe().unwrap();

        assert!(summary.contains("8 prompts"));
        assert!(summary.contains("padding"));
        assert!(summary.contains("invocations"));
        assert!(summary.contains("efficiency"));
    }

    #[test]
    fn adaptive_prefill_probe_selects_per_workload_shape() {
        let summary = adaptive_prefill_probe().unwrap();

        assert!(summary.contains("diverse PowerOfTwo"));
        assert!(summary.contains("cost 284 → 182"));
        assert!(summary.contains("clustered Unbucketed"));
        assert!(summary.contains("cost 64 → 64"));
    }

    #[test]
    fn local_backend_exposes_non_model_work_without_an_artifact() {
        let mut backend = LocalBackend::new(None);
        let snapshot = backend.snapshot().unwrap();

        assert_eq!(snapshot.runtime.connection, ConnectionState::Connected);
        assert!(
            snapshot
                .benchmarks
                .iter()
                .find(|item| item.id == KV_BENCHMARK)
                .is_some_and(|item| item.state == RunState::Ready)
        );
        assert!(
            snapshot
                .benchmarks
                .iter()
                .find(|item| item.id == PREFILL_EXECUTION_BENCHMARK)
                .is_some_and(|item| item.state == RunState::Unavailable)
        );
        assert!(backend.submit_prompt("hello").is_err());
    }

    #[test]
    fn local_backend_runs_the_prefill_bucketing_experiment() {
        let mut backend = LocalBackend::new(None);

        backend.start_experiment(PREFILL_BUCKET_EXPERIMENT).unwrap();
        let snapshot = wait_until(&mut backend, |snapshot| {
            snapshot
                .experiments
                .iter()
                .find(|item| item.id == PREFILL_BUCKET_EXPERIMENT)
                .is_some_and(|item| item.state == RunState::Succeeded)
        });
        let experiment = snapshot
            .experiments
            .iter()
            .find(|item| item.id == PREFILL_BUCKET_EXPERIMENT)
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

        backend.start_experiment(SCHEDULER_EXPERIMENT).unwrap();
        let snapshot = wait_until(&mut backend, |snapshot| {
            snapshot
                .experiments
                .iter()
                .find(|item| item.id == SCHEDULER_EXPERIMENT)
                .is_some_and(|item| item.state == RunState::Succeeded)
        });
        let result = snapshot
            .experiments
            .iter()
            .find(|item| item.id == SCHEDULER_EXPERIMENT)
            .and_then(|item| item.last_result.as_deref())
            .unwrap();

        assert!(result.contains("120 requests, virtual time"));
        assert!(result.contains("shortest-job-first p50/p95"));
        assert!(result.contains("urgent p95 fifo 159.4 ms → priority 23.6 ms"));
    }

    #[test]
    fn local_backend_runs_the_adaptive_prefill_experiment() {
        let mut backend = LocalBackend::new(None);

        backend
            .start_experiment(ADAPTIVE_PREFILL_EXPERIMENT)
            .unwrap();
        let snapshot = wait_until(&mut backend, |snapshot| {
            snapshot
                .experiments
                .iter()
                .find(|item| item.id == ADAPTIVE_PREFILL_EXPERIMENT)
                .is_some_and(|item| item.state == RunState::Succeeded)
        });
        let result = snapshot
            .experiments
            .iter()
            .find(|item| item.id == ADAPTIVE_PREFILL_EXPERIMENT)
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

        backend
            .start_benchmark(PREFILL_EXECUTION_BENCHMARK)
            .unwrap();
        let snapshot = wait_until_for(&mut backend, Duration::from_secs(180), |snapshot| {
            snapshot
                .benchmarks
                .iter()
                .find(|item| item.id == PREFILL_EXECUTION_BENCHMARK)
                .is_some_and(|item| item.state == RunState::Succeeded)
        });
        let result = snapshot
            .benchmarks
            .iter()
            .find(|item| item.id == PREFILL_EXECUTION_BENCHMARK)
            .and_then(|item| item.last_result.as_deref())
            .unwrap();
        assert!(result.contains("calibrated"));
        assert!(result.contains("3 rotating trials + 1 warmup"));
        assert!(result.contains("prefill p50 U/P/A"));
        assert!(result.contains("padding U/P/A 11 / 2 /"));
        assert!(result.contains("saved"));
        assert_eq!(snapshot.runtime.connection, ConnectionState::Connected);

        backend.start_benchmark(CHUNKED_PREFILL_BENCHMARK).unwrap();
        let snapshot = wait_until_for(&mut backend, Duration::from_secs(180), |snapshot| {
            snapshot
                .benchmarks
                .iter()
                .find(|item| item.id == CHUNKED_PREFILL_BENCHMARK)
                .is_some_and(|item| item.state == RunState::Succeeded)
        });
        let chunked = snapshot
            .benchmarks
            .iter()
            .find(|item| item.id == CHUNKED_PREFILL_BENCHMARK)
            .and_then(|item| item.last_result.as_deref())
            .unwrap();
        assert!(chunked.contains("chunk size 3"));
        assert!(chunked.contains("prefill calls 1 → 3"));
        assert!(chunked.contains("tensor calls 1 → 2"));
        assert!(chunked.contains("output preserved"));
        assert!(chunked.contains("fairness (1 warm-up, 3 measured, two active)"));
        assert!(chunked.contains("p50/p95 ms"));

        let paths = fs::read_dir(&results_directory)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect::<Vec<_>>();
        assert_eq!(paths.len(), 1);
        let report = PrefillBenchmarkReport::read_json(&paths[0]).unwrap();
        assert_eq!(report.protocol.warmup_trials, 1);
        assert_eq!(report.protocol.measured_trials, 3);
        assert!(
            report
                .protocol
                .adaptive_invocation_overhead_tokens
                .is_some()
        );
        assert_eq!(report.trials.len(), 3);
        assert!(report.trials.iter().all(|trial| trial.cost_aware.is_some()));
        assert!(report.cost_aware.is_some());
        assert_eq!(report.metadata.model, "ola-rt-tiny-random-llama");
        fs::remove_file(&paths[0]).unwrap();
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
