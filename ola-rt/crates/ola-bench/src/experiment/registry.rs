//! Experiments as data: descriptors, parameters, and the registry.

use std::{
    collections::BTreeMap,
    error::Error,
    fmt,
    io::Write,
    sync::{Arc, Mutex},
};

use ola_model::CausalLanguageModel;
use ola_observe::{Tracer, trace::write_jsonl};
use ola_runtime::{ExecutorIteration, RuntimeState};
use serde::{Deserialize, Serialize};

use super::{ExperimentKind, ExperimentReport, ReportError, TimeDomain};

/// Which workbench list an experiment belongs to.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExperimentCategory {
    /// Measures a component's performance.
    Benchmark,
    /// Tests a policy or behaviour.
    Experiment,
}

/// Something an experiment needs from its frontend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Requirement {
    /// A loaded causal language model.
    LoadedModel,
}

/// A parameter value.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ParameterValue {
    /// A non-negative integer.
    Integer(u64),
    /// Free text, such as a trace name or a comma-separated list.
    Text(String),
}

impl fmt::Display for ParameterValue {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Integer(value) => value.fmt(formatter),
            Self::Text(value) => value.fmt(formatter),
        }
    }
}

/// A declared parameter and its default.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParameterSpec {
    /// Stable snake_case name.
    pub name: &'static str,
    /// What it controls.
    pub description: &'static str,
    /// Value used when not overridden; also fixes the parameter's type.
    pub default: ParameterValue,
}

impl ParameterSpec {
    /// Declares an integer parameter.
    pub const fn integer(name: &'static str, description: &'static str, default: u64) -> Self {
        Self {
            name,
            description,
            default: ParameterValue::Integer(default),
        }
    }

    /// Declares a text parameter.
    pub fn text(name: &'static str, description: &'static str, default: &str) -> Self {
        Self {
            name,
            description,
            default: ParameterValue::Text(default.to_owned()),
        }
    }
}

/// Everything a frontend needs to list, configure, and gate an experiment.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExperimentDescriptor {
    /// Stable ID, such as `scheduler-comparison`.
    pub id: &'static str,
    /// Display name.
    pub name: &'static str,
    /// One-line description.
    pub description: &'static str,
    /// Workbench list.
    pub category: ExperimentCategory,
    /// Probe or study.
    pub kind: ExperimentKind,
    /// Clock its timed metrics come from.
    pub domain: TimeDomain,
    /// Version of its metrics and parameters.
    pub payload_version: u32,
    /// What the frontend must supply.
    pub requirements: Vec<Requirement>,
    /// Declared parameters, in display order.
    pub parameters: Vec<ParameterSpec>,
}

/// Serializable selection and parameter overrides for one experiment run.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ExperimentRunConfig {
    /// Stable experiment identifier.
    pub id: String,
    /// Parameter overrides, keyed by the descriptor's stable parameter names.
    #[serde(default)]
    pub parameters: BTreeMap<String, String>,
}

impl ExperimentRunConfig {
    /// Decodes a run configuration from JSON.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    fn overrides(&self) -> Vec<(String, String)> {
        self.parameters
            .iter()
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect()
    }
}

/// Resolved parameter values for one run.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Parameters(BTreeMap<String, ParameterValue>);

impl Parameters {
    /// Applies `name=value` overrides to the descriptor's defaults.
    ///
    /// # Errors
    ///
    /// Returns [`ExperimentError::InvalidParameter`] for unknown names or
    /// values that do not parse as the parameter's type.
    pub fn resolve(
        descriptor: &ExperimentDescriptor,
        overrides: &[(String, String)],
    ) -> Result<Self, ExperimentError> {
        let mut values = descriptor
            .parameters
            .iter()
            .map(|spec| (spec.name.to_owned(), spec.default.clone()))
            .collect::<BTreeMap<_, _>>();
        for (name, raw) in overrides {
            let invalid = |reason: &str| ExperimentError::InvalidParameter {
                name: name.clone(),
                reason: reason.to_owned(),
            };
            let value = match values.get(name) {
                None => return Err(invalid("unknown parameter")),
                Some(ParameterValue::Integer(_)) => ParameterValue::Integer(
                    raw.parse()
                        .map_err(|_| invalid("expected a non-negative integer"))?,
                ),
                Some(ParameterValue::Text(_)) => ParameterValue::Text(raw.clone()),
            };
            values.insert(name.clone(), value);
        }
        Ok(Self(values))
    }

    /// Returns an integer parameter.
    ///
    /// # Errors
    ///
    /// Returns [`ExperimentError::InvalidParameter`] if it is missing or text.
    pub fn integer(&self, name: &str) -> Result<u64, ExperimentError> {
        match self.0.get(name) {
            Some(ParameterValue::Integer(value)) => Ok(*value),
            _ => Err(ExperimentError::InvalidParameter {
                name: name.to_owned(),
                reason: "expected an integer parameter".into(),
            }),
        }
    }

    /// Returns an integer parameter as `usize`.
    ///
    /// # Errors
    ///
    /// Returns [`ExperimentError::InvalidParameter`] if it is missing, text,
    /// or too large.
    pub fn size(&self, name: &str) -> Result<usize, ExperimentError> {
        usize::try_from(self.integer(name)?).map_err(|_| ExperimentError::InvalidParameter {
            name: name.to_owned(),
            reason: "value is too large".into(),
        })
    }

    /// Returns a text parameter.
    ///
    /// # Errors
    ///
    /// Returns [`ExperimentError::InvalidParameter`] if it is missing or an
    /// integer.
    pub fn text(&self, name: &str) -> Result<&str, ExperimentError> {
        match self.0.get(name) {
            Some(ParameterValue::Text(value)) => Ok(value),
            _ => Err(ExperimentError::InvalidParameter {
                name: name.to_owned(),
                reason: "expected a text parameter".into(),
            }),
        }
    }

    /// Returns every value as a stable string, for workload identity.
    pub fn as_strings(&self) -> BTreeMap<String, String> {
        self.0
            .iter()
            .map(|(name, value)| (name.clone(), value.to_string()))
            .collect()
    }
}

/// What a frontend supplies to a run.
#[derive(Clone, Debug, Default)]
pub struct ExperimentContext {
    /// Loaded model, for experiments that require one.
    pub model: Option<CausalLanguageModel>,
    /// Model identity for reports.
    pub model_label: Option<String>,
    /// Backend and device identity for reports.
    pub device: Option<String>,
    /// Where traceable experiments write request traces.
    pub trace: Option<ExperimentTrace>,
}

/// A shared JSON Lines destination for request traces.
///
/// An experiment may run many executors (policies, seeds, repetitions);
/// each writes its records labelled with a distinct `run`.
#[derive(Clone)]
pub struct ExperimentTrace(Arc<Mutex<Box<dyn Write + Send>>>);

impl ExperimentTrace {
    /// Writes traces to `writer`.
    pub fn new(writer: impl Write + Send + 'static) -> Self {
        Self(Arc::new(Mutex::new(Box::new(writer))))
    }

    /// Flushes buffered records.
    ///
    /// # Errors
    ///
    /// Returns any error from the writer.
    pub fn flush(&self) -> std::io::Result<()> {
        self.0
            .lock()
            .map_err(|_| std::io::Error::other("trace writer poisoned"))?
            .flush()
    }

    /// Starts tracing one executor run labelled `run`.
    pub(crate) fn run(&self, run: impl Into<String>) -> TraceRun {
        TraceRun {
            tracer: Tracer::default(),
            run: run.into(),
            output: self.clone(),
        }
    }
}

impl fmt::Debug for ExperimentTrace {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ExperimentTrace")
    }
}

/// Traces one executor run inside an experiment.
pub(crate) struct TraceRun {
    tracer: Tracer,
    run: String,
    output: ExperimentTrace,
}

impl TraceRun {
    /// Writes the records of one iteration; call with the state right after
    /// it.
    pub(crate) fn observe(
        &mut self,
        iteration: &ExecutorIteration,
        state: &RuntimeState,
    ) -> Result<(), ExperimentError> {
        let mut records = self.tracer.observe_iteration(iteration, state);
        for record in &mut records {
            record.run = Some(self.run.clone());
        }
        let mut writer = self
            .output
            .0
            .lock()
            .map_err(|_| ExperimentError::Run("trace writer poisoned".into()))?;
        write_jsonl(&records, &mut *writer)
            .map_err(|error| ExperimentError::Run(format!("could not write trace: {error}")))
    }
}

/// A finished run.
#[derive(Clone, Debug, PartialEq)]
pub struct ExperimentOutcome {
    /// The validated report.
    pub report: ExperimentReport,
    /// One-line human summary for frontends.
    pub summary: String,
}

/// A runnable experiment.
pub trait Experiment: Send + Sync {
    /// Returns the experiment's identity, requirements, and parameters.
    fn descriptor(&self) -> ExperimentDescriptor;

    /// Whether the experiment writes request traces to
    /// [`ExperimentContext::trace`]. Only experiments that drive an
    /// executor can.
    fn traceable(&self) -> bool {
        false
    }

    /// Runs the experiment.
    ///
    /// # Errors
    ///
    /// Returns [`ExperimentError`] if the run fails.
    fn run(
        &self,
        context: &ExperimentContext,
        parameters: &Parameters,
    ) -> Result<ExperimentOutcome, ExperimentError>;
}

/// Minimum measured repetitions for any study.
pub const STUDY_MIN_REPETITIONS: usize = 3;
/// Minimum measured repetitions for a host-time study.
pub const HOST_STUDY_MIN_REPETITIONS: usize = 5;
/// Minimum warm-ups for a host-time study.
pub const HOST_STUDY_MIN_WARMUPS: usize = 1;

/// Every experiment a frontend can run, in display order.
pub struct ExperimentRegistry {
    experiments: Vec<Box<dyn Experiment>>,
}

impl ExperimentRegistry {
    /// Creates a registry from experiments.
    ///
    /// # Panics
    ///
    /// Panics if two experiments share an ID, which is a programming error.
    pub fn new(experiments: Vec<Box<dyn Experiment>>) -> Self {
        let mut ids = experiments
            .iter()
            .map(|experiment| experiment.descriptor().id)
            .collect::<Vec<_>>();
        ids.sort_unstable();
        let count = ids.len();
        ids.dedup();
        assert_eq!(count, ids.len(), "experiment IDs must be unique");
        Self { experiments }
    }

    /// Returns every descriptor in display order.
    pub fn descriptors(&self) -> Vec<ExperimentDescriptor> {
        self.experiments
            .iter()
            .map(|experiment| experiment.descriptor())
            .collect()
    }

    /// Returns the descriptor for `id`.
    pub fn descriptor(&self, id: &str) -> Option<ExperimentDescriptor> {
        self.find(id).map(|experiment| experiment.descriptor())
    }

    /// Resolves parameters, checks requirements, runs `id`, and checks that
    /// the report identifies the experiment and meets its kind's protocol.
    ///
    /// # Errors
    ///
    /// Returns [`ExperimentError`] for unknown IDs, bad parameters, missing
    /// requirements, run failures, or nonconforming reports.
    pub fn run(
        &self,
        id: &str,
        context: &ExperimentContext,
        overrides: &[(String, String)],
    ) -> Result<ExperimentOutcome, ExperimentError> {
        let experiment = self
            .find(id)
            .ok_or_else(|| ExperimentError::UnknownExperiment(id.to_owned()))?;
        let descriptor = experiment.descriptor();
        let parameters = Parameters::resolve(&descriptor, overrides)?;
        if context.trace.is_some() && !experiment.traceable() {
            return Err(ExperimentError::NotTraceable(id.to_owned()));
        }
        if descriptor.requirements.contains(&Requirement::LoadedModel) && context.model.is_none() {
            return Err(ExperimentError::MissingRequirement(
                Requirement::LoadedModel,
            ));
        }
        let outcome = experiment.run(context, &parameters)?;
        conforms(&descriptor, &outcome.report)?;
        Ok(outcome)
    }

    /// Runs the experiment selected by a serializable configuration.
    pub fn run_config(
        &self,
        config: &ExperimentRunConfig,
        context: &ExperimentContext,
    ) -> Result<ExperimentOutcome, ExperimentError> {
        self.run(&config.id, context, &config.overrides())
    }

    /// Returns whether experiment `id` exists and can write traces.
    pub fn traceable(&self, id: &str) -> bool {
        self.find(id)
            .is_some_and(|experiment| experiment.traceable())
    }

    fn find(&self, id: &str) -> Option<&dyn Experiment> {
        self.experiments
            .iter()
            .find(|experiment| experiment.descriptor().id == id)
            .map(AsRef::as_ref)
    }
}

fn conforms(
    descriptor: &ExperimentDescriptor,
    report: &ExperimentReport,
) -> Result<(), ExperimentError> {
    let nonconforming = |reason: String| Err(ExperimentError::Nonconforming(reason));
    let identity = &report.experiment;
    if identity.id != descriptor.id
        || identity.kind != descriptor.kind
        || identity.domain != descriptor.domain
        || identity.payload_version != descriptor.payload_version
    {
        return nonconforming("report identity does not match the descriptor".into());
    }
    if descriptor.kind == ExperimentKind::Study {
        let (minimum_repetitions, minimum_warmups) = if descriptor.domain == TimeDomain::Host {
            (HOST_STUDY_MIN_REPETITIONS, HOST_STUDY_MIN_WARMUPS)
        } else {
            (STUDY_MIN_REPETITIONS, 0)
        };
        if report.protocol.repetitions < minimum_repetitions
            || report.protocol.warmups < minimum_warmups
        {
            return nonconforming(format!(
                "a {:?} study needs at least {minimum_repetitions} repetitions and \
                 {minimum_warmups} warm-ups",
                descriptor.domain
            ));
        }
    }
    Ok(())
}

/// Failure to configure or run an experiment.
#[derive(Debug)]
pub enum ExperimentError {
    /// No experiment has this ID.
    UnknownExperiment(String),
    /// A parameter is unknown or malformed.
    InvalidParameter {
        /// Parameter name.
        name: String,
        /// What is wrong.
        reason: String,
    },
    /// The frontend did not supply something the experiment needs.
    MissingRequirement(Requirement),
    /// A trace was requested from an experiment that cannot write one.
    NotTraceable(String),
    /// The experiment itself failed.
    Run(String),
    /// The report does not match the descriptor or its kind's protocol.
    Nonconforming(String),
    /// The report could not be built.
    Report(ReportError),
}

impl fmt::Display for ExperimentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownExperiment(id) => write!(formatter, "unknown experiment `{id}`"),
            Self::InvalidParameter { name, reason } => {
                write!(formatter, "parameter `{name}`: {reason}")
            }
            Self::MissingRequirement(Requirement::LoadedModel) => {
                formatter.write_str("this experiment needs a loaded model")
            }
            Self::NotTraceable(id) => write!(
                formatter,
                "experiment `{id}` does not drive an executor, so it cannot write traces"
            ),
            Self::Run(message) => formatter.write_str(message),
            Self::Nonconforming(reason) => write!(formatter, "nonconforming report: {reason}"),
            Self::Report(error) => error.fmt(formatter),
        }
    }
}

impl Error for ExperimentError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Report(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ReportError> for ExperimentError {
    fn from(value: ReportError) -> Self {
        Self::Report(value)
    }
}
