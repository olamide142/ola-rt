//! A stateful wrapper for policy configuration experiments.

use std::{collections::BTreeMap, error::Error, fmt};

use good_lp::{Expression, Solution, SolverModel, variable};
use ola_core::{RequestId, Timestamp};

use crate::{
    ExecutionPolicy, ExecutorEvent, ExecutorIteration, PlanningSnapshot, PolicyDecision,
    PolicyDescriptor, PolicyError,
};

/// Builds an executable policy from an adaptive parameter vector.
pub trait AdaptivePolicyFactory: Send {
    /// Constructs the policy that should be used at the next planning boundary.
    fn build(
        &self,
        theta: &[(&'static str, String)],
    ) -> Result<Box<dyn ExecutionPolicy>, PolicyError>;
}

impl<F> AdaptivePolicyFactory for F
where
    F: Fn(&[(&'static str, String)]) -> Result<Box<dyn ExecutionPolicy>, PolicyError> + Send,
{
    fn build(
        &self,
        theta: &[(&'static str, String)],
    ) -> Result<Box<dyn ExecutionPolicy>, PolicyError> {
        self(theta)
    }
}

/// Runtime ceilings and resource facts available to an adaptive optimizer.
///
/// `None` means that the runtime does not currently publish that constraint;
/// an optimizer must not silently interpret unknown capacity as unlimited.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct AdaptiveConstraints {
    /// Maximum number of simultaneously active requests.
    pub max_active_requests: Option<usize>,
    /// Maximum resumable-prefill chunk, when one is configured.
    pub prefill_chunk_size: Option<usize>,
    /// Available device memory, when telemetry is available.
    pub available_memory_bytes: Option<u64>,
}

/// One serialized adaptive parameter before it is resolved to the static key
/// used by an executable policy factory.
#[derive(Clone, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct AdaptiveParameterConfig {
    /// Stable parameter name understood by the policy factory.
    pub name: String,
    /// Serialized parameter value passed to the factory.
    pub value: String,
}

/// Serializable data configuration for an [`AdaptivePolicy`].
///
/// Factories, optimizers, validators, and update hooks intentionally remain
/// executable Rust components. A caller resolves parameter names explicitly
/// before constructing the policy, keeping configuration loading separate from
/// code execution and preserving the safe fallback boundary.
#[derive(Clone, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct AdaptivePolicyConfig {
    /// Initial adaptive parameter vector.
    #[serde(default)]
    pub theta: Vec<AdaptiveParameterConfig>,
    /// Committed iterations between optimizer proposals. Zero normalizes to
    /// one when applied to an executable policy.
    #[serde(default)]
    pub optimization_interval: u64,
    /// Seed reserved for observation-time updates.
    #[serde(default)]
    pub seed: u64,
    /// Runtime ceilings and resource facts supplied to the optimizer.
    #[serde(default)]
    pub constraints: AdaptiveConstraints,
}

impl Default for AdaptivePolicyConfig {
    fn default() -> Self {
        Self {
            theta: Vec::new(),
            optimization_interval: 1,
            seed: 0,
            constraints: AdaptiveConstraints::default(),
        }
    }
}

/// Configuration failure raised before an executable adaptive policy is built.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AdaptiveConfigError {
    /// A parameter name is empty or repeated.
    InvalidParameter(String),
    /// A configured parameter is not known by the selected factory.
    UnknownParameter(String),
}

impl fmt::Display for AdaptiveConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter(message) => {
                write!(formatter, "invalid adaptive parameter: {message}")
            }
            Self::UnknownParameter(name) => {
                write!(formatter, "unknown adaptive parameter `{name}`")
            }
        }
    }
}

impl Error for AdaptiveConfigError {}

impl AdaptivePolicyConfig {
    /// Resolves serialized names and applies the data configuration to an
    /// executable policy. Factories and optimizers are attached separately.
    pub fn build_policy<F>(
        &self,
        inner: impl ExecutionPolicy + 'static,
        resolve: F,
    ) -> Result<AdaptivePolicy, AdaptiveConfigError>
    where
        F: Fn(&str) -> Option<&'static str>,
    {
        let mut names = std::collections::BTreeSet::new();
        let theta = self
            .theta
            .iter()
            .map(|parameter| {
                if parameter.name.is_empty() || !names.insert(parameter.name.as_str()) {
                    return Err(AdaptiveConfigError::InvalidParameter(
                        parameter.name.clone(),
                    ));
                }
                let name = resolve(&parameter.name)
                    .ok_or_else(|| AdaptiveConfigError::UnknownParameter(parameter.name.clone()))?;
                Ok((name, parameter.value.clone()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(AdaptivePolicy::new(inner, theta)
            .with_constraints(self.constraints)
            .with_optimization_interval(self.optimization_interval)
            .with_seed(self.seed))
    }
}

impl AdaptiveConstraints {
    /// Derives solver ceilings and currently free device memory from a
    /// planning snapshot. A zero-capacity runtime intentionally reports
    /// unknown memory rather than a zero-byte feasible region.
    pub fn from_snapshot(snapshot: &PlanningSnapshot) -> Self {
        Self {
            max_active_requests: Some(snapshot.limits.max_active_requests),
            prefill_chunk_size: snapshot.limits.prefill_chunk_size,
            available_memory_bytes: (snapshot.memory.capacity_bytes > 0)
                .then(|| snapshot.memory.available_bytes()),
        }
    }
}

/// Failure returned when an optimizer cannot produce a feasible parameter
/// vector. The active policy remains installed when this happens.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AdaptiveOptimizationError {
    /// An integer parameter domain is malformed.
    InvalidDomain(String),
    /// The optimizer rejected its inputs or proposed an invalid candidate.
    Infeasible(String),
    /// No candidate was available for the current observation window.
    NoCandidate,
}

impl fmt::Display for AdaptiveOptimizationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDomain(message) => write!(formatter, "invalid adaptive domain: {message}"),
            Self::Infeasible(message) => {
                write!(formatter, "adaptive candidate infeasible: {message}")
            }
            Self::NoCandidate => formatter.write_str("adaptive optimizer produced no candidate"),
        }
    }
}

impl Error for AdaptiveOptimizationError {}

/// Chooses the next adaptive parameter vector after a committed observation.
///
/// Implementations may be exact enumeration, MILP/LP-backed, convex, or
/// seeded stochastic optimizers. They never run during `plan`.
pub trait AdaptiveOptimizer: Send {
    /// Proposes parameters for the next planning boundary.
    fn propose(
        &mut self,
        current: &[(&'static str, String)],
        observations: &[AdaptiveRequestObservation],
        constraints: AdaptiveConstraints,
    ) -> Result<Vec<(&'static str, String)>, AdaptiveOptimizationError>;
}

impl<F> AdaptiveOptimizer for F
where
    F: FnMut(
            &[(&'static str, String)],
            &[AdaptiveRequestObservation],
            AdaptiveConstraints,
        ) -> Result<Vec<(&'static str, String)>, AdaptiveOptimizationError>
        + Send,
{
    fn propose(
        &mut self,
        current: &[(&'static str, String)],
        observations: &[AdaptiveRequestObservation],
        constraints: AdaptiveConstraints,
    ) -> Result<Vec<(&'static str, String)>, AdaptiveOptimizationError> {
        self(current, observations, constraints)
    }
}

/// A bounded integer parameter domain for the finite optimizer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AdaptiveIntegerDomain {
    /// Stable parameter key passed to the policy factory.
    pub name: &'static str,
    /// Inclusive lower bound.
    pub min: i64,
    /// Inclusive upper bound.
    pub max: i64,
    /// Positive enumeration step.
    pub step: i64,
}

impl AdaptiveIntegerDomain {
    fn validate(self) -> Result<(), AdaptiveOptimizationError> {
        if self.name.is_empty() {
            return Err(AdaptiveOptimizationError::InvalidDomain(
                "parameter name cannot be empty".into(),
            ));
        }
        if self.min > self.max {
            return Err(AdaptiveOptimizationError::InvalidDomain(format!(
                "{} has min {} greater than max {}",
                self.name, self.min, self.max
            )));
        }
        if self.step <= 0 {
            return Err(AdaptiveOptimizationError::InvalidDomain(format!(
                "{} has non-positive step {}",
                self.name, self.step
            )));
        }
        Ok(())
    }
}

fn runtime_domain_max(domain: AdaptiveIntegerDomain, constraints: AdaptiveConstraints) -> i64 {
    match domain.name {
        "max_active_requests" | "effective_slots" | "slots" => constraints
            .max_active_requests
            .map_or(domain.max, |value| domain.max.min(value as i64)),
        "prefill_chunk_size" | "effective_chunk_size" | "chunk_size" => constraints
            .prefill_chunk_size
            .map_or(domain.max, |value| domain.max.min(value as i64)),
        _ => domain.max,
    }
}

/// Exact finite-search optimizer used as the dependency-free correctness
/// oracle for later MILP and LP adapters.
pub struct FiniteCandidateOptimizer<F> {
    domains: Vec<AdaptiveIntegerDomain>,
    objective: F,
    memory_constraints: Vec<AdaptiveMemoryConstraint>,
}

/// One upper-bounded linear constraint over integer adaptive parameters.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdaptiveLinearConstraint {
    /// Parameter coefficients in the left-hand side.
    pub terms: Vec<(&'static str, i64)>,
    /// Inclusive upper bound for the left-hand side.
    pub upper_bound: i64,
}

/// A resource constraint expressed in bytes.
///
/// The constraint is active only when `AdaptiveConstraints::available_memory_bytes`
/// is known. This lets a policy describe its memory demand once while still
/// allowing optimization in environments that do not publish memory telemetry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdaptiveMemoryConstraint {
    /// Fixed bytes required before adaptive parameters are applied.
    pub base_bytes: u64,
    /// Additional bytes required per parameter unit.
    pub terms: Vec<(&'static str, u64)>,
}

/// Rebuilds a MILP objective from the latest observations.
type ObjectiveBuilder = Box<
    dyn FnMut(&[AdaptiveRequestObservation], AdaptiveConstraints) -> Vec<(&'static str, i64)>
        + Send,
>;

/// Accepts or rejects a candidate decision before it reaches the executor.
type DecisionValidator = Box<dyn Fn(&PlanningSnapshot, &PolicyDecision) -> bool + Send + Sync>;

/// Adjusts θ directly after each committed iteration.
type ThetaUpdater = Box<
    dyn FnMut(&ExecutorIteration, &[AdaptiveRequestObservation], &mut Vec<(&'static str, String)>)
        + Send,
>;

/// Explicit cost of changing tunable knobs between decision epochs.
///
/// `switch_cost(θ_prev, θ) = Σ_k w_k · 1[θ_k ≠ θ_prev,k]` over the listed
/// knobs only, in the optimizer's own objective units (for the replay
/// optimizer, objective microseconds). Knobs that are not listed, including
/// descriptive θ entries no factory reads, never count as a switch. A knob
/// missing from `θ_prev` counts as changed when `θ` sets it: installing a
/// value where none was known is a reconfiguration. Weights below zero are
/// treated as zero.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SwitchingCost {
    /// Cost charged when each knob changes.
    pub weights: Vec<(&'static str, i64)>,
}

impl SwitchingCost {
    /// Charges `cost` for every change of any of `knobs`.
    pub fn uniform(knobs: &[&'static str], cost: i64) -> Self {
        Self {
            weights: knobs.iter().map(|knob| (*knob, cost)).collect(),
        }
    }

    /// Whether every weight is zero, so selection ignores switching.
    pub fn is_zero(&self) -> bool {
        self.weights.iter().all(|(_, weight)| *weight <= 0)
    }

    /// `switch_cost(previous, candidate)`.
    pub fn cost(
        &self,
        previous: &[(&'static str, String)],
        candidate: &[(&'static str, String)],
    ) -> i64 {
        let value = |theta: &[(&'static str, String)], knob: &str| {
            theta
                .iter()
                .find(|(key, _)| *key == knob)
                .map(|(_, value)| value.trim().to_owned())
        };
        self.weights
            .iter()
            .filter(|(knob, weight)| {
                *weight > 0 && {
                    let next = value(candidate, knob);
                    next.is_some() && next != value(previous, knob)
                }
            })
            .fold(0_i64, |total, (_, weight)| total.saturating_add(*weight))
    }
}

/// A linear integer optimizer backed by `good_lp` and its pure-Rust `microlp`
/// solver. The finite optimizer remains the exact small-domain oracle.
pub struct MilpCandidateOptimizer {
    domains: Vec<AdaptiveIntegerDomain>,
    objective: Vec<(&'static str, i64)>,
    objective_builder: Option<ObjectiveBuilder>,
    constraints: Vec<AdaptiveLinearConstraint>,
    memory_constraints: Vec<AdaptiveMemoryConstraint>,
    constant: i64,
    switching: SwitchingCost,
    last_objective: Option<MilpObjectiveValue>,
}

/// Objective of the last MILP solution, split into its two terms.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MilpObjectiveValue {
    /// Linear operating cost, including the constant term.
    pub operating: i64,
    /// Switching cost from the vector the optimizer was given.
    pub switching: i64,
}

impl MilpObjectiveValue {
    /// The minimized total.
    pub const fn total(self) -> i64 {
        self.operating.saturating_add(self.switching)
    }
}

impl MilpCandidateOptimizer {
    /// Creates a minimization problem over bounded integer parameters.
    pub fn new(
        mut domains: Vec<AdaptiveIntegerDomain>,
        objective: Vec<(&'static str, i64)>,
    ) -> Result<Self, AdaptiveOptimizationError> {
        for domain in &domains {
            domain.validate()?;
        }
        domains.sort_by_key(|domain| domain.name);
        if domains
            .windows(2)
            .any(|window| window[0].name == window[1].name)
        {
            return Err(AdaptiveOptimizationError::InvalidDomain(
                "parameter names must be unique".into(),
            ));
        }
        for (name, _) in &objective {
            if !domains.iter().any(|domain| domain.name == *name) {
                return Err(AdaptiveOptimizationError::InvalidDomain(format!(
                    "objective references unknown parameter {name}"
                )));
            }
        }
        Ok(Self {
            domains,
            objective,
            objective_builder: None,
            constraints: Vec::new(),
            memory_constraints: Vec::new(),
            constant: 0,
            switching: SwitchingCost::default(),
            last_objective: None,
        })
    }

    /// Adds `λ_sw · switch_cost(current, θ)` to the objective.
    ///
    /// Each weighted knob gets a binary `z_k` with
    /// `|x_k − x_prev,k| ≤ M_k z_k`, where `x` counts grid steps and `M_k` is
    /// the domain's step count, so `z_k = 1` exactly when the knob moves
    /// (minimization drives it to zero otherwise). A previous value that is
    /// missing or off the grid fixes `z_k = 1`.
    #[must_use]
    pub fn with_switching_cost(mut self, switching: SwitchingCost) -> Self {
        self.switching = switching;
        self
    }

    /// Objective of the most recent successful solve, as selected.
    pub const fn last_objective(&self) -> Option<MilpObjectiveValue> {
        self.last_objective
    }

    /// Adds an upper-bounded linear constraint.
    #[must_use]
    pub fn with_constraint(mut self, constraint: AdaptiveLinearConstraint) -> Self {
        self.constraints.push(constraint);
        self
    }

    /// Adds a memory-demand constraint enforced when memory telemetry exists.
    #[must_use]
    pub fn with_memory_constraint(mut self, constraint: AdaptiveMemoryConstraint) -> Self {
        self.memory_constraints.push(constraint);
        self
    }

    /// Adds a constant term to the objective.
    #[must_use]
    pub const fn with_constant(mut self, constant: i64) -> Self {
        self.constant = constant;
        self
    }

    /// Rebuilds linear objective coefficients from each observation window.
    #[must_use]
    pub fn with_objective_builder<F>(mut self, builder: F) -> Self
    where
        F: FnMut(&[AdaptiveRequestObservation], AdaptiveConstraints) -> Vec<(&'static str, i64)>
            + Send
            + 'static,
    {
        self.objective_builder = Some(Box::new(builder));
        self
    }
}

impl AdaptiveOptimizer for MilpCandidateOptimizer {
    fn propose(
        &mut self,
        current: &[(&'static str, String)],
        observations: &[AdaptiveRequestObservation],
        constraints: AdaptiveConstraints,
    ) -> Result<Vec<(&'static str, String)>, AdaptiveOptimizationError> {
        let objective_coefficients = self
            .objective_builder
            .as_mut()
            .map(|builder| builder(observations, constraints))
            .unwrap_or_else(|| self.objective.clone());
        let domain_maxima = self
            .domains
            .iter()
            .map(|domain| runtime_domain_max(*domain, constraints))
            .collect::<Vec<_>>();
        if self
            .domains
            .iter()
            .zip(&domain_maxima)
            .any(|(domain, max)| *max < domain.min)
        {
            return Err(AdaptiveOptimizationError::NoCandidate);
        }
        let mut variables = good_lp::ProblemVariables::new();
        let decision_variables = self
            .domains
            .iter()
            .zip(&domain_maxima)
            .map(|(domain, max)| {
                let steps = (max - domain.min) / domain.step;
                variables.add(variable().integer().min(0.0).max(steps as f64))
            })
            .collect::<Vec<_>>();

        let mut objective = Expression::from(self.constant as f64);
        for (name, coefficient) in &objective_coefficients {
            let index = self
                .domains
                .iter()
                .position(|domain| domain.name == *name)
                .ok_or_else(|| {
                    AdaptiveOptimizationError::InvalidDomain(format!(
                        "objective references unknown parameter {name}"
                    ))
                })?;
            let domain = self.domains[index];
            objective += *coefficient as f64 * domain.min as f64;
            objective += *coefficient as f64 * domain.step as f64 * decision_variables[index];
        }

        // Switching indicators: (domain index, binary, previous step or
        // None when the previous value is unknown).
        let mut switches = Vec::new();
        for (index, domain) in self.domains.iter().enumerate() {
            let weight = self
                .switching
                .weights
                .iter()
                .filter(|(knob, _)| *knob == domain.name)
                .map(|(_, weight)| (*weight).max(0))
                .sum::<i64>();
            if weight == 0 {
                continue;
            }
            let steps = (domain_maxima[index] - domain.min) / domain.step;
            let previous = current
                .iter()
                .find(|(key, _)| *key == domain.name)
                .and_then(|(_, value)| value.trim().parse::<i64>().ok())
                .filter(|value| (value - domain.min) % domain.step == 0)
                .map(|value| (value - domain.min) / domain.step)
                .filter(|step| (0..=steps).contains(step));
            let indicator = if previous.is_some() {
                variables.add(variable().binary())
            } else {
                variables.add(variable().integer().min(1.0).max(1.0))
            };
            objective += weight as f64 * indicator;
            switches.push((index, indicator, previous, steps, weight));
        }

        let mut model = variables
            .minimise(objective)
            .using(good_lp::solvers::microlp::microlp);
        for &(index, indicator, previous, steps, _) in &switches {
            if let Some(previous) = previous {
                let big_m = steps.max(1) as f64;
                model = model
                    .with((decision_variables[index] - previous as f64).leq(big_m * indicator))
                    .with((previous as f64 - decision_variables[index]).leq(big_m * indicator));
            }
        }
        for constraint in &self.constraints {
            let mut expression = Expression::from(0.0);
            for (name, coefficient) in &constraint.terms {
                let index = self
                    .domains
                    .iter()
                    .position(|domain| domain.name == *name)
                    .ok_or_else(|| {
                        AdaptiveOptimizationError::InvalidDomain(format!(
                            "constraint references unknown parameter {name}"
                        ))
                    })?;
                let domain = self.domains[index];
                expression += *coefficient as f64 * domain.min as f64;
                expression += *coefficient as f64 * domain.step as f64 * decision_variables[index];
            }
            model = model.with(expression.leq(constraint.upper_bound as f64));
        }
        if let Some(available_bytes) = constraints.available_memory_bytes {
            for constraint in &self.memory_constraints {
                let mut expression = Expression::from(constraint.base_bytes as f64);
                for (name, coefficient) in &constraint.terms {
                    let index = self
                        .domains
                        .iter()
                        .position(|domain| domain.name == *name)
                        .ok_or_else(|| {
                            AdaptiveOptimizationError::InvalidDomain(format!(
                                "memory constraint references unknown parameter {name}"
                            ))
                        })?;
                    let domain = self.domains[index];
                    expression += *coefficient as f64 * domain.min as f64;
                    expression +=
                        *coefficient as f64 * domain.step as f64 * decision_variables[index];
                }
                model = model.with(expression.leq(available_bytes as f64));
            }
        }

        let solution = model.solve().map_err(|error| {
            AdaptiveOptimizationError::Infeasible(format!("MILP solve failed: {error}"))
        })?;
        let mut candidate = current.to_vec();
        for (index, domain) in self.domains.iter().enumerate() {
            let step_count = solution.value(decision_variables[index]).round() as i64;
            let value = domain.min + step_count * domain.step;
            if let Some((_, current_value)) =
                candidate.iter_mut().find(|(key, _)| *key == domain.name)
            {
                *current_value = value.to_string();
            } else {
                candidate.push((domain.name, value.to_string()));
            }
        }
        candidate.sort_by_key(|(key, _)| *key);
        let operating = objective_coefficients
            .iter()
            .filter_map(|(name, coefficient)| {
                let value = candidate
                    .iter()
                    .find(|(key, _)| key == name)?
                    .1
                    .parse::<i64>()
                    .ok()?;
                Some(coefficient.saturating_mul(value))
            })
            .fold(self.constant, i64::saturating_add);
        // Report the term the solver minimized, recomputed on the rounded
        // solution so it matches the vector returned.
        let switching = switches
            .iter()
            .filter(|&&(index, _, previous, _, _)| {
                let step = (candidate
                    .iter()
                    .find(|(key, _)| *key == self.domains[index].name)
                    .and_then(|(_, value)| value.parse::<i64>().ok())
                    .unwrap_or(self.domains[index].min)
                    - self.domains[index].min)
                    / self.domains[index].step;
                previous != Some(step)
            })
            .fold(0_i64, |total, &(_, _, _, _, weight)| {
                total.saturating_add(weight)
            });
        self.last_objective = Some(MilpObjectiveValue {
            operating,
            switching,
        });
        Ok(candidate)
    }
}

impl<F> FiniteCandidateOptimizer<F> {
    /// Creates an optimizer whose objective is minimized; `None` means that a
    /// candidate is infeasible for the supplied observations and constraints.
    pub fn new(
        mut domains: Vec<AdaptiveIntegerDomain>,
        objective: F,
    ) -> Result<Self, AdaptiveOptimizationError> {
        for domain in &domains {
            domain.validate()?;
        }
        domains.sort_by_key(|domain| domain.name);
        if domains
            .windows(2)
            .any(|window| window[0].name == window[1].name)
        {
            return Err(AdaptiveOptimizationError::InvalidDomain(
                "parameter names must be unique".into(),
            ));
        }
        Ok(Self {
            domains,
            objective,
            memory_constraints: Vec::new(),
        })
    }

    /// Adds a memory-demand constraint enforced when memory telemetry exists.
    #[must_use]
    pub fn with_memory_constraint(mut self, constraint: AdaptiveMemoryConstraint) -> Self {
        self.memory_constraints.push(constraint);
        self
    }
}

impl<F> AdaptiveOptimizer for FiniteCandidateOptimizer<F>
where
    F: FnMut(
            &[(&'static str, String)],
            &[AdaptiveRequestObservation],
            AdaptiveConstraints,
        ) -> Result<Option<i64>, AdaptiveOptimizationError>
        + Send,
{
    fn propose(
        &mut self,
        current: &[(&'static str, String)],
        observations: &[AdaptiveRequestObservation],
        constraints: AdaptiveConstraints,
    ) -> Result<Vec<(&'static str, String)>, AdaptiveOptimizationError> {
        if self
            .domains
            .iter()
            .any(|domain| runtime_domain_max(*domain, constraints) < domain.min)
        {
            return Err(AdaptiveOptimizationError::NoCandidate);
        }
        let mut best: Option<(i64, Vec<(&'static str, String)>)> = None;
        let mut assignments = Vec::with_capacity(self.domains.len());
        self.search(
            0,
            &mut assignments,
            current,
            observations,
            constraints,
            &mut best,
        )?;
        best.map(|(_, candidate)| candidate)
            .ok_or(AdaptiveOptimizationError::NoCandidate)
    }
}

impl<F> FiniteCandidateOptimizer<F>
where
    F: FnMut(
            &[(&'static str, String)],
            &[AdaptiveRequestObservation],
            AdaptiveConstraints,
        ) -> Result<Option<i64>, AdaptiveOptimizationError>
        + Send,
{
    /// Solves the finite candidate problem directly.
    pub fn propose(
        &mut self,
        current: &[(&'static str, String)],
        observations: &[AdaptiveRequestObservation],
        constraints: AdaptiveConstraints,
    ) -> Result<Vec<(&'static str, String)>, AdaptiveOptimizationError> {
        <Self as AdaptiveOptimizer>::propose(self, current, observations, constraints)
    }

    fn search(
        &mut self,
        index: usize,
        assignments: &mut Vec<(&'static str, i64)>,
        current: &[(&'static str, String)],
        observations: &[AdaptiveRequestObservation],
        constraints: AdaptiveConstraints,
        best: &mut Option<(i64, Vec<(&'static str, String)>)>,
    ) -> Result<(), AdaptiveOptimizationError> {
        if index == self.domains.len() {
            if let Some(available_bytes) = constraints.available_memory_bytes {
                for constraint in &self.memory_constraints {
                    let mut required_bytes = i128::from(constraint.base_bytes);
                    for (name, coefficient) in &constraint.terms {
                        let value = assignments
                            .iter()
                            .find(|(assignment_name, _)| *assignment_name == *name)
                            .map(|(_, value)| *value)
                            .ok_or_else(|| {
                                AdaptiveOptimizationError::InvalidDomain(format!(
                                    "memory constraint references unknown parameter {name}"
                                ))
                            })?;
                        required_bytes = required_bytes.saturating_add(
                            i128::from(*coefficient).saturating_mul(i128::from(value)),
                        );
                    }
                    if required_bytes > i128::from(available_bytes) {
                        return Ok(());
                    }
                }
            }
            let mut candidate = current.to_vec();
            for (name, value) in assignments.iter().copied() {
                if let Some((_, current_value)) = candidate.iter_mut().find(|(key, _)| *key == name)
                {
                    *current_value = value.to_string();
                } else {
                    candidate.push((name, value.to_string()));
                }
            }
            candidate.sort_by_key(|(key, _)| *key);
            if let Some(score) = (self.objective)(&candidate, observations, constraints)? {
                let replace = best
                    .as_ref()
                    .is_none_or(|(best_score, _)| score < *best_score);
                if replace {
                    *best = Some((score, candidate));
                }
            }
            return Ok(());
        }

        let domain = self.domains[index];
        let max = runtime_domain_max(domain, constraints);
        if max < domain.min {
            return Ok(());
        }
        let mut value = domain.min;
        loop {
            assignments.push((domain.name, value));
            self.search(
                index + 1,
                assignments,
                current,
                observations,
                constraints,
                best,
            )?;
            assignments.pop();
            if value > max.saturating_sub(domain.step) {
                break;
            }
            value += domain.step;
        }
        Ok(())
    }
}

/// Timing observations derived from one committed request iteration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdaptiveRequestObservation {
    /// Request represented by this observation.
    pub request_id: RequestId,
    /// Timestamp at which the request entered the runtime queue.
    pub arrived_at: Timestamp,
    /// Timestamp of the first emitted token, if one was emitted.
    pub first_token_at: Option<Timestamp>,
    /// Timestamp at which the request completed, if it completed.
    pub completed_at: Option<Timestamp>,
    /// Time from arrival to first token, in microseconds.
    pub time_to_first_token_micros: Option<u64>,
    /// Time from arrival to completion, in microseconds.
    pub completion_latency_micros: Option<u64>,
    /// Whether this request missed its caller-supplied deadline.
    pub deadline_missed: bool,
    /// Whether execution used a capability fallback attributable to this
    /// request's iteration.
    pub capability_fallback: bool,
    /// Prompt tokens, reported by the iteration that admitted the request.
    pub prompt_tokens: Option<usize>,
    /// Output limit, reported by the iteration that admitted the request.
    pub max_new_tokens: Option<usize>,
    /// Output tokens produced so far, reported by iterations that produced
    /// one. Together with the two fields above, observations from successive
    /// iterations describe the workload well enough to replay it.
    pub generated_tokens: Option<usize>,
}

/// Aggregate metrics derived from one adaptive observation window.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AdaptiveObjectiveMetrics {
    /// Number of requests represented by the window.
    pub observed_requests: usize,
    /// Number of requests with a completion timestamp.
    pub completed_requests: usize,
    /// Nearest-rank p95 time to first token, when any request emitted one.
    pub p95_ttft_micros: Option<u64>,
    /// Nearest-rank p95 completion latency, when any request completed.
    pub p95_completion_latency_micros: Option<u64>,
    /// Completed requests per second, scaled by 1000 for integer objectives.
    pub throughput_milli_per_second: Option<u64>,
    /// Requests that did not emit a first token in the observation window.
    pub starved_requests: usize,
    /// Requests that completed after their deadlines or remained incomplete.
    pub deadline_misses: usize,
    /// Capability/resource fallbacks observed in the window.
    pub capability_fallbacks: usize,
}

impl AdaptiveObjectiveMetrics {
    /// Computes deterministic aggregate metrics from request observations.
    pub fn from_observations(observations: &[AdaptiveRequestObservation]) -> Self {
        let mut ttft = observations
            .iter()
            .filter_map(|observation| observation.time_to_first_token_micros)
            .collect::<Vec<_>>();
        let mut completion = observations
            .iter()
            .filter_map(|observation| observation.completion_latency_micros)
            .collect::<Vec<_>>();
        ttft.sort_unstable();
        completion.sort_unstable();

        let completed_requests = completion.len();
        let earliest_arrival = observations
            .iter()
            .map(|observation| observation.arrived_at)
            .min();
        let latest_completion = observations
            .iter()
            .filter_map(|observation| observation.completed_at)
            .max();
        let throughput_milli_per_second =
            earliest_arrival
                .zip(latest_completion)
                .and_then(|(start, end)| {
                    let elapsed_micros = end.as_micros().saturating_sub(start.as_micros());
                    (elapsed_micros > 0).then(|| {
                        (completed_requests as u64).saturating_mul(1_000_000_000) / elapsed_micros
                    })
                });

        Self {
            observed_requests: observations.len(),
            completed_requests,
            p95_ttft_micros: nearest_rank_p95(&ttft),
            p95_completion_latency_micros: nearest_rank_p95(&completion),
            throughput_milli_per_second,
            starved_requests: observations
                .iter()
                .filter(|observation| observation.first_token_at.is_none())
                .count(),
            deadline_misses: observations
                .iter()
                .filter(|observation| observation.deadline_missed)
                .count(),
            capability_fallbacks: observations
                .iter()
                .filter(|observation| observation.capability_fallback)
                .count(),
        }
    }

    /// Scores metrics using a minimization objective. Missing metrics return
    /// `None` when their corresponding weight is non-zero.
    pub fn score(&self, weights: AdaptiveObjectiveWeights) -> Option<i64> {
        let ttft = weighted_metric(weights.ttft, self.p95_ttft_micros)?;
        let completion = weighted_metric(weights.completion, self.p95_completion_latency_micros)?;
        let throughput = weighted_metric(weights.throughput, self.throughput_milli_per_second)?;
        let score = i128::from(ttft) + i128::from(completion) - i128::from(throughput)
            + i128::from(weights.starvation) * i128::try_from(self.starved_requests).ok()?
            + i128::from(weights.deadline) * i128::try_from(self.deadline_misses).ok()?
            + i128::from(weights.fallback) * i128::try_from(self.capability_fallbacks).ok()?;
        i64::try_from(score).ok()
    }
}

/// Integer weights for the adaptive minimization objective.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AdaptiveObjectiveWeights {
    /// Weight for p95 TTFT in microseconds.
    pub ttft: i64,
    /// Weight for p95 completion latency in microseconds.
    pub completion: i64,
    /// Weight for throughput in milli-requests per second.
    pub throughput: i64,
    /// Weight for each request without a first token.
    pub starvation: i64,
    /// Weight for each missed deadline.
    pub deadline: i64,
    /// Weight for each capability/resource fallback.
    pub fallback: i64,
}

fn nearest_rank_p95(values: &[u64]) -> Option<u64> {
    if values.is_empty() {
        return None;
    }
    let rank = (values.len().saturating_mul(95).saturating_add(99) / 100).max(1);
    values.get(rank - 1).copied()
}

fn weighted_metric(weight: i64, value: Option<u64>) -> Option<i64> {
    if weight == 0 {
        return Some(0);
    }
    i64::try_from(i128::from(weight) * i128::from(value?)).ok()
}

impl ExecutorIteration {
    /// Derives timing observations using arrival timestamps retained in the
    /// iteration's decision record and token events committed by execution.
    pub fn adaptive_observations(&self) -> Vec<AdaptiveRequestObservation> {
        let mut timings = self
            .decision
            .arrival_times
            .iter()
            .map(|(request_id, arrived_at)| (*request_id, (*arrived_at, None, None)))
            .collect::<BTreeMap<_, _>>();
        let admitted = self
            .decision
            .admissions
            .iter()
            .filter_map(|record| match &record.outcome {
                crate::AdmissionOutcome::Decided {
                    context,
                    decision: crate::AdmissionDecision::Admit { .. },
                    budget_selected: Some(true),
                } => Some((
                    record.request_id,
                    (context.prompt_tokens, context.max_new_tokens),
                )),
                _ => None,
            })
            .collect::<BTreeMap<_, _>>();
        let mut generated = BTreeMap::new();

        for event in &self.events {
            match event {
                ExecutorEvent::Token {
                    request_id,
                    at,
                    finished,
                    generated_tokens,
                    ..
                } => {
                    generated.insert(*request_id, *generated_tokens);
                    if let Some((_, first_token_at, completed_at)) = timings.get_mut(request_id) {
                        *first_token_at = Some(first_token_at.unwrap_or(*at));
                        if *finished {
                            *completed_at = Some(*at);
                        }
                    }
                }
                ExecutorEvent::Failed { request_id, .. } => {
                    if let Some((_, _, completed_at)) = timings.get_mut(request_id) {
                        *completed_at = Some(self.at);
                    }
                }
                _ => {}
            }
        }

        let fallback_count = self.decision.capability_fallbacks.len();
        timings
            .into_iter()
            .enumerate()
            .map(
                |(observation_index, (request_id, (arrived_at, first_token_at, completed_at)))| {
                    AdaptiveRequestObservation {
                        request_id,
                        arrived_at,
                        time_to_first_token_micros: first_token_at
                            .map(|at| at.as_micros().saturating_sub(arrived_at.as_micros())),
                        completion_latency_micros: completed_at
                            .map(|at| at.as_micros().saturating_sub(arrived_at.as_micros())),
                        deadline_missed: false,
                        capability_fallback: observation_index < fallback_count,
                        first_token_at,
                        completed_at,
                        prompt_tokens: admitted.get(&request_id).map(|(prompt, _)| *prompt),
                        max_new_tokens: admitted.get(&request_id).map(|(_, limit)| *limit),
                        generated_tokens: generated.get(&request_id).copied(),
                    }
                },
            )
            .collect()
    }
}

/// Wraps a policy with a stable adaptive configuration and update counter.
///
/// The wrapper delegates planning to its active policy and can rebuild that
/// policy from θ after each committed observation. Its descriptor is copied
/// into every decision record, making the active parameter vector and
/// observation count part of the execution audit trail.
pub struct AdaptivePolicy {
    inner: Box<dyn ExecutionPolicy>,
    fallback: Option<Box<dyn ExecutionPolicy>>,
    factory: Option<Box<dyn AdaptivePolicyFactory>>,
    optimizer: Option<Box<dyn AdaptiveOptimizer>>,
    constraints: AdaptiveConstraints,
    validator: Option<DecisionValidator>,
    updater: Option<ThetaUpdater>,
    theta: Vec<(&'static str, String)>,
    update_count: u64,
    optimization_interval: u64,
    /// Observations since the last optimizer call, which receives all of
    /// them: with an interval above one, a single iteration's observations
    /// would miss most admissions and completions.
    window: Vec<AdaptiveRequestObservation>,
    seed: u64,
    random_state: u64,
}

impl AdaptivePolicy {
    /// Creates a wrapper around `inner` with the supplied adaptive parameters.
    ///
    /// Parameter keys are sorted once so equivalent configurations produce the
    /// same descriptor regardless of construction order.
    pub fn new(
        inner: impl ExecutionPolicy + 'static,
        mut theta: Vec<(&'static str, String)>,
    ) -> Self {
        theta.sort_by_key(|(key, _)| *key);
        Self {
            inner: Box::new(inner),
            fallback: None,
            factory: None,
            optimizer: None,
            constraints: AdaptiveConstraints::default(),
            validator: None,
            updater: None,
            theta,
            update_count: 0,
            window: Vec::new(),
            optimization_interval: 1,
            seed: 0,
            random_state: 0,
        }
    }

    /// Installs a known-safe policy used when the candidate returns an error.
    #[must_use]
    pub fn with_fallback(mut self, fallback: impl ExecutionPolicy + 'static) -> Self {
        self.fallback = Some(Box::new(fallback));
        self
    }

    /// Installs the factory that rebuilds active policy components from θ.
    #[must_use]
    pub fn with_factory(mut self, factory: impl AdaptivePolicyFactory + 'static) -> Self {
        self.factory = Some(Box::new(factory));
        self
    }

    /// Installs a mathematical optimizer for observation-time θ updates.
    #[must_use]
    pub fn with_optimizer(mut self, optimizer: impl AdaptiveOptimizer + 'static) -> Self {
        self.optimizer = Some(Box::new(optimizer));
        self
    }

    /// Supplies executor ceilings and resource facts to the optimizer.
    #[must_use]
    pub const fn with_constraints(mut self, constraints: AdaptiveConstraints) -> Self {
        self.constraints = constraints;
        self
    }

    /// Sets how many committed iterations occur between optimizer proposals.
    ///
    /// An interval of zero is normalized to one. Policy observation hooks
    /// still run on every committed iteration; only the mathematical solver
    /// is rate-limited.
    #[must_use]
    pub const fn with_optimization_interval(mut self, interval: u64) -> Self {
        self.optimization_interval = if interval == 0 { 1 } else { interval };
        self
    }

    /// Installs a pure candidate-plan validator used before commit.
    #[must_use]
    pub fn with_validator<F>(mut self, validator: F) -> Self
    where
        F: Fn(&PlanningSnapshot, &PolicyDecision) -> bool + Send + Sync + 'static,
    {
        self.validator = Some(Box::new(validator));
        self
    }

    /// Installs an update hook invoked only after a committed iteration.
    ///
    /// The hook receives derived timing observations and may update θ for the
    /// next planning boundary. It cannot affect the plan currently committed.
    #[must_use]
    pub fn with_update<F>(mut self, updater: F) -> Self
    where
        F: FnMut(
                &ExecutorIteration,
                &[AdaptiveRequestObservation],
                &mut Vec<(&'static str, String)>,
            ) + Send
            + 'static,
    {
        self.updater = Some(Box::new(updater));
        self
    }

    /// Sets the deterministic seed reserved for observation-time updates.
    #[must_use]
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self.random_state = seed;
        self
    }

    /// Returns the wrapped policy.
    pub fn inner(&self) -> &dyn ExecutionPolicy {
        self.inner.as_ref()
    }

    /// Returns the current adaptive parameter vector in descriptor order.
    pub fn theta(&self) -> &[(&'static str, String)] {
        &self.theta
    }

    /// Returns the number of committed observations received by the wrapper.
    pub const fn update_count(&self) -> u64 {
        self.update_count
    }

    /// Returns the current observation-time random state for diagnostics.
    pub const fn random_state(&self) -> u64 {
        self.random_state
    }

    fn adaptive_descriptor(&self) -> PolicyDescriptor {
        let mut descriptor = PolicyDescriptor::new("adaptive")
            .with_config("update_count", self.update_count)
            .with_config("seed", self.seed)
            .with_config("random_state", self.random_state)
            .with_component(self.inner.descriptor());
        if let Some(fallback) = &self.fallback {
            descriptor = descriptor.with_component(fallback.descriptor());
        }
        for (key, value) in &self.theta {
            descriptor = descriptor.with_config(key, value);
        }
        descriptor
    }
}

fn checked_plan(
    policy: &dyn ExecutionPolicy,
    snapshot: &PlanningSnapshot,
) -> Result<PolicyDecision, PolicyError> {
    let decision = policy.plan(snapshot)?;
    decision
        .plan
        .validate(snapshot, policy.enforces_model_kv_capacity())
        .map_err(|error| PolicyError::InvalidCandidate(error.to_string()))?;
    Ok(decision)
}

impl ExecutionPolicy for AdaptivePolicy {
    fn descriptor(&self) -> PolicyDescriptor {
        self.adaptive_descriptor()
    }

    fn plan(&self, snapshot: &PlanningSnapshot) -> Result<PolicyDecision, PolicyError> {
        let mut decision = match checked_plan(&self.inner, snapshot) {
            Ok(decision) => decision,
            Err(error) => match &self.fallback {
                Some(fallback) => checked_plan(fallback.as_ref(), snapshot)?,
                None => return Err(error),
            },
        };
        if self
            .validator
            .as_ref()
            .is_some_and(|validator| !validator(snapshot, &decision))
        {
            decision = match &self.fallback {
                Some(fallback) => checked_plan(fallback.as_ref(), snapshot)?,
                None => {
                    return Err(PolicyError::InvalidCandidate(
                        "validator returned false".into(),
                    ));
                }
            };
        }
        decision.record.policy = self.adaptive_descriptor();
        Ok(decision)
    }

    fn enforces_model_kv_capacity(&self) -> bool {
        self.inner.enforces_model_kv_capacity()
            && self
                .fallback
                .as_ref()
                .is_none_or(|fallback| fallback.enforces_model_kv_capacity())
    }

    fn observe(&mut self, iteration: &ExecutorIteration) {
        self.inner.observe(iteration);
        if let Some(fallback) = &mut self.fallback {
            fallback.observe(iteration);
        }
        let observations = iteration.adaptive_observations();
        if let Some(updater) = &mut self.updater {
            updater(iteration, &observations, &mut self.theta);
            self.theta.sort_by_key(|(key, _)| *key);
        }
        let should_optimize =
            (self.update_count.saturating_add(1)).is_multiple_of(self.optimization_interval);
        if self.optimizer.is_some() {
            self.window.extend(observations);
        }
        let proposed_theta = if should_optimize {
            let window = std::mem::take(&mut self.window);
            self.optimizer
                .as_mut()
                .and_then(|optimizer| {
                    optimizer
                        .propose(&self.theta, &window, self.constraints)
                        .ok()
                })
                .map(|mut candidate| {
                    candidate.sort_by_key(|(key, _)| *key);
                    candidate
                })
        } else {
            None
        };
        let candidate_theta = proposed_theta.as_deref().unwrap_or(&self.theta);
        if let Some(factory) = &self.factory {
            if let Ok(candidate) = factory.build(candidate_theta) {
                self.inner = candidate;
                if let Some(proposed_theta) = proposed_theta {
                    self.theta = proposed_theta;
                }
            }
        } else if let Some(proposed_theta) = proposed_theta {
            self.theta = proposed_theta;
        }
        self.random_state = splitmix64(self.random_state);
        self.update_count = self.update_count.saturating_add(1);
    }

    fn recovers_kv_exhaustion(&self) -> bool {
        self.inner.recovers_kv_exhaustion()
            || self
                .fallback
                .as_ref()
                .is_some_and(|fallback| fallback.recovers_kv_exhaustion())
    }
}

pub(crate) fn splitmix64(state: u64) -> u64 {
    let mut value = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    use crate::{ComposedPolicy, DecisionRecord, ExecutionPlan, FifoTokenBudgetPolicy};
    use ola_core::{ModelId, Timestamp};
    use ola_model::GeneratedToken;

    struct RejectingPolicy;

    impl ExecutionPolicy for RejectingPolicy {
        fn descriptor(&self) -> PolicyDescriptor {
            PolicyDescriptor::new("rejecting")
        }

        fn plan(&self, _snapshot: &PlanningSnapshot) -> Result<PolicyDecision, PolicyError> {
            Err(PolicyError::KernelSelection {
                model_id: ModelId::new("candidate"),
                message: "candidate rejected".into(),
            })
        }

        fn enforces_model_kv_capacity(&self) -> bool {
            true
        }
    }

    struct AcceptingPolicy;

    impl ExecutionPolicy for AcceptingPolicy {
        fn descriptor(&self) -> PolicyDescriptor {
            PolicyDescriptor::new("fallback")
        }

        fn plan(&self, _snapshot: &PlanningSnapshot) -> Result<PolicyDecision, PolicyError> {
            Ok(PolicyDecision {
                plan: ExecutionPlan::default(),
                record: DecisionRecord::default(),
            })
        }

        fn enforces_model_kv_capacity(&self) -> bool {
            true
        }
    }

    struct StalePolicy;

    impl ExecutionPolicy for StalePolicy {
        fn descriptor(&self) -> PolicyDescriptor {
            PolicyDescriptor::new("stale")
        }

        fn plan(&self, _snapshot: &PlanningSnapshot) -> Result<PolicyDecision, PolicyError> {
            Ok(PolicyDecision {
                plan: ExecutionPlan {
                    generation: 99,
                    ..ExecutionPlan::default()
                },
                record: DecisionRecord::default(),
            })
        }

        fn enforces_model_kv_capacity(&self) -> bool {
            true
        }
    }

    #[test]
    fn descriptor_is_stable_and_contains_theta_and_counter() {
        let policy = AdaptivePolicy::new(
            ComposedPolicy::default(),
            vec![("learning_rate", "0.1".into()), ("exploration", "7".into())],
        );

        assert_eq!(
            policy.descriptor().to_string(),
            "adaptive{update_count=0,seed=0,random_state=0,exploration=7,learning_rate=0.1}[composed{head_of_line=block}[fifo,unconstrained-admission,unbounded-batch,unbucketed-prefill]]"
        );
    }

    #[test]
    fn serialized_config_round_trips_and_builds_with_an_explicit_key_resolver() {
        let json = r#"{
            "theta": [
                {"name": "max_tokens", "value": "32"},
                {"name": "max_prefill_tokens", "value": "8"}
            ],
            "optimization_interval": 4,
            "seed": 19,
            "constraints": {
                "max_active_requests": 6,
                "prefill_chunk_size": 16,
                "available_memory_bytes": 4096
            }
        }"#;
        let config: AdaptivePolicyConfig = serde_json::from_str(json).unwrap();
        let encoded = serde_json::to_string(&config).unwrap();
        let decoded: AdaptivePolicyConfig = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, config);

        let policy = config
            .build_policy(AcceptingPolicy, |name| match name {
                "max_tokens" => Some("max_tokens"),
                "max_prefill_tokens" => Some("max_prefill_tokens"),
                _ => None,
            })
            .unwrap();
        assert_eq!(
            policy.theta(),
            [
                ("max_prefill_tokens", "8".into()),
                ("max_tokens", "32".into())
            ]
        );
        assert_eq!(policy.random_state(), 19);
    }

    #[test]
    fn serialized_config_rejects_duplicate_and_unknown_parameters() {
        let duplicate = AdaptivePolicyConfig {
            theta: vec![
                AdaptiveParameterConfig {
                    name: "budget".into(),
                    value: "1".into(),
                },
                AdaptiveParameterConfig {
                    name: "budget".into(),
                    value: "2".into(),
                },
            ],
            ..AdaptivePolicyConfig::default()
        };
        assert!(matches!(
            duplicate.build_policy(AcceptingPolicy, |_| Some("budget")),
            Err(AdaptiveConfigError::InvalidParameter(name)) if name == "budget"
        ));

        let unknown = AdaptivePolicyConfig {
            theta: vec![AdaptiveParameterConfig {
                name: "unknown".into(),
                value: "1".into(),
            }],
            ..AdaptivePolicyConfig::default()
        };
        assert!(matches!(
            unknown.build_policy(AcceptingPolicy, |_| None),
            Err(AdaptiveConfigError::UnknownParameter(name)) if name == "unknown"
        ));
    }

    #[test]
    fn observation_randomness_is_seeded_and_consumed_only_on_observe() {
        let mut first = AdaptivePolicy::new(AcceptingPolicy, vec![]).with_seed(7);
        let mut second = AdaptivePolicy::new(AcceptingPolicy, vec![]).with_seed(7);
        assert_eq!(first.random_state(), 7);

        let iteration = ExecutorIteration {
            id: crate::IterationId::new(1),
            at: Timestamp::ZERO,
            events: Vec::new(),
            execution_plan: ExecutionPlan::default(),
            decision: DecisionRecord::default(),
            divergences: Vec::new(),
            execution_batches: Vec::new(),
            active_requests: 0,
            queued_requests: 0,
            kv_cache_bytes: 0,
            reserved_kv_bytes: 0,
        };
        first.observe(&iteration);
        second.observe(&iteration);

        assert_eq!(first.update_count(), 1);
        assert_eq!(first.random_state(), second.random_state());
        assert_ne!(first.random_state(), 7);
    }

    #[test]
    fn update_hook_changes_theta_only_after_observation() {
        let mut policy = AdaptivePolicy::new(AcceptingPolicy, vec![("value", "0".into())])
            .with_update(|_, observations, theta| {
                theta.push(("observed", observations.len().to_string()));
            });
        assert!(!policy.descriptor().to_string().contains("observed="));

        let iteration = ExecutorIteration {
            id: crate::IterationId::new(1),
            at: Timestamp::ZERO,
            events: Vec::new(),
            execution_plan: ExecutionPlan::default(),
            decision: DecisionRecord::default(),
            divergences: Vec::new(),
            execution_batches: Vec::new(),
            active_requests: 0,
            queued_requests: 0,
            kv_cache_bytes: 0,
            reserved_kv_bytes: 0,
        };
        policy.observe(&iteration);

        assert!(policy.descriptor().to_string().contains("observed=0"));
    }

    #[test]
    fn factory_rebuilds_the_active_policy_after_observation() {
        let mut policy = AdaptivePolicy::new(RejectingPolicy, vec![("mode", "fallback".into())])
            .with_factory(|theta: &[(&'static str, String)]| {
                assert_eq!(theta, [("mode", "fallback".into())]);
                Ok(Box::new(AcceptingPolicy) as Box<dyn ExecutionPolicy>)
            });

        assert_eq!(policy.inner().descriptor().name, "rejecting");
        let iteration = ExecutorIteration {
            id: crate::IterationId::new(1),
            at: Timestamp::ZERO,
            events: Vec::new(),
            execution_plan: ExecutionPlan::default(),
            decision: DecisionRecord::default(),
            divergences: Vec::new(),
            execution_batches: Vec::new(),
            active_requests: 0,
            queued_requests: 0,
            kv_cache_bytes: 0,
            reserved_kv_bytes: 0,
        };

        policy.observe(&iteration);

        assert_eq!(policy.inner().descriptor().name, "fallback");
    }

    #[test]
    fn rejected_factory_candidate_does_not_mutate_theta() {
        let mut policy = AdaptivePolicy::new(AcceptingPolicy, vec![("budget", "1".into())])
            .with_optimizer(
                |_: &[(&'static str, String)],
                 _: &[AdaptiveRequestObservation],
                 _: AdaptiveConstraints|
                 -> Result<Vec<(&'static str, String)>, AdaptiveOptimizationError> {
                    Ok(vec![("budget", "9".into())])
                },
            )
            .with_factory(|theta: &[(&'static str, String)]| {
                if theta.iter().any(|(_, value)| value == "9") {
                    Err(PolicyError::InvalidCandidate("budget rejected".into()))
                } else {
                    Ok(Box::new(AcceptingPolicy) as Box<dyn ExecutionPolicy>)
                }
            });
        let iteration = ExecutorIteration {
            id: crate::IterationId::new(1),
            at: Timestamp::ZERO,
            events: Vec::new(),
            execution_plan: ExecutionPlan::default(),
            decision: DecisionRecord::default(),
            divergences: Vec::new(),
            execution_batches: Vec::new(),
            active_requests: 0,
            queued_requests: 0,
            kv_cache_bytes: 0,
            reserved_kv_bytes: 0,
        };

        policy.observe(&iteration);

        assert_eq!(policy.theta(), &[("budget", "1".into())]);
        assert_eq!(policy.inner().descriptor().name, "fallback");
    }

    #[test]
    fn optimizer_interval_rate_limits_solver_proposals() {
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let calls_for_optimizer = calls.clone();
        let mut policy = AdaptivePolicy::new(AcceptingPolicy, vec![("budget", "1".into())])
            .with_optimization_interval(2)
            .with_optimizer(
                move |_: &[(&'static str, String)],
                      _: &[AdaptiveRequestObservation],
                      _: AdaptiveConstraints|
                      -> Result<Vec<(&'static str, String)>, AdaptiveOptimizationError> {
                    calls_for_optimizer.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    Ok(vec![("budget", "2".into())])
                },
            );
        let iteration = ExecutorIteration {
            id: crate::IterationId::new(1),
            at: Timestamp::ZERO,
            events: Vec::new(),
            execution_plan: ExecutionPlan::default(),
            decision: DecisionRecord::default(),
            divergences: Vec::new(),
            execution_batches: Vec::new(),
            active_requests: 0,
            queued_requests: 0,
            kv_cache_bytes: 0,
            reserved_kv_bytes: 0,
        };

        policy.observe(&iteration);
        assert_eq!(policy.theta(), &[("budget", "1".into())]);
        policy.observe(&iteration);
        assert_eq!(policy.theta(), &[("budget", "2".into())]);
        assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 1);
    }

    #[test]
    fn optimizer_receives_every_observation_since_its_last_call() {
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let seen_by_optimizer = seen.clone();
        let mut policy = AdaptivePolicy::new(AcceptingPolicy, vec![("budget", "1".into())])
            .with_optimization_interval(3)
            .with_optimizer(
                move |current: &[(&'static str, String)],
                      observations: &[AdaptiveRequestObservation],
                      _: AdaptiveConstraints|
                      -> Result<Vec<(&'static str, String)>, AdaptiveOptimizationError> {
                    seen_by_optimizer.lock().unwrap().push(
                        observations
                            .iter()
                            .map(|observation| observation.request_id.get())
                            .collect::<Vec<_>>(),
                    );
                    Ok(current.to_vec())
                },
            );
        let iteration = |request: u64| {
            let mut decision = DecisionRecord::default();
            decision
                .arrival_times
                .insert(RequestId::new(request), Timestamp::ZERO);
            ExecutorIteration {
                id: crate::IterationId::new(request),
                at: Timestamp::ZERO,
                events: Vec::new(),
                execution_plan: ExecutionPlan::default(),
                decision,
                divergences: Vec::new(),
                execution_batches: Vec::new(),
                active_requests: 0,
                queued_requests: 0,
                kv_cache_bytes: 0,
                reserved_kv_bytes: 0,
            }
        };
        for request in 1..=6 {
            policy.observe(&iteration(request));
        }
        // Each call sees its whole interval, not only its own iteration.
        assert_eq!(*seen.lock().unwrap(), [vec![1, 2, 3], vec![4, 5, 6]]);
    }

    #[test]
    fn optimizer_proposes_sorted_parameters_at_the_observation_boundary() {
        let mut policy = AdaptivePolicy::new(AcceptingPolicy, vec![("budget", "1".into())])
            .with_constraints(AdaptiveConstraints {
                max_active_requests: Some(8),
                prefill_chunk_size: Some(16),
                available_memory_bytes: None,
            })
            .with_optimizer(
                |current: &[(&'static str, String)],
                 observations: &[AdaptiveRequestObservation],
                 constraints: AdaptiveConstraints| {
                    assert_eq!(current, [("budget", "1".into())]);
                    assert!(observations.is_empty());
                    assert_eq!(constraints.max_active_requests, Some(8));
                    Ok(vec![("zeta", "3".into()), ("alpha", "2".into())])
                },
            );

        let iteration = ExecutorIteration {
            id: crate::IterationId::new(1),
            at: Timestamp::ZERO,
            events: Vec::new(),
            execution_plan: ExecutionPlan::default(),
            decision: DecisionRecord::default(),
            divergences: Vec::new(),
            execution_batches: Vec::new(),
            active_requests: 0,
            queued_requests: 0,
            kv_cache_bytes: 0,
            reserved_kv_bytes: 0,
        };

        policy.observe(&iteration);

        assert_eq!(
            policy.theta(),
            &[("alpha", "2".into()), ("zeta", "3".into())]
        );
    }

    #[test]
    fn finite_optimizer_selects_the_lowest_feasible_objective() {
        let mut optimizer = FiniteCandidateOptimizer::new(
            vec![AdaptiveIntegerDomain {
                name: "budget",
                min: 1,
                max: 5,
                step: 1,
            }],
            |candidate: &[(&'static str, String)],
             _: &[AdaptiveRequestObservation],
             _: AdaptiveConstraints|
             -> Result<Option<i64>, AdaptiveOptimizationError> {
                let budget = candidate
                    .iter()
                    .find(|(name, _)| *name == "budget")
                    .and_then(|(_, value)| value.parse::<i64>().ok())
                    .unwrap();
                if budget == 3 {
                    Ok(None)
                } else {
                    Ok(Some((budget - 4).abs()))
                }
            },
        )
        .unwrap();

        let candidate = optimizer
            .propose(
                &[("fixed", "unchanged".into()), ("budget", "1".into())],
                &[],
                AdaptiveConstraints::default(),
            )
            .unwrap();

        assert_eq!(
            candidate,
            &[("budget", "4".into()), ("fixed", "unchanged".into())]
        );
    }

    #[test]
    fn finite_optimizer_breaks_equal_scores_by_enumeration_order() {
        let mut optimizer = FiniteCandidateOptimizer::new(
            vec![AdaptiveIntegerDomain {
                name: "budget",
                min: 2,
                max: 4,
                step: 1,
            }],
            |_: &[(&'static str, String)],
             _: &[AdaptiveRequestObservation],
             _: AdaptiveConstraints| {
                Ok::<Option<i64>, AdaptiveOptimizationError>(Some(0))
            },
        )
        .unwrap();

        let candidate = optimizer
            .propose(&[], &[], AdaptiveConstraints::default())
            .unwrap();

        assert_eq!(candidate, &[("budget", "2".into())]);
    }

    #[test]
    fn finite_optimizer_rejects_malformed_domains() {
        let result = FiniteCandidateOptimizer::new(
            vec![AdaptiveIntegerDomain {
                name: "budget",
                min: 1,
                max: 4,
                step: 0,
            }],
            |_: &[(&'static str, String)],
             _: &[AdaptiveRequestObservation],
             _: AdaptiveConstraints| {
                Ok::<Option<i64>, AdaptiveOptimizationError>(Some(0))
            },
        );
        let error = match result {
            Ok(_) => panic!("zero-step domain should be rejected"),
            Err(error) => error,
        };

        assert!(matches!(
            error,
            AdaptiveOptimizationError::InvalidDomain(message)
                if message.contains("non-positive step")
        ));
    }

    #[test]
    fn milp_optimizer_solves_bounded_integer_problem() {
        let mut optimizer = MilpCandidateOptimizer::new(
            vec![AdaptiveIntegerDomain {
                name: "budget",
                min: 1,
                max: 5,
                step: 2,
            }],
            vec![("budget", -1)],
        )
        .unwrap()
        .with_constraint(AdaptiveLinearConstraint {
            terms: vec![("budget", 1)],
            upper_bound: 4,
        });

        let candidate = optimizer
            .propose(&[], &[], AdaptiveConstraints::default())
            .unwrap();

        assert_eq!(candidate, &[("budget", "3".into())]);
    }

    #[test]
    fn solver_domains_respect_executor_constraints() {
        let mut milp = MilpCandidateOptimizer::new(
            vec![AdaptiveIntegerDomain {
                name: "effective_slots",
                min: 1,
                max: 16,
                step: 1,
            }],
            vec![("effective_slots", -1)],
        )
        .unwrap();
        let candidate = milp
            .propose(
                &[],
                &[],
                AdaptiveConstraints {
                    max_active_requests: Some(4),
                    ..AdaptiveConstraints::default()
                },
            )
            .unwrap();
        assert_eq!(candidate, &[("effective_slots", "4".into())]);

        let mut finite = FiniteCandidateOptimizer::new(
            vec![AdaptiveIntegerDomain {
                name: "chunk_size",
                min: 1,
                max: 16,
                step: 1,
            }],
            |candidate: &[(&'static str, String)],
             _: &[AdaptiveRequestObservation],
             _: AdaptiveConstraints|
             -> Result<Option<i64>, AdaptiveOptimizationError> {
                let value = candidate[0].1.parse::<i64>().unwrap();
                Ok(Some(-value))
            },
        )
        .unwrap();
        let candidate = finite
            .propose(
                &[],
                &[],
                AdaptiveConstraints {
                    prefill_chunk_size: Some(3),
                    ..AdaptiveConstraints::default()
                },
            )
            .unwrap();
        assert_eq!(candidate, &[("chunk_size", "3".into())]);
    }

    #[test]
    fn memory_constraints_bound_finite_and_milp_candidates() {
        let memory = AdaptiveMemoryConstraint {
            base_bytes: 16,
            terms: vec![("slots", 8)],
        };
        let constraints = AdaptiveConstraints {
            available_memory_bytes: Some(40),
            ..AdaptiveConstraints::default()
        };

        let mut finite = FiniteCandidateOptimizer::new(
            vec![AdaptiveIntegerDomain {
                name: "slots",
                min: 1,
                max: 8,
                step: 1,
            }],
            |candidate: &[(&'static str, String)],
             _: &[AdaptiveRequestObservation],
             _: AdaptiveConstraints|
             -> Result<Option<i64>, AdaptiveOptimizationError> {
                Ok(candidate
                    .iter()
                    .find(|(name, _)| *name == "slots")
                    .and_then(|(_, value)| value.parse::<i64>().ok())
                    .map(|value| -value))
            },
        )
        .unwrap()
        .with_memory_constraint(memory.clone());
        assert_eq!(
            finite.propose(&[], &[], constraints).unwrap(),
            &[("slots", "3".into())]
        );

        let mut milp = MilpCandidateOptimizer::new(
            vec![AdaptiveIntegerDomain {
                name: "slots",
                min: 1,
                max: 8,
                step: 1,
            }],
            vec![("slots", -1)],
        )
        .unwrap()
        .with_memory_constraint(memory);
        assert_eq!(
            milp.propose(&[], &[], constraints).unwrap(),
            &[("slots", "3".into())]
        );
    }

    #[test]
    fn milp_and_finite_optimizers_agree_on_a_small_integer_program() {
        let domains = vec![
            AdaptiveIntegerDomain {
                name: "decode_budget",
                min: 0,
                max: 4,
                step: 1,
            },
            AdaptiveIntegerDomain {
                name: "prefill_budget",
                min: 0,
                max: 4,
                step: 1,
            },
        ];
        let current = &[("fixed", "preserved".into())];
        let objective = |candidate: &[(&'static str, String)],
                         _: &[AdaptiveRequestObservation],
                         _: AdaptiveConstraints|
         -> Result<Option<i64>, AdaptiveOptimizationError> {
            let value = |name: &'static str| {
                candidate
                    .iter()
                    .find(|(candidate_name, _)| *candidate_name == name)
                    .and_then(|(_, value)| value.parse::<i64>().ok())
                    .unwrap()
            };
            let decode = value("decode_budget");
            let prefill = value("prefill_budget");
            if decode + prefill > 4 {
                Ok(None)
            } else {
                Ok(Some(-3 * decode - 2 * prefill))
            }
        };

        let mut finite = FiniteCandidateOptimizer::new(domains.clone(), objective).unwrap();
        let finite_candidate = finite
            .propose(current, &[], AdaptiveConstraints::default())
            .unwrap();

        let mut milp = MilpCandidateOptimizer::new(
            domains,
            vec![("decode_budget", -3), ("prefill_budget", -2)],
        )
        .unwrap()
        .with_constraint(AdaptiveLinearConstraint {
            terms: vec![("decode_budget", 1), ("prefill_budget", 1)],
            upper_bound: 4,
        });
        let milp_candidate = milp
            .propose(current, &[], AdaptiveConstraints::default())
            .unwrap();

        assert_eq!(finite_candidate, milp_candidate);
    }

    #[test]
    fn milp_proposal_rebuilds_a_real_batch_policy_component() {
        let mut policy =
            AdaptivePolicy::new(ComposedPolicy::default(), vec![("max_tokens", "1".into())])
                .with_optimizer(
                    MilpCandidateOptimizer::new(
                        vec![AdaptiveIntegerDomain {
                            name: "max_tokens",
                            min: 1,
                            max: 4,
                            step: 1,
                        }],
                        vec![("max_tokens", -1)],
                    )
                    .unwrap(),
                )
                .with_factory(|theta: &[(&'static str, String)]| {
                    let max_tokens = theta
                        .iter()
                        .find(|(name, _)| *name == "max_tokens")
                        .and_then(|(_, value)| value.parse::<usize>().ok())
                        .ok_or_else(|| {
                            PolicyError::InvalidCandidate("missing max_tokens".into())
                        })?;
                    let batch = FifoTokenBudgetPolicy::new(max_tokens).ok_or_else(|| {
                        PolicyError::InvalidCandidate("max_tokens must be positive".into())
                    })?;
                    Ok(Box::new(ComposedPolicy::default().with_batch(batch))
                        as Box<dyn ExecutionPolicy>)
                });

        let iteration = ExecutorIteration {
            id: crate::IterationId::new(1),
            at: Timestamp::ZERO,
            events: Vec::new(),
            execution_plan: ExecutionPlan::default(),
            decision: DecisionRecord::default(),
            divergences: Vec::new(),
            execution_batches: Vec::new(),
            active_requests: 0,
            queued_requests: 0,
            kv_cache_bytes: 0,
            reserved_kv_bytes: 0,
        };
        policy.observe(&iteration);

        assert!(
            policy
                .inner()
                .descriptor()
                .to_string()
                .contains("fifo-token-budget{max_tokens=4}")
        );
    }

    #[test]
    fn milp_objective_builder_consumes_virtual_time_metrics() {
        let mut optimizer = MilpCandidateOptimizer::new(
            vec![AdaptiveIntegerDomain {
                name: "max_tokens",
                min: 1,
                max: 4,
                step: 1,
            }],
            vec![("max_tokens", 1)],
        )
        .unwrap()
        .with_objective_builder(|observations, _| {
            let metrics = AdaptiveObjectiveMetrics::from_observations(observations);
            if metrics.p95_ttft_micros.unwrap_or_default() > 100 {
                vec![("max_tokens", -1)]
            } else {
                vec![("max_tokens", 1)]
            }
        });
        let observation = AdaptiveRequestObservation {
            request_id: ola_core::RequestId::new(7),
            arrived_at: Timestamp::ZERO,
            first_token_at: Some(Timestamp::from_micros(200)),
            completed_at: Some(Timestamp::from_micros(300)),
            time_to_first_token_micros: Some(200),
            completion_latency_micros: Some(300),
            deadline_missed: false,
            capability_fallback: false,
            prompt_tokens: None,
            max_new_tokens: None,
            generated_tokens: None,
        };

        let candidate = optimizer
            .propose(
                &[("max_tokens", "1".into())],
                &[observation],
                AdaptiveConstraints::default(),
            )
            .unwrap();

        assert_eq!(candidate, &[("max_tokens", "4".into())]);
    }

    #[test]
    fn derives_ttft_and_completion_latency_from_committed_events() {
        let mut decision = DecisionRecord::default();
        decision
            .arrival_times
            .insert(ola_core::RequestId::new(4), Timestamp::from_micros(10));
        let iteration = ExecutorIteration {
            id: crate::IterationId::new(1),
            at: Timestamp::from_micros(30),
            events: vec![ExecutorEvent::Token {
                request_id: ola_core::RequestId::new(4),
                token: GeneratedToken {
                    id: 9,
                    is_end_of_sequence: true,
                },
                generated_tokens: 1,
                kv_cache_bytes: 0,
                phase: crate::ExecutionPhase::Prefill,
                finished: true,
                at: Timestamp::from_micros(25),
            }],
            execution_plan: ExecutionPlan::default(),
            decision,
            divergences: Vec::new(),
            execution_batches: Vec::new(),
            active_requests: 0,
            queued_requests: 0,
            kv_cache_bytes: 0,
            reserved_kv_bytes: 0,
        };

        assert_eq!(
            iteration.adaptive_observations(),
            vec![AdaptiveRequestObservation {
                request_id: ola_core::RequestId::new(4),
                arrived_at: Timestamp::from_micros(10),
                first_token_at: Some(Timestamp::from_micros(25)),
                completed_at: Some(Timestamp::from_micros(25)),
                time_to_first_token_micros: Some(15),
                completion_latency_micros: Some(15),
                deadline_missed: false,
                capability_fallback: false,
                prompt_tokens: None,
                max_new_tokens: None,
                generated_tokens: Some(1),
            }]
        );
    }

    #[test]
    fn objective_metrics_are_deterministic_and_preserve_missing_data() {
        let observations = vec![
            AdaptiveRequestObservation {
                request_id: ola_core::RequestId::new(1),
                arrived_at: Timestamp::ZERO,
                first_token_at: Some(Timestamp::from_micros(100)),
                completed_at: Some(Timestamp::from_micros(1_000_000)),
                time_to_first_token_micros: Some(100),
                completion_latency_micros: Some(1_000_000),
                deadline_missed: false,
                capability_fallback: false,
                prompt_tokens: None,
                max_new_tokens: None,
                generated_tokens: None,
            },
            AdaptiveRequestObservation {
                request_id: ola_core::RequestId::new(2),
                arrived_at: Timestamp::ZERO,
                first_token_at: Some(Timestamp::from_micros(200)),
                completed_at: Some(Timestamp::from_micros(2_000_000)),
                time_to_first_token_micros: Some(200),
                completion_latency_micros: Some(2_000_000),
                deadline_missed: true,
                capability_fallback: true,
                prompt_tokens: None,
                max_new_tokens: None,
                generated_tokens: None,
            },
        ];
        let metrics = AdaptiveObjectiveMetrics::from_observations(&observations);

        assert_eq!(metrics.observed_requests, 2);
        assert_eq!(metrics.completed_requests, 2);
        assert_eq!(metrics.p95_ttft_micros, Some(200));
        assert_eq!(metrics.p95_completion_latency_micros, Some(2_000_000));
        assert_eq!(metrics.throughput_milli_per_second, Some(1_000));
        assert_eq!(metrics.starved_requests, 0);
        assert_eq!(metrics.deadline_misses, 1);
        assert_eq!(metrics.capability_fallbacks, 1);
        assert_eq!(
            metrics.score(AdaptiveObjectiveWeights {
                ttft: 1,
                completion: 1,
                throughput: 1,
                starvation: 1,
                deadline: 10,
                fallback: 20,
            }),
            Some(1_999_230)
        );

        let incomplete = AdaptiveObjectiveMetrics::from_observations(&observations[..1]);
        assert_eq!(
            incomplete.score(AdaptiveObjectiveWeights {
                throughput: 1,
                ..AdaptiveObjectiveWeights::default()
            }),
            Some(-1_000)
        );
        assert_eq!(
            AdaptiveObjectiveMetrics::default().score(AdaptiveObjectiveWeights {
                ttft: 1,
                ..AdaptiveObjectiveWeights::default()
            }),
            None
        );
    }

    #[test]
    fn candidate_errors_use_the_safe_fallback_and_keep_adaptive_identity() {
        let policy = AdaptivePolicy::new(RejectingPolicy, vec![]).with_fallback(AcceptingPolicy);
        let snapshot = PlanningSnapshot {
            at: Timestamp::ZERO,
            generation: 0,
            limits: crate::ExecutorLimits {
                max_active_requests: 1,
                prefill_chunk_size: None,
            },
            queued: Vec::new(),
            prefilling: BTreeMap::new(),
            decoding: BTreeMap::new(),
            decode_order: Vec::new(),
            prefill_order: Vec::new(),
            models: BTreeMap::new(),
            memory: crate::MemoryState::default(),
            device: None,
            measurements: crate::PolicyMeasurements::default(),
        };
        let decision = policy.plan(&snapshot).unwrap();

        assert_eq!(decision.record.policy.name, "adaptive");
        assert_eq!(decision.record.policy.components[1].name, "fallback");
    }

    #[test]
    fn rejected_candidates_use_the_safe_fallback() {
        let policy = AdaptivePolicy::new(AcceptingPolicy, vec![])
            .with_fallback(ComposedPolicy::default())
            .with_validator(|_, _| false);
        let snapshot = PlanningSnapshot {
            at: Timestamp::ZERO,
            generation: 0,
            limits: crate::ExecutorLimits {
                max_active_requests: 1,
                prefill_chunk_size: None,
            },
            queued: Vec::new(),
            prefilling: BTreeMap::new(),
            decoding: BTreeMap::new(),
            decode_order: Vec::new(),
            prefill_order: Vec::new(),
            models: BTreeMap::new(),
            memory: crate::MemoryState::default(),
            device: None,
            measurements: crate::PolicyMeasurements::default(),
        };

        let decision = policy.plan(&snapshot).unwrap();
        assert_eq!(decision.record.policy.components[1].name, "composed");
    }

    #[test]
    fn invalid_candidate_plans_are_replaced_before_returning() {
        let policy =
            AdaptivePolicy::new(StalePolicy, vec![]).with_fallback(ComposedPolicy::default());
        let snapshot = PlanningSnapshot {
            at: Timestamp::ZERO,
            generation: 0,
            limits: crate::ExecutorLimits {
                max_active_requests: 1,
                prefill_chunk_size: None,
            },
            queued: Vec::new(),
            prefilling: BTreeMap::new(),
            decoding: BTreeMap::new(),
            decode_order: Vec::new(),
            prefill_order: Vec::new(),
            models: BTreeMap::new(),
            memory: crate::MemoryState::default(),
            device: None,
            measurements: crate::PolicyMeasurements::default(),
        };

        let decision = policy.plan(&snapshot).unwrap();
        assert_eq!(decision.record.policy.components[1].name, "composed");
    }

    fn knob(value: i64) -> Vec<(&'static str, String)> {
        vec![("budget", value.to_string())]
    }

    fn switching_milp(cost: i64) -> MilpCandidateOptimizer {
        MilpCandidateOptimizer::new(
            vec![AdaptiveIntegerDomain {
                name: "budget",
                min: 0,
                max: 10,
                step: 1,
            }],
            // Larger budgets are better by one unit per step.
            vec![("budget", -1)],
        )
        .unwrap()
        .with_switching_cost(SwitchingCost::uniform(&["budget"], cost))
    }

    #[test]
    fn switching_cost_counts_only_listed_changed_knobs() {
        let cost = SwitchingCost::uniform(&["budget"], 7);
        assert_eq!(cost.cost(&knob(3), &knob(3)), 0);
        assert_eq!(cost.cost(&knob(3), &knob(4)), 7);
        // A missing previous value is a reconfiguration.
        assert_eq!(cost.cost(&[], &knob(4)), 7);
        // Descriptive entries nobody listed never count.
        let mut described = knob(3);
        described.push(("label", "new".into()));
        assert_eq!(cost.cost(&knob(3), &described), 0);
        assert!(SwitchingCost::default().is_zero());
    }

    #[test]
    fn milp_keeps_the_current_vector_when_switching_costs_more_than_it_gains() {
        let mut optimizer = switching_milp(20);
        let proposed = optimizer
            .propose(&knob(5), &[], AdaptiveConstraints::default())
            .unwrap();
        assert_eq!(proposed, knob(5));
        assert_eq!(
            optimizer.last_objective(),
            Some(MilpObjectiveValue {
                operating: -5,
                switching: 0
            })
        );
    }

    #[test]
    fn milp_switches_when_the_gain_exceeds_the_switching_cost() {
        let mut optimizer = switching_milp(3);
        let proposed = optimizer
            .propose(&knob(5), &[], AdaptiveConstraints::default())
            .unwrap();
        assert_eq!(proposed, knob(10));
        let objective = optimizer.last_objective().unwrap();
        assert_eq!((objective.operating, objective.switching), (-10, 3));
        assert_eq!(objective.total(), -7);
    }

    #[test]
    fn milp_without_a_previous_value_charges_every_candidate_equally() {
        let mut optimizer = switching_milp(20);
        let proposed = optimizer
            .propose(&[], &[], AdaptiveConstraints::default())
            .unwrap();
        assert_eq!(proposed, knob(10));
        assert_eq!(optimizer.last_objective().unwrap().switching, 20);
    }

    #[test]
    fn milp_switching_cost_matches_the_finite_oracle() {
        for cost in [0, 1, 4, 5, 6, 30] {
            for start in [0, 3, 10] {
                let mut milp = switching_milp(cost);
                let switching = SwitchingCost::uniform(&["budget"], cost);
                let current = knob(start);
                let mut finite = FiniteCandidateOptimizer::new(
                    vec![AdaptiveIntegerDomain {
                        name: "budget",
                        min: 0,
                        max: 10,
                        step: 1,
                    }],
                    |candidate: &[(&'static str, String)],
                     _: &[AdaptiveRequestObservation],
                     _: AdaptiveConstraints| {
                        let value = candidate[0].1.parse::<i64>().unwrap();
                        Ok(Some(-value + switching.cost(&current, candidate)))
                    },
                )
                .unwrap();
                let exact = finite
                    .propose(&current, &[], AdaptiveConstraints::default())
                    .unwrap();
                let solved = milp
                    .propose(&current, &[], AdaptiveConstraints::default())
                    .unwrap();
                let total = |theta: &[(&'static str, String)]| {
                    -theta[0].1.parse::<i64>().unwrap() + switching.cost(&current, theta)
                };
                assert_eq!(total(&solved), total(&exact), "cost {cost}, start {start}");
                assert_eq!(milp.last_objective().unwrap().total(), total(&solved));
            }
        }
    }

    #[test]
    fn failed_optimizer_proposals_keep_the_installed_vector() {
        let mut policy = AdaptivePolicy::new(ComposedPolicy::default(), knob(5))
            .with_factory(|_: &[(&'static str, String)]| {
                Ok(Box::new(ComposedPolicy::default()) as Box<dyn ExecutionPolicy>)
            })
            .with_optimizer(
                |_: &[(&'static str, String)],
                 _: &[AdaptiveRequestObservation],
                 _: AdaptiveConstraints| {
                    Err(AdaptiveOptimizationError::Infeasible(
                        "solver failed".into(),
                    ))
                },
            );
        policy.observe(&ExecutorIteration {
            id: crate::IterationId::default(),
            at: Timestamp::ZERO,
            events: Vec::new(),
            execution_plan: ExecutionPlan::default(),
            decision: DecisionRecord::default(),
            divergences: Vec::new(),
            execution_batches: Vec::new(),
            active_requests: 0,
            queued_requests: 0,
            kv_cache_bytes: 0,
            reserved_kv_bytes: 0,
        });
        assert_eq!(policy.theta(), &knob(5)[..]);
    }
}
