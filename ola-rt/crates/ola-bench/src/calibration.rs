//! Least-squares calibration of iteration cost from measured iterations.
//!
//! Fits `micros = fixed + per_dense_token × dense_tokens + per_invocation ×
//! invocations` over many iteration shapes. Pairing each shape as one
//! batched invocation and as one invocation per request keeps dense tokens
//! and invocation counts from moving together, which a single grouping
//! would not, so all three coefficients are identifiable.
//!
//! The optional padding-aware fit splits dense tokens into real and padded
//! positions: `micros = fixed + per_real_token × real + per_padding_token ×
//! padding + per_invocation × invocations`. Its shapes add batches whose
//! rows have different lengths, so padding varies independently of real
//! tokens.

use std::{error::Error, fmt, time::Instant};

use ola_core::{Input, ModelDescriptor, RequestSpec, Timestamp};
use ola_model::CausalLanguageModel;
use ola_runtime::{
    ComposedPolicy, Executor, PolicyDescriptor, PolicyMeasurements, PrefillBatcher, Runtime,
};

use crate::{CostModel, Interval, random::SplitMix64};

/// ID of the calibration experiment.
pub const COST_CALIBRATION: &str = "cost-calibration";

/// Bootstrap resamples for coefficient intervals.
pub const CALIBRATION_RESAMPLES: usize = 1_000;

/// One measured iteration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CostObservation {
    /// Real plus padding input positions.
    pub dense_tokens: f64,
    /// Padding positions included in `dense_tokens`.
    pub padding_tokens: f64,
    /// Model invocations.
    pub invocations: f64,
    /// Measured iteration time.
    pub micros: f64,
}

/// Fitted iteration cost.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CostFit {
    /// Cost of an iteration independent of work.
    pub fixed_micros: f64,
    /// Cost per dense input position.
    pub per_dense_token_micros: f64,
    /// Cost per model invocation.
    pub per_invocation_micros: f64,
    /// Root-mean-square residual.
    pub residual_rms_micros: f64,
    /// Observations fitted.
    pub observations: usize,
    /// 95% bootstrap intervals, when enough resamples were well conditioned.
    pub intervals: Option<CostIntervals>,
}

/// 95% bootstrap intervals for each coefficient.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CostIntervals {
    /// Interval for `fixed_micros`.
    pub fixed_micros: Interval,
    /// Interval for `per_dense_token_micros`.
    pub per_dense_token_micros: Interval,
    /// Interval for `per_invocation_micros`.
    pub per_invocation_micros: Interval,
}

/// Fitted iteration cost with separate real and padded position costs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PaddedCostFit {
    /// Cost of an iteration independent of work.
    pub fixed_micros: f64,
    /// Cost per real input position.
    pub per_real_token_micros: f64,
    /// Cost per padded input position.
    pub per_padding_token_micros: f64,
    /// Cost per model invocation.
    pub per_invocation_micros: f64,
    /// Root-mean-square residual.
    pub residual_rms_micros: f64,
    /// Coefficient of determination over the fitted observations.
    pub r_squared: f64,
    /// Observations fitted.
    pub observations: usize,
    /// 95% bootstrap intervals, when enough resamples were well conditioned.
    pub intervals: Option<PaddedCostIntervals>,
}

/// 95% bootstrap intervals for each padding-aware coefficient.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PaddedCostIntervals {
    /// Interval for `fixed_micros`.
    pub fixed_micros: Interval,
    /// Interval for `per_real_token_micros`.
    pub per_real_token_micros: Interval,
    /// Interval for `per_padding_token_micros`.
    pub per_padding_token_micros: Interval,
    /// Interval for `per_invocation_micros`.
    pub per_invocation_micros: Interval,
}

impl PaddedCostFit {
    /// Predicted iteration time for one observation's shape.
    pub fn predict(&self, observation: &CostObservation) -> f64 {
        let [_, real, padding, invocations] = padded_row(observation);
        self.fixed_micros
            + self.per_real_token_micros * real
            + self.per_padding_token_micros * padding
            + self.per_invocation_micros * invocations
    }
}

impl CostFit {
    /// Converts the fit into a virtual-time cost model, rounding to whole
    /// microseconds and clamping negative coefficients to zero.
    pub fn cost_model(&self) -> CostModel {
        let micros = |value: f64| {
            // Rounded, clamped, and far below u64::MAX for real timings.
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let rounded = value.max(0.0).round() as u64;
            rounded
        };
        CostModel {
            fixed_micros: micros(self.fixed_micros),
            per_invocation_micros: micros(self.per_invocation_micros),
            per_dense_token_micros: micros(self.per_dense_token_micros),
            per_padding_token_micros: None,
            decode: None,
        }
    }

    /// Returns the invocation cost in token equivalents, the unit
    /// `CostAwarePrefillBatcher` uses, or `None` if token cost is not
    /// positive.
    pub fn invocation_overhead_tokens(&self) -> Option<usize> {
        (self.per_dense_token_micros > 0.0).then(|| {
            let ratio = (self.per_invocation_micros / self.per_dense_token_micros)
                .max(0.0)
                .round();
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let tokens = ratio as usize;
            tokens
        })
    }

    /// Reads the per-repetition mean fit from a `cost-calibration` report,
    /// with the report's 95% intervals across repetitions.
    pub fn from_report(report: &crate::ExperimentReport) -> Option<Self> {
        if report.experiment.id != COST_CALIBRATION {
            return None;
        }
        let estimates = &report.summary("fit")?.estimates;
        let mean = |name: &str| estimates.get(name).map(|estimate| estimate.mean);
        let interval = |name: &str| estimates.get(name).and_then(|estimate| estimate.ci95);
        Some(Self {
            fixed_micros: mean("fixed_micros")?,
            per_dense_token_micros: mean("per_dense_token_micros")?,
            per_invocation_micros: mean("per_invocation_micros")?,
            residual_rms_micros: mean("residual_rms_micros")?,
            observations: report.samples.len(),
            intervals: Some(CostIntervals {
                fixed_micros: interval("fixed_micros")?,
                per_dense_token_micros: interval("per_dense_token_micros")?,
                per_invocation_micros: interval("per_invocation_micros")?,
            }),
        })
    }

    /// Predicted iteration time for one observation's shape.
    pub fn predict(&self, observation: &CostObservation) -> f64 {
        self.fixed_micros
            + self.per_dense_token_micros * observation.dense_tokens
            + self.per_invocation_micros * observation.invocations
    }

    /// Coefficient of determination of this fit over `observations`.
    pub fn r_squared(&self, observations: &[CostObservation]) -> f64 {
        r_squared(observations, |observation| self.predict(observation))
    }

    /// Returns the measurements policies read from the planning snapshot.
    pub fn policy_measurements(&self) -> PolicyMeasurements {
        PolicyMeasurements {
            prefill_invocation_overhead_tokens: self.invocation_overhead_tokens(),
        }
    }
}

/// Calibration could not produce a fit.
#[derive(Clone, Debug, PartialEq)]
pub enum CalibrationError {
    /// Fewer observations than coefficients plus one.
    TooFewObservations(usize),
    /// Dense tokens and invocations do not vary independently.
    Singular,
    /// A padding-aware fit was requested, but no observation had padding.
    NoPadding,
    /// Measuring an iteration failed.
    Measurement(String),
}

impl fmt::Display for CalibrationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewObservations(count) => {
                write!(
                    formatter,
                    "calibration needs at least one more observation than coefficients, got {count}"
                )
            }
            Self::Singular => formatter.write_str(
                "dense tokens and invocations do not vary independently; widen the shape grid",
            ),
            Self::NoPadding => formatter.write_str(
                "no observation has padding; add mixed-length shapes for a padding-aware fit",
            ),
            Self::Measurement(message) => {
                write!(formatter, "calibration measurement failed: {message}")
            }
        }
    }
}

impl Error for CalibrationError {}

/// Fits the cost model by ordinary least squares, with seeded bootstrap
/// intervals over observations.
///
/// # Errors
///
/// Returns [`CalibrationError`] for too few or degenerate observations.
pub fn fit_cost_model(observations: &[CostObservation]) -> Result<CostFit, CalibrationError> {
    if observations.len() < 4 {
        return Err(CalibrationError::TooFewObservations(observations.len()));
    }
    let [fixed, per_dense, per_invocation] =
        least_squares(observations).ok_or(CalibrationError::Singular)?;
    let residual_rms_micros = (observations
        .iter()
        .map(|observation| {
            let predicted = fixed
                + per_dense * observation.dense_tokens
                + per_invocation * observation.invocations;
            (observation.micros - predicted).powi(2)
        })
        .sum::<f64>()
        / count(observations.len()))
    .sqrt();
    Ok(CostFit {
        fixed_micros: fixed,
        per_dense_token_micros: per_dense,
        per_invocation_micros: per_invocation,
        residual_rms_micros,
        observations: observations.len(),
        intervals: bootstrap(observations),
    })
}

/// A measured iteration that least squares and the bootstrap can fit.
trait Observed: Copy {
    /// Measured iteration time.
    fn micros(&self) -> f64;
    /// Values hashed into the bootstrap seed.
    fn seed_values(&self) -> [f64; 3];
}

impl Observed for CostObservation {
    fn micros(&self) -> f64 {
        self.micros
    }

    fn seed_values(&self) -> [f64; 3] {
        [self.dense_tokens, self.invocations, self.micros]
    }
}

/// One measured decode iteration: a single invocation decoding `rows`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecodeObservation {
    /// Rows decoded by the invocation.
    pub rows: f64,
    /// Cached positions the rows attend over, summed across rows.
    pub context_tokens: f64,
    /// Measured iteration time.
    pub micros: f64,
}

impl Observed for DecodeObservation {
    fn micros(&self) -> f64 {
        self.micros
    }

    fn seed_values(&self) -> [f64; 3] {
        [self.rows, self.context_tokens, self.micros]
    }
}

/// Fitted decode iteration cost:
/// `micros = intercept + per_row × rows + per_context_token × context`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecodeFit {
    /// Cost of a decode iteration independent of rows, including the
    /// iteration's fixed cost.
    pub intercept_micros: f64,
    /// Cost per decoded row.
    pub per_row_micros: f64,
    /// Cost per cached position attended over.
    pub per_context_token_micros: f64,
    /// Root-mean-square residual.
    pub residual_rms_micros: f64,
    /// Coefficient of determination over the fitted observations.
    pub r_squared: f64,
    /// Observations fitted.
    pub observations: usize,
    /// 95% bootstrap intervals for intercept, per row, and per context token.
    pub intervals: Option<[Interval; 3]>,
}

impl DecodeFit {
    /// Predicted iteration time for one observation's shape.
    pub fn predict(&self, observation: &DecodeObservation) -> f64 {
        self.intercept_micros
            + self.per_row_micros * observation.rows
            + self.per_context_token_micros * observation.context_tokens
    }
}

/// Fits decode cost by ordinary least squares, with seeded bootstrap
/// intervals over observations.
///
/// # Errors
///
/// Returns [`CalibrationError`] for too few or degenerate observations.
pub fn fit_decode_cost(observations: &[DecodeObservation]) -> Result<DecodeFit, CalibrationError> {
    if observations.len() < 4 {
        return Err(CalibrationError::TooFewObservations(observations.len()));
    }
    let row = |observation: &DecodeObservation| [1.0, observation.rows, observation.context_tokens];
    let [intercept, per_row, per_context] =
        solve(observations, row).ok_or(CalibrationError::Singular)?;
    let mut fit = DecodeFit {
        intercept_micros: intercept,
        per_row_micros: per_row,
        per_context_token_micros: per_context,
        residual_rms_micros: 0.0,
        r_squared: 0.0,
        observations: observations.len(),
        intervals: bootstrap_coefficients(observations, |resample| solve(resample, row)),
    };
    let residual = observations
        .iter()
        .map(|observation| (observation.micros - fit.predict(observation)).powi(2))
        .sum::<f64>();
    let mean = observations.iter().map(|o| o.micros).sum::<f64>() / count(observations.len());
    let total = observations
        .iter()
        .map(|observation| (observation.micros - mean).powi(2))
        .sum::<f64>();
    fit.residual_rms_micros = (residual / count(observations.len())).sqrt();
    fit.r_squared = if total > 0.0 {
        1.0 - residual / total
    } else {
        0.0
    };
    Ok(fit)
}

/// Fits the padding-aware cost model by ordinary least squares, with seeded
/// bootstrap intervals over observations.
///
/// # Errors
///
/// Returns [`CalibrationError`] for too few or degenerate observations, or
/// when no observation contains padding.
pub fn fit_padded_cost_model(
    observations: &[CostObservation],
) -> Result<PaddedCostFit, CalibrationError> {
    if observations.len() < 5 {
        return Err(CalibrationError::TooFewObservations(observations.len()));
    }
    if observations
        .iter()
        .all(|observation| observation.padding_tokens == 0.0)
    {
        return Err(CalibrationError::NoPadding);
    }
    let [fixed, per_real, per_padding, per_invocation] =
        padded_least_squares(observations).ok_or(CalibrationError::Singular)?;
    let mut fit = PaddedCostFit {
        fixed_micros: fixed,
        per_real_token_micros: per_real,
        per_padding_token_micros: per_padding,
        per_invocation_micros: per_invocation,
        residual_rms_micros: 0.0,
        r_squared: 0.0,
        observations: observations.len(),
        intervals: bootstrap_coefficients(observations, padded_least_squares).map(
            |[
                fixed_micros,
                per_real_token_micros,
                per_padding_token_micros,
                per_invocation_micros,
            ]| {
                PaddedCostIntervals {
                    fixed_micros,
                    per_real_token_micros,
                    per_padding_token_micros,
                    per_invocation_micros,
                }
            },
        ),
    };
    fit.residual_rms_micros = (observations
        .iter()
        .map(|observation| (observation.micros - fit.predict(observation)).powi(2))
        .sum::<f64>()
        / count(observations.len()))
    .sqrt();
    fit.r_squared = r_squared(observations, |observation| fit.predict(observation));
    Ok(fit)
}

/// `1 − SS_res / SS_tot` of `predict` over `observations`.
fn r_squared(observations: &[CostObservation], predict: impl Fn(&CostObservation) -> f64) -> f64 {
    let mean = observations
        .iter()
        .map(|observation| observation.micros)
        .sum::<f64>()
        / count(observations.len());
    let (residual, total) =
        observations
            .iter()
            .fold((0.0, 0.0), |(residual, total), observation| {
                (
                    residual + (observation.micros - predict(observation)).powi(2),
                    total + (observation.micros - mean).powi(2),
                )
            });
    if total > 0.0 {
        1.0 - residual / total
    } else {
        0.0
    }
}

fn padded_least_squares(observations: &[CostObservation]) -> Option<[f64; 4]> {
    solve(observations, padded_row)
}

fn least_squares(observations: &[CostObservation]) -> Option<[f64; 3]> {
    solve(observations, dense_row)
}

/// Regressors `[1, dense, invocations]` of the three-coefficient fit.
fn dense_row(observation: &CostObservation) -> [f64; 3] {
    [1.0, observation.dense_tokens, observation.invocations]
}

/// Regressors `[1, real, padding, invocations]` of the padding-aware fit.
fn padded_row(observation: &CostObservation) -> [f64; 4] {
    [
        1.0,
        observation.dense_tokens - observation.padding_tokens,
        observation.padding_tokens,
        observation.invocations,
    ]
}

/// Solves the normal equations `XᵀX β = Xᵀy` for rows produced by `row`.
fn solve<T: Observed, const N: usize>(
    observations: &[T],
    row: impl Fn(&T) -> [f64; N],
) -> Option<[f64; N]> {
    // Each row of `matrix` is one normal equation; the last column is Xᵀy.
    let mut matrix = vec![vec![0.0_f64; N + 1]; N];
    for observation in observations {
        let row = row(observation);
        for (i, left) in row.iter().enumerate() {
            for (j, right) in row.iter().enumerate() {
                matrix[i][j] += left * right;
            }
            matrix[i][N] += left * observation.micros();
        }
    }
    let scale = matrix
        .iter()
        .flat_map(|row| row[..N].iter())
        .fold(0.0_f64, |largest, value| largest.max(value.abs()));
    // Gaussian elimination with partial pivoting.
    for column in 0..N {
        let pivot = (column..N)
            .max_by(|&a, &b| matrix[a][column].abs().total_cmp(&matrix[b][column].abs()))?;
        if matrix[pivot][column].abs() <= scale * 1e-12 {
            return None;
        }
        matrix.swap(column, pivot);
        let pivot_row = matrix[column].clone();
        for (index, row) in matrix.iter_mut().enumerate() {
            if index != column {
                let factor = row[column] / pivot_row[column];
                for (target, source) in row.iter_mut().zip(&pivot_row).skip(column) {
                    *target -= factor * source;
                }
            }
        }
    }
    Some(std::array::from_fn(|index| {
        matrix[index][N] / matrix[index][index]
    }))
}

fn bootstrap(observations: &[CostObservation]) -> Option<CostIntervals> {
    let [fixed_micros, per_dense_token_micros, per_invocation_micros] =
        bootstrap_coefficients(observations, least_squares)?;
    Some(CostIntervals {
        fixed_micros,
        per_dense_token_micros,
        per_invocation_micros,
    })
}

/// Seeded percentile-bootstrap intervals for the coefficients `fit` returns.
fn bootstrap_coefficients<T: Observed, const N: usize>(
    observations: &[T],
    fit: impl Fn(&[T]) -> Option<[f64; N]>,
) -> Option<[Interval; N]> {
    let seed = observations
        .iter()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, observation| {
            observation
                .seed_values()
                .iter()
                .flat_map(|value| value.to_bits().to_le_bytes())
                .fold(hash, |hash, byte| {
                    (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
                })
        });
    let mut random = SplitMix64::new(seed);
    let last = u64::try_from(observations.len() - 1).unwrap_or(u64::MAX);
    let mut fits = Vec::with_capacity(CALIBRATION_RESAMPLES);
    let mut resample = Vec::with_capacity(observations.len());
    for _ in 0..CALIBRATION_RESAMPLES {
        resample.clear();
        resample.extend(
            (0..observations.len())
                .map(|_| observations[usize::try_from(random.between_u64(0, last)).unwrap_or(0)]),
        );
        if let Some(fit) = fit(&resample) {
            fits.push(fit);
        }
    }
    // Too many degenerate resamples means the intervals would be biased.
    if fits.len() < CALIBRATION_RESAMPLES / 2 {
        return None;
    }
    let interval = |coefficient: usize| {
        let mut values = fits.iter().map(|fit| fit[coefficient]).collect::<Vec<_>>();
        values.sort_by(f64::total_cmp);
        let rank = |per_mille: usize| {
            let index = (values.len() * per_mille).div_ceil(1_000).saturating_sub(1);
            values[index.min(values.len() - 1)]
        };
        Interval {
            low: rank(25),
            high: rank(975),
        }
    };
    Some(std::array::from_fn(interval))
}

/// Gives every prompt its own group, forcing one invocation per request.
struct SplitEveryRequest;

impl PrefillBatcher for SplitEveryRequest {
    fn bucket(&self, _prompt_tokens: usize) -> usize {
        0
    }

    fn buckets(&self, _prompt_tokens: &[usize], buckets: &mut [usize]) {
        for (index, bucket) in buckets.iter_mut().enumerate() {
            *bucket = index;
        }
    }

    fn descriptor(&self) -> PolicyDescriptor {
        PolicyDescriptor::new("split-every-request")
    }
}

/// Measures one prefill iteration per (width, length) shape, both batched
/// and split into one invocation per request.
///
/// # Errors
///
/// Returns [`CalibrationError::Measurement`] if the executor fails.
pub fn measure_iteration_costs(
    model: &CausalLanguageModel,
    widths: &[usize],
    lengths: &[usize],
) -> Result<Vec<CostObservation>, CalibrationError> {
    let mut observations = Vec::with_capacity(widths.len() * lengths.len() * 2);
    for &width in widths {
        for &length in lengths {
            let prompts = vec![length.max(1); width.max(1)];
            for split in [false, true] {
                observations.push(measure_one(model, &prompts, split)?);
            }
        }
    }
    Ok(observations)
}

/// Prompt lengths of one mixed-length shape: one row of `length` and
/// `width − 1` rows a quarter as long, or `None` when no padding would result.
pub fn mixed_length_prompts(width: usize, length: usize) -> Option<Vec<usize>> {
    let short = (length / 4).max(1);
    (width >= 2 && short < length).then(|| {
        std::iter::once(length)
            .chain(std::iter::repeat_n(short, width - 1))
            .collect()
    })
}

/// Measures mixed-length shapes, batched and split, for the padding-aware
/// fit. Batched, the short rows are padded to the long one; split, nothing
/// is padded. Widths below two and lengths below two contribute nothing.
///
/// # Errors
///
/// Returns [`CalibrationError::Measurement`] if the executor fails.
pub fn measure_mixed_iteration_costs(
    model: &CausalLanguageModel,
    widths: &[usize],
    lengths: &[usize],
) -> Result<Vec<CostObservation>, CalibrationError> {
    let mut observations = Vec::with_capacity(widths.len() * lengths.len() * 2);
    for &width in widths {
        for &length in lengths {
            let Some(prompts) = mixed_length_prompts(width, length) else {
                continue;
            };
            for split in [false, true] {
                observations.push(measure_one(model, &prompts, split)?);
            }
        }
    }
    Ok(observations)
}

/// Measures one decode iteration per (width, length) shape: `width`
/// requests of `length` tokens are prefilled in one iteration, then the next
/// iteration, which decodes every row in one invocation, is timed.
///
/// # Errors
///
/// Returns [`CalibrationError::Measurement`] if the executor fails or the
/// timed iteration is not a single decode invocation.
pub fn measure_decode_iteration_costs(
    model: &CausalLanguageModel,
    widths: &[usize],
    lengths: &[usize],
) -> Result<Vec<DecodeObservation>, CalibrationError> {
    let mut observations = Vec::with_capacity(widths.len() * lengths.len());
    for &width in widths {
        for &length in lengths {
            observations.push(measure_decode_one(model, width.max(1), length.max(1))?);
        }
    }
    Ok(observations)
}

fn measure_decode_one(
    model: &CausalLanguageModel,
    width: usize,
    length: usize,
) -> Result<DecodeObservation, CalibrationError> {
    const MODEL_ID: &str = "calibration";
    let failed = |error: &dyn fmt::Display| CalibrationError::Measurement(error.to_string());
    let mut executor = Executor::with_max_active_requests(Runtime::default(), width)
        .map_err(|error| failed(&error))?
        .with_policy(ComposedPolicy::default());
    executor
        .register_model(
            ModelDescriptor::language(MODEL_ID, "calibration"),
            model.clone(),
        )
        .map_err(|error| failed(&error))?;
    for _ in 0..width {
        let mut request = RequestSpec::text(MODEL_ID, "");
        request.inputs = vec![Input::Tokens(vec![1; length])];
        request.generation.max_new_tokens = 2;
        request.generation.temperature = 0.0;
        executor
            .submit(request, Timestamp::ZERO)
            .map_err(|error| failed(&error))?;
    }
    executor
        .step_iteration(Timestamp::from_micros(1))
        .map_err(|error| failed(&error))?;
    let started = Instant::now();
    let iteration = executor
        .step_iteration(Timestamp::from_micros(2))
        .map_err(|error| failed(&error))?;
    let micros = started.elapsed().as_secs_f64() * 1_000_000.0;
    let [batch] = iteration.execution_batches.as_slice() else {
        return Err(CalibrationError::Measurement(format!(
            "expected one decode invocation, got {}",
            iteration.execution_batches.len()
        )));
    };
    if batch.phase != ola_runtime::BatchPhase::Decode {
        return Err(CalibrationError::Measurement(
            "the timed iteration did not decode".into(),
        ));
    }
    let rows = batch.request_ids.len();
    Ok(DecodeObservation {
        rows: count(rows),
        // Each row attends over its prompt and its first generated token.
        context_tokens: count(rows * (length + 1)),
        micros,
    })
}

fn measure_one(
    model: &CausalLanguageModel,
    prompt_lengths: &[usize],
    split: bool,
) -> Result<CostObservation, CalibrationError> {
    const MODEL_ID: &str = "calibration";
    let failed = |error: &dyn fmt::Display| CalibrationError::Measurement(error.to_string());
    let policy = if split {
        ComposedPolicy::default().with_grouping(SplitEveryRequest)
    } else {
        ComposedPolicy::default()
    };
    let mut executor = Executor::with_max_active_requests(Runtime::default(), prompt_lengths.len())
        .map_err(|error| failed(&error))?
        .with_policy(policy);
    executor
        .register_model(
            ModelDescriptor::language(MODEL_ID, "calibration"),
            model.clone(),
        )
        .map_err(|error| failed(&error))?;
    for &length in prompt_lengths {
        let mut request = RequestSpec::text(MODEL_ID, "");
        request.inputs = vec![Input::Tokens(vec![1; length])];
        request.generation.max_new_tokens = 1;
        request.generation.temperature = 0.0;
        executor
            .submit(request, Timestamp::ZERO)
            .map_err(|error| failed(&error))?;
    }
    let started = Instant::now();
    let iteration = executor
        .step_iteration(Timestamp::from_micros(1))
        .map_err(|error| failed(&error))?;
    let micros = started.elapsed().as_secs_f64() * 1_000_000.0;
    Ok(CostObservation {
        dense_tokens: count(
            iteration
                .execution_batches
                .iter()
                .map(ola_runtime::ExecutionBatch::dense_input_tokens)
                .sum::<usize>(),
        ),
        padding_tokens: count(
            iteration
                .execution_batches
                .iter()
                .map(|batch| batch.padding_tokens)
                .sum::<usize>(),
        ),
        invocations: count(iteration.execution_batches.len()),
        micros,
    })
}

#[allow(clippy::cast_precision_loss, reason = "counts are tiny")]
fn count(value: usize) -> f64 {
    value as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Observations from known coefficients with small, deterministic noise.
    fn synthetic(fixed: f64, per_dense: f64, per_invocation: f64) -> Vec<CostObservation> {
        let mut observations = Vec::new();
        let mut noise = SplitMix64::new(9);
        for width in [1.0, 2.0, 4.0] {
            for length in [4.0, 8.0, 16.0] {
                for invocations in [1.0, width] {
                    let dense = width * length;
                    let jitter = (noise.between_u64(0, 20) as f64) - 10.0;
                    observations.push(CostObservation {
                        dense_tokens: dense,
                        padding_tokens: 0.0,
                        invocations,
                        micros: fixed + per_dense * dense + per_invocation * invocations + jitter,
                    });
                }
            }
        }
        observations
    }

    #[test]
    fn least_squares_recovers_known_coefficients_with_intervals() {
        let fit = fit_cost_model(&synthetic(200.0, 4.0, 300.0)).unwrap();

        assert!((fit.fixed_micros - 200.0).abs() < 10.0);
        assert!((fit.per_dense_token_micros - 4.0).abs() < 0.5);
        assert!((fit.per_invocation_micros - 300.0).abs() < 10.0);
        assert!(fit.residual_rms_micros < 10.0);
        let intervals = fit.intervals.unwrap();
        for (interval, truth) in [
            (intervals.fixed_micros, 200.0),
            (intervals.per_dense_token_micros, 4.0),
            (intervals.per_invocation_micros, 300.0),
        ] {
            assert!(
                interval.low <= truth && truth <= interval.high,
                "{interval:?} misses {truth}"
            );
        }
        // Noise moves the estimate slightly; the conversions follow the fit.
        assert!(fit.cost_model().per_invocation_micros.abs_diff(300) <= 10);
        let overhead = fit.invocation_overhead_tokens().unwrap();
        assert!(overhead.abs_diff(75) <= 3, "overhead {overhead}");
        assert_eq!(
            fit.policy_measurements().prefill_invocation_overhead_tokens,
            Some(overhead)
        );
        // Deterministic: the same observations give the same fit.
        assert_eq!(fit_cost_model(&synthetic(200.0, 4.0, 300.0)).unwrap(), fit);
    }

    /// Equal-length and mixed-length shapes, batched and split, from known
    /// padding-aware coefficients with small, deterministic noise.
    fn synthetic_padded(
        fixed: f64,
        per_real: f64,
        per_padding: f64,
        per_invocation: f64,
    ) -> Vec<CostObservation> {
        let mut observations = Vec::new();
        let mut noise = SplitMix64::new(11);
        for width in [1_usize, 2, 4] {
            for length in [8_usize, 16, 32] {
                let mut shapes = vec![vec![length; width]];
                shapes.extend(mixed_length_prompts(width, length));
                for prompts in shapes {
                    let real = prompts.iter().sum::<usize>() as f64;
                    let longest = *prompts.iter().max().unwrap() as f64;
                    let batched_padding = longest * prompts.len() as f64 - real;
                    for (invocations, padding) in [(1.0, batched_padding), (width as f64, 0.0)] {
                        let jitter = (noise.between_u64(0, 20) as f64) - 10.0;
                        observations.push(CostObservation {
                            dense_tokens: real + padding,
                            padding_tokens: padding,
                            invocations,
                            micros: fixed
                                + per_real * real
                                + per_padding * padding
                                + per_invocation * invocations
                                + jitter,
                        });
                    }
                }
            }
        }
        observations
    }

    #[test]
    fn padding_aware_fit_separates_real_and_padded_positions() {
        let observations = synthetic_padded(200.0, 4.0, 1.0, 300.0);
        let fit = fit_padded_cost_model(&observations).unwrap();

        assert!((fit.fixed_micros - 200.0).abs() < 10.0);
        assert!((fit.per_real_token_micros - 4.0).abs() < 0.2);
        assert!((fit.per_padding_token_micros - 1.0).abs() < 0.2);
        assert!((fit.per_invocation_micros - 300.0).abs() < 10.0);
        assert!(fit.r_squared > 0.999, "R² {}", fit.r_squared);
        let intervals = fit.intervals.unwrap();
        for (interval, truth) in [
            (intervals.per_real_token_micros, 4.0),
            (intervals.per_padding_token_micros, 1.0),
        ] {
            assert!(
                interval.low <= truth && truth <= interval.high,
                "{interval:?} misses {truth}"
            );
        }
        // The single dense rate cannot express cheap padding, so it fits worse.
        let dense = fit_cost_model(&observations).unwrap();
        assert!(dense.r_squared(&observations) < fit.r_squared);
        assert!(dense.residual_rms_micros > fit.residual_rms_micros);
    }

    #[test]
    fn decode_fit_recovers_per_row_and_context_costs() {
        let mut noise = SplitMix64::new(5);
        let mut observations = Vec::new();
        for rows in [1.0, 2.0, 4.0, 8.0] {
            for length in [16.0, 64.0, 128.0] {
                let context = rows * (length + 1.0);
                let jitter = (noise.between_u64(0, 20) as f64) - 10.0;
                observations.push(DecodeObservation {
                    rows,
                    context_tokens: context,
                    micros: 30_000.0 + 25_000.0 * rows + 2.0 * context + jitter,
                });
            }
        }
        let fit = fit_decode_cost(&observations).unwrap();

        assert!((fit.intercept_micros - 30_000.0).abs() < 20.0);
        assert!((fit.per_row_micros - 25_000.0).abs() < 20.0);
        assert!((fit.per_context_token_micros - 2.0).abs() < 0.1);
        assert!(fit.r_squared > 0.999_99);
        // With twelve points the interval is narrow; it sits on the truth.
        let [_, per_row, _] = fit.intervals.unwrap();
        assert!(per_row.low < per_row.high);
        assert!((per_row.low - 25_000.0).abs() < 20.0 && (per_row.high - 25_000.0).abs() < 20.0);
        assert_eq!(
            fit_decode_cost(&observations[..3]),
            Err(CalibrationError::TooFewObservations(3))
        );
    }

    #[test]
    fn padding_aware_fit_needs_padded_observations() {
        assert_eq!(
            fit_padded_cost_model(&synthetic(200.0, 4.0, 300.0)),
            Err(CalibrationError::NoPadding)
        );
        assert_eq!(
            fit_padded_cost_model(&synthetic_padded(1.0, 1.0, 1.0, 1.0)[..4]),
            Err(CalibrationError::TooFewObservations(4))
        );
    }

    #[test]
    fn mixed_length_shapes_pad_short_rows_to_one_long_row() {
        assert_eq!(mixed_length_prompts(4, 16), Some(vec![16, 4, 4, 4]));
        assert_eq!(mixed_length_prompts(2, 3), Some(vec![3, 1]));
        assert_eq!(mixed_length_prompts(1, 16), None);
        assert_eq!(mixed_length_prompts(4, 1), None);
    }

    #[test]
    fn degenerate_observations_are_rejected() {
        assert_eq!(
            fit_cost_model(&synthetic(1.0, 1.0, 1.0)[..3]),
            Err(CalibrationError::TooFewObservations(3))
        );
        // A single grouping ties invocations to the intercept.
        let one_call = synthetic(200.0, 4.0, 300.0)
            .into_iter()
            .filter(|observation| observation.invocations == 1.0)
            .collect::<Vec<_>>();
        assert_eq!(fit_cost_model(&one_call), Err(CalibrationError::Singular));
    }

    #[test]
    fn negative_coefficients_clamp_in_the_cost_model() {
        let fit = CostFit {
            fixed_micros: -12.0,
            per_dense_token_micros: 2.4,
            per_invocation_micros: 7.6,
            residual_rms_micros: 0.0,
            observations: 4,
            intervals: None,
        };
        assert_eq!(
            fit.cost_model(),
            CostModel {
                fixed_micros: 0,
                per_invocation_micros: 8,
                per_dense_token_micros: 2,
                per_padding_token_micros: None,
                decode: None,
            }
        );
        assert_eq!(fit.invocation_overhead_tokens(), Some(3));
    }

    #[test]
    fn calibration_reports_convert_back_into_fits() {
        use std::collections::BTreeMap;

        use crate::{
            Direction, ExperimentIdentity, ExperimentKind, ExperimentReport, MetricSpec, Protocol,
            ReportParts, RunMetadata, Sample, TimeDomain, Unit, Variant, WorkloadIdentity,
        };

        let metric = |name| MetricSpec::new(name, Unit::Micros, TimeDomain::Host, Direction::Lower);
        let sample = |repetition, fixed: f64, dense: f64, call: f64| Sample {
            repetition,
            variant: "fit".into(),
            values: BTreeMap::from([
                ("fixed_micros".to_owned(), fixed),
                ("per_dense_token_micros".to_owned(), dense),
                ("per_invocation_micros".to_owned(), call),
                ("residual_rms_micros".to_owned(), 1.0),
            ]),
        };
        let report = ExperimentReport::new(ReportParts {
            experiment: ExperimentIdentity {
                id: COST_CALIBRATION.into(),
                payload_version: 1,
                kind: ExperimentKind::Probe,
                domain: TimeDomain::Host,
            },
            metadata: RunMetadata::now(None, None),
            workload: WorkloadIdentity {
                name: "shape-grid".into(),
                seeds: Vec::new(),
                parameters: BTreeMap::new(),
                digest: String::new(),
            },
            protocol: Protocol {
                warmups: 0,
                repetitions: 2,
                ordering: "single run".into(),
            },
            metrics: vec![
                metric("fixed_micros"),
                metric("per_dense_token_micros"),
                metric("per_invocation_micros"),
                metric("residual_rms_micros"),
            ],
            variants: vec![Variant {
                label: "fit".into(),
                policy: None,
                kernel: None,
            }],
            samples: vec![sample(0, 190.0, 3.0, 290.0), sample(1, 210.0, 5.0, 310.0)],
            details: None,
        })
        .unwrap();

        let fit = CostFit::from_report(&report).unwrap();

        assert_eq!(
            fit.cost_model(),
            CostModel {
                fixed_micros: 200,
                per_invocation_micros: 300,
                per_dense_token_micros: 4,
                per_padding_token_micros: None,
                decode: None,
            }
        );
        assert!(fit.intervals.is_some());
        let mut other = report;
        other.experiment.id = "scheduler-comparison".into();
        assert_eq!(CostFit::from_report(&other), None);
    }
}
