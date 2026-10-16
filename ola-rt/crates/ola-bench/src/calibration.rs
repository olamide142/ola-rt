//! Least-squares calibration of iteration cost from measured iterations.
//!
//! Fits `micros = fixed + per_dense_token × dense_tokens + per_invocation ×
//! invocations` over many iteration shapes. Pairing each shape as one
//! batched invocation and as one invocation per request keeps dense tokens
//! and invocation counts from moving together, which a single grouping
//! would not, so all three coefficients are identifiable.

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
    /// Measuring an iteration failed.
    Measurement(String),
}

impl fmt::Display for CalibrationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewObservations(count) => {
                write!(
                    formatter,
                    "calibration needs at least 4 observations, got {count}"
                )
            }
            Self::Singular => formatter.write_str(
                "dense tokens and invocations do not vary independently; widen the shape grid",
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

fn least_squares(observations: &[CostObservation]) -> Option<[f64; 3]> {
    // Normal equations XᵀX β = Xᵀy with X rows [1, dense, invocations].
    let mut matrix = [[0.0_f64; 4]; 3];
    for observation in observations {
        let row = [1.0, observation.dense_tokens, observation.invocations];
        for (i, left) in row.iter().enumerate() {
            for (j, right) in row.iter().enumerate() {
                matrix[i][j] += left * right;
            }
            matrix[i][3] += left * observation.micros;
        }
    }
    let scale = matrix
        .iter()
        .flat_map(|row| row[..3].iter())
        .fold(0.0_f64, |largest, value| largest.max(value.abs()));
    // Gaussian elimination with partial pivoting.
    for column in 0..3 {
        let pivot = (column..3)
            .max_by(|&a, &b| matrix[a][column].abs().total_cmp(&matrix[b][column].abs()))?;
        if matrix[pivot][column].abs() <= scale * 1e-12 {
            return None;
        }
        matrix.swap(column, pivot);
        let pivot_row = matrix[column];
        for (index, row) in matrix.iter_mut().enumerate() {
            if index != column {
                let factor = row[column] / pivot_row[column];
                for (target, source) in row.iter_mut().zip(pivot_row).skip(column) {
                    *target -= factor * source;
                }
            }
        }
    }
    Some([
        matrix[0][3] / matrix[0][0],
        matrix[1][3] / matrix[1][1],
        matrix[2][3] / matrix[2][2],
    ])
}

fn bootstrap(observations: &[CostObservation]) -> Option<CostIntervals> {
    let seed = observations
        .iter()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, observation| {
            [
                observation.dense_tokens,
                observation.invocations,
                observation.micros,
            ]
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
        if let Some(fit) = least_squares(&resample) {
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
    Some(CostIntervals {
        fixed_micros: interval(0),
        per_dense_token_micros: interval(1),
        per_invocation_micros: interval(2),
    })
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
            for split in [false, true] {
                observations.push(measure_one(model, width, length, split)?);
            }
        }
    }
    Ok(observations)
}

fn measure_one(
    model: &CausalLanguageModel,
    width: usize,
    length: usize,
    split: bool,
) -> Result<CostObservation, CalibrationError> {
    const MODEL_ID: &str = "calibration";
    let failed = |error: &dyn fmt::Display| CalibrationError::Measurement(error.to_string());
    let policy = if split {
        ComposedPolicy::default().with_grouping(SplitEveryRequest)
    } else {
        ComposedPolicy::default()
    };
    let mut executor = Executor::with_max_active_requests(Runtime::default(), width.max(1))
        .map_err(|error| failed(&error))?
        .with_policy(policy);
    executor
        .register_model(
            ModelDescriptor::language(MODEL_ID, "calibration"),
            model.clone(),
        )
        .map_err(|error| failed(&error))?;
    for _ in 0..width.max(1) {
        let mut request = RequestSpec::text(MODEL_ID, "");
        request.inputs = vec![Input::Tokens(vec![1; length.max(1)])];
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
            }
        );
        assert!(fit.intervals.is_some());
        let mut other = report;
        other.experiment.id = "scheduler-comparison".into();
        assert_eq!(CostFit::from_report(&other), None);
    }
}
