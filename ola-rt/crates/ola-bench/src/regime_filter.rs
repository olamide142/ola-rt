//! Online regime estimation over the arrival process.
//!
//! The workload regime is hidden; what the executor sees is a stream of
//! arrival times. [`ArrivalWindows`] turns that stream into one feature
//! vector per `size` arrivals (the log mean gap and the gaps' coefficient of
//! variation), a [`GaussianHmm`] models those vectors as emissions of a
//! hidden regime, and [`RegimeFilter`] runs the forward algorithm online,
//! giving a posterior over regimes after every window.
//!
//! Only the arrival process is modeled: it is exogenous. Queue depth and
//! backlog depend on the policy in force, so as emissions they would make
//! the model partly detect its own actions; the replay evaluator reads them
//! directly instead. See `REGIME_DESIGN.md` in `ola-runtime`.
//!
//! [`PageHinkley`] is the scalar change detector the HMM is compared with.

/// Lower bound on a fitted variance, so a constant feature cannot make a
/// state's likelihood infinite.
const VARIANCE_FLOOR: f64 = 1e-3;

/// Groups an arrival stream into feature windows of `size` gaps.
#[derive(Clone, Debug)]
pub struct ArrivalWindows {
    size: usize,
    last: Option<u64>,
    gaps: Vec<f64>,
}

impl ArrivalWindows {
    /// Windows of `size` inter-arrival gaps; `size` below two is raised to
    /// two so a coefficient of variation exists.
    pub fn new(size: usize) -> Self {
        Self {
            size: size.max(2),
            last: None,
            gaps: Vec::new(),
        }
    }

    /// Records one arrival, in non-decreasing time order, and returns the
    /// features of the window it completes, if any.
    pub fn push(&mut self, arrival_micros: u64) -> Option<Vec<f64>> {
        let previous = self.last.replace(arrival_micros)?;
        self.gaps
            .push(arrival_micros.saturating_sub(previous) as f64);
        if self.gaps.len() < self.size {
            return None;
        }
        let features = window_features(&self.gaps);
        self.gaps.clear();
        Some(features)
    }

    /// Features of every complete window of `arrivals`, for fitting.
    pub fn features(arrivals: &[u64], size: usize) -> Vec<Vec<f64>> {
        let mut windows = Self::new(size);
        arrivals
            .iter()
            .filter_map(|&arrival| windows.push(arrival))
            .collect()
    }
}

/// `[ln(1 + mean gap µs), coefficient of variation of the gaps]`.
fn window_features(gaps: &[f64]) -> Vec<f64> {
    let n = gaps.len() as f64;
    let mean = gaps.iter().sum::<f64>() / n;
    let variance = gaps.iter().map(|gap| (gap - mean).powi(2)).sum::<f64>() / n;
    let cv = if mean > 0.0 {
        variance.sqrt() / mean
    } else {
        0.0
    };
    vec![mean.ln_1p(), cv]
}

/// A hidden Markov model with diagonal Gaussian emissions.
#[derive(Clone, Debug, PartialEq)]
pub struct GaussianHmm {
    /// Probability of starting in each state.
    pub initial: Vec<f64>,
    /// `transition[i][j]`: probability of moving from state `i` to `j`
    /// between windows.
    pub transition: Vec<Vec<f64>>,
    /// Per-state emission means, one per feature.
    pub means: Vec<Vec<f64>>,
    /// Per-state emission variances, one per feature.
    pub variances: Vec<Vec<f64>>,
}

impl GaussianHmm {
    /// Number of hidden states.
    pub fn states(&self) -> usize {
        self.initial.len()
    }

    /// Windows a state is expected to last from now, `1 / (1 − p_ii)`:
    /// the remaining-duration term of the switch rule. Infinite for an
    /// absorbing state.
    pub fn expected_remaining_windows(&self, state: usize) -> f64 {
        1.0 / (1.0 - self.transition[state][state])
    }

    fn log_emission(&self, state: usize, x: &[f64]) -> f64 {
        x.iter()
            .zip(&self.means[state])
            .zip(&self.variances[state])
            .map(|((x, mean), variance)| {
                -0.5 * ((2.0 * std::f64::consts::PI * variance).ln()
                    + (x - mean).powi(2) / variance)
            })
            .sum()
    }

    /// Emission likelihoods at `x`, scaled so the largest is one (the
    /// filter normalizes, so only ratios matter).
    fn emissions(&self, x: &[f64]) -> Vec<f64> {
        let logs = (0..self.states())
            .map(|state| self.log_emission(state, x))
            .collect::<Vec<_>>();
        let max = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        logs.iter().map(|log| (log - max).exp()).collect()
    }

    /// Fits `states` regimes to `sequences` of feature windows by
    /// Baum–Welch, starting from quantiles of the first feature, and
    /// orders the states by that feature's mean so fits are comparable.
    /// `None` without enough windows (fewer than `states` in total).
    pub fn fit(sequences: &[Vec<Vec<f64>>], states: usize, iterations: usize) -> Option<Self> {
        let pooled = sequences.iter().flatten().collect::<Vec<_>>();
        let dimensions = pooled.first()?.len();
        if states == 0 || pooled.len() < states {
            return None;
        }
        let mut sorted = pooled.clone();
        sorted.sort_by(|a, b| a[0].total_cmp(&b[0]));
        let chunk = sorted.len() / states;
        let moments = |rows: &[&Vec<f64>]| {
            let n = rows.len() as f64;
            let mean = (0..dimensions)
                .map(|d| rows.iter().map(|row| row[d]).sum::<f64>() / n)
                .collect::<Vec<_>>();
            let variance = (0..dimensions)
                .map(|d| {
                    (rows
                        .iter()
                        .map(|row| (row[d] - mean[d]).powi(2))
                        .sum::<f64>()
                        / n)
                        .max(VARIANCE_FLOOR)
                })
                .collect::<Vec<_>>();
            (mean, variance)
        };
        let (means, variances) = (0..states)
            .map(|state| {
                let end = if state + 1 == states {
                    sorted.len()
                } else {
                    (state + 1) * chunk
                };
                moments(&sorted[state * chunk..end])
            })
            .unzip();
        let stay = 0.9;
        let mut model = Self {
            initial: vec![1.0 / states as f64; states],
            transition: (0..states)
                .map(|i| {
                    (0..states)
                        .map(|j| {
                            if i == j {
                                stay
                            } else {
                                (1.0 - stay) / (states - 1).max(1) as f64
                            }
                        })
                        .collect()
                })
                .collect(),
            means,
            variances,
        };
        for _ in 0..iterations {
            model = model.baum_welch_step(sequences, dimensions);
        }
        model.sort_states();
        Some(model)
    }

    /// One expectation–maximization step over every sequence, with scaled
    /// forward–backward passes.
    fn baum_welch_step(&self, sequences: &[Vec<Vec<f64>>], dimensions: usize) -> Self {
        let k = self.states();
        let mut initial = vec![0.0; k];
        let mut transitions = vec![vec![0.0; k]; k];
        let mut weight = vec![0.0; k];
        let mut sums = vec![vec![0.0; dimensions]; k];
        let mut squares = vec![vec![0.0; dimensions]; k];
        for sequence in sequences.iter().filter(|sequence| !sequence.is_empty()) {
            let emissions = sequence
                .iter()
                .map(|x| self.emissions(x))
                .collect::<Vec<_>>();
            let steps = sequence.len();
            let mut alpha = vec![vec![0.0; k]; steps];
            for t in 0..steps {
                for j in 0..k {
                    let prior = if t == 0 {
                        self.initial[j]
                    } else {
                        (0..k)
                            .map(|i| alpha[t - 1][i] * self.transition[i][j])
                            .sum()
                    };
                    alpha[t][j] = prior * emissions[t][j];
                }
                normalize(&mut alpha[t]);
            }
            let mut beta = vec![vec![1.0; k]; steps];
            for t in (0..steps.saturating_sub(1)).rev() {
                for i in 0..k {
                    beta[t][i] = (0..k)
                        .map(|j| self.transition[i][j] * emissions[t + 1][j] * beta[t + 1][j])
                        .sum();
                }
                normalize(&mut beta[t]);
            }
            for t in 0..steps {
                let mut gamma = (0..k).map(|i| alpha[t][i] * beta[t][i]).collect::<Vec<_>>();
                normalize(&mut gamma);
                for i in 0..k {
                    if t == 0 {
                        initial[i] += gamma[i];
                    }
                    weight[i] += gamma[i];
                    for d in 0..dimensions {
                        sums[i][d] += gamma[i] * sequence[t][d];
                        squares[i][d] += gamma[i] * sequence[t][d].powi(2);
                    }
                }
                if t + 1 < steps {
                    let mut xi = vec![vec![0.0; k]; k];
                    let mut total = 0.0;
                    for (i, row) in xi.iter_mut().enumerate() {
                        for (j, cell) in row.iter_mut().enumerate() {
                            *cell = alpha[t][i]
                                * self.transition[i][j]
                                * emissions[t + 1][j]
                                * beta[t + 1][j];
                            total += *cell;
                        }
                    }
                    if total > 0.0 {
                        for i in 0..k {
                            for j in 0..k {
                                transitions[i][j] += xi[i][j] / total;
                            }
                        }
                    }
                }
            }
        }
        normalize(&mut initial);
        for row in &mut transitions {
            normalize(row);
        }
        let means = (0..k)
            .map(|i| {
                (0..dimensions)
                    .map(|d| {
                        if weight[i] > 0.0 {
                            sums[i][d] / weight[i]
                        } else {
                            self.means[i][d]
                        }
                    })
                    .collect()
            })
            .collect::<Vec<Vec<f64>>>();
        let variances = (0..k)
            .map(|i| {
                (0..dimensions)
                    .map(|d| {
                        if weight[i] > 0.0 {
                            (squares[i][d] / weight[i] - means[i][d].powi(2)).max(VARIANCE_FLOOR)
                        } else {
                            self.variances[i][d]
                        }
                    })
                    .collect()
            })
            .collect();
        Self {
            initial,
            transition: transitions,
            means,
            variances,
        }
    }

    /// Reorders states by the mean of the first feature, ascending.
    fn sort_states(&mut self) {
        let mut order = (0..self.states()).collect::<Vec<_>>();
        order.sort_by(|&a, &b| self.means[a][0].total_cmp(&self.means[b][0]));
        self.initial = order.iter().map(|&i| self.initial[i]).collect();
        self.transition = order
            .iter()
            .map(|&i| order.iter().map(|&j| self.transition[i][j]).collect())
            .collect();
        self.means = order.iter().map(|&i| self.means[i].clone()).collect();
        self.variances = order.iter().map(|&i| self.variances[i].clone()).collect();
    }
}

/// Scales `values` to sum to one; a zero vector becomes uniform.
fn normalize(values: &mut [f64]) {
    let total = values.iter().sum::<f64>();
    if total > 0.0 && total.is_finite() {
        for value in values.iter_mut() {
            *value /= total;
        }
    } else {
        let uniform = 1.0 / values.len().max(1) as f64;
        values.fill(uniform);
    }
}

/// Online forward filtering under a fixed [`GaussianHmm`].
#[derive(Clone, Debug)]
pub struct RegimeFilter {
    model: GaussianHmm,
    posterior: Option<Vec<f64>>,
}

impl RegimeFilter {
    /// A filter that has seen no windows.
    pub const fn new(model: GaussianHmm) -> Self {
        Self {
            model,
            posterior: None,
        }
    }

    /// The model being filtered under.
    pub const fn model(&self) -> &GaussianHmm {
        &self.model
    }

    /// Folds in one feature window and returns the posterior over regimes.
    pub fn update(&mut self, x: &[f64]) -> &[f64] {
        let k = self.model.states();
        let prior = match &self.posterior {
            None => self.model.initial.clone(),
            Some(previous) => (0..k)
                .map(|j| {
                    (0..k)
                        .map(|i| previous[i] * self.model.transition[i][j])
                        .sum()
                })
                .collect(),
        };
        let emissions = self.model.emissions(x);
        let mut posterior = prior
            .iter()
            .zip(&emissions)
            .map(|(prior, emission)| prior * emission)
            .collect::<Vec<_>>();
        normalize(&mut posterior);
        self.posterior.insert(posterior).as_slice()
    }

    /// The most probable regime and its probability, once a window has
    /// been seen.
    pub fn most_probable(&self) -> Option<(usize, f64)> {
        self.posterior.as_ref().map(|posterior| {
            posterior.iter().copied().enumerate().fold(
                (0, f64::NEG_INFINITY),
                |best, (state, p)| {
                    if p > best.1 { (state, p) } else { best }
                },
            )
        })
    }
}

/// Two-sided Page–Hinkley test on a scalar stream: alarms when the
/// cumulative deviation from the running mean, less a tolerance `delta`
/// per sample, moves `threshold` away from its extreme. Restarts after an
/// alarm.
#[derive(Clone, Debug)]
pub struct PageHinkley {
    delta: f64,
    threshold: f64,
    count: f64,
    mean: f64,
    up: (f64, f64),
    down: (f64, f64),
}

impl PageHinkley {
    /// A detector with tolerance `delta` and alarm `threshold`.
    pub const fn new(delta: f64, threshold: f64) -> Self {
        Self {
            delta,
            threshold,
            count: 0.0,
            mean: 0.0,
            up: (0.0, 0.0),
            down: (0.0, 0.0),
        }
    }

    /// Adds one sample; `true` when it raises an alarm.
    pub fn push(&mut self, x: f64) -> bool {
        self.count += 1.0;
        self.mean += (x - self.mean) / self.count;
        // (cumulative sum, its running minimum) for increases, and the
        // mirror image for decreases.
        self.up.0 += x - self.mean - self.delta;
        self.up.1 = self.up.1.min(self.up.0);
        self.down.0 += self.mean - x - self.delta;
        self.down.1 = self.down.1.min(self.down.0);
        let alarm =
            self.up.0 - self.up.1 > self.threshold || self.down.0 - self.down.1 > self.threshold;
        if alarm {
            *self = Self::new(self.delta, self.threshold);
        }
        alarm
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Arrivals with `n` gaps alternating around `gap` by ±`jitter`.
    fn arrivals(start: u64, n: usize, gap: u64, jitter: u64) -> Vec<u64> {
        let mut at = start;
        let mut out = vec![at];
        for i in 0..n {
            at += if i % 2 == 0 {
                gap + jitter
            } else {
                gap - jitter
            };
            out.push(at);
        }
        out
    }

    fn two_regimes() -> Vec<u64> {
        let slow = arrivals(0, 160, 400_000, 100_000);
        let fast = arrivals(*slow.last().unwrap() + 20_000, 160, 20_000, 5_000);
        slow.into_iter().chain(fast).collect()
    }

    #[test]
    fn windows_report_log_mean_gap_and_variation() {
        let features = ArrivalWindows::features(&[0, 100, 300, 400, 600], 2);
        assert_eq!(features.len(), 2);
        assert!((features[0][0] - 151_f64.ln()).abs() < 1e-9);
        assert!((features[0][1] - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn baum_welch_separates_two_arrival_regimes_and_learns_they_persist() {
        let model = GaussianHmm::fit(&[ArrivalWindows::features(&two_regimes(), 8)], 2, 30)
            .expect("enough windows");
        // State 0 is the fast regime: smaller log gap.
        assert!((model.means[0][0] - 20_000_f64.ln()).abs() < 0.1);
        assert!((model.means[1][0] - 400_000_f64.ln()).abs() < 0.1);
        // Twenty windows each, one change: both states are sticky.
        for state in 0..2 {
            assert!(model.transition[state][state] > 0.9);
            assert!(model.expected_remaining_windows(state) > 10.0);
        }
    }

    #[test]
    fn the_filter_flips_within_a_window_of_the_change_and_not_before() {
        let model =
            GaussianHmm::fit(&[ArrivalWindows::features(&two_regimes(), 8)], 2, 30).unwrap();
        let mut filter = RegimeFilter::new(model);
        let features = ArrivalWindows::features(&two_regimes(), 8);
        let states = features
            .iter()
            .map(|x| {
                filter.update(x);
                filter.most_probable().unwrap().0
            })
            .collect::<Vec<_>>();
        // 160 slow gaps fill 20 windows; the 21st straddles the change.
        assert!(states[..20].iter().all(|&s| s == 1));
        assert!(states[21..].iter().all(|&s| s == 0));
    }

    #[test]
    fn page_hinkley_alarms_on_a_level_shift_only() {
        let mut detector = PageHinkley::new(0.1, 5.0);
        let quiet = (0..50).filter(|_| detector.push(12.9)).count();
        assert_eq!(quiet, 0);
        let alarm = (0..10).position(|_| detector.push(9.9));
        assert!(alarm.is_some_and(|step| step < 5));
    }
}
