//! How quickly and how reliably is a regime change detected from arrivals?
//!
//! Two changes, each as 120 `prefill_decode_competition` requests of one
//! regime then 120 of the other (both orders), with gaps scaled to the cost
//! model: `regime_switching`'s overload and burst (a 20× rate change), and
//! moderate load then overload (2×). Two detectors read the same arrival windows
//! (`ArrivalWindows`, 8 gaps each):
//!
//! - `hmm`: a two-state `GaussianHmm` fitted on held-out seeds (101–130),
//!   filtered online, one model per change. A detection is a change of its most probable state
//!   with at least 90% posterior; its first confident state is not one.
//! - `page-hinkley`: Page–Hinkley on the window's log mean gap
//!   (tolerance 0.1, threshold 2.0 in log units).
//!
//! It reports detection delay after the first request of the new phase, in
//! arrivals and in virtual seconds; alarms before the change and on traces
//! with no change (240 requests of one regime); and, for the HMM, the
//! remaining duration `1/(1 − p)` it predicts at detection against the
//! windows actually left.
//!
//! ```sh
//! OLA_COST=10745,30227,482,97,13047,24568 \
//!   cargo run --release -p ola-bench --example regime_detection
//! ```

use ola_bench::{
    ArrivalWindows, CostModel, DecodeCost, GaussianHmm, PageHinkley, RegimeFilter, SchedulingTrace,
    TraceParameters,
};

const PHASE_REQUESTS: usize = 120;
const WINDOW_ARRIVALS: usize = 8;
const CONFIDENCE: f64 = 0.9;

/// The simulator cost model, overridden by `OLA_COST` when set; the same
/// format `reviewer_eval` reads.
fn cost_model() -> CostModel {
    let Ok(value) = std::env::var("OLA_COST") else {
        return CostModel::default();
    };
    let parts: Vec<u64> = value
        .split(',')
        .map(|x| x.trim().parse().expect("OLA_COST must be integers"))
        .collect();
    let (decode, per_padding) = match parts.len() {
        3 => (None, None),
        6 => (
            Some(DecodeCost {
                per_invocation_micros: parts[4],
                per_row_micros: parts[5],
            }),
            Some(parts[3]),
        ),
        _ => panic!(
            "OLA_COST=fixed,per_invocation,per_dense_token\
             [,per_padding_token,decode_per_invocation,decode_per_row]"
        ),
    };
    CostModel {
        fixed_micros: parts[0],
        per_invocation_micros: parts[1],
        per_dense_token_micros: parts[2],
        per_padding_token_micros: per_padding,
        decode,
    }
}

/// As in `budget_regimes`: nominal gaps scale with the cost model.
fn load_scale(cost: CostModel) -> f64 {
    let reference = |c: CostModel| {
        let invocation = match c.decode {
            Some(decode) => decode.per_invocation_micros + 8 * decode.per_row_micros,
            None => c.per_invocation_micros + 8 * c.per_dense_token_micros,
        };
        (c.fixed_micros + invocation) as f64
    };
    reference(cost) / reference(CostModel::default())
}

fn phase(seed: u64, nominal: u64, requests: usize, scale: f64) -> SchedulingTrace {
    SchedulingTrace::prefill_decode_competition(TraceParameters {
        seed,
        requests,
        mean_interarrival_micros: ((nominal as f64) * scale).round().max(1.0) as u64,
    })
}

/// The two regimes of a comparison, by nominal mean gap (scaled by the
/// cost model): `(name, first, second)`.
type Pair = (&'static str, (&'static str, u64), (&'static str, u64));

const PAIRS: [Pair; 2] = [
    ("20x rate", ("overload", 1_000), ("burst", 50)),
    ("2x rate", ("moderate", 2_000), ("overload", 1_000)),
];

/// One evaluated arrangement of a pair's regimes.
#[derive(Clone, Copy)]
enum Arrangement {
    /// The first regime, then the second.
    Forward,
    /// The second regime, then the first.
    Reverse,
    /// Twice as many requests of the first regime alone.
    FirstOnly,
    /// Twice as many requests of the second regime alone.
    SecondOnly,
}

/// Arrival times for `arrangement` of `pair`, and the index of the first
/// request of the second phase (`None` for a single regime).
fn trace(pair: Pair, arrangement: Arrangement, seed: u64, scale: f64) -> (Vec<u64>, Option<usize>) {
    let gap = |nominal: u64| ((nominal as f64) * scale).round().max(1.0) as u64;
    let (_, (_, first), (_, second)) = pair;
    let a = || phase(seed, first, PHASE_REQUESTS, scale);
    let b = || phase(seed + 1_000, second, PHASE_REQUESTS, scale);
    let (trace, boundary) = match arrangement {
        Arrangement::Forward => (
            a().followed_by(b(), "shifted", gap(second)),
            Some(PHASE_REQUESTS),
        ),
        Arrangement::Reverse => (
            b().followed_by(a(), "shifted", gap(first)),
            Some(PHASE_REQUESTS),
        ),
        Arrangement::FirstOnly => (phase(seed, first, 2 * PHASE_REQUESTS, scale), None),
        Arrangement::SecondOnly => (phase(seed + 1_000, second, 2 * PHASE_REQUESTS, scale), None),
    };
    let arrivals = trace.requests.iter().map(|r| r.arrival_micros).collect();
    (arrivals, boundary)
}

/// Arrival indices at which `detect` fires, given one window's features
/// per completed window.
fn alarms(arrivals: &[u64], mut detect: impl FnMut(&[f64]) -> bool) -> Vec<usize> {
    let mut windows = ArrivalWindows::new(WINDOW_ARRIVALS);
    arrivals
        .iter()
        .enumerate()
        .filter_map(|(index, &arrival)| {
            let features = windows.push(arrival)?;
            detect(&features).then_some(index)
        })
        .collect()
}

#[derive(Default)]
struct Tally {
    runs: usize,
    detected: usize,
    delay_arrivals: Vec<f64>,
    delay_micros: Vec<f64>,
    early_alarms: usize,
    extra_alarms: usize,
    predicted_windows: Vec<f64>,
    actual_windows: Vec<f64>,
}

impl Tally {
    /// Scores one run's alarms against its boundary.
    fn score(&mut self, arrivals: &[u64], boundary: Option<usize>, fired: &[usize]) {
        self.runs += 1;
        let Some(boundary) = boundary else {
            self.early_alarms += fired.len();
            return;
        };
        self.early_alarms += fired.iter().filter(|&&at| at < boundary).count();
        let after = fired
            .iter()
            .filter(|&&at| at >= boundary)
            .collect::<Vec<_>>();
        if let Some(&&first) = after.first() {
            self.detected += 1;
            self.delay_arrivals.push((first - boundary) as f64);
            self.delay_micros
                .push(arrivals[first].saturating_sub(arrivals[boundary]) as f64);
            self.extra_alarms += after.len() - 1;
        }
    }
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len().max(1) as f64
}

fn main() {
    let scale = load_scale(cost_model());
    let seeds = std::env::var("SEEDS").map_or(20, |v| v.parse().expect("SEEDS"));
    println!(
        "| change | trace | detector | detected | delay (arrivals) | delay (s) | \
         alarms before change or with none | extra alarms after | \
         predicted / actual remaining windows |"
    );
    println!("|---|---|---|---:|---:|---:|---:|---:|---|");
    let mut models = Vec::new();
    for pair in PAIRS {
        let (change, (first, _), (second, _)) = pair;
        // Fitted on held-out seeds of both orders, as `regime_switching` does.
        let training = [Arrangement::Forward, Arrangement::Reverse]
            .into_iter()
            .flat_map(|arrangement| {
                (101..=130).map(move |seed| trace(pair, arrangement, seed, scale).0)
            })
            .map(|arrivals| ArrivalWindows::features(&arrivals, WINDOW_ARRIVALS))
            .collect::<Vec<_>>();
        let model = GaussianHmm::fit(&training, 2, 50).expect("training windows");
        for (arrangement, label) in [
            (Arrangement::Forward, format!("{first}>{second}")),
            (Arrangement::Reverse, format!("{second}>{first}")),
            (Arrangement::FirstOnly, format!("{first} only")),
            (Arrangement::SecondOnly, format!("{second} only")),
        ] {
            let (mut hmm, mut ph) = (Tally::default(), Tally::default());
            for seed in 1..=seeds {
                let (arrivals, boundary) = trace(pair, arrangement, seed, scale);
                let mut filter = RegimeFilter::new(model.clone());
                let mut regime = None;
                let mut predicted = Vec::new();
                let fired = alarms(&arrivals, |x| {
                    filter.update(x);
                    match filter.most_probable() {
                        Some((state, p)) if p >= CONFIDENCE && regime != Some(state) => {
                            let first = regime.is_none();
                            regime = Some(state);
                            predicted.push(model.expected_remaining_windows(state));
                            !first
                        }
                        _ => false,
                    }
                });
                hmm.score(&arrivals, boundary, &fired);
                if let Some(boundary) = boundary
                    && let Some(&first) = fired.iter().find(|&&at| at >= boundary)
                {
                    hmm.predicted_windows.push(*predicted.last().unwrap());
                    let left = (arrivals.len() - 1 - first) / WINDOW_ARRIVALS;
                    hmm.actual_windows.push(left as f64);
                }
                let mut detector = PageHinkley::new(0.1, 2.0);
                let fired = alarms(&arrivals, |x| detector.push(x[0]));
                ph.score(&arrivals, boundary, &fired);
            }
            for (name, tally) in [("hmm", &hmm), ("page-hinkley", &ph)] {
                let shifted = boundary_expected(arrangement);
                let or_dash = |value: String| {
                    if tally.detected > 0 {
                        value
                    } else {
                        "—".into()
                    }
                };
                let duration = if tally.predicted_windows.is_empty() {
                    "—".to_owned()
                } else {
                    format!(
                        "{:.1} / {:.1}",
                        mean(&tally.predicted_windows),
                        mean(&tally.actual_windows)
                    )
                };
                println!(
                    "| {change} | {label} | {name} | {} | {} | {} | {} | {} | {duration} |",
                    if shifted {
                        format!("{}/{}", tally.detected, tally.runs)
                    } else {
                        "—".into()
                    },
                    or_dash(format!("{:.1}", mean(&tally.delay_arrivals))),
                    or_dash(format!("{:.2}", mean(&tally.delay_micros) / 1e6)),
                    tally.early_alarms,
                    tally.extra_alarms,
                );
            }
        }
        models.push((change, model));
    }
    println!();
    for (change, model) in models {
        println!(
            "{change} model (state 0 = faster arrivals): log-gap means {:.2} / {:.2}, \
             CV means {:.2} / {:.2}, stay {:.3} / {:.3}, expected duration {:.1} / {:.1} windows",
            model.means[0][0],
            model.means[1][0],
            model.means[0][1],
            model.means[1][1],
            model.transition[0][0],
            model.transition[1][1],
            model.expected_remaining_windows(0),
            model.expected_remaining_windows(1),
        );
    }
}

/// Whether `arrangement` contains a change to detect.
const fn boundary_expected(arrangement: Arrangement) -> bool {
    matches!(arrangement, Arrangement::Forward | Arrangement::Reverse)
}
