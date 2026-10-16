//! Seeded, reproducible uncertainty estimates.

use crate::random::SplitMix64;

use super::Interval;

/// Bootstrap resamples per interval.
pub const BOOTSTRAP_RESAMPLES: usize = 2_000;

/// Returns a 95% percentile-bootstrap interval for the mean of `values`, or
/// `None` with fewer than two values, since one run shows no variation.
///
/// The resampling seed is derived from the values themselves, so the same
/// samples always give the same interval and a recomputed report matches
/// its stored one exactly.
pub(crate) fn bootstrap_mean_interval(values: &[f64]) -> Option<Interval> {
    if values.len() < 2 {
        return None;
    }
    let seed = values
        .iter()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, value| {
            value
                .to_bits()
                .to_le_bytes()
                .iter()
                .fold(hash, |hash, byte| {
                    (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
                })
        });
    let mut random = SplitMix64::new(seed);
    let count = values.len();
    let last = u64::try_from(count - 1).unwrap_or(u64::MAX);
    let mut means = (0..BOOTSTRAP_RESAMPLES)
        .map(|_| {
            let total = (0..count)
                .map(|_| {
                    let index = usize::try_from(random.between_u64(0, last)).unwrap_or(0);
                    values[index]
                })
                .sum::<f64>();
            total / as_f64(count)
        })
        .collect::<Vec<_>>();
    means.sort_by(f64::total_cmp);
    // Nearest-rank 2.5th and 97.5th percentiles.
    let rank = |per_mille: usize| {
        let index = (BOOTSTRAP_RESAMPLES * per_mille)
            .div_ceil(1_000)
            .saturating_sub(1);
        means[index.min(BOOTSTRAP_RESAMPLES - 1)]
    };
    Some(Interval {
        low: rank(25),
        high: rank(975),
    })
}

#[allow(clippy::cast_precision_loss, reason = "sample counts are tiny")]
pub(crate) fn as_f64(value: usize) -> f64 {
    value as f64
}
