//! Inter-token latency from per-token timestamps.

use ola_core::Timestamp;

/// Distribution of gaps between consecutive tokens.
///
/// A gap is the time between two consecutive token timestamps of one
/// request, so the first token (prefill) contributes no gap and `n` tokens
/// give `n - 1` gaps; this is the time per output token (TPOT) after the
/// first. Percentiles use the nearest-rank method on integer microseconds.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TokenGaps {
    /// Number of gaps summarized.
    pub gaps: usize,
    /// Mean gap, rounded down.
    pub mean_micros: u64,
    /// Median gap.
    pub p50_micros: u64,
    /// 90th-percentile gap.
    pub p90_micros: u64,
    /// 99th-percentile gap.
    pub p99_micros: u64,
    /// Longest gap.
    pub max_micros: u64,
}

impl TokenGaps {
    /// Summarizes one request's token times; `None` with fewer than two.
    ///
    /// Times are expected in emission order; a time earlier than its
    /// predecessor counts as a zero gap rather than wrapping.
    pub fn of(times: &[Timestamp]) -> Option<Self> {
        Self::summarize(gaps(times).collect())
    }

    /// Summarizes the gaps of several requests together. Gaps never span two
    /// requests. `None` when no request has two tokens.
    pub fn pooled<'a>(requests: impl IntoIterator<Item = &'a [Timestamp]>) -> Option<Self> {
        Self::summarize(requests.into_iter().flat_map(gaps).collect())
    }

    fn summarize(mut gaps: Vec<u64>) -> Option<Self> {
        if gaps.is_empty() {
            return None;
        }
        gaps.sort_unstable();
        let total = gaps
            .iter()
            .fold(0_u128, |total, gap| total + u128::from(*gap));
        let count = gaps.len();
        Some(Self {
            gaps: count,
            mean_micros: u64::try_from(total / count as u128).unwrap_or(u64::MAX),
            p50_micros: nearest_rank(&gaps, 50),
            p90_micros: nearest_rank(&gaps, 90),
            p99_micros: nearest_rank(&gaps, 99),
            max_micros: gaps[count - 1],
        })
    }
}

fn gaps(times: &[Timestamp]) -> impl Iterator<Item = u64> + '_ {
    times
        .windows(2)
        .map(|pair| pair[1].as_micros().saturating_sub(pair[0].as_micros()))
}

/// The smallest value with at least `percent`% of samples at or below it.
fn nearest_rank(sorted: &[u64], percent: usize) -> u64 {
    let rank = (percent * sorted.len()).div_ceil(100).max(1);
    sorted[rank - 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn times(micros: &[u64]) -> Vec<Timestamp> {
        micros.iter().copied().map(Timestamp::from_micros).collect()
    }

    #[test]
    fn summarizes_every_gap_after_the_first_token() {
        // Gaps 10, 20, 30, 40, then a 400 µs stall.
        let gaps = TokenGaps::of(&times(&[100, 110, 130, 160, 200, 600])).unwrap();

        assert_eq!(gaps.gaps, 5);
        assert_eq!(gaps.mean_micros, 100);
        assert_eq!(gaps.p50_micros, 30);
        assert_eq!(gaps.p90_micros, 400);
        assert_eq!(gaps.max_micros, 400);
        assert_eq!(TokenGaps::of(&times(&[5])), None);
        assert_eq!(TokenGaps::of(&[]), None);
    }

    #[test]
    fn a_stall_shows_in_the_tail_but_not_the_median() {
        let mut steady = (0..100).map(|step| step * 10).collect::<Vec<_>>();
        steady.push(990 + 5_000);
        let gaps = TokenGaps::of(&times(&steady)).unwrap();

        assert_eq!(gaps.p50_micros, 10);
        assert_eq!(gaps.p99_micros, 10);
        assert_eq!(gaps.max_micros, 5_000);
        // An average over the decode span hides the stall as ~60 µs.
        assert_eq!(gaps.mean_micros, 59);
    }

    #[test]
    fn pooling_never_bridges_requests() {
        let first = times(&[0, 10, 20]);
        let second = times(&[1_000, 1_050]);
        let pooled = TokenGaps::pooled([first.as_slice(), second.as_slice(), &[]]).unwrap();

        assert_eq!(pooled.gaps, 3);
        assert_eq!(pooled.max_micros, 50);
        assert_eq!(TokenGaps::pooled([&times(&[3])[..]]), None);
    }
}
