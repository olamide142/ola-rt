//! Deterministic pseudo-random numbers shared by traces and statistics.

/// SplitMix64: a tiny, well-distributed, seedable generator.
pub(crate) struct SplitMix64(u64);

impl SplitMix64 {
    pub(crate) const fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub(crate) fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    /// Uniform in `low..=high`; `low` when the range is empty.
    pub(crate) fn between_u64(&mut self, low: u64, high: u64) -> u64 {
        if high <= low {
            return low;
        }
        let span = high - low;
        if span == u64::MAX {
            return self.next();
        }
        low + self.next() % (span + 1)
    }

    pub(crate) fn between(&mut self, low: usize, high: usize) -> usize {
        let low64 = u64::try_from(low).unwrap_or(u64::MAX);
        let high64 = u64::try_from(high).unwrap_or(u64::MAX);
        usize::try_from(self.between_u64(low64, high64)).unwrap_or(usize::MAX)
    }

    /// True with probability `percent` / 100.
    pub(crate) fn chance(&mut self, percent: u64) -> bool {
        self.next() % 100 < percent
    }
}
