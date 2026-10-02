use std::{fmt, time::Duration};

/// Monotonic time relative to the start of a runtime or workload replay.
///
/// Keeping wall-clock time out of the core API makes experiments deterministic.
#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub struct Timestamp(u64);

impl Timestamp {
    /// The start of a runtime epoch or workload trace.
    pub const ZERO: Self = Self(0);

    /// Constructs a timestamp from microseconds since the caller-defined epoch.
    pub const fn from_micros(micros: u64) -> Self {
        Self(micros)
    }

    /// Returns microseconds since the caller-defined epoch.
    pub const fn as_micros(self) -> u64 {
        self.0
    }

    /// Adds a duration, returning `None` when microsecond conversion or addition
    /// would overflow.
    pub fn checked_add(self, duration: Duration) -> Option<Self> {
        let micros = u64::try_from(duration.as_micros()).ok()?;
        self.0.checked_add(micros).map(Self)
    }

    /// Returns elapsed time, clamping to zero when `earlier` is later than this
    /// timestamp.
    pub const fn saturating_duration_since(self, earlier: Self) -> Duration {
        Duration::from_micros(self.0.saturating_sub(earlier.0))
    }
}

impl fmt::Display for Timestamp {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}us", self.0)
    }
}
