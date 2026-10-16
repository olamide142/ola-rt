use std::{
    fmt,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

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

/// Source of OlaRT timestamps for a live runtime.
///
/// A clock is monotonic: successive readings never decrease. Long-running
/// drivers read it at every lifecycle boundary, so tests substitute a
/// [`ManualClock`] to make timing deterministic.
pub trait Clock: Send + Sync {
    /// Returns the current time in the clock's epoch.
    fn now(&self) -> Timestamp;
}

/// Process monotonic clock whose epoch is the moment it was created.
#[derive(Clone, Copy, Debug)]
pub struct MonotonicClock {
    epoch: Instant,
}

impl MonotonicClock {
    /// Starts a clock at [`Timestamp::ZERO`].
    pub fn start() -> Self {
        Self {
            epoch: Instant::now(),
        }
    }

    /// Returns the instant that maps to [`Timestamp::ZERO`].
    pub const fn epoch(&self) -> Instant {
        self.epoch
    }

    /// Translates an instant into this clock's epoch, clamping instants
    /// before the epoch to zero and saturating after about 584,000 years.
    pub fn timestamp_of(&self, instant: Instant) -> Timestamp {
        let micros = instant.saturating_duration_since(self.epoch).as_micros();
        Timestamp::from_micros(u64::try_from(micros).unwrap_or(u64::MAX))
    }
}

impl Clock for MonotonicClock {
    fn now(&self) -> Timestamp {
        self.timestamp_of(Instant::now())
    }
}

/// Clock advanced explicitly by its owner.
///
/// Clones share one reading, so a test can hold one clone while a driver
/// thread reads another.
#[derive(Clone, Debug, Default)]
pub struct ManualClock {
    micros: Arc<AtomicU64>,
}

impl ManualClock {
    /// Starts a manual clock at `at`.
    pub fn new(at: Timestamp) -> Self {
        Self {
            micros: Arc::new(AtomicU64::new(at.as_micros())),
        }
    }

    /// Moves the clock forward by `duration`, saturating at the maximum
    /// timestamp, and returns the new reading.
    pub fn advance(&self, duration: Duration) -> Timestamp {
        let step = u64::try_from(duration.as_micros()).unwrap_or(u64::MAX);
        let previous = self
            .micros
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |micros| {
                Some(micros.saturating_add(step))
            })
            .unwrap_or_else(|micros| micros);
        Timestamp::from_micros(previous.saturating_add(step))
    }

    /// Moves the clock to `at`; readings never move backwards, so an earlier
    /// `at` leaves the clock unchanged.
    pub fn set(&self, at: Timestamp) {
        self.micros.fetch_max(at.as_micros(), Ordering::SeqCst);
    }
}

impl Clock for ManualClock {
    fn now(&self) -> Timestamp {
        Timestamp::from_micros(self.micros.load(Ordering::SeqCst))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monotonic_clock_translates_instants_relative_to_its_epoch() {
        let clock = MonotonicClock::start();
        let later = clock.epoch() + Duration::from_micros(1_500);

        assert_eq!(clock.timestamp_of(later), Timestamp::from_micros(1_500));
        assert_eq!(clock.timestamp_of(clock.epoch()), Timestamp::ZERO);
        let first = clock.now();
        assert!(clock.now() >= first);
    }

    #[test]
    fn manual_clock_clones_share_one_monotonic_reading() {
        let clock = ManualClock::new(Timestamp::from_micros(10));
        let reader = clock.clone();

        assert_eq!(
            clock.advance(Duration::from_micros(5)),
            Timestamp::from_micros(15)
        );
        assert_eq!(reader.now(), Timestamp::from_micros(15));
        clock.set(Timestamp::from_micros(12));
        assert_eq!(reader.now(), Timestamp::from_micros(15));
        clock.set(Timestamp::from_micros(40));
        assert_eq!(reader.now(), Timestamp::from_micros(40));
    }
}
