//! Seeded request traces for scheduling comparisons.
//!
//! Every generator is a pure function of its parameters: the same seed gives
//! the same trace on every platform. Arrival gaps are drawn uniformly from
//! `[0, 2 × mean]` with integer arithmetic only, avoiding floating-point
//! functions whose results may differ between platforms.

/// One request in a scheduling trace.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TraceRequest {
    /// Virtual arrival time in microseconds.
    pub arrival_micros: u64,
    /// Prompt length in tokens.
    pub prompt_tokens: usize,
    /// Output limit in tokens.
    pub max_new_tokens: usize,
    /// Relative priority; larger is more important.
    pub priority: i32,
    /// Absolute virtual deadline in microseconds, if any.
    pub deadline_micros: Option<u64>,
    /// Generated-token step at which the request emits EOS, if before its
    /// output limit.
    pub eos_after: Option<usize>,
}

/// Parameters shared by every trace generator.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TraceParameters {
    /// Seed for the deterministic generator.
    pub seed: u64,
    /// Number of requests to generate.
    pub requests: usize,
    /// Mean gap between arrivals in microseconds.
    pub mean_interarrival_micros: u64,
}

/// A named, seeded sequence of requests ordered by arrival.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SchedulingTrace {
    /// Stable generator name.
    pub name: &'static str,
    /// Parameters the trace was generated from.
    pub parameters: TraceParameters,
    /// Requests in nondecreasing arrival order.
    pub requests: Vec<TraceRequest>,
}

impl SchedulingTrace {
    /// 80% short requests and 20% long prompts, a quarter of which stop
    /// early. Exercises head-of-line blocking and size-aware scheduling.
    pub fn mixed_short_long(parameters: TraceParameters) -> Self {
        Self::generate("mixed_short_long", parameters, |random, _| {
            let mut request = if random.chance(20) {
                shaped(random, 96..=192, 8..=32)
            } else {
                shaped(random, 4..=16, 4..=16)
            };
            if random.chance(25) {
                request.eos_after = Some(random.between(1, request.max_new_tokens));
            }
            request
        })
    }

    /// A first quarter of short prompts with long outputs (many live
    /// decodes), then long prompts with short outputs competing with them.
    pub fn prefill_decode_competition(parameters: TraceParameters) -> Self {
        let decoders = parameters.requests / 4;
        Self::generate("prefill_decode_competition", parameters, |random, index| {
            if index < decoders {
                shaped(random, 4..=8, 48..=96)
            } else {
                shaped(random, 128..=256, 2..=4)
            }
        })
    }

    /// Medium requests where one in five has priority 5 and the rest 0.
    pub fn priority_classes(parameters: TraceParameters) -> Self {
        Self::generate("priority_classes", parameters, |random, _| {
            let mut request = shaped(random, 8..=64, 8..=32);
            if random.chance(20) {
                request.priority = 5;
            }
            request
        })
    }

    /// Medium requests with deadlines of 2 to 40 mean gaps after arrival;
    /// one in ten has a 1 µs deadline and cannot be met.
    pub fn deadlines(parameters: TraceParameters) -> Self {
        let gap = parameters.mean_interarrival_micros.max(1);
        Self::generate("deadlines", parameters, |random, _| {
            let mut request = shaped(random, 8..=64, 8..=32);
            let slack = if random.chance(10) {
                1
            } else {
                gap.saturating_mul(random.between_u64(2, 40))
            };
            // Arrival is filled in by `generate`; store the slack for now.
            request.deadline_micros = Some(slack);
            request
        })
        .with_relative_deadlines()
    }

    /// The `mixed_short_long` shape arriving eight times faster than the
    /// given mean, so queues grow and starvation becomes visible.
    pub fn overload(parameters: TraceParameters) -> Self {
        let fast = TraceParameters {
            mean_interarrival_micros: (parameters.mean_interarrival_micros / 8).max(1),
            ..parameters
        };
        let mut trace = Self::mixed_short_long(fast);
        trace.name = "overload";
        trace.parameters = parameters;
        trace
    }

    /// Short requests where one in ten needs more KV tokens than a pool of
    /// `capacity_tokens` holds, so admission must reject it.
    pub fn impossible(parameters: TraceParameters, capacity_tokens: usize) -> Self {
        Self::generate("impossible", parameters, |random, _| {
            if random.chance(10) {
                let prompt = capacity_tokens.saturating_add(random.between(1, 64));
                TraceRequest {
                    prompt_tokens: prompt,
                    ..shaped(random, 1..=1, 4..=8)
                }
            } else {
                shaped(random, 4..=16, 4..=16)
            }
        })
    }

    fn generate(
        name: &'static str,
        parameters: TraceParameters,
        mut request: impl FnMut(&mut SplitMix64, usize) -> TraceRequest,
    ) -> Self {
        let mut random = SplitMix64::new(parameters.seed);
        let mut arrival = 0_u64;
        let requests = (0..parameters.requests)
            .map(|index| {
                if index > 0 {
                    let gap = parameters.mean_interarrival_micros.saturating_mul(2);
                    arrival = arrival.saturating_add(random.between_u64(0, gap));
                }
                TraceRequest {
                    arrival_micros: arrival,
                    ..request(&mut random, index)
                }
            })
            .collect();
        Self {
            name,
            parameters,
            requests,
        }
    }

    fn with_relative_deadlines(mut self) -> Self {
        for request in &mut self.requests {
            request.deadline_micros = request
                .deadline_micros
                .map(|slack| request.arrival_micros.saturating_add(slack));
        }
        self
    }
}

fn shaped(
    random: &mut SplitMix64,
    prompt: std::ops::RangeInclusive<usize>,
    output: std::ops::RangeInclusive<usize>,
) -> TraceRequest {
    TraceRequest {
        arrival_micros: 0,
        prompt_tokens: random.between(*prompt.start(), *prompt.end()),
        max_new_tokens: random.between(*output.start(), *output.end()),
        priority: 0,
        deadline_micros: None,
        eos_after: None,
    }
}

/// SplitMix64: a tiny, well-distributed, seedable generator.
struct SplitMix64(u64);

impl SplitMix64 {
    const fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    /// Uniform in `low..=high`; `low` when the range is empty.
    fn between_u64(&mut self, low: u64, high: u64) -> u64 {
        if high <= low {
            return low;
        }
        let span = high - low;
        if span == u64::MAX {
            return self.next();
        }
        low + self.next() % (span + 1)
    }

    fn between(&mut self, low: usize, high: usize) -> usize {
        let low64 = u64::try_from(low).unwrap_or(u64::MAX);
        let high64 = u64::try_from(high).unwrap_or(u64::MAX);
        usize::try_from(self.between_u64(low64, high64)).unwrap_or(usize::MAX)
    }

    /// True with probability `percent` / 100.
    fn chance(&mut self, percent: u64) -> bool {
        self.next() % 100 < percent
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PARAMETERS: TraceParameters = TraceParameters {
        seed: 7,
        requests: 200,
        mean_interarrival_micros: 1_000,
    };

    fn all() -> Vec<SchedulingTrace> {
        vec![
            SchedulingTrace::mixed_short_long(PARAMETERS),
            SchedulingTrace::prefill_decode_competition(PARAMETERS),
            SchedulingTrace::priority_classes(PARAMETERS),
            SchedulingTrace::deadlines(PARAMETERS),
            SchedulingTrace::overload(PARAMETERS),
            SchedulingTrace::impossible(PARAMETERS, 1_024),
        ]
    }

    #[test]
    fn traces_are_reproducible_ordered_and_seed_dependent() {
        let reseeded = TraceParameters {
            seed: 8,
            ..PARAMETERS
        };
        for (first, second) in all().into_iter().zip(all()) {
            assert_eq!(first, second, "{} is not reproducible", first.name);
            assert_eq!(first.requests.len(), PARAMETERS.requests);
            assert!(
                first
                    .requests
                    .windows(2)
                    .all(|pair| pair[0].arrival_micros <= pair[1].arrival_micros)
            );
            assert!(
                first
                    .requests
                    .iter()
                    .all(|request| request.prompt_tokens > 0)
            );
        }
        assert_ne!(
            SchedulingTrace::mixed_short_long(PARAMETERS).requests,
            SchedulingTrace::mixed_short_long(reseeded).requests
        );
    }

    #[test]
    fn traces_contain_the_cases_they_are_named_for() {
        let mixed = SchedulingTrace::mixed_short_long(PARAMETERS);
        assert!(
            mixed
                .requests
                .iter()
                .any(|request| request.prompt_tokens >= 96)
        );
        assert!(
            mixed
                .requests
                .iter()
                .any(|request| request.eos_after.is_some())
        );

        let priorities = SchedulingTrace::priority_classes(PARAMETERS);
        assert!(
            priorities
                .requests
                .iter()
                .any(|request| request.priority == 5)
        );

        let deadlines = SchedulingTrace::deadlines(PARAMETERS);
        assert!(deadlines.requests.iter().all(|request| {
            request
                .deadline_micros
                .is_some_and(|deadline| deadline > request.arrival_micros)
        }));
        assert!(
            deadlines
                .requests
                .iter()
                .any(|request| { request.deadline_micros == Some(request.arrival_micros + 1) })
        );

        let impossible = SchedulingTrace::impossible(PARAMETERS, 1_024);
        assert!(
            impossible
                .requests
                .iter()
                .any(|request| request.prompt_tokens > 1_024)
        );

        let overload = SchedulingTrace::overload(PARAMETERS);
        let last = |trace: &SchedulingTrace| trace.requests.last().unwrap().arrival_micros;
        assert!(last(&overload) < last(&mixed) / 4);
    }
}
