//! Seeded request traces for scheduling comparisons.
//!
//! Every generator is a pure function of its parameters: the same seed gives
//! the same trace on every platform. Arrival gaps are drawn uniformly from
//! `[0, 2 × mean]` with integer arithmetic only, avoiding floating-point
//! functions whose results may differ between platforms.

use crate::random::SplitMix64;

/// One request in a scheduling trace.
#[derive(Clone, Eq, PartialEq)]
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
    /// When set, the prompt starts with a prefix shared by every request
    /// that has one, followed by tokens unique to this request; otherwise
    /// prompts are uniform. Requires `eos_after` to be `None`, because the
    /// end-of-sequence marker is the prompt's first token.
    pub shared_prefix: Option<SharedPrefix>,
}

/// A prompt made of a common prefix and a request-specific remainder.
#[derive(Clone, Copy, Eq, PartialEq)]
pub struct SharedPrefix {
    /// Prompt tokens identical across requests in the same group.
    pub tokens: usize,
    /// Seed for this request's unique remainder.
    pub seed: u64,
    /// Which prefix the request shares; requests in different groups share
    /// no prompt tokens beyond the first.
    pub group: u64,
}

// Written out so group 0 prints, and therefore digests, exactly as it did
// before the field existed.
impl std::fmt::Debug for SharedPrefix {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug = formatter.debug_struct("SharedPrefix");
        debug
            .field("tokens", &self.tokens)
            .field("seed", &self.seed);
        if self.group != 0 {
            debug.field("group", &self.group);
        }
        debug.finish()
    }
}

// Written out so a trace without shared prefixes prints, and therefore
// digests, exactly as it did before the field existed.
impl std::fmt::Debug for TraceRequest {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug = formatter.debug_struct("TraceRequest");
        debug
            .field("arrival_micros", &self.arrival_micros)
            .field("prompt_tokens", &self.prompt_tokens)
            .field("max_new_tokens", &self.max_new_tokens)
            .field("priority", &self.priority)
            .field("deadline_micros", &self.deadline_micros)
            .field("eos_after", &self.eos_after);
        if let Some(shared_prefix) = &self.shared_prefix {
            debug.field("shared_prefix", shared_prefix);
        }
        debug.finish()
    }
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
    /// Reads a trace from JSON Lines in `replay_trace export`'s format, as
    /// `lab/replay/servegen.py` writes it: one object per request with
    /// `arrival_us`, `prompt_tokens` and `max_new_tokens`, and optionally
    /// `priority`, `deadline_us` (relative to arrival), `eos_after` and
    /// `window`. Lines without `arrival_us`, such as a provenance header,
    /// are skipped. Requests are ordered by arrival; the second value holds
    /// each request's `window` label in that order, for scoring by regime.
    ///
    /// # Errors
    ///
    /// Fails on a line that is not JSON or a request missing a required
    /// field.
    pub fn from_json_lines(
        name: &'static str,
        text: &str,
    ) -> Result<(Self, Vec<Option<u64>>), String> {
        let mut rows = Vec::new();
        for (number, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let value: serde_json::Value = serde_json::from_str(line)
                .map_err(|error| format!("line {}: {error}", number + 1))?;
            let Some(arrival_micros) = value.get("arrival_us").and_then(serde_json::Value::as_u64)
            else {
                continue;
            };
            let number_field = |key: &str| value.get(key).and_then(serde_json::Value::as_u64);
            let required = |key: &str| {
                number_field(key)
                    .and_then(|n| usize::try_from(n).ok())
                    .ok_or_else(|| format!("line {}: missing {key}", number + 1))
            };
            let request = TraceRequest {
                arrival_micros,
                prompt_tokens: required("prompt_tokens")?.max(1),
                max_new_tokens: required("max_new_tokens")?.max(1),
                priority: value
                    .get("priority")
                    .and_then(serde_json::Value::as_i64)
                    .and_then(|p| i32::try_from(p).ok())
                    .unwrap_or(0),
                deadline_micros: number_field("deadline_us")
                    .map(|relative| arrival_micros.saturating_add(relative)),
                eos_after: number_field("eos_after").and_then(|n| usize::try_from(n).ok()),
                shared_prefix: None,
            };
            rows.push((request, number_field("window")));
        }
        rows.sort_by_key(|(request, _)| request.arrival_micros);
        let span = rows.last().map_or(0, |(request, _)| request.arrival_micros);
        let count = rows.len();
        let (requests, windows) = rows.into_iter().unzip();
        Ok((
            Self {
                name,
                parameters: TraceParameters {
                    seed: 0,
                    requests: count,
                    mean_interarrival_micros: span / (count as u64).max(1),
                },
                requests,
            },
            windows,
        ))
    }

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

    /// Chat-like requests whose callers allow a generous output limit
    /// (128–256 tokens) that most never reach: three quarters stop after
    /// 8–64 tokens. Worst-case KV reservation strands memory on exactly
    /// this workload.
    pub fn early_stopping(parameters: TraceParameters) -> Self {
        Self::generate("early_stopping", parameters, |random, _| {
            let mut request = shaped(random, 16..=64, 128..=256);
            if random.chance(75) {
                request.eos_after = Some(random.between(8, 64));
            }
            request
        })
    }

    /// The `early_stopping` shape with no early stops: every request runs
    /// to its 128–256-token limit. Reserving less than the worst case only
    /// defers the memory each request will eventually need.
    pub fn run_to_limit(parameters: TraceParameters) -> Self {
        Self::generate("run_to_limit", parameters, |random, _| {
            shaped(random, 16..=64, 128..=256)
        })
    }

    /// `self` followed by `next`, whose arrivals start `gap_micros` after
    /// the last of `self`'s. Request `i` of the result belongs to `self`
    /// exactly when `i < self.requests.len()`.
    #[must_use]
    pub fn followed_by(self, next: Self, name: &'static str, gap_micros: u64) -> Self {
        let offset = self
            .requests
            .last()
            .map_or(0, |request| request.arrival_micros)
            .saturating_add(gap_micros);
        let mut requests = self.requests;
        requests.extend(next.requests.into_iter().map(|request| TraceRequest {
            arrival_micros: request.arrival_micros.saturating_add(offset),
            ..request
        }));
        Self {
            name,
            parameters: TraceParameters {
                requests: requests.len(),
                ..self.parameters
            },
            requests,
        }
    }

    /// Every prompt starts with the same `prefix_tokens` (a system prompt)
    /// followed by 8–32 tokens unique to the request; outputs are 8–32
    /// tokens. No request stops early, so prefixes stay comparable.
    pub fn shared_prefix(parameters: TraceParameters, prefix_tokens: usize) -> Self {
        Self::generate("shared_prefix", parameters, |random, _| {
            let mut request = shaped(random, 8..=32, 8..=32);
            request.prompt_tokens += prefix_tokens;
            request.shared_prefix = Some(SharedPrefix {
                tokens: prefix_tokens,
                seed: random.between_u64(0, u64::MAX - 1),
                group: 0,
            });
            request
        })
    }

    /// Like [`shared_prefix`](Self::shared_prefix), but each request draws
    /// one of `groups` distinct `prefix_tokens`-token prefixes uniformly, as
    /// tenants with different system prompts would. Routing that keeps a
    /// group on one worker lets its prefix cache serve the group.
    pub fn prefix_groups(parameters: TraceParameters, groups: u64, prefix_tokens: usize) -> Self {
        Self::generate("prefix_groups", parameters, |random, _| {
            let mut request = shaped(random, 8..=32, 8..=32);
            request.prompt_tokens += prefix_tokens;
            let group = random.between_u64(0, groups.max(1) - 1);
            request.shared_prefix = Some(SharedPrefix {
                tokens: prefix_tokens,
                seed: random.between_u64(0, u64::MAX - 1),
                group,
            });
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

    /// Medium requests whose deadlines allow 1× to 4× an estimate of their
    /// time running alone; one in ten has a 1 µs deadline and cannot be met.
    ///
    /// The estimate is `(output limit + 1) × iteration_micros`: one
    /// iteration per generated token plus one for the prompt. Pass the
    /// cost model's approximate single-request iteration time, so deadlines
    /// are feasible without load and become hard under it.
    pub fn deadlines(parameters: TraceParameters, iteration_micros: u64) -> Self {
        Self::generate("deadlines", parameters, |random, _| {
            let mut request = shaped(random, 8..=64, 8..=32);
            let solo = u64::try_from(request.max_new_tokens.saturating_add(1))
                .unwrap_or(u64::MAX)
                .saturating_mul(iteration_micros.max(1));
            let slack = if random.chance(10) {
                1
            } else {
                // 2..=8 halves: 1× to 4× the solo estimate.
                solo.saturating_mul(random.between_u64(2, 8)) / 2
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
        shared_prefix: None,
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
            SchedulingTrace::deadlines(PARAMETERS, 500),
            SchedulingTrace::overload(PARAMETERS),
            SchedulingTrace::impossible(PARAMETERS, 1_024),
        ]
    }

    #[test]
    fn json_lines_traces_skip_headers_and_sort_by_arrival() {
        let text = r#"{"servegen": {"seed": 1}}
{"arrival_us": 500, "prompt_tokens": 30, "max_new_tokens": 4, "window": 600}
{"arrival_us": 100, "prompt_tokens": 10, "max_new_tokens": 2, "priority": 5, "deadline_us": 50, "window": 0}
"#;
        let (trace, windows) = SchedulingTrace::from_json_lines("servegen", text).unwrap();
        assert_eq!(windows, [Some(0), Some(600)]);
        assert_eq!(trace.requests.len(), 2);
        let first = &trace.requests[0];
        assert_eq!(
            (
                first.arrival_micros,
                first.prompt_tokens,
                first.max_new_tokens,
                first.priority
            ),
            (100, 10, 2, 5)
        );
        assert_eq!(first.deadline_micros, Some(150));
        assert!(SchedulingTrace::from_json_lines("bad", "{\"arrival_us\": 1}").is_err());
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

        let deadlines = SchedulingTrace::deadlines(PARAMETERS, 500);
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
