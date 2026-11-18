use std::{fmt, sync::Arc};

use ola_core::{ModelId, RequestId};
use ola_model::KvCacheProfile;

use crate::{MemoryState, OutputLengthBelief, PolicyDescriptor};

/// Read-only inputs supplied to an executor admission policy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdmissionContext {
    /// Scheduler-selected queued request.
    pub request_id: RequestId,
    /// Model whose shared KV pool would own the request cache.
    pub model_id: ModelId,
    /// Tokenized prompt length.
    pub prompt_tokens: usize,
    /// Leading prompt tokens the model's prefix cache already holds.
    pub cached_prefix_tokens: usize,
    /// Maximum number of generated tokens requested by the caller.
    pub max_new_tokens: usize,
    /// Architecture-provided physical KV layout, when available.
    pub kv_cache: Option<KvCacheProfile>,
    /// Device accounting currently visible to runtime policies.
    pub memory: MemoryState,
    /// Worst-case KV bytes reserved by every live request.
    pub reserved_kv_bytes: u64,
    /// Worst-case KV bytes reserved by live requests for this model.
    pub model_reserved_kv_bytes: u64,
}

/// Typed reason that a queued request remained deferred.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AdmissionReason {
    /// The model did not publish enough cache geometry to estimate demand.
    MissingKvCacheProfile,
    /// Prompt plus output limit could not be represented safely.
    EstimateOverflow,
    /// One request can never fit in the model's complete physical KV pool.
    RequestExceedsModelKvCapacity {
        /// Page-rounded bytes required by the request.
        required_bytes: u64,
        /// Total physical capacity of the model pool.
        capacity_bytes: u64,
    },
    /// The model's shared physical KV pool cannot satisfy the reservation.
    ModelKvCapacity {
        /// Page-rounded bytes required by the request.
        required_bytes: u64,
        /// Unreserved bytes remaining in the model pool.
        available_bytes: u64,
    },
    /// The configured device budget cannot satisfy the reservation.
    DeviceMemoryCapacity {
        /// Page-rounded bytes required by the request.
        required_bytes: u64,
        /// Unreserved device bytes after non-KV accounting.
        available_bytes: u64,
    },
    /// The model's pool has too few free blocks for the request's first
    /// prefill after this iteration's live growth.
    KvPoolFull {
        /// Blocks the first prefill would allocate.
        needed_blocks: usize,
        /// Free blocks left after live growth and earlier admissions.
        available_blocks: usize,
    },
    /// One request cannot fit in the configured device budget even alone.
    RequestExceedsDeviceMemory {
        /// Page-rounded bytes required by the request.
        required_bytes: u64,
        /// Device bytes available after non-KV accounting.
        capacity_bytes: u64,
    },
}

impl fmt::Display for AdmissionReason {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingKvCacheProfile => {
                formatter.write_str("model has no KV-cache resource profile")
            }
            Self::EstimateOverflow => formatter.write_str("KV-cache demand estimate overflowed"),
            Self::RequestExceedsModelKvCapacity {
                required_bytes,
                capacity_bytes,
            } => write!(
                formatter,
                "request needs {required_bytes} KV bytes but the model pool holds only {capacity_bytes}"
            ),
            Self::ModelKvCapacity {
                required_bytes,
                available_bytes,
            } => write!(
                formatter,
                "model KV pool needs {required_bytes} bytes but only {available_bytes} are unreserved"
            ),
            Self::DeviceMemoryCapacity {
                required_bytes,
                available_bytes,
            } => write!(
                formatter,
                "device budget needs {required_bytes} KV bytes but only {available_bytes} are unreserved"
            ),
            Self::KvPoolFull {
                needed_blocks,
                available_blocks,
            } => write!(
                formatter,
                "first prefill needs {needed_blocks} KV blocks but only {available_blocks} are free"
            ),
            Self::RequestExceedsDeviceMemory {
                required_bytes,
                capacity_bytes,
            } => write!(
                formatter,
                "request needs {required_bytes} KV bytes but the device budget permits only {capacity_bytes}"
            ),
        }
    }
}

/// Outcome returned by an [`AdmissionPolicy`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AdmissionDecision {
    /// Start the request and hold this conservative KV reservation while live.
    Admit {
        /// Page-rounded worst-case KV bytes for the request.
        reserved_kv_bytes: u64,
    },
    /// Leave the request queued without changing its lifecycle state.
    Defer(AdmissionReason),
    /// Mark a request terminal because it cannot ever fit this configuration.
    Reject(AdmissionReason),
}

/// Decides whether a scheduler-selected request may start now.
pub trait AdmissionPolicy: Send {
    /// Evaluates one queued request against current observable resources.
    fn decide(&self, context: &AdmissionContext) -> AdmissionDecision;

    /// Returns the policy's stable name and configuration.
    fn descriptor(&self) -> PolicyDescriptor;

    /// Whether admitted reservations never exceed a model's KV pool.
    ///
    /// When true, plan validation rejects admissions that would exceed the
    /// pool. Permissive policies leave allocation failures to execution.
    fn enforces_model_kv_capacity(&self) -> bool {
        false
    }
}

/// Compatibility policy that never defers scheduler-selected work.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnconstrainedAdmission;

impl AdmissionPolicy for UnconstrainedAdmission {
    fn descriptor(&self) -> PolicyDescriptor {
        PolicyDescriptor::new("unconstrained-admission")
    }

    fn decide(&self, context: &AdmissionContext) -> AdmissionDecision {
        AdmissionDecision::Admit {
            reserved_kv_bytes: estimate_kv_bytes(context).unwrap_or(0),
        }
    }
}

/// Strict FIFO baseline using page-rounded worst-case KV reservations.
#[derive(Clone, Copy, Debug, Default)]
pub struct MemoryAwareAdmission;

impl AdmissionPolicy for MemoryAwareAdmission {
    fn descriptor(&self) -> PolicyDescriptor {
        PolicyDescriptor::new("memory-aware-admission")
    }

    fn enforces_model_kv_capacity(&self) -> bool {
        true
    }

    fn decide(&self, context: &AdmissionContext) -> AdmissionDecision {
        let Some(profile) = context.kv_cache else {
            return AdmissionDecision::Reject(AdmissionReason::MissingKvCacheProfile);
        };
        let Some(required_bytes) = estimate_kv_bytes(context) else {
            return AdmissionDecision::Reject(AdmissionReason::EstimateOverflow);
        };

        if required_bytes > profile.capacity_bytes() {
            return AdmissionDecision::Reject(AdmissionReason::RequestExceedsModelKvCapacity {
                required_bytes,
                capacity_bytes: profile.capacity_bytes(),
            });
        }

        let model_available = profile
            .capacity_bytes()
            .saturating_sub(context.model_reserved_kv_bytes);
        if required_bytes > model_available {
            return AdmissionDecision::Defer(AdmissionReason::ModelKvCapacity {
                required_bytes,
                available_bytes: model_available,
            });
        }

        // Zero means the caller has not configured a device-wide budget. The
        // model pool remains enforced and is already a hard physical bound.
        if context.memory.capacity_bytes > 0 {
            let non_kv_bytes = context
                .memory
                .weights_bytes
                .saturating_add(context.memory.activations_bytes)
                .saturating_add(context.memory.workspace_bytes);
            let device_kv_capacity = context.memory.capacity_bytes.saturating_sub(non_kv_bytes);
            if required_bytes > device_kv_capacity {
                return AdmissionDecision::Reject(AdmissionReason::RequestExceedsDeviceMemory {
                    required_bytes,
                    capacity_bytes: device_kv_capacity,
                });
            }
            let device_available = device_kv_capacity.saturating_sub(context.reserved_kv_bytes);
            if required_bytes > device_available {
                return AdmissionDecision::Defer(AdmissionReason::DeviceMemoryCapacity {
                    required_bytes,
                    available_bytes: device_available,
                });
            }
        }

        AdmissionDecision::Admit {
            reserved_kv_bytes: required_bytes,
        }
    }
}

/// Admits on the prompt plus a fixed headroom instead of the worst case.
///
/// Reservations cover only `prompt + headroom_tokens` positions, so the
/// model pool is deliberately over-committed: requests that end early never
/// held memory for tokens they did not produce. When live requests outgrow
/// their reservations, the pool runs short, and the planner must preempt.
/// Use it only with [`ComposedPolicy::with_preemption`](crate::ComposedPolicy::with_preemption),
/// which also checks each admission against the pool's free blocks.
///
/// A request whose worst case could never fit the pool is still rejected,
/// since no amount of preemption would let it finish. The device-wide
/// budget is not consulted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WatermarkAdmission {
    /// Positions reserved beyond the prompt.
    pub headroom_tokens: usize,
}

impl AdmissionPolicy for WatermarkAdmission {
    fn descriptor(&self) -> PolicyDescriptor {
        PolicyDescriptor::new("watermark-admission")
            .with_config("headroom_tokens", self.headroom_tokens.to_string())
    }

    fn enforces_model_kv_capacity(&self) -> bool {
        true
    }

    fn decide(&self, context: &AdmissionContext) -> AdmissionDecision {
        let Some(profile) = context.kv_cache else {
            return AdmissionDecision::Reject(AdmissionReason::MissingKvCacheProfile);
        };
        let Some(worst_case_bytes) = estimate_kv_bytes(context) else {
            return AdmissionDecision::Reject(AdmissionReason::EstimateOverflow);
        };
        if worst_case_bytes > profile.capacity_bytes() {
            return AdmissionDecision::Reject(AdmissionReason::RequestExceedsModelKvCapacity {
                required_bytes: worst_case_bytes,
                capacity_bytes: profile.capacity_bytes(),
            });
        }
        let Some(required_bytes) = context
            .prompt_tokens
            .checked_add(self.headroom_tokens)
            .and_then(|tokens| profile.estimate_bytes(tokens))
            .map(|bytes| bytes.min(worst_case_bytes))
        else {
            return AdmissionDecision::Reject(AdmissionReason::EstimateOverflow);
        };
        let model_available = profile
            .capacity_bytes()
            .saturating_sub(context.model_reserved_kv_bytes);
        if required_bytes > model_available {
            return AdmissionDecision::Defer(AdmissionReason::ModelKvCapacity {
                required_bytes,
                available_bytes: model_available,
            });
        }
        AdmissionDecision::Admit {
            reserved_kv_bytes: required_bytes,
        }
    }
}

/// Reserves the prompt plus a quantile of the request's remaining-output
/// belief, and the worst case when there is no belief.
///
/// With a usable [`OutputLengthBelief`] it behaves exactly like
/// [`WatermarkAdmission`] with `headroom = q_p(R) − 1`, where `q_p(R)` is the
/// `quantile` of the remaining output (the last token never enters the
/// cache): the pool is over-committed, so it must run with
/// [`ComposedPolicy::with_preemption`](crate::ComposedPolicy::with_preemption),
/// which checks free blocks and preempts when the bet fails. Without a
/// belief (cold start, too few samples) it is [`MemoryAwareAdmission`].
///
/// The belief only sizes reservations. Requests whose worst case can never
/// fit are still rejected, plan validation still bounds the sum of
/// reservations by the pool, and the physical block check and allocation
/// recovery are unchanged. The quantile is not a calibrated guarantee.
#[derive(Clone, Debug)]
pub struct QuantileAdmission {
    belief: Arc<OutputLengthBelief>,
    quantile: f64,
}

impl QuantileAdmission {
    /// Admits with the `quantile` of each request's remaining output;
    /// quantiles outside `(0, 1)` reserve the worst case.
    pub fn new(belief: Arc<OutputLengthBelief>, quantile: f64) -> Self {
        Self { belief, quantile }
    }

    /// Headroom beyond the prompt this policy would reserve, or `None` when
    /// it falls back to the worst case.
    pub fn headroom_tokens(&self, context: &AdmissionContext) -> Option<usize> {
        if !(self.quantile > 0.0 && self.quantile < 1.0) {
            return None;
        }
        let remaining = self
            .belief
            .remaining_for(
                context.request_id,
                &context.model_id,
                context.max_new_tokens,
            )
            .ok()?
            .quantile(self.quantile);
        Some(remaining.saturating_sub(1))
    }
}

impl AdmissionPolicy for QuantileAdmission {
    fn descriptor(&self) -> PolicyDescriptor {
        PolicyDescriptor::new("quantile-admission")
            .with_config("quantile", self.quantile)
            .with_config("samples", self.belief.recorded_samples())
    }

    fn enforces_model_kv_capacity(&self) -> bool {
        true
    }

    fn decide(&self, context: &AdmissionContext) -> AdmissionDecision {
        match self.headroom_tokens(context) {
            Some(headroom_tokens) => WatermarkAdmission { headroom_tokens }.decide(context),
            None => MemoryAwareAdmission.decide(context),
        }
    }
}

fn estimate_kv_bytes(context: &AdmissionContext) -> Option<u64> {
    let profile = context.kv_cache?;
    // The first selected output token is not a decode input yet, so at most
    // max_new_tokens - 1 generated positions enter the cache.
    let cache_tokens = context
        .prompt_tokens
        .checked_add(context.max_new_tokens.saturating_sub(1))?;
    profile.estimate_bytes(cache_tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context() -> AdmissionContext {
        AdmissionContext {
            request_id: RequestId::new(1),
            model_id: "tiny".into(),
            prompt_tokens: 17,
            cached_prefix_tokens: 0,
            max_new_tokens: 16,
            kv_cache: KvCacheProfile::new(8, 16, 512),
            memory: MemoryState::default(),
            reserved_kv_bytes: 0,
            model_reserved_kv_bytes: 0,
        }
    }

    #[test]
    fn rounds_worst_case_demand_to_physical_pages() {
        let decision = MemoryAwareAdmission.decide(&context());

        assert_eq!(
            decision,
            AdmissionDecision::Admit {
                reserved_kv_bytes: 256
            }
        );
    }

    #[test]
    fn enforces_model_pool_and_device_budgets_independently() {
        let mut model_limited = context();
        model_limited.model_reserved_kv_bytes = 384;
        assert_eq!(
            MemoryAwareAdmission.decide(&model_limited),
            AdmissionDecision::Defer(AdmissionReason::ModelKvCapacity {
                required_bytes: 256,
                available_bytes: 128,
            })
        );

        let mut device_limited = context();
        device_limited.memory.capacity_bytes = 1_024;
        device_limited.memory.weights_bytes = 700;
        device_limited.reserved_kv_bytes = 128;
        assert_eq!(
            MemoryAwareAdmission.decide(&device_limited),
            AdmissionDecision::Defer(AdmissionReason::DeviceMemoryCapacity {
                required_bytes: 256,
                available_bytes: 196,
            })
        );
    }

    #[test]
    fn rejects_requests_that_cannot_fit_even_without_competition() {
        let mut too_large = context();
        too_large.max_new_tokens = 128;

        assert_eq!(
            MemoryAwareAdmission.decide(&too_large),
            AdmissionDecision::Reject(AdmissionReason::RequestExceedsModelKvCapacity {
                required_bytes: 1_152,
                capacity_bytes: 512,
            })
        );
    }

    #[test]
    fn watermark_reserves_the_prompt_plus_headroom_but_rejects_what_can_never_fit() {
        let watermark = WatermarkAdmission { headroom_tokens: 4 };
        // Four 16-token pages: a 10-token prompt allowed 40 tokens is 49
        // positions at worst (four pages) but 14 with headroom (one page).
        let context = AdmissionContext {
            prompt_tokens: 10,
            max_new_tokens: 40,
            ..context()
        };
        assert_eq!(
            watermark.decide(&context),
            AdmissionDecision::Admit {
                reserved_kv_bytes: 128,
            }
        );
        assert_eq!(
            MemoryAwareAdmission.decide(&context),
            AdmissionDecision::Admit {
                reserved_kv_bytes: 512,
            }
        );

        let crowded = AdmissionContext {
            model_reserved_kv_bytes: 448,
            ..context.clone()
        };
        assert_eq!(
            watermark.decide(&crowded),
            AdmissionDecision::Defer(AdmissionReason::ModelKvCapacity {
                required_bytes: 128,
                available_bytes: 64,
            })
        );
        let impossible = AdmissionContext {
            max_new_tokens: 60,
            ..context
        };
        assert!(matches!(
            watermark.decide(&impossible),
            AdmissionDecision::Reject(AdmissionReason::RequestExceedsModelKvCapacity { .. })
        ));
    }

    fn learned_belief(lengths: &[usize]) -> Arc<OutputLengthBelief> {
        let mut belief = OutputLengthBelief::new(crate::OutputBeliefConfig {
            min_samples: lengths.len(),
            ..crate::OutputBeliefConfig::default()
        });
        for (index, &length) in lengths.iter().enumerate() {
            let request = RequestId::new(100 + index as u64);
            belief.admit(request, &"tiny".into(), 16);
            belief.token(request, length, true, length < 16);
        }
        Arc::new(belief)
    }

    #[test]
    fn quantile_admission_without_a_belief_is_worst_case() {
        let cold = QuantileAdmission::new(Arc::new(OutputLengthBelief::default()), 0.9);
        assert_eq!(cold.headroom_tokens(&context()), None);
        assert_eq!(
            cold.decide(&context()),
            MemoryAwareAdmission.decide(&context())
        );
        let degenerate = QuantileAdmission::new(learned_belief(&[2, 2, 2, 2]), 1.0);
        assert_eq!(
            degenerate.decide(&context()),
            MemoryAwareAdmission.decide(&context())
        );
    }

    #[test]
    fn quantile_admission_reserves_the_prompt_plus_the_remaining_quantile() {
        let admission = QuantileAdmission::new(learned_belief(&[2, 2, 2, 12]), 0.5);
        // Median remaining output is 2 tokens: one generated position enters
        // the cache, so 17 + 1 = 18 positions, two 16-token pages.
        assert_eq!(admission.headroom_tokens(&context()), Some(1));
        assert_eq!(
            admission.decide(&context()),
            AdmissionDecision::Admit {
                reserved_kv_bytes: 256,
            }
        );
        // A worst case that can never fit is still rejected, whatever the
        // belief says.
        let impossible = AdmissionContext {
            prompt_tokens: 70,
            ..context()
        };
        assert!(matches!(
            admission.decide(&impossible),
            AdmissionDecision::Reject(AdmissionReason::RequestExceedsModelKvCapacity { .. })
        ));
    }
}
