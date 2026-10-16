use std::fmt;

use ola_core::{ModelId, RequestId};
use ola_model::KvCacheProfile;

use crate::{MemoryState, PolicyDescriptor};

/// Read-only inputs supplied to an executor admission policy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdmissionContext {
    /// Scheduler-selected queued request.
    pub request_id: RequestId,
    /// Model whose shared KV pool would own the request cache.
    pub model_id: ModelId,
    /// Tokenized prompt length.
    pub prompt_tokens: usize,
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
}
