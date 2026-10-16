//! Stable identities for runtime policies.
//!
//! See `EXECUTION_POLICY.md` in this crate for the design.

use std::fmt;

/// Stable name and configuration of a policy, for reports and comparisons.
///
/// Names are lowercase, hyphenated, and never change for a given behaviour;
/// configuration keys are listed in a fixed order. Two policies with equal
/// descriptors must make identical decisions from identical state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PolicyDescriptor {
    /// Stable policy name, such as `fifo-token-budget`.
    pub name: &'static str,
    /// Configuration that affects decisions, in a fixed order.
    pub config: Vec<(&'static str, String)>,
    /// Descriptors of composed sub-policies, in a fixed order.
    pub components: Vec<PolicyDescriptor>,
}

impl PolicyDescriptor {
    /// Describes an unconfigured policy.
    pub const fn new(name: &'static str) -> Self {
        Self {
            name,
            config: Vec::new(),
            components: Vec::new(),
        }
    }

    /// Adds one configuration entry.
    #[must_use]
    pub fn with_config(mut self, key: &'static str, value: impl fmt::Display) -> Self {
        self.config.push((key, value.to_string()));
        self
    }

    /// Adds one composed sub-policy.
    #[must_use]
    pub fn with_component(mut self, component: Self) -> Self {
        self.components.push(component);
        self
    }
}

impl fmt::Display for PolicyDescriptor {
    /// Writes `name{key=value,…}[component,…]`, omitting empty parts.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name)?;
        if !self.config.is_empty() {
            formatter.write_str("{")?;
            for (index, (key, value)) in self.config.iter().enumerate() {
                if index > 0 {
                    formatter.write_str(",")?;
                }
                write!(formatter, "{key}={value}")?;
            }
            formatter.write_str("}")?;
        }
        if !self.components.is_empty() {
            formatter.write_str("[")?;
            for (index, component) in self.components.iter().enumerate() {
                if index > 0 {
                    formatter.write_str(",")?;
                }
                component.fmt(formatter)?;
            }
            formatter.write_str("]")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptors_render_name_config_and_components() {
        let descriptor = PolicyDescriptor::new("composed")
            .with_component(PolicyDescriptor::new("fifo"))
            .with_component(
                PolicyDescriptor::new("fifo-token-budget").with_config("max_tokens", 4),
            );

        assert_eq!(
            descriptor.to_string(),
            "composed[fifo,fifo-token-budget{max_tokens=4}]"
        );
    }

    #[test]
    fn built_in_policies_have_stable_descriptors() {
        use crate::{
            AdmissionPolicy, BatchPolicy, CostAwarePrefillBatcher, FifoScheduler,
            FifoTokenBudgetPolicy, MemoryAwareAdmission, PowerOfTwoPrefillBatcher, PrefillBatcher,
            Scheduler, UnboundedBatchPolicy, UnbucketedPrefillBatcher, UnconstrainedAdmission,
        };

        let names = [
            FifoScheduler.descriptor(),
            UnconstrainedAdmission.descriptor(),
            MemoryAwareAdmission.descriptor(),
            UnboundedBatchPolicy.descriptor(),
            FifoTokenBudgetPolicy::new(4).unwrap().descriptor(),
            UnbucketedPrefillBatcher.descriptor(),
            PowerOfTwoPrefillBatcher.descriptor(),
            CostAwarePrefillBatcher::new(12).descriptor(),
        ]
        .map(|descriptor| descriptor.to_string());

        assert_eq!(
            names,
            [
                "fifo",
                "unconstrained-admission",
                "memory-aware-admission",
                "unbounded-batch",
                "fifo-token-budget{max_tokens=4}",
                "unbucketed-prefill",
                "power-of-two-prefill",
                "cost-aware-prefill{invocation_overhead_tokens=12}",
            ]
        );
    }
}
