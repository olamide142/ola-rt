use std::{fmt, sync::Arc};

use crate::{
    KernelCapability, KernelDescriptor, KernelError, KernelIncompatibility, KernelOperation,
    KernelRequirements, KernelTarget, PagedAttentionKernel, PortablePagedAttention,
};

/// Factory and capability declaration for a paged-attention implementation.
pub trait PagedAttentionProvider: Send + Sync {
    /// Returns the provider's declarative compatibility boundary.
    fn capability(&self) -> KernelCapability;

    /// Returns the deterministic preference used among compatible providers.
    ///
    /// Higher values win. Equal priorities are ordered by stable descriptor
    /// name, so registration order cannot change a plan.
    fn priority(&self) -> u16 {
        0
    }

    /// Creates a stateless or independently owned kernel instance.
    fn create(&self) -> Box<dyn PagedAttentionKernel>;
}

/// A rejected provider retained to explain a portable fallback.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KernelRejection {
    /// Provider that was considered.
    pub descriptor: KernelDescriptor,
    /// First incompatible constraint in stable evaluation order.
    pub reason: KernelIncompatibility,
}

/// Typed explanation for selecting the portable implementation.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum KernelFallbackReason {
    /// No registered optimized provider satisfied all constraints.
    NoCompatibleOptimizedProvider {
        /// Optimized candidates and their first rejected constraint, sorted by
        /// stable provider name.
        rejections: Vec<KernelRejection>,
    },
}

impl KernelFallbackReason {
    /// Returns rejected optimized candidates in deterministic order.
    pub fn rejections(&self) -> &[KernelRejection] {
        match self {
            Self::NoCompatibleOptimizedProvider { rejections } => rejections,
        }
    }
}

/// Deterministic result of matching requirements against registered providers.
#[derive(Clone)]
pub struct KernelSelection {
    descriptor: KernelDescriptor,
    fallback_reason: Option<KernelFallbackReason>,
    provider: Arc<dyn PagedAttentionProvider>,
}

impl KernelSelection {
    /// Returns the chosen provider descriptor.
    pub const fn descriptor(&self) -> KernelDescriptor {
        self.descriptor
    }

    /// Returns why the portable provider replaced optimized candidates.
    pub const fn fallback_reason(&self) -> Option<&KernelFallbackReason> {
        self.fallback_reason.as_ref()
    }

    /// Returns whether selection fell back from a physical target.
    pub const fn is_fallback(&self) -> bool {
        self.fallback_reason.is_some()
    }

    /// Creates the exact provider implementation retained by this selection.
    ///
    /// The executable handle is deliberately omitted from equality and debug
    /// output; stable plan identity is the descriptor plus fallback record.
    pub fn create(&self) -> Box<dyn PagedAttentionKernel> {
        self.provider.create()
    }
}

impl fmt::Debug for KernelSelection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("KernelSelection")
            .field("descriptor", &self.descriptor)
            .field("fallback_reason", &self.fallback_reason)
            .finish_non_exhaustive()
    }
}

impl PartialEq for KernelSelection {
    fn eq(&self, other: &Self) -> bool {
        self.descriptor == other.descriptor && self.fallback_reason == other.fallback_reason
    }
}

impl Eq for KernelSelection {}

struct RegisteredProvider {
    provider: Arc<dyn PagedAttentionProvider>,
    capability: KernelCapability,
    priority: u16,
}

/// Registry of paged-attention providers with an unconditional portable base.
pub struct KernelRegistry {
    providers: Vec<RegisteredProvider>,
}

impl Default for KernelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl KernelRegistry {
    /// Creates a registry containing the portable correctness provider.
    pub fn new() -> Self {
        let mut registry = Self {
            providers: Vec::new(),
        };
        registry
            .register(PortablePagedAttention::new())
            .expect("the built-in portable provider declaration is valid");
        registry
    }

    /// Creates the portable registry plus every provider compiled into this
    /// build.
    ///
    /// Capability matching still decides whether a compiled provider can run
    /// a particular model invocation; feature availability alone never forces
    /// its selection.
    pub fn with_compiled_providers() -> Self {
        let registry = Self::new();
        #[cfg(any(feature = "cubecl-cpu", feature = "wgpu", feature = "cuda"))]
        let mut registry = registry;
        #[cfg(feature = "cubecl-cpu")]
        registry
            .register(
                crate::CubePagedAttention::new(KernelTarget::Cpu)
                    .expect("the CubeCL CPU feature supports the CPU target"),
            )
            .expect("the built-in CubeCL CPU provider declaration is valid");
        #[cfg(feature = "wgpu")]
        registry
            .register(
                crate::CubePagedAttention::new(KernelTarget::Wgpu)
                    .expect("the CubeCL WGPU feature supports the WGPU target"),
            )
            .expect("the built-in CubeCL WGPU provider declaration is valid");
        #[cfg(feature = "cuda")]
        registry
            .register(
                crate::CubePagedAttention::new(KernelTarget::Cuda)
                    .expect("the CubeCL CUDA feature supports the CUDA target"),
            )
            .expect("the built-in CubeCL CUDA provider declaration is valid");
        registry
    }

    /// Registers a provider after validating and canonicalizing its capability.
    pub fn register(
        &mut self,
        provider: impl PagedAttentionProvider + 'static,
    ) -> Result<(), KernelError> {
        let mut capability = provider.capability();
        capability.validate()?;
        let descriptor = capability.descriptor();
        if descriptor.operation() != KernelOperation::PagedAttentionDecode {
            return Err(KernelError::new(format!(
                "paged-attention provider {} declares the wrong operation",
                descriptor.name()
            )));
        }
        if descriptor.target() != KernelTarget::Portable
            && !capability.targets().contains(&descriptor.target())
        {
            return Err(KernelError::new(format!(
                "kernel provider {} does not include its descriptor target in its capability",
                descriptor.name()
            )));
        }
        if self
            .providers
            .iter()
            .any(|registered| registered.capability.descriptor().name() == descriptor.name())
        {
            return Err(KernelError::new(format!(
                "kernel provider name {} is already registered",
                descriptor.name()
            )));
        }
        self.providers.push(RegisteredProvider {
            priority: provider.priority(),
            provider: Arc::new(provider),
            capability,
        });
        Ok(())
    }

    /// Returns provider capabilities in stable descriptor-name order.
    pub fn capabilities(&self) -> Vec<&KernelCapability> {
        let mut capabilities: Vec<_> = self
            .providers
            .iter()
            .map(|provider| &provider.capability)
            .collect();
        capabilities.sort_by_key(|capability| capability.descriptor().name());
        capabilities
    }

    /// Selects the highest-priority compatible optimized provider or the
    /// portable provider with a typed fallback explanation.
    pub fn select(&self, requirements: KernelRequirements) -> Result<KernelSelection, KernelError> {
        let mut compatible: Vec<_> = self
            .providers
            .iter()
            .filter(|provider| {
                provider.capability.descriptor().target() != KernelTarget::Portable
                    && provider.capability.supports(requirements)
            })
            .collect();
        compatible.sort_by(|left, right| {
            right.priority.cmp(&left.priority).then_with(|| {
                left.capability
                    .descriptor()
                    .name()
                    .cmp(right.capability.descriptor().name())
            })
        });
        if let Some(provider) = compatible.first() {
            return Ok(KernelSelection {
                descriptor: provider.capability.descriptor(),
                fallback_reason: None,
                provider: Arc::clone(&provider.provider),
            });
        }

        let portable = self
            .providers
            .iter()
            .find(|provider| {
                provider.capability.descriptor().target() == KernelTarget::Portable
                    && provider.capability.supports(requirements)
            })
            .ok_or_else(|| {
                KernelError::new(format!(
                    "no portable paged-attention provider supports {requirements:?}"
                ))
            })?;
        let fallback_reason = if requirements.target() == KernelTarget::Portable {
            None
        } else {
            let mut rejections: Vec<_> = self
                .providers
                .iter()
                .filter(|provider| {
                    provider.capability.descriptor().target() != KernelTarget::Portable
                })
                .filter_map(|provider| {
                    provider
                        .capability
                        .incompatibility(requirements)
                        .map(|reason| KernelRejection {
                            descriptor: provider.capability.descriptor(),
                            reason,
                        })
                })
                .collect();
            rejections.sort_by_key(|rejection| rejection.descriptor.name());
            Some(KernelFallbackReason::NoCompatibleOptimizedProvider { rejections })
        };
        Ok(KernelSelection {
            descriptor: portable.capability.descriptor(),
            fallback_reason,
            provider: Arc::clone(&portable.provider),
        })
    }

    /// Creates the kernel named by a prior selection.
    pub fn create(
        &self,
        selection: &KernelSelection,
    ) -> Result<Box<dyn PagedAttentionKernel>, KernelError> {
        self.providers
            .iter()
            .find(|provider| provider.capability.descriptor() == selection.descriptor)
            .map(|_| selection.create())
            .ok_or_else(|| {
                KernelError::new(format!(
                    "selected kernel provider {} is not registered",
                    selection.descriptor.name()
                ))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::KernelDType;

    struct TestProvider {
        capability: KernelCapability,
        priority: u16,
    }

    impl PagedAttentionProvider for TestProvider {
        fn capability(&self) -> KernelCapability {
            self.capability.clone()
        }

        fn priority(&self) -> u16 {
            self.priority
        }

        fn create(&self) -> Box<dyn PagedAttentionKernel> {
            Box::new(PortablePagedAttention::new())
        }
    }

    fn provider(
        name: &'static str,
        dtype: KernelDType,
        max_pages: usize,
        priority: u16,
    ) -> TestProvider {
        TestProvider {
            capability: KernelCapability::new(
                KernelDescriptor::new(
                    name,
                    KernelOperation::PagedAttentionDecode,
                    KernelTarget::Wgpu,
                ),
                [KernelTarget::Wgpu],
                [dtype],
            )
            .with_head_dimensions([128])
            .with_gqa_ratios([4])
            .with_max_pages(max_pages),
            priority,
        }
    }

    fn requirements(dtype: KernelDType, pages: usize) -> KernelRequirements {
        KernelRequirements::paged_attention_decode(KernelTarget::Wgpu, dtype, 128, 4, pages)
            .unwrap()
    }

    #[test]
    fn selection_uses_priority_then_name_not_registration_order() {
        let mut first = KernelRegistry::new();
        first
            .register(provider("z-provider", KernelDType::F32, 32, 7))
            .unwrap();
        first
            .register(provider("a-provider", KernelDType::F32, 32, 7))
            .unwrap();
        let mut second = KernelRegistry::new();
        second
            .register(provider("a-provider", KernelDType::F32, 32, 7))
            .unwrap();
        second
            .register(provider("z-provider", KernelDType::F32, 32, 7))
            .unwrap();

        let left = first.select(requirements(KernelDType::F32, 8)).unwrap();
        let right = second.select(requirements(KernelDType::F32, 8)).unwrap();

        assert_eq!(left, right);
        assert_eq!(left.descriptor().name(), "a-provider");
        assert!(!left.is_fallback());
    }

    #[test]
    fn unsupported_shape_falls_back_with_stable_rejections() {
        let mut registry = KernelRegistry::new();
        registry
            .register(provider("limited-pages", KernelDType::F32, 4, 10))
            .unwrap();
        registry
            .register(provider("wrong-dtype", KernelDType::F16, 32, 20))
            .unwrap();

        let selection = registry.select(requirements(KernelDType::F32, 8)).unwrap();
        let rejections = selection.fallback_reason().unwrap().rejections();

        assert_eq!(selection.descriptor().target(), KernelTarget::Portable);
        assert_eq!(rejections[0].descriptor.name(), "limited-pages");
        assert_eq!(rejections[0].reason, KernelIncompatibility::PageCount);
        assert_eq!(rejections[1].descriptor.name(), "wrong-dtype");
        assert_eq!(rejections[1].reason, KernelIncompatibility::DType);
    }

    #[test]
    fn duplicate_stable_names_are_rejected() {
        let mut registry = KernelRegistry::new();
        registry
            .register(provider("same-name", KernelDType::F32, 4, 1))
            .unwrap();

        let error = registry
            .register(provider("same-name", KernelDType::F16, 8, 2))
            .unwrap_err();

        assert!(error.message().contains("already registered"));
    }

    #[test]
    fn selected_provider_can_create_a_kernel() {
        let requirements = KernelRequirements::paged_attention_decode(
            KernelTarget::Portable,
            KernelDType::F32,
            64,
            1,
            1,
        )
        .unwrap();

        let selection = KernelRegistry::new().select(requirements).unwrap();
        let kernel = selection.create();

        assert_eq!(kernel.descriptor(), selection.descriptor());
        assert!(!selection.is_fallback());
    }

    #[cfg(feature = "cubecl-cpu")]
    #[test]
    fn compiled_provider_registry_selects_cubecl_for_its_target() {
        let registry = KernelRegistry::with_compiled_providers();
        let requirements = KernelRequirements::paged_attention_decode(
            KernelTarget::Cpu,
            KernelDType::F32,
            128,
            4,
            2,
        )
        .unwrap();

        let selection = registry.select(requirements).unwrap();

        assert_eq!(selection.descriptor().name(), "cubecl-cpu-paged-attention");
        assert!(!selection.is_fallback());
    }
}
