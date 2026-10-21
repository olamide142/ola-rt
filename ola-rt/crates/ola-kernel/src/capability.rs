use burn::tensor::DType;

use crate::{KernelDescriptor, KernelError, KernelOperation, KernelTarget};

/// Floating-point storage type used when planning a kernel invocation.
///
/// This metadata type keeps Burn's tensor types out of runtime plans while
/// still allowing the tensor boundary to translate its concrete dtype.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum KernelDType {
    /// 32-bit IEEE floating point.
    F32,
    /// Burn's backend-selected flexible 32-bit floating-point type.
    Flex32,
    /// 16-bit IEEE floating point.
    F16,
    /// 16-bit brain floating point.
    Bf16,
}

impl TryFrom<DType> for KernelDType {
    type Error = KernelError;

    fn try_from(value: DType) -> Result<Self, Self::Error> {
        match value {
            DType::F32 => Ok(Self::F32),
            DType::Flex32 => Ok(Self::Flex32),
            DType::F16 => Ok(Self::F16),
            DType::BF16 => Ok(Self::Bf16),
            unsupported => Err(KernelError::new(format!(
                "kernel planning requires a floating-point dtype, found {unsupported:?}"
            ))),
        }
    }
}

/// Backend-neutral requirements for one kernel invocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KernelRequirements {
    operation: KernelOperation,
    target: KernelTarget,
    dtype: KernelDType,
    head_dim: usize,
    gqa_ratio: usize,
    page_count: usize,
}

impl KernelRequirements {
    /// Describes a paged decode-attention invocation.
    pub fn paged_attention_decode(
        target: KernelTarget,
        dtype: KernelDType,
        head_dim: usize,
        gqa_ratio: usize,
        page_count: usize,
    ) -> Result<Self, KernelError> {
        if head_dim == 0 {
            return Err(KernelError::new(
                "kernel requirement head dimension must be greater than zero",
            ));
        }
        if gqa_ratio == 0 {
            return Err(KernelError::new(
                "kernel requirement GQA ratio must be greater than zero",
            ));
        }
        if page_count == 0 {
            return Err(KernelError::new(
                "kernel requirement page count must be greater than zero",
            ));
        }
        Ok(Self {
            operation: KernelOperation::PagedAttentionDecode,
            target,
            dtype,
            head_dim,
            gqa_ratio,
            page_count,
        })
    }

    /// Returns the requested operation family.
    pub const fn operation(self) -> KernelOperation {
        self.operation
    }

    /// Returns the physical execution target.
    pub const fn target(self) -> KernelTarget {
        self.target
    }

    /// Returns the requested floating-point dtype.
    pub const fn dtype(self) -> KernelDType {
        self.dtype
    }

    /// Returns the width of one attention head.
    pub const fn head_dim(self) -> usize {
        self.head_dim
    }

    /// Returns the number of query heads sharing each key/value head.
    pub const fn gqa_ratio(self) -> usize {
        self.gqa_ratio
    }

    /// Returns the number of populated logical KV pages.
    pub const fn page_count(self) -> usize {
        self.page_count
    }
}

/// One reason that a provider cannot execute a requirement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum KernelIncompatibility {
    /// The provider implements another operation family.
    Operation,
    /// The provider cannot execute on the requested target.
    Target,
    /// The provider does not implement the requested dtype.
    DType,
    /// The attention head width is unsupported.
    HeadDimension,
    /// The grouped-query-attention ratio is unsupported.
    GqaRatio,
    /// The logical page count exceeds the provider limit.
    PageCount,
}

/// Declarative limits published by one kernel provider.
///
/// Empty head-dimension and GQA-ratio lists mean that every positive value is
/// supported. Registration validates all other structural invariants.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelCapability {
    descriptor: KernelDescriptor,
    targets: Vec<KernelTarget>,
    dtypes: Vec<KernelDType>,
    head_dimensions: Vec<usize>,
    gqa_ratios: Vec<usize>,
    max_pages: Option<usize>,
}

impl KernelCapability {
    /// Starts a capability declaration for the given targets and dtypes.
    pub fn new(
        descriptor: KernelDescriptor,
        targets: impl IntoIterator<Item = KernelTarget>,
        dtypes: impl IntoIterator<Item = KernelDType>,
    ) -> Self {
        Self {
            descriptor,
            targets: targets.into_iter().collect(),
            dtypes: dtypes.into_iter().collect(),
            head_dimensions: Vec::new(),
            gqa_ratios: Vec::new(),
            max_pages: None,
        }
    }

    /// Restricts the provider to the listed positive attention head widths.
    pub fn with_head_dimensions(mut self, values: impl IntoIterator<Item = usize>) -> Self {
        self.head_dimensions = values.into_iter().collect();
        self
    }

    /// Restricts the provider to the listed positive GQA ratios.
    pub fn with_gqa_ratios(mut self, values: impl IntoIterator<Item = usize>) -> Self {
        self.gqa_ratios = values.into_iter().collect();
        self
    }

    /// Restricts the number of logical pages accepted by one invocation.
    pub const fn with_max_pages(mut self, max_pages: usize) -> Self {
        self.max_pages = Some(max_pages);
        self
    }

    /// Returns stable implementation metadata.
    pub const fn descriptor(&self) -> KernelDescriptor {
        self.descriptor
    }

    /// Returns supported physical execution targets.
    pub fn targets(&self) -> &[KernelTarget] {
        &self.targets
    }

    /// Returns supported floating-point dtypes.
    pub fn dtypes(&self) -> &[KernelDType] {
        &self.dtypes
    }

    /// Returns supported head widths, or an empty slice when unrestricted.
    pub fn head_dimensions(&self) -> &[usize] {
        &self.head_dimensions
    }

    /// Returns supported GQA ratios, or an empty slice when unrestricted.
    pub fn gqa_ratios(&self) -> &[usize] {
        &self.gqa_ratios
    }

    /// Returns the maximum logical page count, if bounded.
    pub const fn max_pages(&self) -> Option<usize> {
        self.max_pages
    }

    /// Reports the first constraint that rejects a requirement.
    pub fn incompatibility(
        &self,
        requirements: KernelRequirements,
    ) -> Option<KernelIncompatibility> {
        if self.descriptor.operation() != requirements.operation {
            return Some(KernelIncompatibility::Operation);
        }
        if !self.targets.contains(&requirements.target) {
            return Some(KernelIncompatibility::Target);
        }
        if !self.dtypes.contains(&requirements.dtype) {
            return Some(KernelIncompatibility::DType);
        }
        if !self.head_dimensions.is_empty()
            && !self.head_dimensions.contains(&requirements.head_dim)
        {
            return Some(KernelIncompatibility::HeadDimension);
        }
        if !self.gqa_ratios.is_empty() && !self.gqa_ratios.contains(&requirements.gqa_ratio) {
            return Some(KernelIncompatibility::GqaRatio);
        }
        if self
            .max_pages
            .is_some_and(|max_pages| requirements.page_count > max_pages)
        {
            return Some(KernelIncompatibility::PageCount);
        }
        None
    }

    /// Returns whether all declared constraints accept a requirement.
    pub fn supports(&self, requirements: KernelRequirements) -> bool {
        self.incompatibility(requirements).is_none()
    }

    pub(crate) fn validate(&mut self) -> Result<(), KernelError> {
        sort_and_deduplicate(&mut self.targets);
        sort_and_deduplicate(&mut self.dtypes);
        sort_and_deduplicate(&mut self.head_dimensions);
        sort_and_deduplicate(&mut self.gqa_ratios);

        if self.targets.is_empty() {
            return Err(KernelError::new(format!(
                "kernel provider {} must declare at least one target",
                self.descriptor.name()
            )));
        }
        if self.dtypes.is_empty() {
            return Err(KernelError::new(format!(
                "kernel provider {} must declare at least one dtype",
                self.descriptor.name()
            )));
        }
        if self.head_dimensions.contains(&0) {
            return Err(KernelError::new(format!(
                "kernel provider {} declares a zero head dimension",
                self.descriptor.name()
            )));
        }
        if self.gqa_ratios.contains(&0) {
            return Err(KernelError::new(format!(
                "kernel provider {} declares a zero GQA ratio",
                self.descriptor.name()
            )));
        }
        if self.max_pages == Some(0) {
            return Err(KernelError::new(format!(
                "kernel provider {} declares a zero page limit",
                self.descriptor.name()
            )));
        }
        Ok(())
    }
}

fn sort_and_deduplicate<T: Ord>(values: &mut Vec<T>) {
    values.sort_unstable();
    values.dedup();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor() -> KernelDescriptor {
        KernelDescriptor::new(
            "bounded-test-kernel",
            KernelOperation::PagedAttentionDecode,
            KernelTarget::Wgpu,
        )
    }

    #[test]
    fn capability_checks_every_paged_attention_constraint() {
        let capability =
            KernelCapability::new(descriptor(), [KernelTarget::Wgpu], [KernelDType::F16])
                .with_head_dimensions([64, 128])
                .with_gqa_ratios([1, 4])
                .with_max_pages(16);
        let supported = KernelRequirements::paged_attention_decode(
            KernelTarget::Wgpu,
            KernelDType::F16,
            128,
            4,
            16,
        )
        .unwrap();

        assert!(capability.supports(supported));

        let too_many_pages = KernelRequirements::paged_attention_decode(
            KernelTarget::Wgpu,
            KernelDType::F16,
            128,
            4,
            17,
        )
        .unwrap();
        assert_eq!(
            capability.incompatibility(too_many_pages),
            Some(KernelIncompatibility::PageCount)
        );
    }

    #[test]
    fn registration_validation_canonicalizes_declared_sets() {
        let mut capability = KernelCapability::new(
            descriptor(),
            [KernelTarget::Wgpu, KernelTarget::Wgpu],
            [KernelDType::F32, KernelDType::F16, KernelDType::F32],
        )
        .with_head_dimensions([128, 64, 128]);

        capability.validate().unwrap();

        assert_eq!(capability.targets(), &[KernelTarget::Wgpu]);
        assert_eq!(capability.dtypes(), &[KernelDType::F32, KernelDType::F16]);
        assert_eq!(capability.head_dimensions(), &[64, 128]);
    }
}
