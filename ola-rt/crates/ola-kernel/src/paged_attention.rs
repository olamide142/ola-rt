use burn::tensor::{DType, Device, Int, Tensor, TensorData};

use crate::{
    KernelCapability, KernelDType, KernelDescriptor, KernelError, KernelOperation, KernelTarget,
    PagedAttentionProvider,
};

/// Dimensions shared by paged decode-attention implementations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PagedAttentionConfig {
    query_heads: usize,
    key_value_heads: usize,
    head_dim: usize,
}

impl PagedAttentionConfig {
    /// Creates and validates an attention configuration.
    pub fn new(
        query_heads: usize,
        key_value_heads: usize,
        head_dim: usize,
    ) -> Result<Self, KernelError> {
        if query_heads == 0 {
            return Err(KernelError::new(
                "query head count must be greater than zero",
            ));
        }
        if key_value_heads == 0 {
            return Err(KernelError::new(
                "key/value head count must be greater than zero",
            ));
        }
        if head_dim == 0 {
            return Err(KernelError::new(
                "attention head dimension must be greater than zero",
            ));
        }
        if !query_heads.is_multiple_of(key_value_heads) {
            return Err(KernelError::new(
                "query head count must be divisible by key/value head count",
            ));
        }
        Ok(Self {
            query_heads,
            key_value_heads,
            head_dim,
        })
    }

    /// Returns the number of query heads.
    pub const fn query_heads(self) -> usize {
        self.query_heads
    }

    /// Returns the number of key/value heads.
    pub const fn key_value_heads(self) -> usize {
        self.key_value_heads
    }

    /// Returns the width of one attention head.
    pub const fn head_dim(self) -> usize {
        self.head_dim
    }

    pub(crate) const fn groups(self) -> usize {
        self.query_heads / self.key_value_heads
    }
}

/// One populated logical KV-cache page.
#[derive(Clone)]
pub struct PagedKvPage {
    key: Tensor<4>,
    value: Tensor<4>,
}

impl PagedKvPage {
    /// Pairs key and value tensors for one logical page.
    ///
    /// Shape validation occurs when pages are assembled into
    /// [`PagedAttentionInput`].
    pub const fn new(key: Tensor<4>, value: Tensor<4>) -> Self {
        Self { key, value }
    }

    /// Returns the key tensor.
    pub const fn key(&self) -> &Tensor<4> {
        &self.key
    }

    /// Returns the value tensor.
    pub const fn value(&self) -> &Tensor<4> {
        &self.value
    }

    pub(crate) fn into_parts(self) -> (Tensor<4>, Tensor<4>) {
        (self.key, self.value)
    }
}

/// Validated logical-to-physical page indices resident on the compute device.
#[derive(Clone)]
pub struct PagedKvBlockTable {
    tensor: Tensor<1, Int>,
    block_ids: Vec<usize>,
    physical_blocks: usize,
}

impl PagedKvBlockTable {
    /// Uploads a non-empty logical block table after validating every index.
    pub fn new(
        block_ids: Vec<usize>,
        physical_blocks: usize,
        device: &Device,
    ) -> Result<Self, KernelError> {
        if physical_blocks == 0 {
            return Err(KernelError::new(
                "physical KV block count must be greater than zero",
            ));
        }
        if block_ids.is_empty() {
            return Err(KernelError::new(
                "a populated KV cache requires at least one logical block",
            ));
        }
        for (logical, physical) in block_ids.iter().copied().enumerate() {
            if physical >= physical_blocks {
                return Err(KernelError::new(format!(
                    "logical KV block {logical} maps to physical block {physical}, outside {physical_blocks} blocks"
                )));
            }
        }
        let values = block_ids
            .iter()
            .map(|block| {
                i64::try_from(*block)
                    .map_err(|_| KernelError::new("physical KV block index exceeds i64"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let length = values.len();
        Ok(Self {
            tensor: Tensor::from_data(TensorData::new(values, [length]), device),
            block_ids,
            physical_blocks,
        })
    }

    /// Returns a view covering the first `logical_blocks` entries.
    #[allow(clippy::single_range_in_vec_init)]
    pub fn prefix(&self, logical_blocks: usize) -> Result<Self, KernelError> {
        if logical_blocks == 0 || logical_blocks > self.block_ids.len() {
            return Err(KernelError::new(format!(
                "logical block prefix {logical_blocks} is outside a {}-block table",
                self.block_ids.len()
            )));
        }
        Ok(Self {
            tensor: self.tensor.clone().slice([0..logical_blocks]),
            block_ids: self.block_ids[..logical_blocks].to_vec(),
            physical_blocks: self.physical_blocks,
        })
    }

    /// Returns the number of populated logical blocks.
    pub const fn len(&self) -> usize {
        self.block_ids.len()
    }

    /// Reports whether this table contains no logical blocks.
    pub const fn is_empty(&self) -> bool {
        self.block_ids.is_empty()
    }

    /// Returns the device integer tensor consumed by optimized kernels.
    pub const fn tensor(&self) -> &Tensor<1, Int> {
        &self.tensor
    }

    #[cfg(feature = "cubecl")]
    pub(crate) fn block_ids(&self) -> &[usize] {
        &self.block_ids
    }
}

/// Device-resident physical KV blocks and the logical-to-physical block table.
///
/// The logical page views remain part of [`PagedAttentionInput`] for portable
/// implementations. Optimized kernels can use this representation to traverse
/// the cache without concatenating those views into a sequence-sized tensor.
#[derive(Clone)]
pub struct PagedKvBlocks {
    key: Tensor<4>,
    value: Tensor<4>,
    block_table: PagedKvBlockTable,
    tokens_per_block: usize,
    token_count: usize,
}

impl PagedKvBlocks {
    /// Creates physical cache metadata.
    ///
    /// Key and value tensors use shape
    /// `[physical_blocks, key_value_heads, tokens_per_block, head_dim]`.
    /// The block table contains one physical block index per populated logical
    /// block, in sequence order.
    pub fn new(
        key: Tensor<4>,
        value: Tensor<4>,
        block_table: PagedKvBlockTable,
        tokens_per_block: usize,
        token_count: usize,
    ) -> Result<Self, KernelError> {
        let key_shape = key.dims();
        let value_shape = value.dims();
        if key_shape != value_shape {
            return Err(KernelError::new(format!(
                "physical KV blocks have different key {key_shape:?} and value {value_shape:?} shapes"
            )));
        }
        if key_shape.contains(&0) {
            return Err(KernelError::new(
                "physical KV block dimensions must be greater than zero",
            ));
        }
        if tokens_per_block == 0 || key_shape[2] != tokens_per_block {
            return Err(KernelError::new(format!(
                "physical KV block token dimension must equal tokens_per_block {tokens_per_block}, found {}",
                key_shape[2]
            )));
        }
        let logical_blocks = block_table.len();
        if logical_blocks == 0 || token_count == 0 {
            return Err(KernelError::new(
                "physical KV metadata requires populated blocks and tokens",
            ));
        }
        if token_count > logical_blocks.saturating_mul(tokens_per_block)
            || token_count <= (logical_blocks - 1).saturating_mul(tokens_per_block)
        {
            return Err(KernelError::new(format!(
                "token count {token_count} does not match {logical_blocks} logical blocks of {tokens_per_block} tokens"
            )));
        }
        let device = key.device();
        if value.device() != device || block_table.tensor.device() != device {
            return Err(KernelError::new(
                "physical KV blocks and block table must live on the same device",
            ));
        }
        if value.dtype() != key.dtype() {
            return Err(KernelError::new(
                "physical key and value blocks must use the same dtype",
            ));
        }
        if block_table.physical_blocks != key_shape[0] {
            return Err(KernelError::new(format!(
                "block table addresses {} physical blocks but K/V storage contains {}",
                block_table.physical_blocks, key_shape[0]
            )));
        }
        Ok(Self {
            key,
            value,
            block_table,
            tokens_per_block,
            token_count,
        })
    }

    /// Returns all physical key blocks.
    pub const fn key(&self) -> &Tensor<4> {
        &self.key
    }

    /// Returns all physical value blocks.
    pub const fn value(&self) -> &Tensor<4> {
        &self.value
    }

    /// Returns the logical-to-physical block table.
    pub const fn block_table(&self) -> &PagedKvBlockTable {
        &self.block_table
    }

    /// Returns the physical page width.
    pub const fn tokens_per_block(&self) -> usize {
        self.tokens_per_block
    }

    /// Returns the populated logical token count.
    pub const fn token_count(&self) -> usize {
        self.token_count
    }

    #[cfg(feature = "cubecl")]
    pub(crate) fn into_parts(self) -> (Tensor<4>, Tensor<4>, Tensor<1, Int>, usize, usize) {
        (
            self.key,
            self.value,
            self.block_table.tensor,
            self.tokens_per_block,
            self.token_count,
        )
    }
}

/// Validated input to single-sequence paged decode attention.
#[derive(Clone)]
pub struct PagedAttentionInput {
    query: Tensor<4>,
    pages: Vec<PagedKvPage>,
    blocks: Option<PagedKvBlocks>,
    config: PagedAttentionConfig,
}

impl PagedAttentionInput {
    /// Validates and constructs an operator input.
    pub fn new(
        query: Tensor<4>,
        pages: Vec<PagedKvPage>,
        config: PagedAttentionConfig,
    ) -> Result<Self, KernelError> {
        let expected_query = [1, config.query_heads, 1, config.head_dim];
        let actual_query = query.dims();
        if actual_query != expected_query {
            return Err(KernelError::new(format!(
                "query shape must be {expected_query:?}, found {actual_query:?}"
            )));
        }
        if pages.is_empty() {
            return Err(KernelError::new(
                "paged decode attention requires at least one populated KV page",
            ));
        }
        let query_device = query.device();
        let query_dtype = query.dtype();
        for (index, page) in pages.iter().enumerate() {
            let key = page.key.dims();
            let value = page.value.dims();
            if key != value {
                return Err(KernelError::new(format!(
                    "KV page {index} has different key {key:?} and value {value:?} shapes"
                )));
            }
            if key[0] != 1
                || key[1] != config.key_value_heads
                || key[2] == 0
                || key[3] != config.head_dim
            {
                return Err(KernelError::new(format!(
                    "KV page {index} shape must be [1, {}, populated_tokens, {}], found {key:?}",
                    config.key_value_heads, config.head_dim
                )));
            }
            if page.key.device() != query_device || page.value.device() != query_device {
                return Err(KernelError::new(format!(
                    "KV page {index} and query must live on the same device"
                )));
            }
            if page.key.dtype() != query_dtype || page.value.dtype() != query_dtype {
                return Err(KernelError::new(format!(
                    "KV page {index} and query must use the same dtype"
                )));
            }
        }
        Ok(Self {
            query,
            pages,
            blocks: None,
            config,
        })
    }

    /// Attaches device-resident physical blocks for optimized implementations.
    pub fn with_blocks(mut self, blocks: PagedKvBlocks) -> Result<Self, KernelError> {
        let shape = blocks.key.dims();
        if shape[1] != self.config.key_value_heads || shape[3] != self.config.head_dim {
            return Err(KernelError::new(format!(
                "physical KV block shape must be [physical_blocks, {}, tokens_per_block, {}], found {shape:?}",
                self.config.key_value_heads, self.config.head_dim
            )));
        }
        if blocks.key.device() != self.query.device() || blocks.key.dtype() != self.query.dtype() {
            return Err(KernelError::new(
                "physical KV blocks and query must use the same device and dtype",
            ));
        }
        let page_tokens = self
            .pages
            .iter()
            .map(|page| page.key.dims()[2])
            .sum::<usize>();
        if page_tokens != blocks.token_count {
            return Err(KernelError::new(format!(
                "logical pages contain {page_tokens} tokens but physical metadata contains {}",
                blocks.token_count
            )));
        }
        self.blocks = Some(blocks);
        Ok(self)
    }

    /// Returns the query tensor.
    pub const fn query(&self) -> &Tensor<4> {
        &self.query
    }

    /// Returns logical KV pages in sequence order.
    pub fn pages(&self) -> &[PagedKvPage] {
        &self.pages
    }

    /// Returns physical cache metadata when the producer owns a flattened pool.
    pub const fn blocks(&self) -> Option<&PagedKvBlocks> {
        self.blocks.as_ref()
    }

    /// Returns the validated dimensions.
    pub const fn config(&self) -> PagedAttentionConfig {
        self.config
    }

    /// Returns the common floating-point dtype validated for query and pages.
    pub fn dtype(&self) -> DType {
        self.query.dtype()
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        Tensor<4>,
        Vec<PagedKvPage>,
        Option<PagedKvBlocks>,
        PagedAttentionConfig,
    ) {
        (self.query, self.pages, self.blocks, self.config)
    }
}

/// Replaceable implementation of single-sequence paged decode attention.
pub trait PagedAttentionKernel: Send + Sync {
    /// Returns stable metadata for diagnostics and benchmark reports.
    fn descriptor(&self) -> KernelDescriptor;

    /// Produces an attention context with shape `[1, query_heads, 1, head_dim]`.
    fn execute(&self, input: PagedAttentionInput) -> Result<Tensor<4>, KernelError>;

    /// Executes independent one-token queries as one logical operation.
    ///
    /// Providers may override this to use a genuinely batched dispatch. The
    /// default deliberately preserves the portable request-local semantics.
    fn execute_batch(
        &self,
        inputs: Vec<PagedAttentionInput>,
    ) -> Result<Vec<Tensor<4>>, KernelError> {
        inputs
            .into_iter()
            .map(|input| self.execute(input))
            .collect()
    }
}

/// Reference implementation composed from portable Burn tensor operations.
#[derive(Clone, Copy, Debug, Default)]
pub struct PortablePagedAttention;

impl PortablePagedAttention {
    /// Creates the stateless portable implementation.
    pub const fn new() -> Self {
        Self
    }
}

impl PagedAttentionKernel for PortablePagedAttention {
    fn descriptor(&self) -> KernelDescriptor {
        KernelDescriptor::new(
            "portable-burn-paged-attention",
            KernelOperation::PagedAttentionDecode,
            KernelTarget::Portable,
        )
    }

    fn execute(&self, input: PagedAttentionInput) -> Result<Tensor<4>, KernelError> {
        let (query, pages, _blocks, config) = input.into_parts();
        let scale = (config.head_dim as f32).sqrt();
        let mut pages = pages.into_iter();
        let (first_key, first_value) = pages
            .next()
            .expect("validated paged attention input contains a page")
            .into_parts();
        let first_key = repeat_key_value(first_key, config.groups());
        let first_value = repeat_key_value(first_value, config.groups());
        let scores = query
            .clone()
            .matmul(first_key.swap_dims(2, 3))
            .div_scalar(scale);
        let mut running_max = scores.clone().max_dim(3);
        let weights = (scores - running_max.clone()).exp();
        let mut running_sum = weights.clone().sum_dim(3);
        let mut running_output = weights.matmul(first_value);

        for page in pages {
            let (key, value) = page.into_parts();
            let key = repeat_key_value(key, config.groups());
            let value = repeat_key_value(value, config.groups());
            let scores = query.clone().matmul(key.swap_dims(2, 3)).div_scalar(scale);
            let page_max = scores.clone().max_dim(3);
            let new_max = running_max.clone().max_pair(page_max);
            let previous_scale = (running_max - new_max.clone()).exp();
            let weights = (scores - new_max.clone()).exp();
            running_sum = running_sum * previous_scale.clone() + weights.clone().sum_dim(3);
            running_output = running_output * previous_scale + weights.matmul(value);
            running_max = new_max;
        }

        Ok(running_output / running_sum)
    }
}

impl PagedAttentionProvider for PortablePagedAttention {
    fn capability(&self) -> KernelCapability {
        KernelCapability::new(
            self.descriptor(),
            [
                KernelTarget::Portable,
                KernelTarget::Cpu,
                KernelTarget::Wgpu,
                KernelTarget::Cuda,
            ],
            [
                KernelDType::F32,
                KernelDType::Flex32,
                KernelDType::F16,
                KernelDType::Bf16,
            ],
        )
    }

    fn create(&self) -> Box<dyn PagedAttentionKernel> {
        Box::new(*self)
    }
}

/// Repeats each key/value head contiguously into its query-head group.
fn repeat_key_value(tensor: Tensor<4>, groups: usize) -> Tensor<4> {
    if groups == 1 {
        return tensor;
    }
    let [batch, key_value_heads, sequence, head_dim] = tensor.dims();
    tensor
        .reshape([batch, key_value_heads, 1, sequence, head_dim])
        .repeat_dim(2, groups)
        .reshape([batch, key_value_heads * groups, sequence, head_dim])
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;

    use super::*;

    #[test]
    fn rejects_invalid_grouped_query_attention() {
        let error = PagedAttentionConfig::new(3, 2, 4).unwrap_err();

        assert!(error.message().contains("must be divisible"));
    }

    #[test]
    fn rejects_empty_page_input() {
        let device = Default::default();
        let query = Tensor::<4>::zeros([1, 1, 1, 1], &device);
        let config = PagedAttentionConfig::new(1, 1, 1).unwrap();

        let error = PagedAttentionInput::new(query, Vec::new(), config)
            .err()
            .expect("empty pages must fail validation");

        assert!(error.message().contains("at least one"));
    }

    #[test]
    fn rejects_a_block_table_that_addresses_outside_the_physical_pool() {
        let device = Default::default();
        let error = PagedKvBlockTable::new(vec![0, 2], 2, &device)
            .err()
            .expect("an out-of-range physical block must be rejected");

        assert!(error.message().contains("outside 2 blocks"));
    }

    #[test]
    fn online_softmax_combines_pages_without_concatenation() {
        let device = Default::default();
        let query = Tensor::<4>::from_data(TensorData::new(vec![1.0_f32], [1, 1, 1, 1]), &device);
        let first = PagedKvPage::new(
            Tensor::from_data(TensorData::new(vec![0.0_f32], [1, 1, 1, 1]), &device),
            Tensor::from_data(TensorData::new(vec![10.0_f32], [1, 1, 1, 1]), &device),
        );
        let second = PagedKvPage::new(
            Tensor::from_data(TensorData::new(vec![3.0_f32.ln()], [1, 1, 1, 1]), &device),
            Tensor::from_data(TensorData::new(vec![20.0_f32], [1, 1, 1, 1]), &device),
        );
        let config = PagedAttentionConfig::new(1, 1, 1).unwrap();
        let input = PagedAttentionInput::new(query, vec![first, second], config).unwrap();

        let output = PortablePagedAttention::new().execute(input).unwrap();
        let values = output.into_data().try_to_vec::<f32>().unwrap();

        assert!((values[0] - 17.5).abs() < 1e-5);
    }

    #[test]
    fn grouped_query_heads_share_their_corresponding_kv_head() {
        let device = Default::default();
        let query = Tensor::<4>::zeros([1, 4, 1, 1], &device);
        let page = PagedKvPage::new(
            Tensor::zeros([1, 2, 1, 1], &device),
            Tensor::from_data(TensorData::new(vec![10.0_f32, 20.0], [1, 2, 1, 1]), &device),
        );
        let config = PagedAttentionConfig::new(4, 2, 1).unwrap();
        let input = PagedAttentionInput::new(query, vec![page], config).unwrap();

        let output = PortablePagedAttention::new().execute(input).unwrap();

        assert_eq!(
            output.into_data().try_to_vec::<f32>().unwrap(),
            vec![10.0, 10.0, 20.0, 20.0]
        );
    }

    #[test]
    fn portable_descriptor_is_stable() {
        let descriptor = PortablePagedAttention::new().descriptor();

        assert_eq!(descriptor.name(), "portable-burn-paged-attention");
        assert_eq!(
            descriptor.operation(),
            KernelOperation::PagedAttentionDecode
        );
        assert_eq!(descriptor.target(), KernelTarget::Portable);
    }
}
