use std::sync::{Arc, Mutex, MutexGuard};

use burn::tensor::Tensor;
use ola_kernel::{PagedKvBlockTable, PagedKvBlocks};
use ola_model::KvCacheProfile;
use ola_runtime::{KvBlockError, KvBlockManager, KvSequenceId};

use super::LlamaConfig;

/// Capacity and page geometry for the Llama KV-cache pool.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LlamaKvCacheConfig {
    capacity_tokens: usize,
    tokens_per_block: usize,
    prefix_cache: bool,
}

impl LlamaKvCacheConfig {
    /// Creates a cache configuration.
    ///
    /// Capacity is shared by every live session created from one loaded model.
    /// Invalid or overflowing values are reported when the model is loaded.
    pub const fn new(capacity_tokens: usize, tokens_per_block: usize) -> Self {
        Self {
            capacity_tokens,
            tokens_per_block,
            prefix_cache: false,
        }
    }

    /// Keeps full KV blocks after their sessions end so later sessions with
    /// the same prompt prefix reuse them instead of recomputing.
    #[must_use]
    pub const fn with_prefix_cache(mut self) -> Self {
        self.prefix_cache = true;
        self
    }

    /// Uses one model context of capacity and 16 token positions per block.
    pub const fn for_model(config: &LlamaConfig) -> Self {
        Self::new(config.max_position_embeddings, 16)
    }

    /// Returns the requested minimum token capacity shared by all sessions.
    ///
    /// The physical pool rounds this up to a whole number of blocks.
    pub const fn capacity_tokens(self) -> usize {
        self.capacity_tokens
    }

    /// Returns the number of token positions in each physical block.
    pub const fn tokens_per_block(self) -> usize {
        self.tokens_per_block
    }

    pub(crate) fn profile(self, model: &LlamaConfig) -> Result<KvCacheProfile, KvBlockError> {
        if self.capacity_tokens == 0 {
            return Err(KvBlockError::InvalidConfiguration(
                "capacity_tokens must be greater than zero",
            ));
        }
        if self.tokens_per_block == 0 {
            return Err(KvBlockError::InvalidConfiguration(
                "tokens_per_block must be greater than zero",
            ));
        }
        let element_bytes = std::mem::size_of::<f32>() as u64;
        let bytes_per_token = checked_product(&[
            model.num_hidden_layers as u64,
            2,
            model.num_key_value_heads as u64,
            model.attention_head_dim() as u64,
            element_bytes,
        ])?;
        let total_blocks = self.capacity_tokens.div_ceil(self.tokens_per_block);
        let capacity_bytes = checked_product(&[
            total_blocks as u64,
            self.tokens_per_block as u64,
            bytes_per_token,
        ])?;
        KvCacheProfile::new(bytes_per_token, self.tokens_per_block, capacity_bytes).ok_or(
            KvBlockError::InvalidConfiguration("KV-cache profile must be non-empty"),
        )
    }
}

pub(crate) struct LlamaKvCache {
    pool: Arc<Mutex<LlamaKvPool>>,
    sequence: KvSequenceId,
    tokens: usize,
    block_ids: Vec<usize>,
    block_table: Option<PagedKvBlockTable>,
}

impl LlamaKvCache {
    pub(crate) fn allocate(pool: Arc<Mutex<LlamaKvPool>>) -> Result<Self, KvBlockError> {
        let sequence = lock_pool(&pool).manager.allocate()?;
        Ok(Self {
            pool,
            sequence,
            tokens: 0,
            block_ids: Vec::new(),
            block_table: None,
        })
    }

    pub(crate) const fn tokens(&self) -> usize {
        self.tokens
    }

    /// Attaches the longest cached prefix of `tokens` to this empty cache.
    pub(crate) fn attach_prefix(&mut self, tokens: &[u32]) -> Result<usize, KvBlockError> {
        let attached = lock_pool(&self.pool)
            .manager
            .attach_prefix(self.sequence, tokens)?;
        self.tokens = attached;
        self.refresh_block_table();
        Ok(attached)
    }

    /// Reports the content of positions whose KV has been written.
    pub(crate) fn record_tokens(&self, tokens: &[u32]) {
        let result = lock_pool(&self.pool)
            .manager
            .record_tokens(self.sequence, tokens);
        debug_assert!(result.is_ok(), "recorded content follows written positions");
    }

    pub(crate) fn reserve(&mut self, additional_tokens: usize) -> Result<(), KvBlockError> {
        let new_token_count = self
            .tokens
            .checked_add(additional_tokens)
            .ok_or(KvBlockError::TokenCountOverflow(self.sequence))?;
        lock_pool(&self.pool)
            .manager
            .append(self.sequence, additional_tokens)?;
        self.tokens = new_token_count;
        self.refresh_block_table();
        Ok(())
    }

    /// Atomically grows several caches backed by the same physical pool.
    pub(crate) fn reserve_batch(
        caches: &mut [&mut Self],
        additional_tokens: usize,
    ) -> Result<(), KvBlockError> {
        let additional_tokens = vec![additional_tokens; caches.len()];
        Self::reserve_batch_lengths(caches, &additional_tokens)
    }

    /// Atomically grows several caches by request-specific token counts.
    ///
    /// Variable-length prefill uses this boundary before any layer writes so a
    /// capacity failure cannot leave only a prefix of the batch reserved.
    pub(crate) fn reserve_batch_lengths(
        caches: &mut [&mut Self],
        additional_tokens: &[usize],
    ) -> Result<(), KvBlockError> {
        if caches.len() != additional_tokens.len() {
            return Err(KvBlockError::InvalidConfiguration(
                "batched cache and token-count lengths must match",
            ));
        }
        let Some(first) = caches.first() else {
            return Ok(());
        };
        let pool = first.pool.clone();
        if caches.iter().any(|cache| !Arc::ptr_eq(&pool, &cache.pool)) {
            return Err(KvBlockError::InvalidConfiguration(
                "batched caches must share one physical pool",
            ));
        }
        let mut reservations = Vec::with_capacity(caches.len());
        for (cache, &additional_tokens) in caches.iter().zip(additional_tokens) {
            cache
                .tokens
                .checked_add(additional_tokens)
                .ok_or(KvBlockError::TokenCountOverflow(cache.sequence))?;
            reservations.push((cache.sequence, additional_tokens));
        }
        lock_pool(&pool).manager.append_batch(&reservations)?;
        for (cache, &additional_tokens) in caches.iter_mut().zip(additional_tokens) {
            cache.tokens += additional_tokens;
            cache.refresh_block_table();
        }
        Ok(())
    }

    pub(crate) fn write_layer(
        &self,
        layer: usize,
        token_offset: usize,
        key: Tensor<4>,
        value: Tensor<4>,
    ) {
        lock_pool(&self.pool).write_layer(self.sequence, layer, token_offset, key, value);
    }

    pub(crate) fn read_layer(&self, layer: usize) -> (Vec<(Tensor<4>, Tensor<4>)>, PagedKvBlocks) {
        self.read_layer_prefix(layer, self.tokens)
    }

    /// Reads only the populated prefix of an atomically pre-reserved segment.
    pub(crate) fn read_layer_prefix(
        &self,
        layer: usize,
        token_count: usize,
    ) -> (Vec<(Tensor<4>, Tensor<4>)>, PagedKvBlocks) {
        let tokens_per_block = lock_pool(&self.pool).manager.tokens_per_block();
        let logical_blocks = token_count.div_ceil(tokens_per_block);
        let block_table = self
            .block_table
            .as_ref()
            .expect("a populated cache owns a device block table")
            .prefix(logical_blocks)
            .expect("a populated prefix must fit its reserved block table");
        lock_pool(&self.pool).read_layer_prefix(self.sequence, layer, token_count, block_table)
    }

    /// Returns physical page bytes attributed to this sequence.
    pub(crate) fn bytes(&self) -> u64 {
        let pool = lock_pool(&self.pool);
        let blocks = pool
            .manager
            .block_table(self.sequence)
            .expect("a live cache owns a block table")
            .len() as u64;
        blocks * pool.manager.metrics().block_bytes
    }

    fn refresh_block_table(&mut self) {
        let (block_ids, device) = {
            let pool = lock_pool(&self.pool);
            let block_ids = pool
                .manager
                .block_table(self.sequence)
                .expect("a live cache owns a block table")
                .iter()
                .map(|block| block.get())
                .collect::<Vec<_>>();
            (block_ids, pool.device.clone())
        };
        if block_ids == self.block_ids {
            return;
        }
        self.block_ids = block_ids;
        self.block_table = if self.block_ids.is_empty() {
            None
        } else {
            let physical_blocks = lock_pool(&self.pool).manager.metrics().total_blocks;
            Some(
                PagedKvBlockTable::new(self.block_ids.clone(), physical_blocks, &device)
                    .expect("the runtime block manager must produce valid physical indices"),
            )
        };
    }

    #[cfg(test)]
    pub(crate) fn block_ids(&self) -> Vec<usize> {
        lock_pool(&self.pool)
            .manager
            .block_table(self.sequence)
            .expect("a live cache owns a block table")
            .iter()
            .map(|block| block.get())
            .collect()
    }
}

impl Drop for LlamaKvCache {
    fn drop(&mut self) {
        let result = lock_pool(&self.pool).manager.free(self.sequence);
        debug_assert!(result.is_ok(), "a live cache sequence must be releasable");
    }
}

pub(crate) struct LlamaKvPool {
    manager: KvBlockManager,
    layers: Vec<LlamaLayerPages>,
    num_key_value_heads: usize,
    head_dim: usize,
    device: burn::tensor::Device,
}

struct LlamaLayerPages {
    keys: Tensor<4>,
    values: Tensor<4>,
}

impl LlamaKvPool {
    pub(crate) fn status(&self) -> ola_model::KvPoolStatus {
        let metrics = self.manager.metrics();
        ola_model::KvPoolStatus {
            total_blocks: metrics.total_blocks,
            free_blocks: metrics.free_blocks,
            allocated_blocks: metrics.allocated_blocks,
            cached_blocks: metrics.cached_blocks,
            tokens_per_block: self.manager.tokens_per_block(),
            live_sequences: metrics.live_sequences,
            evictions: metrics.evictions,
            prefix_hit_tokens: metrics.prefix_hit_tokens,
        }
    }

    pub(crate) fn cached_prefix(&self, tokens: &[u32]) -> ola_model::CachedPrefix {
        let found = self.manager.lookup_prefix(tokens);
        ola_model::CachedPrefix {
            tokens: found.tokens,
            evictable_blocks: found.evictable_blocks,
        }
    }

    pub(crate) fn new(
        model: &LlamaConfig,
        config: LlamaKvCacheConfig,
        device: burn::tensor::Device,
    ) -> Result<Self, KvBlockError> {
        let profile = config.profile(model)?;
        let mut manager = KvBlockManager::new(
            profile.capacity_bytes(),
            profile.tokens_per_block(),
            profile.bytes_per_token(),
        )?;
        if config.prefix_cache {
            manager = manager.with_prefix_cache();
        }
        let physical_blocks = manager.metrics().total_blocks;
        let page_shape = [
            physical_blocks,
            model.num_key_value_heads,
            config.tokens_per_block,
            model.attention_head_dim(),
        ];
        let layers = (0..model.num_hidden_layers)
            .map(|_| LlamaLayerPages {
                keys: Tensor::zeros(page_shape, &device),
                values: Tensor::zeros(page_shape, &device),
            })
            .collect();
        Ok(Self {
            manager,
            layers,
            num_key_value_heads: model.num_key_value_heads,
            head_dim: model.attention_head_dim(),
            device,
        })
    }

    fn write_layer(
        &mut self,
        sequence: KvSequenceId,
        layer: usize,
        token_offset: usize,
        key: Tensor<4>,
        value: Tensor<4>,
    ) {
        let [batch, heads, token_count, head_dim] = key.dims();
        debug_assert_eq!(value.dims(), key.dims());
        debug_assert_eq!(batch, 1);
        debug_assert_eq!(heads, self.num_key_value_heads);
        debug_assert_eq!(head_dim, self.head_dim);
        debug_assert!(token_offset + token_count <= self.manager.cached_tokens(sequence).unwrap());

        let tokens_per_block = self.manager.tokens_per_block();
        let block_table = self
            .manager
            .block_table(sequence)
            .expect("a reserved cache owns a block table")
            .to_vec();
        let layer = self
            .layers
            .get_mut(layer)
            .expect("model and cache layer counts must agree");
        let mut source_offset = 0;
        while source_offset < token_count {
            let logical_position = token_offset + source_offset;
            let logical_block = logical_position / tokens_per_block;
            let block_offset = logical_position % tokens_per_block;
            let physical_block = block_table[logical_block].get();
            let count = (tokens_per_block - block_offset).min(token_count - source_offset);
            let key_values = key.clone().slice([
                0..1,
                0..heads,
                source_offset..source_offset + count,
                0..head_dim,
            ]);
            let value_values = value.clone().slice([
                0..1,
                0..heads,
                source_offset..source_offset + count,
                0..head_dim,
            ]);
            layer.keys = layer.keys.clone().slice_assign(
                [
                    physical_block..physical_block + 1,
                    0..heads,
                    block_offset..block_offset + count,
                    0..head_dim,
                ],
                key_values,
            );
            layer.values = layer.values.clone().slice_assign(
                [
                    physical_block..physical_block + 1,
                    0..heads,
                    block_offset..block_offset + count,
                    0..head_dim,
                ],
                value_values,
            );
            source_offset += count;
        }
    }

    fn read_layer_prefix(
        &self,
        sequence: KvSequenceId,
        layer: usize,
        token_count: usize,
        block_table: PagedKvBlockTable,
    ) -> (Vec<(Tensor<4>, Tensor<4>)>, PagedKvBlocks) {
        let reserved_tokens = self
            .manager
            .cached_tokens(sequence)
            .expect("a live cache owns token accounting");
        debug_assert!(token_count <= reserved_tokens);
        debug_assert!(token_count > 0);
        let tokens_per_block = self.manager.tokens_per_block();
        let layer = self
            .layers
            .get(layer)
            .expect("model and cache layer counts must agree");
        let mut pages = Vec::new();
        // Blocks reserved past the populated prefix (a chunk is reserved
        // whole, then read position by position) are not read.
        for (logical_block, block) in self
            .manager
            .block_table(sequence)
            .expect("a live cache owns a block table")
            .iter()
            .enumerate()
            .take(token_count.div_ceil(tokens_per_block))
        {
            let remaining = token_count - logical_block * tokens_per_block;
            let used = remaining.min(tokens_per_block);
            let physical = block.get();
            let key = layer.keys.clone().slice([
                physical..physical + 1,
                0..self.num_key_value_heads,
                0..used,
                0..self.head_dim,
            ]);
            let value = layer.values.clone().slice([
                physical..physical + 1,
                0..self.num_key_value_heads,
                0..used,
                0..self.head_dim,
            ]);
            pages.push((key, value));
        }
        let blocks = PagedKvBlocks::new(
            layer.keys.clone(),
            layer.values.clone(),
            block_table,
            tokens_per_block,
            token_count,
        )
        .expect("Llama cache storage must satisfy the paged-attention block contract");
        (pages, blocks)
    }
}

fn checked_product(values: &[u64]) -> Result<u64, KvBlockError> {
    values.iter().try_fold(1_u64, |product, value| {
        product
            .checked_mul(*value)
            .ok_or(KvBlockError::ConfigurationOverflow)
    })
}

fn lock_pool(pool: &Mutex<LlamaKvPool>) -> MutexGuard<'_, LlamaKvPool> {
    pool.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;

    use super::*;

    fn tiny_config() -> LlamaConfig {
        LlamaConfig {
            model_type: "llama".into(),
            vocab_size: 8,
            hidden_size: 4,
            intermediate_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            max_position_embeddings: 8,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            head_dim: None,
            hidden_act: "silu".into(),
            attention_bias: false,
            mlp_bias: false,
            tie_word_embeddings: false,
            rope_scaling: None,
            bos_token_id: Some(1),
            eos_token_id: vec![2],
        }
    }

    #[test]
    fn writes_and_reads_across_non_contiguous_physical_pages() {
        let pool = LlamaKvPool::new(
            &tiny_config(),
            LlamaKvCacheConfig::new(6, 2),
            Default::default(),
        )
        .unwrap();
        let device = pool.device.clone();
        let pool = Arc::new(Mutex::new(pool));
        let mut first = LlamaKvCache::allocate(pool.clone()).unwrap();
        let mut second = LlamaKvCache::allocate(pool).unwrap();

        first.reserve(3).unwrap();
        second.reserve(2).unwrap();
        assert_eq!(first.block_ids(), vec![0, 1]);
        assert_eq!(second.block_ids(), vec![2]);

        let keys = Tensor::from_data(
            TensorData::new(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], [1, 1, 3, 2]),
            &device,
        );
        let values = Tensor::from_data(
            TensorData::new(vec![11.0_f32, 12.0, 13.0, 14.0, 15.0, 16.0], [1, 1, 3, 2]),
            &device,
        );
        first.write_layer(0, 0, keys, values);
        let (pages, _blocks) = first.read_layer(0);
        let (keys, values): (Vec<_>, Vec<_>) = pages.into_iter().unzip();
        let keys = Tensor::cat(keys, 2);
        let values = Tensor::cat(values, 2);

        assert_eq!(
            keys.into_data().try_to_vec::<f32>().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
        assert_eq!(
            values.into_data().try_to_vec::<f32>().unwrap(),
            vec![11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
        );
    }

    #[test]
    fn released_pages_are_reused_and_capacity_failure_is_atomic() {
        let pool = LlamaKvPool::new(
            &tiny_config(),
            LlamaKvCacheConfig::new(4, 2),
            Default::default(),
        )
        .unwrap();
        let pool = Arc::new(Mutex::new(pool));
        let mut first = LlamaKvCache::allocate(pool.clone()).unwrap();
        let mut waiting = LlamaKvCache::allocate(pool.clone()).unwrap();
        first.reserve(4).unwrap();

        assert!(matches!(
            waiting.reserve(1),
            Err(KvBlockError::InsufficientCapacity { .. })
        ));
        assert_eq!(waiting.tokens(), 0);
        assert!(waiting.block_ids().is_empty());

        drop(first);
        waiting.reserve(1).unwrap();
        assert_eq!(waiting.block_ids(), vec![0]);
    }

    #[test]
    fn variable_length_batch_reservation_is_atomic() {
        let pool = LlamaKvPool::new(
            &tiny_config(),
            LlamaKvCacheConfig::new(8, 2),
            Default::default(),
        )
        .unwrap();
        let pool = Arc::new(Mutex::new(pool));
        let mut first = LlamaKvCache::allocate(pool.clone()).unwrap();
        let mut second = LlamaKvCache::allocate(pool).unwrap();

        LlamaKvCache::reserve_batch_lengths(&mut [&mut first, &mut second], &[1, 3]).unwrap();
        assert_eq!((first.tokens(), second.tokens()), (1, 3));
        assert_eq!(
            (first.block_ids(), second.block_ids()),
            (vec![0], vec![1, 2])
        );

        let error = LlamaKvCache::reserve_batch_lengths(&mut [&mut first, &mut second], &[2, 2])
            .unwrap_err();

        assert!(matches!(
            error,
            KvBlockError::BatchInsufficientCapacity { .. }
        ));
        assert_eq!((first.tokens(), second.tokens()), (1, 3));
        assert_eq!(
            (first.block_ids(), second.block_ids()),
            (vec![0], vec![1, 2])
        );
    }
}
