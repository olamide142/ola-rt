use std::{collections::BTreeMap, collections::BTreeSet, error::Error, fmt};

/// Identifies one live logical sequence in a [`KvBlockManager`].
///
/// This identifier is deliberately distinct from `RequestId`: a request may
/// eventually own multiple cache entries, and prefix-cache entries may outlive
/// the request that first computed them.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KvSequenceId(u64);

impl KvSequenceId {
    /// Returns the manager-local numeric identifier.
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl fmt::Display for KvSequenceId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

/// Identifies one fixed-size physical block in a KV-cache pool.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KvBlockId(usize);

impl KvBlockId {
    /// Returns the zero-based physical block index.
    pub const fn get(self) -> usize {
        self.0
    }
}

impl fmt::Display for KvBlockId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SequenceAllocation {
    cached_tokens: usize,
    blocks: Vec<KvBlockId>,
    /// Token content whose KV has been written, recorded only while the
    /// prefix cache is enabled.
    content: Vec<u32>,
    /// Leading blocks whose content has been hashed.
    hashed_blocks: usize,
    /// Chained hash of the last hashed block.
    last_hash: u64,
}

impl SequenceAllocation {
    const fn empty() -> Self {
        Self {
            cached_tokens: 0,
            blocks: Vec::new(),
            content: Vec::new(),
            hashed_blocks: 0,
            last_hash: PREFIX_ROOT,
        }
    }
}

/// Identity of a full, immutable block in the prefix cache: its content and
/// everything before it.
#[derive(Clone, Debug, Eq, PartialEq)]
struct CachedBlock {
    hash: u64,
    parent: u64,
    tokens: Vec<u32>,
}

/// Chain seed for the first block of every sequence.
const PREFIX_ROOT: u64 = 0xcbf2_9ce4_8422_2325;

/// FNV-1a over the parent hash and one block's token IDs.
fn chain_hash(parent: u64, tokens: &[u32]) -> u64 {
    let mut hash = PREFIX_ROOT;
    for byte in parent
        .to_le_bytes()
        .into_iter()
        .chain(tokens.iter().flat_map(|token| token.to_le_bytes()))
    {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

/// A cached prefix found for a token sequence.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PrefixMatch {
    /// Leading tokens whose KV is cached, a whole number of blocks.
    pub tokens: usize,
    /// Matched blocks no live sequence holds, which attaching takes off the
    /// evictable list.
    pub evictable_blocks: usize,
}

/// Snapshot of block-pool occupancy and internal fragmentation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct KvBlockMetrics {
    /// Bytes supplied when the pool was constructed.
    pub capacity_bytes: u64,
    /// Bytes in one allocatable block.
    pub block_bytes: u64,
    /// Number of blocks represented by the configured capacity.
    pub total_blocks: usize,
    /// Blocks currently assigned to live sequences.
    pub allocated_blocks: usize,
    /// Blocks immediately available for allocation.
    pub free_blocks: usize,
    /// Bytes reserved by allocated blocks.
    pub allocated_bytes: u64,
    /// Bytes represented by blocks on the free list.
    pub free_bytes: u64,
    /// Capacity remainder too small to form a complete block.
    pub unusable_bytes: u64,
    /// Number of logical cache sequences, including empty sequences.
    pub live_sequences: usize,
    /// Actual token positions stored across all live sequences; a shared
    /// prefix counts once per sequence holding it.
    pub cached_tokens: usize,
    /// Token slots reserved by allocated blocks.
    pub reserved_token_slots: usize,
    /// Reserved token slots not yet occupied by tokens.
    pub internal_fragmentation_tokens: usize,
    /// Bytes corresponding to unoccupied slots in allocated blocks.
    pub internal_fragmentation_bytes: u64,
    /// Full blocks no live sequence holds, kept for prefix reuse and evicted
    /// least recently used first when an allocation needs them.
    pub cached_blocks: usize,
    /// Cached blocks evicted to satisfy allocations.
    pub evictions: u64,
    /// Prefix attachments attempted.
    pub prefix_lookups: u64,
    /// Attachments that reused at least one block.
    pub prefix_hits: u64,
    /// Tokens whose KV was reused instead of recomputed.
    pub prefix_hit_tokens: u64,
}

/// Summary returned after releasing one logical sequence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvRelease {
    /// Number of cached token positions discarded.
    pub cached_tokens: usize,
    /// Number of physical blocks returned to the pool.
    pub released_blocks: usize,
    /// Bytes returned to the pool.
    pub released_bytes: u64,
}

/// Deterministic metadata allocator for fixed-size KV-cache blocks.
///
/// The manager owns block assignment and accounting, not tensor memory. A
/// backend can use each sequence's block table to map logical token positions
/// onto CPU, CUDA, or simulated physical storage. Allocation always chooses the
/// lowest free block ID, making experiments and tests reproducible.
#[derive(Clone, Debug)]
pub struct KvBlockManager {
    capacity_bytes: u64,
    bytes_per_token: u64,
    tokens_per_block: usize,
    block_bytes: u64,
    total_blocks: usize,
    free_blocks: BTreeSet<KvBlockId>,
    sequences: BTreeMap<KvSequenceId, SequenceAllocation>,
    next_sequence_id: Option<u64>,
    prefix_cache: bool,
    /// Live sequences holding each block.
    references: Vec<u32>,
    /// Identity of every block registered in `index`.
    cached: BTreeMap<KvBlockId, CachedBlock>,
    /// Chained content hash to the block holding that content.
    index: BTreeMap<u64, KvBlockId>,
    /// Unreferenced cached blocks by last use, oldest first.
    lru: BTreeMap<u64, KvBlockId>,
    lru_tick: BTreeMap<KvBlockId, u64>,
    tick: u64,
    evictions: u64,
    prefix_lookups: u64,
    prefix_hits: u64,
    prefix_hit_tokens: u64,
}

impl KvBlockManager {
    /// Creates a fixed-size block pool.
    ///
    /// `bytes_per_token` must include keys and values for every layer managed by
    /// this pool. Capacity that cannot form a complete block remains visible as
    /// `unusable_bytes` in [`metrics`](Self::metrics).
    pub fn new(
        capacity_bytes: u64,
        tokens_per_block: usize,
        bytes_per_token: u64,
    ) -> Result<Self, KvBlockError> {
        if capacity_bytes == 0 {
            return Err(KvBlockError::InvalidConfiguration(
                "capacity_bytes must be greater than zero",
            ));
        }
        if tokens_per_block == 0 {
            return Err(KvBlockError::InvalidConfiguration(
                "tokens_per_block must be greater than zero",
            ));
        }
        if bytes_per_token == 0 {
            return Err(KvBlockError::InvalidConfiguration(
                "bytes_per_token must be greater than zero",
            ));
        }
        let block_bytes = u64::try_from(tokens_per_block)
            .ok()
            .and_then(|tokens| tokens.checked_mul(bytes_per_token))
            .ok_or(KvBlockError::ConfigurationOverflow)?;
        let total_blocks_u64 = capacity_bytes / block_bytes;
        if total_blocks_u64 == 0 {
            return Err(KvBlockError::CapacityTooSmall {
                capacity_bytes,
                block_bytes,
            });
        }
        let total_blocks =
            usize::try_from(total_blocks_u64).map_err(|_| KvBlockError::ConfigurationOverflow)?;
        total_blocks
            .checked_mul(tokens_per_block)
            .ok_or(KvBlockError::ConfigurationOverflow)?;
        let free_blocks = (0..total_blocks).map(KvBlockId).collect();
        Ok(Self {
            capacity_bytes,
            bytes_per_token,
            tokens_per_block,
            block_bytes,
            total_blocks,
            free_blocks,
            sequences: BTreeMap::new(),
            next_sequence_id: Some(0),
            prefix_cache: false,
            references: vec![0; total_blocks],
            cached: BTreeMap::new(),
            index: BTreeMap::new(),
            lru: BTreeMap::new(),
            lru_tick: BTreeMap::new(),
            tick: 0,
            evictions: 0,
            prefix_lookups: 0,
            prefix_hits: 0,
            prefix_hit_tokens: 0,
        })
    }

    /// Keeps full blocks after their sequences end so later sequences with
    /// the same prefix can reuse their KV.
    ///
    /// Only full blocks whose content was reported through
    /// [`record_tokens`](Self::record_tokens) are shared, and a shared block
    /// is never written again: a sequence appends only after its last block,
    /// which a prefix attachment always leaves full. Matches are verified
    /// token by token, so a hash collision cannot share the wrong KV.
    #[must_use]
    pub const fn with_prefix_cache(mut self) -> Self {
        self.prefix_cache = true;
        self
    }

    /// Returns whether the prefix cache is enabled.
    pub const fn prefix_cache_enabled(&self) -> bool {
        self.prefix_cache
    }

    /// Blocks an allocation can obtain: free ones plus evictable cached ones.
    fn available_blocks(&self) -> usize {
        self.free_blocks.len() + self.lru.len()
    }

    /// Takes `count` blocks, free ones first, then evicting the least
    /// recently used cached blocks. The caller checked availability.
    fn take_blocks(&mut self, count: usize) -> Vec<KvBlockId> {
        let mut taken = Vec::with_capacity(count);
        for _ in 0..count {
            let block = match self.free_blocks.pop_first() {
                Some(block) => block,
                None => {
                    let (_, block) = self
                        .lru
                        .pop_first()
                        .expect("availability was checked before allocation");
                    self.lru_tick.remove(&block);
                    if let Some(identity) = self.cached.remove(&block) {
                        self.index.remove(&identity.hash);
                    }
                    self.evictions += 1;
                    block
                }
            };
            self.references[block.0] = 1;
            taken.push(block);
        }
        taken
    }

    /// Walks the cached chain for `tokens`, returning matched blocks.
    fn matching_blocks(&self, tokens: &[u32]) -> Vec<KvBlockId> {
        let mut matched = Vec::new();
        if !self.prefix_cache {
            return matched;
        }
        let mut parent = PREFIX_ROOT;
        for chunk in tokens.chunks_exact(self.tokens_per_block) {
            let hash = chain_hash(parent, chunk);
            let Some(&block) = self.index.get(&hash) else {
                break;
            };
            let identity = &self.cached[&block];
            if identity.parent != parent || identity.tokens != chunk {
                break;
            }
            matched.push(block);
            parent = hash;
        }
        matched
    }

    /// Returns the longest cached prefix of `tokens` without changing the
    /// pool. At most `tokens.len()` positions are matched; callers that must
    /// still compute a token pass a shorter slice.
    pub fn lookup_prefix(&self, tokens: &[u32]) -> PrefixMatch {
        let matched = self.matching_blocks(tokens);
        PrefixMatch {
            tokens: matched.len() * self.tokens_per_block,
            evictable_blocks: matched
                .iter()
                .filter(|block| self.references[block.0] == 0)
                .count(),
        }
    }

    /// Attaches the longest cached prefix of `tokens` to an empty sequence
    /// and returns how many tokens it covers; their KV need not be
    /// recomputed. Returns 0 when the cache is disabled or nothing matches.
    pub fn attach_prefix(
        &mut self,
        sequence: KvSequenceId,
        tokens: &[u32],
    ) -> Result<usize, KvBlockError> {
        let allocation = self
            .sequences
            .get(&sequence)
            .ok_or(KvBlockError::UnknownSequence(sequence))?;
        if allocation.cached_tokens != 0 {
            return Err(KvBlockError::InvalidConfiguration(
                "a prefix can only be attached to an empty sequence",
            ));
        }
        if !self.prefix_cache {
            return Ok(0);
        }
        self.prefix_lookups += 1;
        let matched = self.matching_blocks(tokens);
        for block in &matched {
            if self.references[block.0] == 0
                && let Some(tick) = self.lru_tick.remove(block)
            {
                self.lru.remove(&tick);
            }
            self.references[block.0] += 1;
        }
        let reused = matched.len() * self.tokens_per_block;
        if reused > 0 {
            self.prefix_hits += 1;
            self.prefix_hit_tokens += reused as u64;
        }
        let last_hash = matched
            .last()
            .map_or(PREFIX_ROOT, |block| self.cached[block].hash);
        let allocation = self
            .sequences
            .get_mut(&sequence)
            .expect("sequence existence was checked");
        allocation.hashed_blocks = matched.len();
        allocation.blocks = matched;
        allocation.cached_tokens = reused;
        allocation.content = tokens[..reused].to_vec();
        allocation.last_hash = last_hash;
        Ok(reused)
    }

    /// Records the token content of positions whose KV has been written,
    /// making each newly completed block available to later sequences.
    /// Content must be reported in order and never beyond the positions
    /// already appended. Does nothing while the prefix cache is disabled.
    pub fn record_tokens(
        &mut self,
        sequence: KvSequenceId,
        tokens: &[u32],
    ) -> Result<(), KvBlockError> {
        if !self.prefix_cache {
            return Ok(());
        }
        let block = self.tokens_per_block;
        let allocation = self
            .sequences
            .get_mut(&sequence)
            .ok_or(KvBlockError::UnknownSequence(sequence))?;
        if allocation.content.len() + tokens.len() > allocation.cached_tokens {
            return Err(KvBlockError::InvalidConfiguration(
                "token content cannot exceed the positions appended",
            ));
        }
        allocation.content.extend_from_slice(tokens);
        while (allocation.hashed_blocks + 1) * block <= allocation.content.len() {
            let index = allocation.hashed_blocks;
            let content = &allocation.content[index * block..(index + 1) * block];
            let hash = chain_hash(allocation.last_hash, content);
            let physical = allocation.blocks[index];
            // The first block to hold some content is the one shared; an
            // identical later block stays private and is freed normally.
            if let std::collections::btree_map::Entry::Vacant(entry) = self.index.entry(hash) {
                entry.insert(physical);
                self.cached.insert(
                    physical,
                    CachedBlock {
                        hash,
                        parent: allocation.last_hash,
                        tokens: content.to_vec(),
                    },
                );
            }
            allocation.last_hash = hash;
            allocation.hashed_blocks += 1;
        }
        Ok(())
    }

    /// Creates an empty logical cache and returns its manager-local ID.
    pub fn allocate(&mut self) -> Result<KvSequenceId, KvBlockError> {
        let value = self
            .next_sequence_id
            .ok_or(KvBlockError::SequenceIdExhausted)?;
        self.next_sequence_id = value.checked_add(1);
        let sequence = KvSequenceId(value);
        self.sequences.insert(sequence, SequenceAllocation::empty());
        Ok(sequence)
    }

    /// Returns the number of logical token positions stored in one block.
    pub const fn tokens_per_block(&self) -> usize {
        self.tokens_per_block
    }

    /// Returns the configured storage cost of one token position.
    pub const fn bytes_per_token(&self) -> u64 {
        self.bytes_per_token
    }

    /// Reserves enough blocks to append `additional_tokens` atomically.
    ///
    /// On failure, neither the block table nor the cached-token count changes.
    pub fn append(
        &mut self,
        sequence: KvSequenceId,
        additional_tokens: usize,
    ) -> Result<(), KvBlockError> {
        let allocation = self
            .sequences
            .get(&sequence)
            .ok_or(KvBlockError::UnknownSequence(sequence))?;
        let new_token_count = allocation
            .cached_tokens
            .checked_add(additional_tokens)
            .ok_or(KvBlockError::TokenCountOverflow(sequence))?;
        let required_blocks = blocks_for_tokens(new_token_count, self.tokens_per_block);
        let additional_blocks = required_blocks - allocation.blocks.len();
        if additional_blocks > self.available_blocks() {
            return Err(KvBlockError::InsufficientCapacity {
                sequence,
                requested_blocks: additional_blocks,
                available_blocks: self.available_blocks(),
            });
        }

        let acquired = self.take_blocks(additional_blocks);
        let allocation = self
            .sequences
            .get_mut(&sequence)
            .expect("sequence existence was checked before block allocation");
        allocation.blocks.extend(acquired);
        allocation.cached_tokens = new_token_count;
        Ok(())
    }

    /// Atomically reserves token growth for several logical sequences.
    ///
    /// Every sequence is validated and total block demand is checked before
    /// any block table or token count changes. Sequence IDs may appear only
    /// once in a batch. This is the allocator boundary used by tensor-batched
    /// decode so one capacity failure cannot advance only part of a batch.
    pub fn append_batch(&mut self, requests: &[(KvSequenceId, usize)]) -> Result<(), KvBlockError> {
        let mut seen = BTreeSet::new();
        let mut total_additional_blocks = 0_usize;
        let mut plans = Vec::with_capacity(requests.len());
        for &(sequence, additional_tokens) in requests {
            if !seen.insert(sequence) {
                return Err(KvBlockError::DuplicateSequence(sequence));
            }
            let allocation = self
                .sequences
                .get(&sequence)
                .ok_or(KvBlockError::UnknownSequence(sequence))?;
            let new_token_count = allocation
                .cached_tokens
                .checked_add(additional_tokens)
                .ok_or(KvBlockError::TokenCountOverflow(sequence))?;
            let required_blocks = blocks_for_tokens(new_token_count, self.tokens_per_block);
            let additional_blocks = required_blocks - allocation.blocks.len();
            total_additional_blocks = total_additional_blocks
                .checked_add(additional_blocks)
                .ok_or(KvBlockError::ConfigurationOverflow)?;
            plans.push((sequence, new_token_count, additional_blocks));
        }
        if total_additional_blocks > self.available_blocks() {
            return Err(KvBlockError::BatchInsufficientCapacity {
                requested_blocks: total_additional_blocks,
                available_blocks: self.available_blocks(),
            });
        }

        for (sequence, new_token_count, additional_blocks) in plans {
            let acquired = self.take_blocks(additional_blocks);
            let allocation = self
                .sequences
                .get_mut(&sequence)
                .expect("batch sequence existence was validated before mutation");
            allocation.blocks.extend(acquired);
            allocation.cached_tokens = new_token_count;
        }
        Ok(())
    }

    /// Releases a sequence. A block no other sequence holds returns to the
    /// free list, or, when it holds cached content, to the evictable list.
    pub fn free(&mut self, sequence: KvSequenceId) -> Result<KvRelease, KvBlockError> {
        let allocation = self
            .sequences
            .remove(&sequence)
            .ok_or(KvBlockError::UnknownSequence(sequence))?;
        let released_blocks = allocation.blocks.len();
        for block in allocation.blocks {
            let references = &mut self.references[block.0];
            debug_assert!(*references > 0, "a held block has a reference");
            *references = references.saturating_sub(1);
            if *references > 0 {
                continue;
            }
            if self.cached.contains_key(&block) {
                self.tick += 1;
                self.lru.insert(self.tick, block);
                self.lru_tick.insert(block, self.tick);
            } else {
                let inserted = self.free_blocks.insert(block);
                debug_assert!(inserted, "an allocated block cannot already be free");
            }
        }
        Ok(KvRelease {
            cached_tokens: allocation.cached_tokens,
            released_blocks,
            released_bytes: released_blocks as u64 * self.block_bytes,
        })
    }

    /// Returns the physical block table for a logical sequence.
    pub fn block_table(&self, sequence: KvSequenceId) -> Result<&[KvBlockId], KvBlockError> {
        self.sequences
            .get(&sequence)
            .map(|allocation| allocation.blocks.as_slice())
            .ok_or(KvBlockError::UnknownSequence(sequence))
    }

    /// Returns the number of cached tokens owned by a logical sequence.
    pub fn cached_tokens(&self, sequence: KvSequenceId) -> Result<usize, KvBlockError> {
        self.sequences
            .get(&sequence)
            .map(|allocation| allocation.cached_tokens)
            .ok_or(KvBlockError::UnknownSequence(sequence))
    }

    /// Returns a cheap pool-wide accounting snapshot.
    pub fn metrics(&self) -> KvBlockMetrics {
        let allocated_blocks = self.total_blocks - self.free_blocks.len() - self.lru.len();
        let cached_tokens = self
            .sequences
            .values()
            .map(|allocation| allocation.cached_tokens)
            .sum::<usize>();
        let reserved_token_slots = allocated_blocks * self.tokens_per_block;
        // Shared prefix blocks count once in slots but once per sequence in
        // cached tokens, so with sharing this undercounts and can reach zero.
        let internal_fragmentation_tokens = reserved_token_slots.saturating_sub(cached_tokens);
        let represented_bytes = self.total_blocks as u64 * self.block_bytes;
        KvBlockMetrics {
            capacity_bytes: self.capacity_bytes,
            block_bytes: self.block_bytes,
            total_blocks: self.total_blocks,
            allocated_blocks,
            free_blocks: self.free_blocks.len(),
            allocated_bytes: allocated_blocks as u64 * self.block_bytes,
            free_bytes: self.free_blocks.len() as u64 * self.block_bytes,
            unusable_bytes: self.capacity_bytes - represented_bytes,
            live_sequences: self.sequences.len(),
            cached_tokens,
            reserved_token_slots,
            internal_fragmentation_tokens,
            internal_fragmentation_bytes: internal_fragmentation_tokens as u64
                * self.bytes_per_token,
            cached_blocks: self.lru.len(),
            evictions: self.evictions,
            prefix_lookups: self.prefix_lookups,
            prefix_hits: self.prefix_hits,
            prefix_hit_tokens: self.prefix_hit_tokens,
        }
    }
}

fn blocks_for_tokens(tokens: usize, tokens_per_block: usize) -> usize {
    tokens / tokens_per_block + usize::from(!tokens.is_multiple_of(tokens_per_block))
}

/// Failure to configure or mutate a KV block pool.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KvBlockError {
    /// One configuration field was zero or otherwise invalid.
    InvalidConfiguration(&'static str),
    /// Configuration arithmetic or platform conversion overflowed.
    ConfigurationOverflow,
    /// Capacity cannot hold even one complete block.
    CapacityTooSmall {
        /// Configured pool capacity.
        capacity_bytes: u64,
        /// Bytes required by one block.
        block_bytes: u64,
    },
    /// Monotonic logical sequence IDs were exhausted.
    SequenceIdExhausted,
    /// No live cache has the supplied identifier.
    UnknownSequence(KvSequenceId),
    /// Appending would overflow the logical token count.
    TokenCountOverflow(KvSequenceId),
    /// The append requires more blocks than the pool can provide.
    InsufficientCapacity {
        /// Sequence whose growth was attempted.
        sequence: KvSequenceId,
        /// Additional blocks needed for the append.
        requested_blocks: usize,
        /// Blocks free at the time of the request.
        available_blocks: usize,
    },
    /// One atomic batch referenced the same sequence more than once.
    DuplicateSequence(KvSequenceId),
    /// Combined batch growth exceeded immediately free physical blocks.
    BatchInsufficientCapacity {
        /// Total additional blocks needed by the batch.
        requested_blocks: usize,
        /// Blocks free before any batch mutation.
        available_blocks: usize,
    },
}

impl fmt::Display for KvBlockError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfiguration(message) => {
                write!(formatter, "invalid KV block configuration: {message}")
            }
            Self::ConfigurationOverflow => {
                formatter.write_str("KV block configuration exceeds supported numeric limits")
            }
            Self::CapacityTooSmall {
                capacity_bytes,
                block_bytes,
            } => write!(
                formatter,
                "KV capacity {capacity_bytes} bytes cannot hold one {block_bytes}-byte block"
            ),
            Self::SequenceIdExhausted => formatter.write_str("KV sequence ID space is exhausted"),
            Self::UnknownSequence(sequence) => {
                write!(formatter, "KV sequence `{sequence}` does not exist")
            }
            Self::TokenCountOverflow(sequence) => {
                write!(
                    formatter,
                    "token count for KV sequence `{sequence}` overflowed"
                )
            }
            Self::InsufficientCapacity {
                sequence,
                requested_blocks,
                available_blocks,
            } => write!(
                formatter,
                "KV sequence `{sequence}` needs {requested_blocks} additional blocks but only {available_blocks} are free"
            ),
            Self::DuplicateSequence(sequence) => write!(
                formatter,
                "KV batch references sequence `{sequence}` more than once"
            ),
            Self::BatchInsufficientCapacity {
                requested_blocks,
                available_blocks,
            } => write!(
                formatter,
                "KV batch needs {requested_blocks} additional blocks but only {available_blocks} are free"
            ),
        }
    }
}

impl Error for KvBlockError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocates_logical_pages_and_reuses_freed_physical_blocks() {
        let mut manager = KvBlockManager::new(32, 4, 2).unwrap();
        let first = manager.allocate().unwrap();
        let second = manager.allocate().unwrap();

        manager.append(first, 5).unwrap();
        manager.append(second, 4).unwrap();
        manager.append(first, 4).unwrap();

        assert_eq!(manager.cached_tokens(first), Ok(9));
        assert_eq!(
            manager.block_table(first).unwrap(),
            [KvBlockId(0), KvBlockId(1), KvBlockId(3)]
        );
        assert_eq!(manager.block_table(second).unwrap(), [KvBlockId(2)]);

        let before_failure = manager.metrics();
        assert_eq!(
            manager.append(second, 1),
            Err(KvBlockError::InsufficientCapacity {
                sequence: second,
                requested_blocks: 1,
                available_blocks: 0,
            })
        );
        assert_eq!(manager.metrics(), before_failure);
        assert_eq!(manager.cached_tokens(second), Ok(4));

        assert_eq!(
            manager.free(first),
            Ok(KvRelease {
                cached_tokens: 9,
                released_blocks: 3,
                released_bytes: 24,
            })
        );
        manager.append(second, 1).unwrap();
        assert_eq!(
            manager.block_table(second).unwrap(),
            [KvBlockId(2), KvBlockId(0)]
        );
    }

    #[test]
    fn reports_capacity_occupancy_and_internal_fragmentation() {
        let mut manager = KvBlockManager::new(35, 4, 2).unwrap();
        let first = manager.allocate().unwrap();
        let empty = manager.allocate().unwrap();
        manager.append(first, 5).unwrap();

        assert_eq!(manager.block_table(empty), Ok([].as_slice()));
        assert_eq!(
            manager.metrics(),
            KvBlockMetrics {
                capacity_bytes: 35,
                block_bytes: 8,
                total_blocks: 4,
                allocated_blocks: 2,
                free_blocks: 2,
                allocated_bytes: 16,
                free_bytes: 16,
                unusable_bytes: 3,
                live_sequences: 2,
                cached_tokens: 5,
                reserved_token_slots: 8,
                internal_fragmentation_tokens: 3,
                internal_fragmentation_bytes: 6,
                ..KvBlockMetrics::default()
            }
        );
    }

    #[test]
    fn batch_append_is_atomic_across_sequences() {
        let mut manager = KvBlockManager::new(16, 4, 1).unwrap();
        let first = manager.allocate().unwrap();
        let second = manager.allocate().unwrap();

        manager.append_batch(&[(first, 1), (second, 5)]).unwrap();
        assert_eq!(manager.cached_tokens(first), Ok(1));
        assert_eq!(manager.cached_tokens(second), Ok(5));
        assert_eq!(manager.metrics().free_blocks, 1);

        let before_failure = manager.metrics();
        assert_eq!(
            manager.append_batch(&[(first, 4), (second, 4)]),
            Err(KvBlockError::BatchInsufficientCapacity {
                requested_blocks: 2,
                available_blocks: 1,
            })
        );
        assert_eq!(manager.metrics(), before_failure);
        assert_eq!(manager.cached_tokens(first), Ok(1));
        assert_eq!(manager.cached_tokens(second), Ok(5));
        assert_eq!(
            manager.append_batch(&[(first, 1), (first, 1)]),
            Err(KvBlockError::DuplicateSequence(first))
        );
    }

    #[test]
    fn rejects_invalid_pool_configurations() {
        assert!(matches!(
            KvBlockManager::new(0, 4, 2),
            Err(KvBlockError::InvalidConfiguration(_))
        ));
        assert!(matches!(
            KvBlockManager::new(8, 0, 2),
            Err(KvBlockError::InvalidConfiguration(_))
        ));
        assert!(matches!(
            KvBlockManager::new(7, 4, 2),
            Err(KvBlockError::CapacityTooSmall {
                capacity_bytes: 7,
                block_bytes: 8,
            })
        ));
        assert!(matches!(
            KvBlockManager::new(u64::MAX, usize::MAX, u64::MAX),
            Err(KvBlockError::ConfigurationOverflow)
        ));
    }

    #[test]
    fn rejects_unknown_or_already_freed_sequences() {
        let mut manager = KvBlockManager::new(32, 4, 2).unwrap();
        let sequence = manager.allocate().unwrap();
        manager.append(sequence, 0).unwrap();
        assert_eq!(manager.free(sequence).unwrap().released_blocks, 0);
        assert_eq!(
            manager.append(sequence, 1),
            Err(KvBlockError::UnknownSequence(sequence))
        );
        assert_eq!(
            manager.free(sequence),
            Err(KvBlockError::UnknownSequence(sequence))
        );
    }

    /// Six 4-token blocks of 1 byte per token, with prefix caching.
    fn caching() -> KvBlockManager {
        KvBlockManager::new(24, 4, 1).unwrap().with_prefix_cache()
    }

    /// Allocates a sequence holding `tokens`, recording their content.
    fn filled(manager: &mut KvBlockManager, tokens: &[u32]) -> KvSequenceId {
        let sequence = manager.allocate().unwrap();
        manager.append(sequence, tokens.len()).unwrap();
        manager.record_tokens(sequence, tokens).unwrap();
        sequence
    }

    #[test]
    fn freed_full_blocks_are_reused_by_a_matching_prefix() {
        let mut manager = caching();
        let prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let first = filled(&mut manager, &prompt);
        let first_blocks = manager.block_table(first).unwrap().to_vec();
        manager.free(first).unwrap();

        // Two full blocks are cached; the partial third went back to free.
        let metrics = manager.metrics();
        assert_eq!((metrics.cached_blocks, metrics.free_blocks), (2, 4));
        assert_eq!(
            manager.lookup_prefix(&prompt),
            PrefixMatch {
                tokens: 8,
                evictable_blocks: 2,
            }
        );

        let second = manager.allocate().unwrap();
        assert_eq!(manager.attach_prefix(second, &prompt).unwrap(), 8);
        assert_eq!(manager.block_table(second).unwrap(), &first_blocks[..2]);
        assert_eq!(manager.cached_tokens(second).unwrap(), 8);
        manager.append(second, 2).unwrap();
        manager.record_tokens(second, &prompt[8..]).unwrap();
        let metrics = manager.metrics();
        assert_eq!(metrics.cached_blocks, 0);
        assert_eq!(
            (
                metrics.prefix_lookups,
                metrics.prefix_hits,
                metrics.prefix_hit_tokens
            ),
            (1, 1, 8)
        );
    }

    #[test]
    fn live_sequences_share_blocks_until_the_last_one_ends() {
        let mut manager = caching();
        let first = filled(&mut manager, &[1, 2, 3, 4, 5]);
        let second = manager.allocate().unwrap();
        assert_eq!(manager.attach_prefix(second, &[1, 2, 3, 4, 9]).unwrap(), 4);
        assert_eq!(
            manager.lookup_prefix(&[1, 2, 3, 4]).evictable_blocks,
            0,
            "a block a live sequence holds is not evictable"
        );

        manager.free(first).unwrap();
        // The shared block is still held; only the private one is freed.
        assert_eq!(manager.metrics().cached_blocks, 0);
        assert_eq!(manager.metrics().allocated_blocks, 1);
        manager.free(second).unwrap();
        assert_eq!(manager.metrics().cached_blocks, 1);
    }

    #[test]
    fn allocation_evicts_the_least_recently_used_cached_block() {
        let mut manager = caching();
        let old = filled(&mut manager, &[1, 1, 1, 1]);
        let recent = filled(&mut manager, &[2, 2, 2, 2]);
        manager.free(old).unwrap();
        manager.free(recent).unwrap();

        // Four free blocks plus two cached: five new blocks evict one.
        let big = manager.allocate().unwrap();
        manager.append(big, 20).unwrap();

        let metrics = manager.metrics();
        assert_eq!((metrics.evictions, metrics.cached_blocks), (1, 1));
        assert_eq!(
            manager.lookup_prefix(&[1, 1, 1, 1]).tokens,
            0,
            "oldest evicted"
        );
        assert_eq!(manager.lookup_prefix(&[2, 2, 2, 2]).tokens, 4);
        assert!(matches!(
            manager.append(big, 8),
            Err(KvBlockError::InsufficientCapacity {
                available_blocks: 1,
                ..
            })
        ));
    }

    #[test]
    fn a_block_matches_only_after_the_same_prefix() {
        let mut manager = caching();
        let sequence = filled(&mut manager, &[1, 2, 3, 4, 5, 6, 7, 8]);
        manager.free(sequence).unwrap();

        // Same second block, different first block: no match at all.
        assert_eq!(manager.lookup_prefix(&[9, 9, 9, 9, 5, 6, 7, 8]).tokens, 0);
        assert_eq!(manager.lookup_prefix(&[1, 2, 3, 4, 5, 6, 7, 0]).tokens, 4);
        assert_eq!(
            manager.lookup_prefix(&[1, 2, 3]).tokens,
            0,
            "only full blocks"
        );
    }

    #[test]
    fn without_the_prefix_cache_nothing_is_kept_or_reused() {
        let mut manager = KvBlockManager::new(24, 4, 1).unwrap();
        let sequence = filled(&mut manager, &[1, 2, 3, 4]);
        manager.free(sequence).unwrap();
        let next = manager.allocate().unwrap();

        assert_eq!(manager.attach_prefix(next, &[1, 2, 3, 4]).unwrap(), 0);
        let metrics = manager.metrics();
        assert_eq!((metrics.cached_blocks, metrics.free_blocks), (0, 6));
        assert_eq!(metrics.prefix_lookups, 0);
        manager.append(next, 1).unwrap();
        assert!(
            manager.attach_prefix(next, &[1]).is_err(),
            "only empty sequences"
        );
    }
}
