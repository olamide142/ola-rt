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
    /// Actual token positions stored across all live sequences.
    pub cached_tokens: usize,
    /// Token slots reserved by allocated blocks.
    pub reserved_token_slots: usize,
    /// Reserved token slots not yet occupied by tokens.
    pub internal_fragmentation_tokens: usize,
    /// Bytes corresponding to unoccupied slots in allocated blocks.
    pub internal_fragmentation_bytes: u64,
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
        })
    }

    /// Creates an empty logical cache and returns its manager-local ID.
    pub fn allocate(&mut self) -> Result<KvSequenceId, KvBlockError> {
        let value = self
            .next_sequence_id
            .ok_or(KvBlockError::SequenceIdExhausted)?;
        self.next_sequence_id = value.checked_add(1);
        let sequence = KvSequenceId(value);
        self.sequences.insert(
            sequence,
            SequenceAllocation {
                cached_tokens: 0,
                blocks: Vec::new(),
            },
        );
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
        if additional_blocks > self.free_blocks.len() {
            return Err(KvBlockError::InsufficientCapacity {
                sequence,
                requested_blocks: additional_blocks,
                available_blocks: self.free_blocks.len(),
            });
        }

        let mut acquired = Vec::with_capacity(additional_blocks);
        for _ in 0..additional_blocks {
            acquired.push(
                self.free_blocks
                    .pop_first()
                    .expect("capacity was checked before block allocation"),
            );
        }
        let allocation = self
            .sequences
            .get_mut(&sequence)
            .expect("sequence existence was checked before block allocation");
        allocation.blocks.extend(acquired);
        allocation.cached_tokens = new_token_count;
        Ok(())
    }

    /// Releases a sequence and returns all of its blocks to the free list.
    pub fn free(&mut self, sequence: KvSequenceId) -> Result<KvRelease, KvBlockError> {
        let allocation = self
            .sequences
            .remove(&sequence)
            .ok_or(KvBlockError::UnknownSequence(sequence))?;
        let released_blocks = allocation.blocks.len();
        for block in allocation.blocks {
            let inserted = self.free_blocks.insert(block);
            debug_assert!(inserted, "an allocated block cannot already be free");
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
        let allocated_blocks = self.total_blocks - self.free_blocks.len();
        let cached_tokens = self
            .sequences
            .values()
            .map(|allocation| allocation.cached_tokens)
            .sum::<usize>();
        let reserved_token_slots = allocated_blocks * self.tokens_per_block;
        let internal_fragmentation_tokens = reserved_token_slots - cached_tokens;
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
            }
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
}
