//! OpenRaft type configuration for the coordinator consensus adapter.
//!
//! This module defines the protocol types only. Log storage, state-machine
//! storage, peer transport, membership management, and leader fencing remain
//! separate implementation steps; this alias is not by itself a running
//! consensus cluster.

use std::io::Cursor;

use crate::CoordinatorInput;

openraft::declare_raft_types!(
    /// OpenRaft generic configuration used by coordinator replicas.
    pub CoordinatorRaftTypeConfig:
        D = CoordinatorInput,
        R = (),
        NodeId = u64,
        Node = openraft::BasicNode,
);

/// OpenRaft coordinator handle once a storage and network adapter is supplied.
pub type CoordinatorRaft = openraft::Raft<CoordinatorRaftTypeConfig>;
