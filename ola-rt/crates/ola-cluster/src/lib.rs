#![doc = include_str!("../README.md")]

mod coordinator;
mod adaptive;
mod client;
mod local;
mod network;
mod plan;
mod policy;
mod routing;
mod snapshot;
mod summary;
mod session;
mod worker;

pub use coordinator::{
    ClientEvent, ClusterIteration, ClusterRequestStatus, ClusterTraceEvent, ClusterTraceRecord,
    Coordinator, CoordinatorError, CoordinatorOutput, WorkerCommand, WorkerMessage,
};
pub use adaptive::{AdaptiveClusterPolicy, ClusterWeights};
pub use client::{CoordinatorLinkError, CoordinatorLinkEvent, CoordinatorWorkerLink};
pub use network::{TcpCoordinator, TcpCoordinatorError};
pub use local::{LocalCluster, LocalClusterError, LocalClusterStep};
pub use plan::{ClusterAction, ClusterPlan, ClusterPlanError, DeferReason, RejectReason};
pub use policy::{
    ClusterDecision, ClusterDecisionRecord, ClusterPolicy, ClusterPolicyError, RouteCandidate,
    RouteReason, RouteRecord,
};
pub use routing::{
    KvHeadroomRouting, LeastLoadedRouting, PowerOfTwoRouting, PrefixAffinityRouting,
    RoundRobinRouting,
};
pub use snapshot::{ClusterSnapshot, PendingRequest, WorkerView, token_demand};
pub use summary::{WorkerHealth, WorkerSummary};
pub use session::{WorkerSession, WorkerSessionError};
pub use worker::WorkerHost;
