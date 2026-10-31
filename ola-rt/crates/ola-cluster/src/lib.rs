#![doc = include_str!("../README.md")]

mod adaptive;
mod client;
mod coordinator;
mod local;
mod network;
mod placement;
mod plan;
mod policy;
mod routing;
mod routing_optimizer;
mod session;
mod snapshot;
mod summary;
mod transfer;
mod worker;

pub use adaptive::{AdaptiveClusterPolicy, ClusterWeights};
pub use client::{CoordinatorLinkError, CoordinatorLinkEvent, CoordinatorWorkerLink};
pub use coordinator::{
    ClientEvent, ClusterIteration, ClusterRequestStatus, ClusterTraceEvent, ClusterTraceRecord,
    Coordinator, CoordinatorError, CoordinatorOutput, WorkerCommand, WorkerMessage,
};
pub use local::{LocalCluster, LocalClusterError, LocalClusterStep};
pub use network::{LinkCalibration, TcpCoordinator, TcpCoordinatorError};
pub use placement::{
    PlacementError, PlacementModel, PlacementSolution, PlacementWorker, optimize_placement,
};
pub use plan::{ClusterAction, ClusterPlan, ClusterPlanError, DeferReason, RejectReason};
pub use policy::{
    ClusterDecision, ClusterDecisionRecord, ClusterPolicy, ClusterPolicyError, RouteCandidate,
    RouteReason, RouteRecord,
};
pub use routing::{
    KvHeadroomRouting, LeastLoadedRouting, PowerOfTwoRouting, PrefixAffinityRouting,
    RoundRobinRouting,
};
pub use routing_optimizer::{
    RoutingOptimizationError, RoutingRequest, RoutingSolution, RoutingWorker,
    optimize_batch_routing,
};
pub use session::{WorkerSession, WorkerSessionError};
pub use snapshot::{ClusterSnapshot, PendingRequest, WorkerView, token_demand};
pub use summary::{WorkerHealth, WorkerSummary};
pub use transfer::{HandoffCostModel, SessionHandoffMode, choose_handoff, session_state_bytes};
pub use worker::WorkerHost;
