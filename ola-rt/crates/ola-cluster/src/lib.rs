#![doc = include_str!("../README.md")]

mod adaptive;
mod client;
mod consensus;
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
pub use client::{CoordinatorLinkError, CoordinatorLinkEvent, CoordinatorWorkerLink, LinkFrame};
pub use consensus::{CoordinatorRaft, CoordinatorRaftTypeConfig};
pub use coordinator::{
    ClientEvent, ClusterFeedbackSnapshot, ClusterIteration, ClusterMigrationObservation,
    ClusterMigrationOutcome, ClusterPolicyUpdateObservation, ClusterRequestObservation,
    ClusterRequestOutcome, ClusterRequestStatus, ClusterRouteObservation, ClusterTraceEvent,
    ClusterTraceRecord, ClusterWorkerObservation, Coordinator, CoordinatorApplyResult,
    CoordinatorError, CoordinatorInput, CoordinatorOutput, WorkerCommand, WorkerMessage,
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
pub use summary::{WorkerHealth, WorkerRole, WorkerSummary};
pub use transfer::{
    HandoffCostModel, SESSION_STATE_CONTENT_TYPE, SessionFrameError, SessionHandoffMode,
    choose_handoff, session_state_bytes, session_state_from_frame, session_state_to_frame,
};
pub use worker::WorkerHost;
