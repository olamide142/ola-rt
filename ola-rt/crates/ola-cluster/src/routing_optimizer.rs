//! Exact small-batch routing optimizer for a cluster planning tick.

use std::{collections::BTreeMap, error::Error, fmt};

use good_lp::{Expression, Solution, SolverModel, variable};
use ola_core::{ClusterRequestId, WorkerId};

/// Capacity and calibrated cost facts for one routing target.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RoutingWorker {
    /// Worker identity.
    pub worker: WorkerId,
    /// Maximum requests this batch may add to the worker.
    pub capacity: usize,
    /// Requests already queued or live.
    pub current_load: usize,
    /// KV tokens available for this batch.
    pub kv_headroom_tokens: usize,
    /// Cost of adding one request to this worker.
    pub load_cost: u64,
}

/// One request competing for a worker in the same planning tick.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RoutingRequest {
    /// Request identity.
    pub request: ClusterRequestId,
    /// Worst-case KV tokens it needs.
    pub kv_tokens: usize,
}

/// Feasible batch assignment.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RoutingSolution {
    /// Request-to-worker assignment in deterministic request order.
    pub assignments: BTreeMap<ClusterRequestId, WorkerId>,
    /// Integer objective value of the selected assignment.
    pub objective: u64,
}

/// Why batch optimization could not produce an assignment.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RoutingOptimizationError {
    /// At least one request has no feasible worker.
    Infeasible(ClusterRequestId),
    /// The solver returned an unusable result.
    Solver(String),
}

impl fmt::Display for RoutingOptimizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Infeasible(request) => write!(f, "no worker can route request `{request}`"),
            Self::Solver(message) => write!(f, "routing solver failed: {message}"),
        }
    }
}

impl Error for RoutingOptimizationError {}

/// Solves a capacity- and KV-constrained assignment for one batch.
pub fn optimize_batch_routing(
    workers: &[RoutingWorker],
    requests: &[RoutingRequest],
) -> Result<RoutingSolution, RoutingOptimizationError> {
    for request in requests {
        if !workers.iter().any(|worker| {
            worker.current_load < worker.capacity && request.kv_tokens <= worker.kv_headroom_tokens
        }) {
            return Err(RoutingOptimizationError::Infeasible(request.request));
        }
    }
    if requests.is_empty() {
        return Ok(RoutingSolution {
            assignments: BTreeMap::new(),
            objective: 0,
        });
    }
    let mut variables = good_lp::ProblemVariables::new();
    let decision = requests
        .iter()
        .map(|_| {
            workers
                .iter()
                .map(|_| variables.add(variable().integer().min(0.0).max(1.0)))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut objective = Expression::from(0.0);
    for row in &decision {
        for (worker_index, variable) in row.iter().enumerate() {
            objective += workers[worker_index].load_cost as f64 * *variable;
        }
    }
    let mut problem = variables
        .minimise(objective.clone())
        .using(good_lp::solvers::microlp::microlp);
    for row in &decision {
        let mut exactly_one = Expression::from(0.0);
        for variable in row {
            exactly_one += *variable;
        }
        problem = problem.with(exactly_one.eq(1.0));
    }
    for (worker_index, worker) in workers.iter().enumerate() {
        let mut load = Expression::from(worker.current_load as f64);
        let mut kv = Expression::from(0.0);
        for (request_index, request) in requests.iter().enumerate() {
            load += decision[request_index][worker_index];
            kv += request.kv_tokens as f64 * decision[request_index][worker_index];
        }
        problem = problem.with(load.leq(worker.capacity as f64));
        problem = problem.with(kv.leq(worker.kv_headroom_tokens as f64));
    }
    let solution = problem
        .solve()
        .map_err(|error| RoutingOptimizationError::Solver(error.to_string()))?;
    let mut assignments = BTreeMap::new();
    for (request_index, row) in decision.iter().enumerate() {
        let worker_index = row
            .iter()
            .position(|variable| solution.value(*variable) > 0.5)
            .ok_or_else(|| RoutingOptimizationError::Solver("solution omitted a request".into()))?;
        assignments.insert(
            requests[request_index].request,
            workers[worker_index].worker.clone(),
        );
    }
    Ok(RoutingSolution {
        assignments,
        objective: solution.eval(&objective).ceil().max(0.0) as u64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn worker(id: &str, capacity: usize, kv: usize, cost: u64) -> RoutingWorker {
        RoutingWorker {
            worker: WorkerId::new(id),
            capacity,
            current_load: 0,
            kv_headroom_tokens: kv,
            load_cost: cost,
        }
    }

    #[test]
    fn routes_a_batch_under_load_and_kv_constraints() {
        let solution = optimize_batch_routing(
            &[worker("slow", 2, 100, 5), worker("fast", 1, 20, 1)],
            &[
                RoutingRequest {
                    request: ClusterRequestId::new(1),
                    kv_tokens: 20,
                },
                RoutingRequest {
                    request: ClusterRequestId::new(2),
                    kv_tokens: 20,
                },
            ],
        )
        .unwrap();
        assert_eq!(
            solution.assignments[&ClusterRequestId::new(1)],
            WorkerId::new("fast")
        );
        assert_eq!(
            solution.assignments[&ClusterRequestId::new(2)],
            WorkerId::new("slow")
        );
    }

    #[test]
    fn rejects_a_batch_that_cannot_fit() {
        let result = optimize_batch_routing(
            &[worker("a", 1, 4, 1)],
            &[RoutingRequest {
                request: ClusterRequestId::new(9),
                kv_tokens: 5,
            }],
        );
        assert!(matches!(
            result,
            Err(RoutingOptimizationError::Infeasible(request))
                if request == ClusterRequestId::new(9)
        ));
    }
}
