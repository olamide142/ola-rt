//! Small exact placement optimizer for model residency.

use std::{collections::BTreeMap, error::Error, fmt};

use good_lp::{Expression, Solution, SolverModel, variable};
use ola_core::{ModelId, WorkerId};

/// Capacity and current load for one placement target.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlacementWorker {
    /// Target worker identity.
    pub worker: WorkerId,
    /// Total bytes available for model residency.
    pub capacity_bytes: u64,
    /// Bytes already occupied by resident models.
    pub resident_bytes: u64,
    /// Relative expected serving cost; lower is preferred.
    pub load_cost: u64,
}

/// One model residency demand.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlacementModel {
    /// Model identity.
    pub model: ModelId,
    /// Bytes needed when resident.
    pub bytes: u64,
}

/// Feasible placement selected by the optimizer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlacementSolution {
    /// Model-to-worker assignments.
    pub assignments: BTreeMap<ModelId, WorkerId>,
    /// Objective value of the selected assignment.
    pub objective: u64,
}

/// Placement failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PlacementError {
    /// No target can hold one model.
    Infeasible(ModelId),
    /// The solver failed to produce a solution.
    Solver(String),
}

impl fmt::Display for PlacementError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Infeasible(model) => write!(f, "no worker can hold model `{model}`"),
            Self::Solver(message) => write!(f, "placement solver failed: {message}"),
        }
    }
}

impl Error for PlacementError {}

/// Solves a binary model-placement problem with worker memory constraints.
pub fn optimize_placement(
    workers: &[PlacementWorker],
    models: &[PlacementModel],
) -> Result<PlacementSolution, PlacementError> {
    for model in models {
        if !workers.iter().any(|worker| {
            model.bytes <= worker.capacity_bytes.saturating_sub(worker.resident_bytes)
        }) {
            return Err(PlacementError::Infeasible(model.model.clone()));
        }
    }
    if models.is_empty() {
        return Ok(PlacementSolution {
            assignments: BTreeMap::new(),
            objective: 0,
        });
    }
    let mut variables = good_lp::ProblemVariables::new();
    let decision = models
        .iter()
        .map(|_| {
            workers
                .iter()
                .map(|_| variables.add(variable().integer().min(0.0).max(1.0)))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut objective = Expression::from(0.0);
    for row in decision.iter() {
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
        let mut used = Expression::from(worker.resident_bytes as f64);
        for (model_index, model) in models.iter().enumerate() {
            used += model.bytes as f64 * decision[model_index][worker_index];
        }
        problem = problem.with(used.leq(worker.capacity_bytes as f64));
    }
    let solution = problem
        .solve()
        .map_err(|error| PlacementError::Solver(error.to_string()))?;
    let mut assignments = BTreeMap::new();
    for (model_index, row) in decision.iter().enumerate() {
        let worker_index = row
            .iter()
            .position(|variable| solution.value(*variable) > 0.5)
            .ok_or_else(|| PlacementError::Solver("solution omitted a model".into()))?;
        assignments.insert(
            models[model_index].model.clone(),
            workers[worker_index].worker.clone(),
        );
    }
    let objective = solution.eval(&objective).ceil().max(0.0) as u64;
    Ok(PlacementSolution {
        assignments,
        objective,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn places_models_under_capacity_at_lowest_load_cost() {
        let solution = optimize_placement(
            &[
                PlacementWorker {
                    worker: WorkerId::new("slow"),
                    capacity_bytes: 10,
                    resident_bytes: 0,
                    load_cost: 4,
                },
                PlacementWorker {
                    worker: WorkerId::new("fast"),
                    capacity_bytes: 10,
                    resident_bytes: 0,
                    load_cost: 1,
                },
            ],
            &[PlacementModel {
                model: ModelId::from("m"),
                bytes: 5,
            }],
        )
        .unwrap();
        assert_eq!(
            solution.assignments[&ModelId::from("m")],
            WorkerId::new("fast")
        );
    }

    #[test]
    fn rejects_a_model_that_fits_nowhere() {
        let result = optimize_placement(
            &[PlacementWorker {
                worker: WorkerId::new("a"),
                capacity_bytes: 4,
                resident_bytes: 4,
                load_cost: 1,
            }],
            &[PlacementModel {
                model: ModelId::from("m"),
                bytes: 1,
            }],
        );
        assert!(
            matches!(result, Err(PlacementError::Infeasible(model)) if model == ModelId::from("m"))
        );
    }
}
