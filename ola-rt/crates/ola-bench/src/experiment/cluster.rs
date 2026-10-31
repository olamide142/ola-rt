//! Built-in cluster experiments, in virtual time.

use std::collections::BTreeMap;

use ola_cluster::{
    AdaptiveClusterPolicy, ClusterPolicy, ClusterWeights, KvHeadroomRouting, LeastLoadedRouting,
    PowerOfTwoRouting, PrefixAffinityRouting, RoundRobinRouting,
};

use super::builtin::{as_f64, finish, run_error};
use super::{
    Direction, Experiment, ExperimentCategory, ExperimentContext, ExperimentDescriptor,
    ExperimentError, ExperimentKind, ExperimentOutcome, MetricSpec, ParameterSpec, Parameters,
    TimeDomain, Unit, Variant,
};
use crate::{
    ClusterSimulationConfig, DEFAULT_STARVATION_FACTOR, KvGeometry, NetworkModel, SchedulingTrace,
    SimulationConfig, TraceParameters, isolated_end_to_end, run_cluster_trial,
    summarize_cluster_run,
};

/// Routing baselines over one topology, network, and trace generator, across
/// several seeds.
pub struct ClusterRouting;

/// Selects adaptive cluster weights on one virtual trace and evaluates them on
/// independent traces without letting held-out results influence selection.
pub struct AdaptiveClusterRoutingStudy;

impl Experiment for AdaptiveClusterRoutingStudy {
    fn descriptor(&self) -> ExperimentDescriptor {
        ExperimentDescriptor {
            id: "adaptive-cluster-routing-study",
            name: "Adaptive cluster routing study",
            description: "Training-selected cluster weights evaluated on held-out traces",
            category: ExperimentCategory::Experiment,
            kind: ExperimentKind::Study,
            domain: TimeDomain::Virtual,
            payload_version: 1,
            requirements: Vec::new(),
            parameters: vec![
                ParameterSpec::integer("seeds", "Held-out trace seeds", 3),
                ParameterSpec::integer("requests", "Requests per trace", 20),
                ParameterSpec::integer(
                    "mean_interarrival_micros",
                    "Mean arrival gap in virtual microseconds",
                    400,
                ),
            ],
        }
    }

    fn traceable(&self) -> bool {
        true
    }

    fn run(
        &self,
        context: &ExperimentContext,
        parameters: &Parameters,
    ) -> Result<ExperimentOutcome, ExperimentError> {
        let seeds = (1..=parameters.integer("seeds")?).collect::<Vec<_>>();
        let base = TraceParameters {
            seed: 0,
            requests: parameters.size("requests")?,
            mean_interarrival_micros: parameters.integer("mean_interarrival_micros")?,
        };
        let worker = SimulationConfig {
            max_active_requests: 4,
            ..SimulationConfig::default()
        };
        let config = ClusterSimulationConfig::uniform(2, worker, NetworkModel::default());
        let generate = |seed| SchedulingTrace::overload(TraceParameters { seed, ..base });
        let candidates = [
            ClusterWeights {
                load: 1,
                kv_headroom: 1,
                network: 0,
            },
            ClusterWeights {
                load: 2,
                kv_headroom: 1,
                network: 0,
            },
            ClusterWeights {
                load: 1,
                kv_headroom: 2,
                network: 0,
            },
            ClusterWeights {
                load: 1,
                kv_headroom: 4,
                network: 0,
            },
        ];
        let training = generate(0);
        let training_isolated = vec![None; training.requests.len()];
        let mut selected = candidates[0];
        let mut selected_score = u128::MAX;
        for weights in candidates {
            let policy = AdaptiveClusterPolicy::new(weights).map_err(run_error)?;
            let run = run_cluster_trial(&training, policy, &config).map_err(run_error)?;
            let summary =
                summarize_cluster_run(&run, &training_isolated, DEFAULT_STARVATION_FACTOR);
            let score = u128::from(summary.scheduling.completion.p95_micros).saturating_add(
                u128::from(summary.scheduling.time_to_first_token.p95_micros),
            );
            if score < selected_score {
                selected = weights;
                selected_score = score;
            }
        }
        let adaptive_descriptor = AdaptiveClusterPolicy::new(selected)
            .map_err(run_error)?
            .descriptor()
            .to_string();
        let variants = [
            ("adaptive", adaptive_descriptor),
            ("least-loaded", LeastLoadedRouting.descriptor().to_string()),
            ("kv-headroom", KvHeadroomRouting.descriptor().to_string()),
        ];
        let mut samples = variants
            .iter()
            .map(|_| Vec::new())
            .collect::<Vec<Vec<BTreeMap<String, f64>>>>();
        let mut content = String::new();
        for seed in &seeds {
            let trace = generate(*seed);
            for request in &trace.requests {
                content.push_str(&format!("{request:?};"));
            }
            let isolated = vec![None; trace.requests.len()];
            let policies: Vec<Box<dyn ClusterPolicy>> = vec![
                Box::new(AdaptiveClusterPolicy::new(selected).map_err(run_error)?),
                Box::new(LeastLoadedRouting),
                Box::new(KvHeadroomRouting),
            ];
            for (variant, policy) in samples.iter_mut().zip(policies) {
                let run = run_cluster_trial(&trace, policy, &config).map_err(run_error)?;
                let summary = summarize_cluster_run(&run, &isolated, DEFAULT_STARVATION_FACTOR);
                variant.push(
                    [
                        (
                            "completion_p95",
                            as_f64(summary.scheduling.completion.p95_micros),
                        ),
                        (
                            "ttft_p95",
                            as_f64(summary.scheduling.time_to_first_token.p95_micros),
                        ),
                        (
                            "throughput",
                            summary.scheduling.output_tokens_per_second().unwrap_or(0.0),
                        ),
                    ]
                    .into_iter()
                    .map(|(name, value)| (name.to_owned(), value))
                    .collect(),
                );
            }
        }
        let metrics = vec![
            MetricSpec::new(
                "completion_p95",
                Unit::Micros,
                TimeDomain::Virtual,
                Direction::Lower,
            ),
            MetricSpec::new(
                "ttft_p95",
                Unit::Micros,
                TimeDomain::Virtual,
                Direction::Lower,
            ),
            MetricSpec::new(
                "throughput",
                Unit::TokensPerSecond,
                TimeDomain::Virtual,
                Direction::Higher,
            ),
        ];
        let outcome = finish(
            &self.descriptor(),
            context,
            parameters,
            "adaptive-cluster-routing-study",
            seeds.clone(),
            metrics,
            variants
                .into_iter()
                .zip(samples)
                .map(|((label, policy), values)| (Variant::policy(label, policy), values))
                .collect(),
            format!(
                "train seed 0, {} held-out seeds, selected load_weight={}, kv_weight={}",
                seeds.len(),
                selected.load,
                selected.kv_headroom
            ),
        )?
        .with_content_digest(&content);
        Ok(ExperimentOutcome {
            summary: format!(
                "{} held-out traces; selected adaptive weights load={} kv={}",
                seeds.len(),
                selected.load,
                selected.kv_headroom
            ),
            ..outcome
        })
    }
}

impl ClusterRouting {
    fn policies(
        prefix_tokens: usize,
        load_slack: usize,
    ) -> Vec<(&'static str, Box<dyn ClusterPolicy>)> {
        vec![
            ("round-robin", Box::new(RoundRobinRouting)),
            ("jsq", Box::new(LeastLoadedRouting)),
            ("power-of-two", Box::new(PowerOfTwoRouting::new(1))),
            ("kv-headroom", Box::new(KvHeadroomRouting)),
            (
                "prefix-affinity",
                Box::new(PrefixAffinityRouting::new(prefix_tokens, load_slack)),
            ),
            ("adaptive", Box::new(AdaptiveClusterPolicy::default())),
        ]
    }
}

impl Experiment for ClusterRouting {
    fn descriptor(&self) -> ExperimentDescriptor {
        ExperimentDescriptor {
            id: "cluster-routing",
            name: "Cluster routing",
            description: "Routing baselines across simulated workers and links in virtual time",
            category: ExperimentCategory::Experiment,
            kind: ExperimentKind::Study,
            domain: TimeDomain::Virtual,
            payload_version: 1,
            requirements: Vec::new(),
            parameters: vec![
                ParameterSpec::text(
                    "trace",
                    "prefix_groups, mixed_short_long, overload, prefill_decode_competition, \
                     or priority_classes",
                    "prefix_groups",
                ),
                ParameterSpec::integer("seeds", "Trace seeds, one per repetition", 5),
                ParameterSpec::integer("requests", "Requests per trace", 160),
                ParameterSpec::integer(
                    "mean_interarrival_micros",
                    "Mean arrival gap in virtual microseconds",
                    400,
                ),
                ParameterSpec::text(
                    "workers",
                    "Comma-separated executor slots per worker; one entry per worker",
                    "4,4,4,4",
                ),
                ParameterSpec::integer(
                    "kv_capacity_tokens",
                    "KV pool size per worker in tokens",
                    1_024,
                ),
                ParameterSpec::integer("prefix_cache", "1 to give workers a prefix cache", 1),
                ParameterSpec::integer("latency_micros", "One-way link latency", 50),
                ParameterSpec::integer(
                    "bytes_per_micro",
                    "Link bandwidth in bytes per microsecond; 0 is unlimited",
                    1_250,
                ),
                ParameterSpec::integer("prefix_groups", "Distinct prefixes (prefix_groups)", 8),
                ParameterSpec::integer("prefix_tokens", "Shared prefix length (prefix_groups)", 96),
                ParameterSpec::integer(
                    "affinity_prefix_tokens",
                    "Prompt tokens prefix affinity hashes",
                    32,
                ),
                ParameterSpec::integer(
                    "affinity_load_slack",
                    "Extra load prefix affinity tolerates on a preferred worker",
                    4,
                ),
            ],
        }
    }

    fn run(
        &self,
        context: &ExperimentContext,
        parameters: &Parameters,
    ) -> Result<ExperimentOutcome, ExperimentError> {
        let trace_name = parameters.text("trace")?.to_owned();
        let seeds = (1..=parameters.integer("seeds")?).collect::<Vec<_>>();
        let slots = super::builtin::lengths(parameters, "workers")?;
        if slots.is_empty() || slots.contains(&0) {
            return Err(ExperimentError::InvalidParameter {
                name: "workers".into(),
                reason: "expected at least one worker with at least one slot".into(),
            });
        }
        let geometry = KvGeometry::default();
        let base_worker = SimulationConfig {
            kv: KvGeometry {
                capacity_bytes: parameters
                    .integer("kv_capacity_tokens")?
                    .saturating_mul(geometry.bytes_per_token),
                ..geometry
            },
            prefix_cache: parameters.integer("prefix_cache")? != 0,
            ..SimulationConfig::default()
        };
        let config = ClusterSimulationConfig {
            workers: slots
                .iter()
                .map(|&max_active_requests| SimulationConfig {
                    max_active_requests,
                    ..base_worker
                })
                .collect(),
            ..ClusterSimulationConfig::uniform(
                0,
                base_worker,
                NetworkModel {
                    latency_micros: parameters.integer("latency_micros")?,
                    bytes_per_micro: parameters.integer("bytes_per_micro")?,
                },
            )
        };
        let base = TraceParameters {
            seed: 0,
            requests: parameters.size("requests")?,
            mean_interarrival_micros: parameters.integer("mean_interarrival_micros")?,
        };
        let groups = parameters.integer("prefix_groups")?;
        let prefix_tokens = parameters.size("prefix_tokens")?;
        let generate = |seed| {
            let parameters = TraceParameters { seed, ..base };
            Ok(match trace_name.as_str() {
                "prefix_groups" => {
                    SchedulingTrace::prefix_groups(parameters, groups, prefix_tokens)
                }
                "mixed_short_long" => SchedulingTrace::mixed_short_long(parameters),
                "overload" => SchedulingTrace::overload(parameters),
                "prefill_decode_competition" => {
                    SchedulingTrace::prefill_decode_competition(parameters)
                }
                "priority_classes" => SchedulingTrace::priority_classes(parameters),
                _ => {
                    return Err(ExperimentError::InvalidParameter {
                        name: "trace".into(),
                        reason: format!("unknown trace `{trace_name}`"),
                    });
                }
            })
        };
        let affinity_tokens = parameters.size("affinity_prefix_tokens")?;
        let load_slack = parameters.size("affinity_load_slack")?;
        let policies = || Self::policies(affinity_tokens, load_slack);

        let labels = policies()
            .into_iter()
            .map(|(label, policy)| (label, policy.descriptor().to_string()))
            .collect::<Vec<_>>();
        let mut samples = labels
            .iter()
            .map(|_| Vec::new())
            .collect::<Vec<Vec<BTreeMap<String, f64>>>>();
        let mut content = String::new();
        for &seed in &seeds {
            let trace = generate(seed)?;
            for request in &trace.requests {
                content.push_str(&format!("{request:?};"));
            }
            let isolated = isolated_end_to_end(&trace, &config.workers[0]).map_err(run_error)?;
            for (variant, (_, policy)) in samples.iter_mut().zip(policies()) {
                let run = run_cluster_trial(&trace, policy, &config).map_err(run_error)?;
                let summary = summarize_cluster_run(&run, &isolated, DEFAULT_STARVATION_FACTOR);
                let scheduling = &summary.scheduling;
                variant.push(
                    [
                        ("completion_p50", as_f64(scheduling.completion.p50_micros)),
                        ("completion_p95", as_f64(scheduling.completion.p95_micros)),
                        (
                            "ttft_p95",
                            as_f64(scheduling.time_to_first_token.p95_micros),
                        ),
                        ("queue_p95", as_f64(scheduling.queue.p95_micros)),
                        (
                            "throughput",
                            scheduling.output_tokens_per_second().unwrap_or(0.0),
                        ),
                        ("jain_fairness", scheduling.slowdown.jain_fairness),
                        ("unfinished", as_f64(scheduling.unfinished)),
                        ("prefix_hit_tokens", as_f64(summary.prefix_hit_tokens)),
                        ("busy_imbalance", summary.busy_imbalance),
                        ("network_bytes", as_f64(summary.network_bytes)),
                    ]
                    .into_iter()
                    .map(|(name, value)| (name.to_owned(), value))
                    .collect(),
                );
            }
        }

        let virtual_metric =
            |name, unit, better| MetricSpec::new(name, unit, TimeDomain::Virtual, better);
        let latency = |name| virtual_metric(name, Unit::Micros, Direction::Lower);
        let outcome = finish(
            &self.descriptor(),
            context,
            parameters,
            &trace_name,
            seeds.clone(),
            vec![
                latency("completion_p50"),
                latency("completion_p95"),
                latency("ttft_p95"),
                latency("queue_p95"),
                virtual_metric("throughput", Unit::TokensPerSecond, Direction::Higher),
                virtual_metric("jain_fairness", Unit::Ratio, Direction::Higher),
                virtual_metric("unfinished", Unit::Count, Direction::Lower),
                virtual_metric("prefix_hit_tokens", Unit::Tokens, Direction::Higher),
                virtual_metric("busy_imbalance", Unit::Ratio, Direction::Lower),
                virtual_metric("network_bytes", Unit::Bytes, Direction::Neither),
            ],
            labels
                .iter()
                .zip(samples)
                .map(|((label, policy), samples)| {
                    (Variant::policy(*label, policy.clone()), samples)
                })
                .collect(),
            String::new(),
        )?
        .with_content_digest(&content);
        let summary = format!(
            "{trace_name}, {} workers, {} seeds × {} requests, virtual time · \
             completion p95: {}",
            slots.len(),
            seeds.len(),
            base.requests,
            outcome
                .report
                .summaries
                .iter()
                .map(|summary| format!(
                    "{} {:.1} ms",
                    summary.variant,
                    summary.estimates["completion_p95"].mean / 1_000.0
                ))
                .collect::<Vec<_>>()
                .join(" · ")
        );
        Ok(ExperimentOutcome { summary, ..outcome })
    }
}
