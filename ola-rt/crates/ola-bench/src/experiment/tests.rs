use std::collections::BTreeMap;

use super::*;

fn metrics() -> Vec<MetricSpec> {
    vec![
        MetricSpec::new(
            "completion_p95",
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
    ]
}

fn sample(variant: &str, repetition: usize, completion: f64, throughput: f64) -> Sample {
    Sample {
        repetition,
        variant: variant.into(),
        values: BTreeMap::from([
            ("completion_p95".to_owned(), completion),
            ("throughput".to_owned(), throughput),
        ]),
    }
}

fn parts() -> ReportParts {
    ReportParts {
        experiment: ExperimentIdentity {
            id: "unit-test".into(),
            payload_version: 1,
            kind: ExperimentKind::Study,
            domain: TimeDomain::Virtual,
        },
        metadata: RunMetadata::now(None, None),
        workload: WorkloadIdentity {
            name: "trace".into(),
            seeds: vec![1, 2],
            parameters: BTreeMap::from([("requests".to_owned(), "10".to_owned())]),
            digest: content_digest(b"trace"),
        },
        protocol: Protocol {
            warmups: 0,
            repetitions: 2,
            ordering: "by-seed".into(),
        },
        metrics: metrics(),
        variants: vec![
            Variant::policy("fifo", "composed[fifo]"),
            Variant::policy("sjf", "composed[sjf]"),
        ],
        samples: vec![
            sample("fifo", 0, 100.0, 10.0),
            sample("fifo", 1, 300.0, 12.5),
            sample("sjf", 1, 80.0, 11.0),
            sample("sjf", 0, 60.0, 9.0),
        ],
        details: Some(serde_json::json!({ "note": "kept verbatim" })),
    }
}

#[test]
fn reports_derive_summaries_from_samples() {
    let report = ExperimentReport::new(parts()).unwrap();

    assert_eq!(report.schema, EXPERIMENT_SCHEMA);
    let fifo = &report.summary("fifo").unwrap().estimates["completion_p95"];
    assert_eq!(
        (fifo.n, fifo.mean, fifo.min, fifo.max),
        (2, 200.0, 100.0, 300.0)
    );
    let sjf = &report.summary("sjf").unwrap().estimates["throughput"];
    assert_eq!(sjf.mean, 10.0);
    assert!(!report.metadata.build.revision.is_empty());
}

#[test]
fn estimates_use_nearest_rank_percentiles() {
    let estimate = Estimate::of(&[3.0, 1.0, 2.0, 4.0]).unwrap();

    assert_eq!(
        (
            estimate.mean,
            estimate.p50,
            estimate.p95,
            estimate.min,
            estimate.max
        ),
        (2.5, 2.0, 4.0, 1.0, 4.0)
    );
    assert_eq!(Estimate::of(&[]), None);
}

#[test]
fn reports_round_trip_exactly_and_never_overwrite() {
    let mut parts = parts();
    // A value that is not exactly representable in short decimal.
    parts.samples[0]
        .values
        .insert("throughput".into(), 0.1 + 0.2);
    let report = ExperimentReport::new(parts).unwrap();
    assert_eq!(
        ExperimentReport::from_json(&report.to_json_pretty().unwrap()).unwrap(),
        report
    );

    let directory = std::env::temp_dir().join(format!(
        "ola-experiment-test-{}-{}",
        std::process::id(),
        report.metadata.recorded_at_unix_micros
    ));
    let first = report.write_new(&directory).unwrap();
    let second = report.write_new(&directory).unwrap();

    assert_ne!(first, second);
    assert_eq!(ExperimentReport::read(&first).unwrap(), report);
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn validation_rejects_inconsistent_reports() {
    let rejects = |change: fn(&mut ReportParts)| {
        let mut parts = parts();
        change(&mut parts);
        assert!(
            matches!(
                ExperimentReport::new(parts),
                Err(ReportError::Inconsistent(_))
            ),
            "an inconsistent report was accepted"
        );
    };
    rejects(|parts| {
        parts.samples.pop();
    });
    rejects(|parts| parts.samples[3].repetition = 1);
    rejects(|parts| parts.samples[0].variant = "unknown".into());
    rejects(|parts| parts.samples[0].repetition = 2);
    rejects(|parts| {
        parts.samples[0].values.insert("undeclared".into(), 1.0);
    });
    rejects(|parts| {
        parts.samples[0]
            .values
            .insert("throughput".into(), f64::NAN);
    });
    rejects(|parts| parts.variants.push(Variant::policy("fifo", "again")));
    rejects(|parts| parts.protocol.repetitions = 0);
}

#[test]
fn reading_rejects_tampered_summaries_and_unknown_schemas() {
    let report = ExperimentReport::new(parts()).unwrap();

    let mut tampered = report.clone();
    tampered.summaries[0]
        .estimates
        .get_mut("completion_p95")
        .unwrap()
        .mean = 1.0;
    let json = tampered.to_json_pretty().unwrap();
    assert!(matches!(
        ExperimentReport::from_json(&json),
        Err(ReportError::Inconsistent(_))
    ));

    let mut future = report;
    future.schema = "ola-experiment/2".into();
    assert!(matches!(
        ExperimentReport::from_json(&future.to_json_pretty().unwrap()),
        Err(ReportError::UnsupportedSchema(_))
    ));
}

#[test]
fn content_digests_match_fnv1a_test_vectors() {
    assert_eq!(content_digest(b""), "cbf29ce484222325");
    assert_eq!(content_digest(b"a"), "af63dc4c8601ec8c");
}

fn report_from(variants: &[(&str, &[f64], &[f64])]) -> ExperimentReport {
    let repetitions = variants[0].1.len();
    let mut parts = parts();
    parts.protocol.repetitions = repetitions;
    parts.metrics.push(MetricSpec::new(
        "iterations",
        Unit::Count,
        TimeDomain::Untimed,
        Direction::Neither,
    ));
    parts.variants = variants
        .iter()
        .map(|(label, _, _)| Variant::policy(*label, format!("policy-{label}")))
        .collect();
    parts.samples = variants
        .iter()
        .flat_map(|(label, completion, throughput)| {
            completion.iter().zip(*throughput).enumerate().map(
                |(repetition, (completion, throughput))| {
                    let mut sample = sample(label, repetition, *completion, *throughput);
                    sample.values.insert("iterations".into(), *completion);
                    sample
                },
            )
        })
        .collect();
    ExperimentReport::new(parts).unwrap()
}

fn verdicts(comparison: &ReportComparison) -> Vec<(&str, Change)> {
    comparison
        .metrics
        .iter()
        .map(|metric| (metric.metric.name.as_str(), metric.change))
        .collect()
}

#[test]
fn bootstrap_intervals_are_reproducible_and_bracket_the_mean() {
    assert_eq!(Estimate::of(&[5.0]).unwrap().ci95, None);
    let flat = Estimate::of(&[2.0, 2.0, 2.0]).unwrap().ci95.unwrap();
    assert_eq!((flat.low, flat.high), (2.0, 2.0));

    let values = [10.0, 12.0, 9.0, 14.0, 11.0, 13.0];
    let estimate = Estimate::of(&values).unwrap();
    let interval = estimate.ci95.unwrap();
    assert!(interval.low <= estimate.mean && estimate.mean <= interval.high);
    assert!(estimate.min <= interval.low && interval.high <= estimate.max);
    let mut shuffled = values;
    shuffled.reverse();
    assert_eq!(Estimate::of(&shuffled).unwrap().ci95, Some(interval));

    let width = |values: &[f64]| {
        let interval = Estimate::of(values).unwrap().ci95.unwrap();
        interval.high - interval.low
    };
    let many = values.iter().cycle().take(60).copied().collect::<Vec<_>>();
    assert!(width(&many) < width(&values));
}

#[test]
fn comparisons_classify_by_direction_intervals_and_threshold() {
    let baseline = report_from(&[(
        "fifo",
        &[100.0, 102.0, 98.0, 101.0, 99.0],
        &[10.0, 10.2, 9.8, 10.1, 9.9],
    )]);
    let faster_but_slower_throughput = report_from(&[(
        "fifo",
        &[80.0, 81.0, 79.0, 82.0, 78.0],
        &[8.0, 8.1, 7.9, 8.2, 7.8],
    )]);
    let comparison = compare_reports(
        &baseline,
        &faster_but_slower_throughput,
        CompareOptions::default(),
    )
    .unwrap();
    assert_eq!(
        verdicts(&comparison),
        [
            ("completion_p95", Change::Improved),
            ("throughput", Change::Regressed),
            ("iterations", Change::Informational),
        ]
    );
    assert_eq!(comparison.regressions().count(), 1);
    assert!(comparison.identity_differences.is_empty());

    // Overlapping intervals.
    let noisy = report_from(&[(
        "fifo",
        &[90.0, 110.0, 95.0, 105.0, 100.0],
        &[9.0, 11.0, 9.5, 10.5, 10.0],
    )]);
    // Separated, but under the 5% threshold.
    let slight = report_from(&[(
        "fifo",
        &[103.0, 103.1, 102.9, 103.0, 103.0],
        &[10.3, 10.31, 10.29, 10.3, 10.3],
    )]);
    for candidate in [&noisy, &slight] {
        let comparison = compare_reports(&baseline, candidate, CompareOptions::default()).unwrap();
        assert!(
            comparison.metrics.iter().all(|metric| matches!(
                metric.change,
                Change::Inconclusive | Change::Informational
            ))
        );
    }
}

#[test]
fn single_runs_never_yield_a_verdict() {
    let baseline = report_from(&[("fifo", &[100.0], &[10.0])]);
    let candidate = report_from(&[("fifo", &[50.0], &[20.0])]);

    let comparison = compare_reports(&baseline, &candidate, CompareOptions::default()).unwrap();

    assert!(
        comparison
            .metrics
            .iter()
            .all(|metric| matches!(metric.change, Change::Inconclusive | Change::Informational))
    );
}

#[test]
fn comparisons_report_identity_differences_and_refuse_incomparable_reports() {
    let baseline = report_from(&[("fifo", &[1.0, 2.0], &[1.0, 2.0])]);
    let mut candidate = baseline.clone();
    candidate.metadata.model = Some("tiny-llama".into());
    candidate.variants[0].policy = Some("policy-changed".into());
    candidate.variants[0].kernel = Some("kernel-changed".into());
    let comparison = compare_reports(&baseline, &candidate, CompareOptions::default()).unwrap();
    let fields = comparison
        .identity_differences
        .iter()
        .map(|difference| difference.field.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        fields,
        ["model", "variant fifo policy", "variant fifo kernel"]
    );

    let mut other_workload = baseline.clone();
    other_workload.workload.digest = content_digest(b"other");
    assert!(matches!(
        compare_reports(&baseline, &other_workload, CompareOptions::default()),
        Err(CompareError::DifferentWorkload { .. })
    ));
    let allowed = compare_reports(
        &baseline,
        &other_workload,
        CompareOptions {
            allow_workload_mismatch: true,
            ..CompareOptions::default()
        },
    )
    .unwrap();
    assert_eq!(allowed.identity_differences[0].field, "workload");

    let mut other_experiment = baseline.clone();
    other_experiment.experiment.id = "different".into();
    assert!(matches!(
        compare_reports(&baseline, &other_experiment, CompareOptions::default()),
        Err(CompareError::DifferentExperiment { .. })
    ));
    let mut host = baseline.clone();
    host.experiment.domain = TimeDomain::Host;
    assert_eq!(
        compare_reports(&baseline, &host, CompareOptions::default()),
        Err(CompareError::DifferentDomain)
    );

    let mut extra_variant = report_from(&[
        ("fifo", &[1.0, 2.0], &[1.0, 2.0]),
        ("sjf", &[1.0, 2.0], &[1.0, 2.0]),
    ]);
    extra_variant.workload = baseline.workload.clone();
    let comparison = compare_reports(&baseline, &extra_variant, CompareOptions::default()).unwrap();
    assert_eq!(comparison.unmatched, ["variant sjf (candidate only)"]);
}

fn run(id: &str, overrides: &[(&str, &str)]) -> Result<ExperimentOutcome, ExperimentError> {
    let overrides = overrides
        .iter()
        .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
        .collect::<Vec<_>>();
    ExperimentRegistry::builtin().run(id, &ExperimentContext::default(), &overrides)
}

#[test]
fn a_json_run_config_selects_and_overrides_an_experiment() {
    let config =
        ExperimentRunConfig::from_json(r#"{"id":"fifo-workload","parameters":{"requests":"10"}}"#)
            .unwrap();
    let outcome = ExperimentRegistry::builtin()
        .run_config(&config, &ExperimentContext::default())
        .unwrap();

    assert_eq!(outcome.report.experiment.id, "fifo-workload");
    assert!(outcome.summary.starts_with("10 requests"));
}

#[test]
fn a_json_run_config_rejects_malformed_json() {
    assert!(ExperimentRunConfig::from_json("{not-json").is_err());
}

#[test]
fn every_builtin_experiment_runs_with_its_defaults() {
    let registry = ExperimentRegistry::builtin();
    let descriptors = registry.descriptors();
    let actual = descriptors
        .iter()
        .map(|descriptor| descriptor.id)
        .collect::<Vec<_>>();
    #[allow(unused_mut)]
    let mut expected = vec![
        "llama-generation",
        "llama-prefill-strategies",
        "llama-chunked-prefill",
        "cost-calibration",
        "kv-block-pressure",
        "fifo-workload",
        "prefill-bucketing",
        "adaptive-prefill-cost",
        "scheduler-comparison",
        "adaptive-policy-study",
        "kv-pressure",
        "prefix-cache",
        "kv-page-size",
        "cluster-routing",
    ];
    #[cfg(any(feature = "cubecl-cpu", feature = "wgpu"))]
    expected.push("paged-attention-kernel");
    assert_eq!(actual, expected);
    for descriptor in descriptors {
        if descriptor.requirements.contains(&Requirement::LoadedModel) {
            assert!(matches!(
                run(descriptor.id, &[]),
                Err(ExperimentError::MissingRequirement(
                    Requirement::LoadedModel
                ))
            ));
            continue;
        }
        let outcome = run(descriptor.id, &[]).unwrap();
        assert_eq!(outcome.report.experiment.id, descriptor.id);
        assert!(
            !outcome.summary.is_empty(),
            "{} has no summary",
            descriptor.id
        );
        outcome.report.validate().unwrap();
    }
}

#[cfg(feature = "wgpu")]
#[test]
#[ignore = "requires a local WGPU adapter"]
fn paged_attention_study_runs_on_wgpu() {
    let outcome = run(
        "paged-attention-kernel",
        &[
            ("page_sizes", "2"),
            ("batch", "1"),
            ("pool_slack_blocks", "3"),
            ("warmups", "1"),
            ("repetitions", "5"),
        ],
    )
    .unwrap();
    assert_eq!(outcome.report.protocol.warmups, 1);
    assert_eq!(outcome.report.protocol.repetitions, 5);
    outcome.report.validate().unwrap();
}

#[cfg(feature = "cubecl-cpu")]
#[test]
fn paged_attention_study_records_pool_fragmentation_on_cpu() {
    let outcome = run(
        "paged-attention-kernel",
        &[
            ("page_sizes", "2"),
            ("sequence_tokens", "5"),
            ("batch", "1"),
            ("pool_slack_blocks", "3"),
            ("warmups", "1"),
            ("repetitions", "5"),
        ],
    )
    .unwrap();
    outcome.report.validate().unwrap();
    let summary = outcome.report.summary("cubecl-page=2").unwrap();
    assert_eq!(summary.estimates["physical_blocks"].mean, 7.0);
    assert_eq!(summary.estimates["unused_blocks"].mean, 4.0);
    assert_eq!(summary.estimates["pool_fragmentation"].mean, 4.0 / 7.0);
}

#[test]
fn ported_probes_keep_their_results() {
    let bucketing = run("prefill-bucketing", &[]).unwrap();
    assert!(bucketing.summary.contains("8 prompts · padding 324 → 52"));
    let unbucketed = &bucketing.report.summary("unbucketed").unwrap().estimates;
    assert_eq!(unbucketed["padding_tokens"].mean, 324.0);

    let adaptive = run("adaptive-prefill-cost", &[]).unwrap();
    assert!(adaptive.summary.contains("diverse PowerOfTwo"));
    assert!(adaptive.summary.contains("clustered Unbucketed"));

    let fifo = run("fifo-workload", &[("requests", "10")]).unwrap();
    assert!(
        fifo.summary
            .starts_with("10 requests · FIFO order verified")
    );
    assert_eq!(fifo.report.experiment.kind, ExperimentKind::Probe);
}

#[test]
fn the_scheduler_study_repeats_over_seeds_with_intervals() {
    let first = run("scheduler-comparison", &[("requests", "40")]).unwrap();
    let report = &first.report;

    assert_eq!(report.experiment.kind, ExperimentKind::Study);
    assert_eq!(report.experiment.domain, TimeDomain::Virtual);
    assert_eq!(report.protocol.repetitions, 5);
    assert_eq!(report.workload.seeds, [1, 2, 3, 4, 5]);
    assert_eq!(report.variants.len(), 5);
    assert!(
        report
            .summaries
            .iter()
            .all(|summary| summary.estimates["completion_p95"].ci95.is_some())
    );
    assert!(
        first
            .summary
            .starts_with("overload, 5 seeds × 40 requests, virtual time")
    );

    // Virtual time: a second run differs only in when it was recorded.
    let mut second = run("scheduler-comparison", &[("requests", "40")]).unwrap();
    second.report.metadata.recorded_at_unix_micros = report.metadata.recorded_at_unix_micros;
    assert_eq!(second, first);
}

#[test]
fn the_registry_validates_parameters_requirements_and_protocols() {
    assert!(matches!(
        run("missing", &[]),
        Err(ExperimentError::UnknownExperiment(_))
    ));
    assert!(matches!(
        run("fifo-workload", &[("requests", "many")]),
        Err(ExperimentError::InvalidParameter { .. })
    ));
    assert!(matches!(
        run("fifo-workload", &[("unknown", "1")]),
        Err(ExperimentError::InvalidParameter { .. })
    ));
    assert!(matches!(
        run(
            "scheduler-comparison",
            &[("trace", "nonsense"), ("requests", "10")]
        ),
        Err(ExperimentError::InvalidParameter { .. })
    ));
    // Two seeds are too few repetitions for a study.
    assert!(matches!(
        run(
            "scheduler-comparison",
            &[("seeds", "2"), ("requests", "10")]
        ),
        Err(ExperimentError::Nonconforming(_))
    ));

    struct NeedsModel;
    impl Experiment for NeedsModel {
        fn descriptor(&self) -> ExperimentDescriptor {
            ExperimentDescriptor {
                id: "needs-model",
                name: "Needs model",
                description: "",
                category: ExperimentCategory::Benchmark,
                kind: ExperimentKind::Probe,
                domain: TimeDomain::Host,
                payload_version: 1,
                requirements: vec![Requirement::LoadedModel],
                parameters: Vec::new(),
            }
        }

        fn run(
            &self,
            _context: &ExperimentContext,
            _parameters: &Parameters,
        ) -> Result<ExperimentOutcome, ExperimentError> {
            unreachable!("the registry checks requirements first")
        }
    }
    let registry = ExperimentRegistry::new(vec![Box::new(NeedsModel)]);
    assert!(matches!(
        registry.run("needs-model", &ExperimentContext::default(), &[]),
        Err(ExperimentError::MissingRequirement(
            Requirement::LoadedModel
        ))
    ));
}

#[test]
fn watermark_admission_beats_worst_case_reservation_when_requests_stop_early() {
    let outcome = run("kv-pressure", &[("requests", "60")]).unwrap();
    let estimate = |variant: &str, metric: &str| {
        outcome
            .report
            .summaries
            .iter()
            .find(|summary| summary.variant == variant)
            .unwrap()
            .estimates[metric]
    };

    let (worst, watermark) = (
        estimate("worst-case", "completion_p95"),
        estimate("watermark", "completion_p95"),
    );
    // Separated intervals: the gain is not noise across seeds.
    assert!(watermark.ci95.unwrap().high < worst.ci95.unwrap().low);
    assert!(estimate("watermark", "preemptions").mean > 0.0);
    assert_eq!(estimate("worst-case", "preemptions").mean, 0.0);
    for variant in ["worst-case", "watermark"] {
        assert_eq!(estimate(variant, "failed").mean, 0.0);
        assert_eq!(estimate(variant, "unfinished").mean, 0.0);
    }
}

#[test]
fn the_prefix_cache_cuts_prefill_work_on_shared_system_prompts() {
    let outcome = run("prefix-cache", &[("requests", "40"), ("seeds", "3")]).unwrap();
    let estimate = |variant: &str, metric: &str| {
        outcome
            .report
            .summaries
            .iter()
            .find(|summary| summary.variant == variant)
            .unwrap()
            .estimates[metric]
    };

    let (plain, cached) = (
        estimate("no-cache", "input_tokens"),
        estimate("prefix-cache", "input_tokens"),
    );
    assert!(cached.ci95.unwrap().high < plain.ci95.unwrap().low);
    assert!(
        cached.mean < plain.mean / 2.0,
        "most of each prompt is shared"
    );
    assert_eq!(estimate("no-cache", "completed").mean, 40.0);
    assert_eq!(estimate("prefix-cache", "completed").mean, 40.0);
}

#[test]
fn larger_pages_waste_more_and_reuse_less() {
    let outcome = run(
        "kv-page-size",
        &[
            ("page_sizes", "4,16,64"),
            ("requests", "40"),
            ("seeds", "3"),
        ],
    )
    .unwrap();
    let mean = |variant: &str, metric: &str| {
        outcome
            .report
            .summaries
            .iter()
            .find(|summary| summary.variant == variant)
            .unwrap()
            .estimates[metric]
            .mean
    };

    let waste = ["page=4", "page=16", "page=64"].map(|page| mean(page, "kv_fragmentation"));
    assert!(waste[0] < waste[1] && waste[1] < waste[2], "{waste:?}");
    // A 100-token prefix: 4-token pages can reuse all of it and 64-token
    // pages only one page, for each of the 39 requests after the first. A
    // request admitted while an earlier one is mid-prefill finds only the
    // blocks written so far, so reuse can fall slightly short of the bound.
    let (small, large) = (
        mean("page=4", "prefix_hit_tokens"),
        mean("page=64", "prefix_hit_tokens"),
    );
    assert!(
        small <= 39.0 * 100.0 && small > 0.95 * 39.0 * 100.0,
        "{small}"
    );
    assert!(large <= 39.0 * 64.0 && large < small, "{large}");
}
