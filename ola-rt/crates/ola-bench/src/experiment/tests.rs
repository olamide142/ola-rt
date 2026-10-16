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
