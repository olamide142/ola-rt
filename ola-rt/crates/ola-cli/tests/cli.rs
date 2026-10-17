//! End-to-end tests of the `ola` command line.

use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use ola_bench::{ExperimentReport, ReportParts};
use ola_cli::{EXIT_ERROR, EXIT_OK, EXIT_REGRESSION, run};
use ola_ui::{LocalBackend, LocalDevice, RunState, UiBackend};

fn ola(args: &[&str]) -> (u8, String) {
    let mut out = Vec::new();
    let code = run(args.iter().map(|arg| (*arg).to_owned()), &mut out);
    (code, String::from_utf8(out).unwrap())
}

fn scratch(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "ola-cli-{name}-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

fn only_report(directory: &Path) -> ExperimentReport {
    let paths = std::fs::read_dir(directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect::<Vec<_>>();
    assert_eq!(
        paths.len(),
        1,
        "expected exactly one report in {directory:?}"
    );
    ExperimentReport::read(&paths[0]).unwrap()
}

#[test]
fn list_shows_every_experiment_with_requirements_and_parameters() {
    let (code, output) = ola(&["experiment", "list"]);

    assert_eq!(code, EXIT_OK);
    for id in [
        "llama-generation",
        "llama-prefill-strategies",
        "llama-chunked-prefill",
        "cost-calibration",
        "kv-block-pressure",
        "fifo-workload",
        "prefill-bucketing",
        "adaptive-prefill-cost",
        "scheduler-comparison",
        "kv-pressure",
    ] {
        assert!(output.contains(id), "{id} missing from:\n{output}");
    }
    assert!(output.contains(
        "llama-generation  [benchmark, probe, not for publication, host time, needs --model, traceable]"
    ));
    assert!(output.contains("--set seeds=5"));
    assert!(output.contains("scheduler-comparison  [experiment, study, virtual time, traceable]"));
    assert!(
        output.contains("prefill-bucketing  [experiment, probe, not for publication, untimed]")
    );
}

#[test]
fn traceable_experiments_write_labelled_json_lines_traces() {
    let directory = scratch("trace");
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join("trace.jsonl");
    let path_text = path.to_str().unwrap();

    let (code, output) = ola(&[
        "experiment",
        "run",
        "scheduler-comparison",
        "--set",
        "requests=12",
        "--set",
        "seeds=3",
        "--trace",
        path_text,
    ]);
    assert_eq!(code, EXIT_OK, "{output}");
    let records = ola_observe::trace::read_jsonl(std::io::BufReader::new(
        std::fs::File::open(&path).unwrap(),
    ))
    .unwrap();
    let runs = records
        .iter()
        .map(|record| record.run.clone().unwrap())
        .collect::<std::collections::BTreeSet<_>>();
    // Five policies, three seeds: one run each.
    assert_eq!(runs.len(), 15, "{runs:?}");
    assert!(runs.contains("fifo/seed=1"));
    // Every request in every run reaches an outcome.
    let completed = records
        .iter()
        .filter(|record| record.event.is_terminal())
        .count();
    assert_eq!(completed, 15 * 12);

    // Tracing is read-only: the same study without a trace reports the
    // same samples.
    let study = |trace: Option<&str>| {
        let mut args = vec![
            "experiment",
            "run",
            "scheduler-comparison",
            "--set",
            "requests=12",
            "--set",
            "seeds=3",
            "--json",
        ];
        if let Some(trace) = trace {
            args.extend(["--trace", trace]);
        }
        let (code, output) = ola(&args);
        assert_eq!(code, EXIT_OK, "{output}");
        ExperimentReport::from_json(&output).unwrap().samples
    };
    let traced_again = directory.join("again.jsonl");
    assert_eq!(study(Some(traced_again.to_str().unwrap())), study(None));

    // The file is never overwritten, and untraceable experiments refuse.
    let (code, output) = ola(&[
        "experiment",
        "run",
        "scheduler-comparison",
        "--trace",
        path_text,
    ]);
    assert_eq!(code, EXIT_ERROR);
    assert!(output.contains("could not create trace"), "{output}");
    let other = directory.join("other.jsonl");
    let (code, output) = ola(&[
        "experiment",
        "run",
        "fifo-workload",
        "--trace",
        other.to_str().unwrap(),
    ]);
    assert_eq!(code, EXIT_ERROR);
    assert!(output.contains("cannot write traces"), "{output}");
    assert!(!other.exists());
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn run_prints_labelled_summaries_and_honours_overrides() {
    let (code, output) = ola(&["experiment", "run", "prefill-bucketing"]);
    assert_eq!(code, EXIT_OK);
    assert!(output.contains("8 prompts · padding 324 → 52"));
    assert!(
        output
            .trim_end()
            .ends_with("probe, not for publication, untimed")
    );

    let (code, output) = ola(&[
        "experiment",
        "run",
        "prefill-bucketing",
        "--set",
        "prompt_lengths=5,9",
        "--json",
    ]);
    assert_eq!(code, EXIT_OK);
    let report = ExperimentReport::from_json(&output).unwrap();
    assert_eq!(report.workload.parameters["prompt_lengths"], "5,9");
}

#[test]
fn errors_exit_with_code_two() {
    let (code, output) = ola(&["experiment", "run", "no-such-experiment"]);
    assert_eq!(code, EXIT_ERROR);
    assert!(output.contains("unknown experiment"));

    let (code, output) = ola(&["experiment", "run", "llama-generation"]);
    assert_eq!(code, EXIT_ERROR);
    assert!(output.contains("needs a loaded model"));

    let (code, _) = ola(&["experiment", "run", "fifo-workload", "--set", "requests"]);
    assert_eq!(code, EXIT_ERROR);
    let (code, _) = ola(&["bogus"]);
    assert_eq!(code, EXIT_ERROR);

    let (code, output) = ola(&["serve"]);
    assert_eq!(code, EXIT_ERROR);
    assert!(output.contains("serve needs --model DIR"));
    let (code, output) = ola(&["serve", "--model", "x", "--max-active", "0"]);
    assert_eq!(code, EXIT_ERROR);
    assert!(output.contains("--max-active expects a positive integer"));
}

#[test]
fn compare_flags_regressions_with_exit_code_one() {
    let directory = scratch("compare");
    let study = [
        "experiment",
        "run",
        "scheduler-comparison",
        "--set",
        "requests=40",
        "--results-dir",
    ];
    let mut args = study.to_vec();
    let path = directory.join("baseline");
    let path = path.to_str().unwrap();
    args.push(path);
    assert_eq!(ola(&args).0, EXIT_OK);
    let baseline = only_report(&directory.join("baseline"));
    let baseline_path = directory.join("baseline.json");
    std::fs::write(&baseline_path, baseline.to_json_pretty().unwrap()).unwrap();

    // Identical studies: intervals overlap, so nothing regresses.
    let (code, output) = ola(&[
        "experiment",
        "compare",
        baseline_path.to_str().unwrap(),
        baseline_path.to_str().unwrap(),
    ]);
    assert_eq!(code, EXIT_OK, "{output}");
    assert!(output.contains("identity: unchanged"));
    assert!(output.contains("regressions: 0"));

    // Doubling FIFO's completion times is a clear, separated regression.
    let mut slower = ReportParts {
        experiment: baseline.experiment.clone(),
        metadata: baseline.metadata.clone(),
        workload: baseline.workload.clone(),
        protocol: baseline.protocol.clone(),
        metrics: baseline.metrics.clone(),
        variants: baseline.variants.clone(),
        samples: baseline.samples.clone(),
        details: None,
    };
    for sample in slower
        .samples
        .iter_mut()
        .filter(|sample| sample.variant == "fifo")
    {
        *sample.values.get_mut("completion_p95").unwrap() *= 2.0;
    }
    let candidate_path = directory.join("candidate.json");
    std::fs::write(
        &candidate_path,
        ExperimentReport::new(slower)
            .unwrap()
            .to_json_pretty()
            .unwrap(),
    )
    .unwrap();
    let (code, output) = ola(&[
        "experiment",
        "compare",
        baseline_path.to_str().unwrap(),
        candidate_path.to_str().unwrap(),
    ]);
    assert_eq!(code, EXIT_REGRESSION, "{output}");
    assert!(output.lines().any(|line| line.starts_with("fifo")
        && line.contains("completion_p95")
        && line.ends_with("REGRESSED")));
    assert!(output.contains("regressions: 1"));
    std::fs::remove_dir_all(directory).unwrap();
}

/// Runs `id` in the terminal workbench with a results directory and returns
/// the saved report.
fn run_in_ui(id: &str, directory: &Path) -> ExperimentReport {
    let mut backend = LocalBackend::with_options(None, LocalDevice::Cpu, Some(directory.into()));
    backend.start_experiment(id).unwrap();
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        let snapshot = backend.snapshot().unwrap();
        let item = snapshot
            .experiments
            .iter()
            .find(|item| item.id == id)
            .unwrap();
        match item.state {
            RunState::Succeeded => break,
            RunState::Failed => panic!("{id} failed: {:?}", item.last_result),
            _ => {}
        }
        assert!(Instant::now() < deadline, "{id} did not finish");
        std::thread::sleep(Duration::from_millis(20));
    }
    only_report(directory)
}

#[test]
fn the_cli_and_the_workbench_produce_the_same_report() {
    for id in ["prefill-bucketing", "scheduler-comparison"] {
        let directory = scratch(id);
        let mut from_ui = run_in_ui(id, &directory.join("ui"));
        let cli_directory = directory.join("cli");
        let (code, output) = ola(&[
            "experiment",
            "run",
            id,
            "--results-dir",
            cli_directory.to_str().unwrap(),
        ]);
        assert_eq!(code, EXIT_OK, "{output}");
        let from_cli = only_report(&cli_directory);

        // Only the recording time may differ.
        from_ui.metadata.recorded_at_unix_micros = from_cli.metadata.recorded_at_unix_micros;
        assert_eq!(from_ui, from_cli, "{id} differs between frontends");
        std::fs::remove_dir_all(directory).unwrap();
    }
}
