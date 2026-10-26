//! Command-line frontend for OlaRT experiments and serving.
//!
//! The terminal workbench and this CLI share `ola_bench::ExperimentRegistry`,
//! so a named experiment runs identically from either. `ola serve` serves a
//! model over the `ola-serve` HTTP API. Exit codes: `0` success, `1` a
//! comparison found a regression, `2` usage or run errors.

mod serve;

use std::{collections::BTreeMap, io::Write, path::PathBuf};

use burn::tensor::Device;
use ola_bench::{
    Change, CompareOptions, ExperimentCategory, ExperimentContext, ExperimentKind,
    ExperimentRegistry, ExperimentReport, ExperimentRunConfig, ExperimentTrace, ParameterValue,
    Requirement, TimeDomain, compare_reports,
};
use ola_burn::llama::LlamaFactory;
use ola_model::CausalLanguageModel;

/// Success.
pub const EXIT_OK: u8 = 0;
/// A comparison found at least one regression.
pub const EXIT_REGRESSION: u8 = 1;
/// Usage, configuration, or run error.
pub const EXIT_ERROR: u8 = 2;

const USAGE: &str = "usage:
  ola experiment list
  ola experiment run <id> [--config FILE] [--model DIR] [--backend cpu|wgpu|cuda] [--set KEY=VALUE]...
                         [--results-dir DIR] [--json] [--trace FILE]
  ola experiment compare <baseline.json> <candidate.json>
                         [--threshold FRACTION] [--allow-workload-mismatch]
  ola serve --model DIR [--backend cpu|wgpu|cuda] [--name ID] [--addr HOST:PORT]
            [--max-active N] [--max-in-flight N] [--drain-secs S] [--trace FILE]

Exit codes: 0 success, 1 regression found by compare, 2 error.";

/// Runs the CLI with `args` (excluding the program name), writing output to
/// `out`, and returns the process exit code.
pub fn run(args: impl IntoIterator<Item = String>, out: &mut impl Write) -> u8 {
    let args = args.into_iter().collect::<Vec<_>>();
    let result = match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        ["experiment", "list"] => list(out),
        ["experiment", "run", id, rest @ ..] => run_experiment(id, rest, out),
        ["experiment", "compare", baseline, candidate, rest @ ..] => {
            compare(baseline, candidate, rest, out)
        }
        ["serve", rest @ ..] => serve::serve(rest, out),
        ["--help" | "-h" | "help"] | [] => {
            let _ = writeln!(out, "{USAGE}");
            Ok(EXIT_OK)
        }
        _ => Err(format!("unrecognized command\n\n{USAGE}")),
    };
    match result {
        Ok(code) => code,
        Err(message) => {
            let _ = writeln!(out, "error: {message}");
            EXIT_ERROR
        }
    }
}

fn list(out: &mut impl Write) -> Result<u8, String> {
    let registry = ExperimentRegistry::builtin();
    for descriptor in registry.descriptors() {
        let category = match descriptor.category {
            ExperimentCategory::Benchmark => "benchmark",
            ExperimentCategory::Experiment => "experiment",
        };
        let model = if descriptor.requirements.contains(&Requirement::LoadedModel) {
            ", needs --model"
        } else {
            ""
        };
        let traceable = if registry.traceable(descriptor.id) {
            ", traceable"
        } else {
            ""
        };
        write_line(
            out,
            format!(
                "{}  [{category}, {}, {}{model}{traceable}]\n    {}",
                descriptor.id,
                kind(descriptor.kind),
                domain(descriptor.domain),
                descriptor.description
            ),
        )?;
        for parameter in &descriptor.parameters {
            let default = match &parameter.default {
                ParameterValue::Integer(value) => value.to_string(),
                ParameterValue::Text(value) => format!("\"{value}\""),
            };
            write_line(
                out,
                format!(
                    "    --set {}={default}  {}",
                    parameter.name, parameter.description
                ),
            )?;
        }
    }
    Ok(EXIT_OK)
}

fn run_experiment(id: &str, rest: &[&str], out: &mut impl Write) -> Result<u8, String> {
    let mut model_directory = None;
    let mut backend = "cpu".to_owned();
    let mut overrides = Vec::new();
    let mut config_path = None;
    let mut results_directory = None;
    let mut json = false;
    let mut trace_path = None;
    let mut iter = rest.iter();
    while let Some(flag) = iter.next() {
        let mut value = || {
            iter.next()
                .map(|value| (*value).to_owned())
                .ok_or_else(|| format!("{flag} requires a value"))
        };
        match *flag {
            "--config" => config_path = Some(PathBuf::from(value()?)),
            "--model" => model_directory = Some(PathBuf::from(value()?)),
            "--backend" => backend = value()?,
            "--set" => {
                let pair = value()?;
                let (name, value) = pair
                    .split_once('=')
                    .ok_or_else(|| format!("--set expects KEY=VALUE, got `{pair}`"))?;
                overrides.push((name.to_owned(), value.to_owned()));
            }
            "--results-dir" => results_directory = Some(PathBuf::from(value()?)),
            "--json" => json = true,
            "--trace" => trace_path = Some(PathBuf::from(value()?)),
            other => return Err(format!("unknown option `{other}`\n\n{USAGE}")),
        }
    }

    let mut configured = BTreeMap::new();
    if let Some(path) = config_path {
        let json = std::fs::read_to_string(&path)
            .map_err(|error| format!("could not read config {}: {error}", path.display()))?;
        let config = ExperimentRunConfig::from_json(&json)
            .map_err(|error| format!("could not parse config {}: {error}", path.display()))?;
        if config.id != id {
            return Err(format!(
                "config selects experiment `{}`, but command selects `{id}`",
                config.id
            ));
        }
        configured.extend(config.parameters);
    }
    configured.extend(overrides);
    let overrides = configured.into_iter().collect::<Vec<_>>();

    let registry = ExperimentRegistry::builtin();
    // Checked before the file is created, so a refused run leaves nothing.
    if trace_path.is_some() && !registry.traceable(id) {
        return Err(format!(
            "experiment `{id}` cannot write traces; see `ola experiment list`"
        ));
    }
    let mut context = match &model_directory {
        Some(directory) => ExperimentContext {
            model: Some(load_model(directory, &backend)?),
            model_label: Some(
                directory
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("model")
                    .to_owned(),
            ),
            device: Some(
                std::env::var("OLA_BENCH_DEVICE_LABEL")
                    .unwrap_or_else(|_| format!("ola-cli / Burn / {backend}")),
            ),
            trace: None,
        },
        None => ExperimentContext::default(),
    };
    if let Some(path) = &trace_path {
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|error| format!("could not create trace {}: {error}", path.display()))?;
        context.trace = Some(ExperimentTrace::new(std::io::BufWriter::new(file)));
    }
    let outcome = registry
        .run(id, &context, &overrides)
        .map_err(|error| error.to_string())?;
    if let Some(trace) = &context.trace {
        trace
            .flush()
            .map_err(|error| format!("could not write trace: {error}"))?;
    }
    let report = &outcome.report;
    if json {
        write_line(
            out,
            report.to_json_pretty().map_err(|error| error.to_string())?,
        )?;
    } else {
        write_line(
            out,
            format!(
                "{} · {}, {}",
                outcome.summary,
                kind(report.experiment.kind),
                domain(report.experiment.domain)
            ),
        )?;
    }
    if let Some(directory) = results_directory {
        let path = report
            .write_new(&directory)
            .map_err(|error| format!("could not save report: {error}"))?;
        if !json {
            write_line(out, format!("saved {}", path.display()))?;
        }
    }
    Ok(EXIT_OK)
}

fn compare(
    baseline: &str,
    candidate: &str,
    rest: &[&str],
    out: &mut impl Write,
) -> Result<u8, String> {
    let mut options = CompareOptions::default();
    let mut iter = rest.iter();
    while let Some(flag) = iter.next() {
        match *flag {
            "--threshold" => {
                let value = iter
                    .next()
                    .ok_or("--threshold requires a value")?
                    .parse::<f64>()
                    .map_err(|_| "--threshold expects a fraction such as 0.05")?;
                options.relative_threshold = value;
            }
            "--allow-workload-mismatch" => options.allow_workload_mismatch = true,
            other => return Err(format!("unknown option `{other}`\n\n{USAGE}")),
        }
    }
    let read = |path: &str| {
        ExperimentReport::read_compatible(path).map_err(|error| format!("{path}: {error}"))
    };
    let (before, after) = (read(baseline)?, read(candidate)?);
    let comparison =
        compare_reports(&before, &after, options).map_err(|error| error.to_string())?;

    write_line(
        out,
        format!(
            "{} · threshold {:.1}% · intervals must separate for a verdict",
            comparison.experiment,
            options.relative_threshold * 100.0
        ),
    )?;
    if comparison.identity_differences.is_empty() {
        write_line(out, "identity: unchanged".to_owned())?;
    } else {
        for difference in &comparison.identity_differences {
            write_line(
                out,
                format!(
                    "identity: {}: {} → {}",
                    difference.field, difference.baseline, difference.candidate
                ),
            )?;
        }
    }
    for metric in &comparison.metrics {
        let relative = metric.relative.map_or_else(
            || "n/a".to_owned(),
            |relative| format!("{:+.1}%", relative * 100.0),
        );
        let verdict = match metric.change {
            Change::Improved => "improved",
            Change::Regressed => "REGRESSED",
            Change::Inconclusive => "inconclusive",
            Change::Informational => "info",
        };
        write_line(
            out,
            format!(
                "{:<14} {:<28} {:>14.2} → {:>14.2}  {:>8}  {verdict}",
                metric.variant,
                metric.metric.name,
                metric.baseline.mean,
                metric.candidate.mean,
                relative
            ),
        )?;
    }
    for unmatched in &comparison.unmatched {
        write_line(out, format!("unmatched: {unmatched}"))?;
    }
    let regressions = comparison.regressions().count();
    write_line(out, format!("regressions: {regressions}"))?;
    Ok(if regressions > 0 {
        EXIT_REGRESSION
    } else {
        EXIT_OK
    })
}

fn load_model(directory: &PathBuf, backend: &str) -> Result<CausalLanguageModel, String> {
    let factory = match backend {
        "cpu" => LlamaFactory::new(Device::flex()),
        #[cfg(feature = "wgpu")]
        "wgpu" => LlamaFactory::new(Device::wgpu(Default::default())),
        #[cfg(feature = "cuda")]
        "cuda" => LlamaFactory::new(Device::cuda(0)),
        #[cfg(not(feature = "wgpu"))]
        "wgpu" => return Err(not_compiled(backend)),
        #[cfg(not(feature = "cuda"))]
        "cuda" => return Err(not_compiled(backend)),
        other => return Err(format!("unknown backend `{other}`; use cpu, wgpu, or cuda")),
    };
    factory
        .load_local(directory)
        .map_err(|error| format!("could not load {}: {error}", directory.display()))
}

#[cfg(any(not(feature = "wgpu"), not(feature = "cuda")))]
fn not_compiled(backend: &str) -> String {
    format!("{backend} support is not compiled; rebuild ola-cli with --features {backend}")
}

const fn kind(kind: ExperimentKind) -> &'static str {
    match kind {
        ExperimentKind::Probe => "probe, not for publication",
        ExperimentKind::Study => "study",
    }
}

const fn domain(domain: TimeDomain) -> &'static str {
    match domain {
        TimeDomain::Virtual => "virtual time",
        TimeDomain::Host => "host time",
        TimeDomain::Untimed => "untimed",
    }
}

fn write_line(out: &mut impl Write, line: String) -> Result<(), String> {
    writeln!(out, "{line}").map_err(|error| error.to_string())
}
