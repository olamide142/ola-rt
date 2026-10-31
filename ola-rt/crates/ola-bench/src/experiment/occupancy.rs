//! Profiler-backed occupancy and utilization for CubeCL paged attention.
//!
//! Host latency cannot reveal how much of the accelerator a dispatch uses, so
//! this study measures host latency itself and attaches hardware counters
//! from a separate Nsight Compute capture of the same deterministic workload.
//! Every homogeneous `execute_batch` launches exactly one `paged_attention`
//! CubeCL kernel, so profiled launches map onto measurements by order; launch
//! geometry is checked against each shape's work-item count to catch a
//! capture taken with different parameters.
//!
//! Without an accelerator feature the study cannot run, but its capture
//! parsing still compiles and is tested.
#![cfg_attr(
    not(any(feature = "cubecl-cpu", feature = "wgpu", feature = "cuda")),
    allow(dead_code, reason = "only accelerator builds run the study")
)]

use std::collections::BTreeMap;

use super::{
    Direction, Experiment, ExperimentCategory, ExperimentContext, ExperimentDescriptor,
    ExperimentError, ExperimentKind, ExperimentOutcome, MetricSpec, ParameterSpec, Parameters,
    TimeDomain, Unit,
};

/// Paged-attention latency with optional Nsight Compute occupancy counters.
pub struct PagedAttentionOccupancy;

/// Launch-geometry counters every profile must carry.
const GRID_COUNTER: &str = "launch__grid_size";
const BLOCK_COUNTER: &str = "launch__block_size";

/// How a profiler value converts into its report unit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Scale {
    /// A percentage, reported as a 0–1 ratio.
    Percent,
    /// A duration, reported in microseconds.
    Duration,
    /// Reported as captured.
    Plain,
}

/// One Nsight Compute counter the study ingests.
struct ProfilerMetric {
    counter: &'static str,
    name: &'static str,
    unit: Unit,
    domain: TimeDomain,
    better: Direction,
    scale: Scale,
}

const PROFILER_METRICS: [ProfilerMetric; 6] = [
    ProfilerMetric {
        counter: "gpu__time_duration.sum",
        name: "gpu_duration_micros",
        unit: Unit::Micros,
        domain: TimeDomain::Device,
        better: Direction::Lower,
        scale: Scale::Duration,
    },
    ProfilerMetric {
        counter: "sm__warps_active.avg.pct_of_peak_sustained_active",
        name: "achieved_occupancy",
        unit: Unit::Ratio,
        domain: TimeDomain::Untimed,
        better: Direction::Higher,
        scale: Scale::Percent,
    },
    ProfilerMetric {
        counter: "sm__maximum_warps_per_active_cycle_pct",
        name: "theoretical_occupancy",
        unit: Unit::Ratio,
        domain: TimeDomain::Untimed,
        better: Direction::Higher,
        scale: Scale::Percent,
    },
    ProfilerMetric {
        counter: "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        name: "sm_throughput",
        unit: Unit::Ratio,
        domain: TimeDomain::Untimed,
        better: Direction::Higher,
        scale: Scale::Percent,
    },
    ProfilerMetric {
        counter: "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
        name: "memory_throughput",
        unit: Unit::Ratio,
        domain: TimeDomain::Untimed,
        better: Direction::Higher,
        scale: Scale::Percent,
    },
    ProfilerMetric {
        counter: "launch__registers_per_thread",
        name: "registers_per_thread",
        unit: Unit::Count,
        domain: TimeDomain::Untimed,
        better: Direction::Neither,
        scale: Scale::Plain,
    },
];

/// The `--metrics` list a capture must request.
pub fn nsight_metrics() -> String {
    PROFILER_METRICS
        .iter()
        .map(|metric| metric.counter)
        .chain([GRID_COUNTER, BLOCK_COUNTER])
        .collect::<Vec<_>>()
        .join(",")
}

/// One workload shape: a variant of the study.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Shape {
    page: usize,
    batch: usize,
    sequence: usize,
}

impl Shape {
    fn label(self) -> String {
        format!(
            "page={},batch={},seq={}",
            self.page, self.batch, self.sequence
        )
    }

    /// Threads the kernel needs: one per request, query head, and output dim.
    fn work_items(self, query_heads: usize, head_dim: usize) -> usize {
        self.batch * query_heads * head_dim
    }
}

/// Page sizes outermost, then batch sizes, then sequence lengths, so a
/// capture and a measurement with the same parameters launch identically.
fn shapes(pages: &[usize], batches: &[usize], sequences: &[usize]) -> Vec<Shape> {
    pages
        .iter()
        .flat_map(|&page| {
            batches.iter().flat_map(move |&batch| {
                sequences.iter().map(move |&sequence| Shape {
                    page,
                    batch,
                    sequence,
                })
            })
        })
        .collect()
}

/// One profiled kernel launch.
#[derive(Clone, Debug, PartialEq)]
struct ProfiledLaunch {
    kernel: String,
    threads: u64,
    /// Report metric name to converted value.
    values: BTreeMap<String, f64>,
}

/// Matching launches from one Nsight Compute capture.
#[derive(Clone, Debug, PartialEq)]
struct KernelProfile {
    launches: Vec<ProfiledLaunch>,
    device: Option<String>,
}

/// Parses `ncu --csv --page raw` output, keeping launches whose kernel name
/// contains `kernel_filter`.
fn parse_nsight_csv(text: &str, kernel_filter: &str) -> Result<KernelProfile, ExperimentError> {
    let invalid = |reason: String| ExperimentError::InvalidParameter {
        name: "profile".into(),
        reason,
    };
    // ncu interleaves `==PROF==` and `==WARNING==` lines with the table.
    let mut rows = text
        .lines()
        .map(str::trim_end)
        .filter(|line| !line.is_empty() && !line.starts_with("=="))
        .map(split_csv_line);
    let header = rows
        .next()
        .ok_or_else(|| invalid("profile has no CSV header".into()))?
        .map_err(invalid)?;
    let column = |name: &str| header.iter().position(|field| field == name);
    let kernel_column = column("Kernel Name")
        .or_else(|| column("Function Name"))
        .ok_or_else(|| {
            invalid("profile has no `Kernel Name` column; capture with --page raw".into())
        })?;
    let id_column = column("ID");
    let device_column = column("Device");
    let mut counters = Vec::new();
    let mut missing = Vec::new();
    for counter in PROFILER_METRICS
        .iter()
        .map(|metric| metric.counter)
        .chain([GRID_COUNTER, BLOCK_COUNTER])
    {
        match column(counter) {
            Some(index) => counters.push((counter, index)),
            None => missing.push(counter),
        }
    }
    if !missing.is_empty() {
        return Err(invalid(format!(
            "profile lacks counters {}; capture with --metrics {}",
            missing.join(","),
            nsight_metrics()
        )));
    }

    let mut units = BTreeMap::new();
    let mut launches = Vec::new();
    let mut device = None;
    for (line, row) in rows.enumerate() {
        let row = row.map_err(invalid)?;
        let field = |index: usize| row.get(index).map_or("", String::as_str);
        // The raw page's second row carries units and leaves `ID` empty.
        if line == 0 && id_column.is_some_and(|index| field(index).is_empty()) {
            for &(counter, index) in &counters {
                units.insert(counter, field(index).to_owned());
            }
            continue;
        }
        let kernel = field(kernel_column);
        if !kernel.contains(kernel_filter) {
            continue;
        }
        let number = |counter: &str| -> Result<f64, ExperimentError> {
            let index = counters
                .iter()
                .find(|(name, _)| *name == counter)
                .map(|(_, index)| *index)
                .expect("every counter has a column");
            let raw = field(index).replace(',', "");
            raw.trim()
                .parse::<f64>()
                .ok()
                .filter(|value| value.is_finite())
                .ok_or_else(|| {
                    invalid(format!(
                        "launch {} has a non-numeric `{counter}`: `{}`",
                        launches.len(),
                        field(index)
                    ))
                })
        };
        let mut values = BTreeMap::new();
        for metric in &PROFILER_METRICS {
            let raw = number(metric.counter)?;
            let unit = units.get(metric.counter).map_or("", String::as_str);
            let value = match metric.scale {
                Scale::Percent => raw / 100.0,
                Scale::Duration => raw * duration_micros(unit).map_err(&invalid)?,
                Scale::Plain => raw,
            };
            values.insert(metric.name.to_owned(), value);
        }
        let grid = number(GRID_COUNTER)?;
        let block = number(BLOCK_COUNTER)?;
        if device.is_none() {
            device = device_column
                .map(|index| field(index).to_owned())
                .filter(|name| !name.is_empty());
        }
        launches.push(ProfiledLaunch {
            kernel: kernel.to_owned(),
            threads: as_count(grid * block),
            values,
        });
    }
    Ok(KernelProfile { launches, device })
}

/// Microseconds per unit of an Nsight duration. An empty unit is the base
/// unit, nanoseconds, as with `--print-units base`.
fn duration_micros(unit: &str) -> Result<f64, String> {
    match unit {
        "" | "ns" | "nsecond" => Ok(1e-3),
        "us" | "usecond" => Ok(1.0),
        "ms" | "msecond" => Ok(1e3),
        "s" | "second" => Ok(1e6),
        other => Err(format!("unrecognized duration unit `{other}`")),
    }
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "launch geometry is a small non-negative integer"
)]
fn as_count(value: f64) -> u64 {
    value.max(0.0).round() as u64
}

/// Splits one CSV record, honouring quotes and doubled-quote escapes.
fn split_csv_line(line: &str) -> Result<Vec<String>, String> {
    let mut fields = Vec::new();
    let mut field = String::new();
    let mut quoted = false;
    let mut chars = line.chars().peekable();
    while let Some(character) = chars.next() {
        match (character, quoted) {
            ('"', true) if chars.peek() == Some(&'"') => {
                field.push('"');
                chars.next();
            }
            ('"', _) => quoted = !quoted,
            (',', false) => fields.push(std::mem::take(&mut field)),
            (character, _) => field.push(character),
        }
    }
    if quoted {
        return Err(format!("unterminated quote in `{line}`"));
    }
    fields.push(field);
    Ok(fields)
}

/// Assigns profiled launches to measured repetitions.
///
/// Each shape launches once for the oracle check, then `warmups` times, then
/// `repetitions` times. Returns, per shape, the profiler values for each
/// measured repetition.
fn assign_launches(
    profile: &KernelProfile,
    shapes: &[Shape],
    query_heads: usize,
    head_dim: usize,
    warmups: usize,
    repetitions: usize,
) -> Result<Vec<Vec<BTreeMap<String, f64>>>, ExperimentError> {
    let per_shape = 1 + warmups + repetitions;
    let expected = shapes.len() * per_shape;
    if profile.launches.len() != expected {
        return Err(ExperimentError::InvalidParameter {
            name: "profile".into(),
            reason: format!(
                "profile has {} matching launches, but {} shapes × (1 oracle + {warmups} \
                 warm-ups + {repetitions} repetitions) need {expected}; capture with \
                 identical parameters",
                profile.launches.len(),
                shapes.len()
            ),
        });
    }
    shapes
        .iter()
        .zip(profile.launches.chunks(per_shape))
        .map(|(shape, launches)| {
            let work_items = shape.work_items(query_heads, head_dim) as u64;
            for launch in launches {
                // Elementwise launches cover every item and overshoot by less
                // than the item count again; anything else is another shape.
                if launch.threads < work_items || launch.threads >= 2 * work_items + 1024 {
                    return Err(ExperimentError::InvalidParameter {
                        name: "profile".into(),
                        reason: format!(
                            "launch `{}` has {} threads, but shape {} needs {work_items}; \
                             capture with identical parameters",
                            launch.kernel,
                            launch.threads,
                            shape.label()
                        ),
                    });
                }
            }
            Ok(launches[1 + warmups..]
                .iter()
                .map(|launch| launch.values.clone())
                .collect())
        })
        .collect()
}

fn profiler_metric_specs() -> impl Iterator<Item = MetricSpec> {
    PROFILER_METRICS
        .iter()
        .map(|metric| MetricSpec::new(metric.name, metric.unit, metric.domain, metric.better))
}

impl Experiment for PagedAttentionOccupancy {
    fn descriptor(&self) -> ExperimentDescriptor {
        ExperimentDescriptor {
            id: "paged-attention-occupancy",
            name: "Paged-attention occupancy",
            description: "CubeCL paged-attention latency across page, batch, and sequence \
                          shapes, with Nsight Compute occupancy and utilization",
            category: ExperimentCategory::Benchmark,
            kind: ExperimentKind::Study,
            domain: TimeDomain::Host,
            payload_version: 1,
            requirements: Vec::new(),
            parameters: vec![
                ParameterSpec::text("page_sizes", "Tokens per logical page", "8,16,32"),
                ParameterSpec::text("batch_sizes", "Requests per dispatch", "1,8"),
                ParameterSpec::text("sequence_lengths", "Populated tokens per request", "64,256"),
                ParameterSpec::integer("query_heads", "Query heads", 8),
                ParameterSpec::integer("kv_heads", "Key/value heads", 2),
                ParameterSpec::integer("head_dim", "Width of one head", 64),
                ParameterSpec::integer("warmups", "Discarded dispatches per shape", 1),
                ParameterSpec::integer("repetitions", "Measured dispatches per shape", 5),
                ParameterSpec::text(
                    "profile",
                    "Nsight Compute `--csv --page raw` capture of an identical run; \
                     empty for latency only",
                    "",
                ),
                ParameterSpec::text(
                    "profile_kernel",
                    "Substring selecting the attention kernel in the capture",
                    "paged_attention",
                ),
            ],
        }
    }

    fn run(
        &self,
        context: &ExperimentContext,
        parameters: &Parameters,
    ) -> Result<ExperimentOutcome, ExperimentError> {
        #[cfg(not(any(feature = "cubecl-cpu", feature = "wgpu", feature = "cuda")))]
        {
            let _ = (context, parameters);
            Err(ExperimentError::Run(
                "paged-attention-occupancy requires the ola-bench cuda, wgpu, or cubecl-cpu \
                 feature"
                    .into(),
            ))
        }

        #[cfg(any(feature = "cubecl-cpu", feature = "wgpu", feature = "cuda"))]
        accelerated::run(&self.descriptor(), context, parameters)
    }
}

#[cfg(any(feature = "cubecl-cpu", feature = "wgpu", feature = "cuda"))]
mod accelerated {
    use std::time::Duration;

    use burn::tensor::Device;
    #[cfg(all(not(feature = "cuda"), feature = "wgpu"))]
    use burn::tensor::DeviceKind;
    use ola_kernel::{
        CubePagedAttention, KernelTarget, PagedAttentionConfig, PagedAttentionKernel,
        PortablePagedAttention,
    };

    use super::super::{
        builtin::{
            finish, kernel_input, lengths, max_error, metric, plain_variant, run_error, run_kernel,
            values,
        },
        content_digest,
    };
    use super::*;

    /// The most capable compiled target: CUDA, then WGPU, then CubeCL CPU.
    fn target() -> (Device, KernelTarget, &'static str) {
        #[cfg(feature = "cuda")]
        return (Device::cuda(0), KernelTarget::Cuda, "cubecl-cuda:0");
        #[cfg(all(not(feature = "cuda"), feature = "wgpu"))]
        return (
            Device::wgpu(DeviceKind::IntegratedGpu(0)),
            KernelTarget::Wgpu,
            "cubecl-wgpu:integrated-0",
        );
        #[cfg(all(not(feature = "cuda"), not(feature = "wgpu")))]
        return (Device::cpu(), KernelTarget::Cpu, "cubecl-cpu");
    }

    fn micros(duration: Duration) -> f64 {
        duration.as_secs_f64() * 1e6
    }

    fn positive(parameters: &Parameters, name: &str) -> Result<Vec<usize>, ExperimentError> {
        let values = lengths(parameters, name)?;
        if values.is_empty() || values.contains(&0) {
            return Err(ExperimentError::InvalidParameter {
                name: name.to_owned(),
                reason: "expected comma-separated positive integers".into(),
            });
        }
        Ok(values)
    }

    #[allow(
        clippy::too_many_lines,
        reason = "one linear protocol: validate, profile, measure, report"
    )]
    pub(super) fn run(
        descriptor: &ExperimentDescriptor,
        context: &ExperimentContext,
        parameters: &Parameters,
    ) -> Result<ExperimentOutcome, ExperimentError> {
        let shapes = shapes(
            &positive(parameters, "page_sizes")?,
            &positive(parameters, "batch_sizes")?,
            &positive(parameters, "sequence_lengths")?,
        );
        let query_heads = parameters.size("query_heads")?;
        let kv_heads = parameters.size("kv_heads")?;
        let head_dim = parameters.size("head_dim")?;
        let config =
            PagedAttentionConfig::new(query_heads, kv_heads, head_dim).map_err(|error| {
                ExperimentError::InvalidParameter {
                    name: "query_heads".into(),
                    reason: error.to_string(),
                }
            })?;
        let warmups = parameters.size("warmups")?;
        let repetitions = parameters.size("repetitions")?;
        if repetitions == 0 {
            return Err(ExperimentError::Run(
                "repetitions must be greater than zero".into(),
            ));
        }

        // Read and match the capture before any dispatch, so a mismatched
        // profile fails fast.
        let profile_path = parameters.text("profile")?;
        let profile = if profile_path.is_empty() {
            None
        } else {
            let text = std::fs::read_to_string(profile_path).map_err(|error| {
                ExperimentError::InvalidParameter {
                    name: "profile".into(),
                    reason: format!("could not read {profile_path}: {error}"),
                }
            })?;
            let parsed = parse_nsight_csv(&text, parameters.text("profile_kernel")?)?;
            let assigned = assign_launches(
                &parsed,
                &shapes,
                query_heads,
                head_dim,
                warmups,
                repetitions,
            )?;
            Some((content_digest(text.as_bytes()), parsed, assigned))
        };

        let (device, target, device_label) = target();
        let portable = PortablePagedAttention::new();
        let cube = CubePagedAttention::new(target).map_err(run_error)?;
        let mut variants = Vec::with_capacity(shapes.len());
        let mut content = String::new();
        for (index, shape) in shapes.iter().enumerate() {
            let input = kernel_input(&device, config, shape.page, shape.sequence, 0)?;
            let inputs = vec![input; shape.batch];
            let logical_pages = shape.sequence.div_ceil(shape.page);
            let padding_tokens = logical_pages * shape.page - shape.sequence;
            let work_items = shape.work_items(query_heads, head_dim);
            content.push_str(&format!("{};", shape.label()));

            // Launch order matches `assign_launches`: oracle, warm-ups, measured.
            let expected = run_kernel(&portable, &inputs)?.1;
            let error = max_error(&expected, &run_kernel(&cube, &inputs)?.1);
            for _ in 0..warmups {
                let _ = run_kernel(&cube, &inputs)?;
            }
            let mut samples = Vec::with_capacity(repetitions);
            for repetition in 0..repetitions {
                let (elapsed, _) = run_kernel(&cube, &inputs)?;
                let mut sample = values(&[
                    ("latency_micros", micros(elapsed)),
                    ("max_abs_error", error),
                    ("batch_size", shape.batch as f64),
                    ("page_tokens", shape.page as f64),
                    ("sequence_tokens", shape.sequence as f64),
                    ("logical_pages", logical_pages as f64),
                    ("padding_tokens", padding_tokens as f64),
                    ("work_items", work_items as f64),
                ]);
                if let Some((_, _, assigned)) = &profile {
                    sample.extend(assigned[index][repetition].clone());
                }
                samples.push(sample);
            }
            let mut variant = plain_variant(&shape.label());
            variant.kernel = Some(cube.descriptor().name().to_owned());
            variants.push((variant, samples));
        }

        let mut metrics = vec![
            metric(
                "latency_micros",
                Unit::Micros,
                TimeDomain::Host,
                Direction::Lower,
            ),
            metric(
                "max_abs_error",
                Unit::Ratio,
                TimeDomain::Untimed,
                Direction::Lower,
            ),
            metric(
                "batch_size",
                Unit::Count,
                TimeDomain::Untimed,
                Direction::Neither,
            ),
            metric(
                "page_tokens",
                Unit::Tokens,
                TimeDomain::Untimed,
                Direction::Neither,
            ),
            metric(
                "sequence_tokens",
                Unit::Tokens,
                TimeDomain::Untimed,
                Direction::Neither,
            ),
            metric(
                "logical_pages",
                Unit::Count,
                TimeDomain::Untimed,
                Direction::Neither,
            ),
            metric(
                "padding_tokens",
                Unit::Tokens,
                TimeDomain::Untimed,
                Direction::Lower,
            ),
            metric(
                "work_items",
                Unit::Count,
                TimeDomain::Untimed,
                Direction::Neither,
            ),
        ];
        if profile.is_some() {
            metrics.extend(profiler_metric_specs());
        }
        let mut outcome = finish(
            descriptor,
            context,
            parameters,
            "paged-attention-occupancy",
            (0..repetitions as u64).collect(),
            metrics,
            variants,
            String::new(),
        )?;
        outcome.report.protocol.warmups = warmups;
        outcome.report.protocol.repetitions = repetitions;
        outcome.report.protocol.ordering =
            "per shape: oracle check, warm-ups, measured; pages, then batches, then lengths".into();
        if outcome.report.metadata.device.is_none() {
            outcome.report.metadata.device = Some(device_label.to_owned());
        }
        if let Some((digest, parsed, _)) = &profile {
            content.push_str(&format!("profile={digest};"));
            outcome.report.details = Some(serde_json::json!({
                "profiler": {
                    "tool": "nsight-compute",
                    "device": parsed.device,
                    "kernel_filter": parameters.text("profile_kernel")?,
                    "launches": parsed.launches.len(),
                    "digest": digest,
                }
            }));
        }
        outcome.report.workload.digest = content_digest(content.as_bytes());
        let summary = outcome
            .report
            .summaries
            .iter()
            .map(|summary| {
                let latency = summary.estimates["latency_micros"].mean;
                match summary.estimates.get("achieved_occupancy") {
                    Some(occupancy) => format!(
                        "{}: {latency:.0} µs, occupancy {:.1}%",
                        summary.variant,
                        occupancy.mean * 100.0
                    ),
                    None => format!("{}: {latency:.0} µs", summary.variant),
                }
            })
            .collect::<Vec<_>>()
            .join(" · ");
        let summary = if profile.is_some() {
            summary
        } else {
            format!("{summary} · no profile attached")
        };
        Ok(ExperimentOutcome { summary, ..outcome })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A capture in the shape `ncu --csv --page raw` writes, with one shape
    /// of 2 × 1 × 1 work items per launch (`threads` per row).
    fn capture(threads: &[(&str, u64)]) -> String {
        let mut text = String::from("==PROF== Connected to process 1\n");
        let counters = nsight_metrics();
        let counters = counters.split(',').collect::<Vec<_>>();
        let mut header = vec!["ID", "Kernel Name", "Device"];
        header.extend(&counters);
        text.push_str(
            &header
                .iter()
                .map(|field| format!("\"{field}\""))
                .collect::<Vec<_>>()
                .join(","),
        );
        text.push('\n');
        let mut units = vec!["", "", ""];
        units.extend(["usecond", "%", "%", "%", "%", "register/thread", "", ""]);
        text.push_str(
            &units
                .iter()
                .map(|field| format!("\"{field}\""))
                .collect::<Vec<_>>()
                .join(","),
        );
        text.push('\n');
        for (id, (kernel, threads)) in threads.iter().enumerate() {
            text.push_str(&format!(
                "\"{id}\",\"{kernel}\",\"NVIDIA GeForce GTX 1650\",\"1,250.5\",\"25\",\"50\",\
                 \"10\",\"40\",\"32\",\"{threads}\",\"1\"\n"
            ));
        }
        text
    }

    #[test]
    fn csv_fields_keep_quoted_commas_and_escaped_quotes() {
        assert_eq!(
            split_csv_line(r#""a, b","say ""hi""",,3"#).unwrap(),
            vec!["a, b", "say \"hi\"", "", "3"]
        );
        assert!(split_csv_line("\"open").is_err());
    }

    #[test]
    fn parses_counters_with_units_and_filters_kernels() {
        let text = capture(&[
            ("paged_attention_f32", 64),
            ("matmul<float, 4>", 128),
            ("paged_attention_f32", 64),
        ]);
        let profile = parse_nsight_csv(&text, "paged_attention").unwrap();
        assert_eq!(profile.launches.len(), 2);
        assert_eq!(profile.device.as_deref(), Some("NVIDIA GeForce GTX 1650"));
        let launch = &profile.launches[0];
        assert_eq!(launch.threads, 64);
        assert_eq!(launch.values["gpu_duration_micros"], 1250.5);
        assert_eq!(launch.values["achieved_occupancy"], 0.25);
        assert_eq!(launch.values["theoretical_occupancy"], 0.5);
        assert_eq!(launch.values["sm_throughput"], 0.1);
        assert_eq!(launch.values["memory_throughput"], 0.4);
        assert_eq!(launch.values["registers_per_thread"], 32.0);
    }

    #[test]
    fn nanosecond_base_units_convert_to_micros() {
        let text = capture(&[("paged_attention", 64)]).replace("\"usecond\"", "\"nsecond\"");
        let profile = parse_nsight_csv(&text, "paged_attention").unwrap();
        assert!((profile.launches[0].values["gpu_duration_micros"] - 1.2505).abs() < 1e-12);
    }

    #[test]
    fn missing_counters_name_the_capture_command() {
        let text = "\"ID\",\"Kernel Name\",\"gpu__time_duration.sum\"\n\"0\",\"k\",\"1\"\n";
        let error = parse_nsight_csv(text, "k").unwrap_err().to_string();
        assert!(error.contains("sm__warps_active"), "{error}");
        assert!(error.contains(&nsight_metrics()), "{error}");
    }

    #[test]
    fn launches_map_to_measured_repetitions_after_oracle_and_warmups() {
        let shapes = shapes(&[4], &[1, 2], &[8]);
        // One query head of width 64: 64 then 128 work items.
        let mut threads = vec![("paged_attention", 64); 4];
        threads.extend(vec![("paged_attention", 128); 4]);
        let mut profile = parse_nsight_csv(&capture(&threads), "paged_attention").unwrap();
        for (index, launch) in profile.launches.iter_mut().enumerate() {
            launch
                .values
                .insert("registers_per_thread".into(), index as f64);
        }
        let assigned = assign_launches(&profile, &shapes, 1, 64, 1, 2).unwrap();
        let registers =
            |shape: usize, repetition: usize| assigned[shape][repetition]["registers_per_thread"];
        assert_eq!(assigned.len(), 2);
        assert_eq!((registers(0, 0), registers(0, 1)), (2.0, 3.0));
        assert_eq!((registers(1, 0), registers(1, 1)), (6.0, 7.0));
    }

    #[test]
    fn mismatched_captures_are_rejected() {
        let shapes = shapes(&[4], &[1, 2], &[8]);
        let short =
            parse_nsight_csv(&capture(&[("paged_attention", 64); 7]), "paged_attention").unwrap();
        let error = assign_launches(&short, &shapes, 1, 64, 1, 2)
            .unwrap_err()
            .to_string();
        assert!(error.contains("need 8"), "{error}");

        // Right count, but the second shape's launches cover too few items.
        let wrong =
            parse_nsight_csv(&capture(&[("paged_attention", 64); 8]), "paged_attention").unwrap();
        let error = assign_launches(&wrong, &shapes, 1, 64, 1, 2)
            .unwrap_err()
            .to_string();
        assert!(error.contains("page=4,batch=2,seq=8"), "{error}");
    }

    #[cfg(feature = "cubecl-cpu")]
    fn run_study(extra: &[(&str, &str)]) -> Result<ExperimentOutcome, ExperimentError> {
        let mut overrides = vec![
            ("page_sizes", "2"),
            ("batch_sizes", "1,2"),
            ("sequence_lengths", "5"),
            ("query_heads", "4"),
            ("kv_heads", "2"),
            ("head_dim", "4"),
            ("warmups", "1"),
            ("repetitions", "5"),
        ];
        overrides.extend(extra);
        let overrides = overrides
            .into_iter()
            .map(|(name, value)| (name.to_owned(), value.to_owned()))
            .collect::<Vec<_>>();
        super::super::ExperimentRegistry::builtin().run(
            "paged-attention-occupancy",
            &ExperimentContext::default(),
            &overrides,
        )
    }

    #[cfg(feature = "cubecl-cpu")]
    #[test]
    fn study_attaches_a_matching_capture_on_cpu() {
        let latency = run_study(&[]).unwrap();
        latency.report.validate().unwrap();
        assert!(latency.summary.ends_with("no profile attached"));
        let estimates = &latency
            .report
            .summary("page=2,batch=2,seq=5")
            .unwrap()
            .estimates;
        assert_eq!(estimates["work_items"].mean, 32.0);
        assert_eq!(estimates["padding_tokens"].mean, 1.0);
        assert!(estimates["max_abs_error"].mean < 1e-5);
        assert!(!estimates.contains_key("achieved_occupancy"));

        // 4 heads × 4 dims: 16 then 32 work items, 7 launches per shape.
        let mut threads = vec![("paged_attention", 16); 7];
        threads.extend(vec![("paged_attention", 32); 7]);
        let path = std::env::temp_dir().join(format!("ola-occupancy-{}.csv", std::process::id()));
        std::fs::write(&path, capture(&threads)).unwrap();
        let profiled = run_study(&[("profile", path.to_str().unwrap())]);
        std::fs::remove_file(&path).unwrap();
        let profiled = profiled.unwrap();
        profiled.report.validate().unwrap();
        let estimates = &profiled
            .report
            .summary("page=2,batch=1,seq=5")
            .unwrap()
            .estimates;
        assert_eq!(estimates["achieved_occupancy"].mean, 0.25);
        assert_eq!(estimates["gpu_duration_micros"].mean, 1250.5);
        let domain = |name: &str| {
            profiled
                .report
                .metrics
                .iter()
                .find(|metric| metric.name == name)
                .unwrap()
                .domain
        };
        assert_eq!(domain("gpu_duration_micros"), TimeDomain::Device);
        assert_eq!(domain("latency_micros"), TimeDomain::Host);
        assert_ne!(
            profiled.report.workload.digest,
            latency.report.workload.digest
        );
        assert_eq!(
            profiled.report.details.as_ref().unwrap()["profiler"]["launches"],
            14
        );
        assert!(profiled.summary.contains("occupancy 25.0%"));
    }

    #[cfg(feature = "cubecl-cpu")]
    #[test]
    fn study_rejects_a_capture_from_other_parameters() {
        let path =
            std::env::temp_dir().join(format!("ola-occupancy-short-{}.csv", std::process::id()));
        std::fs::write(&path, capture(&[("paged_attention", 16); 7])).unwrap();
        let result = run_study(&[("profile", path.to_str().unwrap())]);
        std::fs::remove_file(&path).unwrap();
        assert!(matches!(
            result,
            Err(ExperimentError::InvalidParameter { ref name, .. }) if name == "profile"
        ));
    }

    #[test]
    fn shapes_vary_sequence_fastest() {
        let labels = shapes(&[8, 16], &[1], &[64, 256])
            .into_iter()
            .map(Shape::label)
            .collect::<Vec<_>>();
        assert_eq!(
            labels,
            [
                "page=8,batch=1,seq=64",
                "page=8,batch=1,seq=256",
                "page=16,batch=1,seq=64",
                "page=16,batch=1,seq=256",
            ]
        );
    }
}
