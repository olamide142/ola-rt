//! Trace export and simulation for sim-versus-real validation.
//!
//! ```text
//! replay_trace export <trace> <seed> <requests> <load> <out.jsonl>
//! replay_trace simulate <observed.jsonl> <max_active> <kv> <out.jsonl>
//! ```
//!
//! * `export` writes a `SchedulingTrace` (`overload` or `priority_classes`)
//!   as JSON Lines: `arrival_us`, `prompt_tokens`, `max_new_tokens`,
//!   `priority`, and `deadline_us` relative to arrival. Loads `light`,
//!   `moderate`, and `overload` use mean interarrival times of 4000, 2000,
//!   and 1000 µs, scaled like `reviewer_eval` by the ratio of a reference
//!   decode iteration's cost under `OLA_COST` to the default cost model.
//! * `simulate` reads what `lab/replay/client.py` observed from `ola serve`
//!   and replays it in virtual time with the server's policy (FIFO with
//!   memory-aware admission, no prefill chunking), the same slot limit, and
//!   the server's KV pool (`kv` is `bytes_per_token,tokens_per_block,
//!   capacity_bytes`, from `GET /v1/models`). Each request arrives at its
//!   server queue time and stops after the number of tokens it really
//!   generated. Writes one JSON line per request with simulated times.
//!
//! Set `OLA_COST=fixed,per_invocation,per_dense_token` (microseconds) to the
//! calibrated coefficients for both commands. Three more values,
//! `per_padding_token,decode_per_invocation,decode_per_row`, price padding
//! and decode invocations separately (see `CostModel`).

#![allow(missing_docs)]

use std::{
    error::Error,
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
};

use ola_bench::{
    CostModel, DecodeCost, KvGeometry, SchedulingTrace, SimulationConfig, TraceParameters,
    TraceRequest, TrialOutcome, run_observed_trial,
};
use ola_runtime::{ComposedPolicy, MemoryAwareAdmission};
use serde_json::{Value, json};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("export") if args.len() == 6 => export(&args[1..]),
        Some("simulate") if args.len() == 5 => simulate(&args[1..]),
        _ => Err(
            "usage: replay_trace export <trace> <seed> <requests> <load> <out.jsonl> | \
                  simulate <observed.jsonl> <max_active> <kv> <out.jsonl>"
                .into(),
        ),
    }
}

fn cost_model() -> Result<CostModel, Box<dyn Error>> {
    let Ok(text) = std::env::var("OLA_COST") else {
        return Ok(CostModel::default());
    };
    let parts = numbers(&text)?;
    let (&[fixed, per_invocation, per_dense], rest) = parts.split_at(parts.len().min(3)) else {
        return Err(COST_USAGE.into());
    };
    let (per_padding, decode) = match *rest {
        [] => (None, None),
        [per_padding, decode_per_invocation, decode_per_row] => (
            Some(per_padding),
            Some(DecodeCost {
                per_invocation_micros: decode_per_invocation,
                per_row_micros: decode_per_row,
            }),
        ),
        _ => return Err(COST_USAGE.into()),
    };
    Ok(CostModel {
        fixed_micros: fixed,
        per_invocation_micros: per_invocation,
        per_dense_token_micros: per_dense,
        per_padding_token_micros: per_padding,
        decode,
    })
}

const COST_USAGE: &str = "OLA_COST=fixed,per_invocation,per_dense_token\
     [,per_padding_token,decode_per_invocation,decode_per_row]";

fn numbers(text: &str) -> Result<Vec<u64>, Box<dyn Error>> {
    Ok(text
        .split(',')
        .map(|part| part.trim().parse())
        .collect::<Result<_, _>>()?)
}

/// Same scaling as `reviewer_eval`: a reference decode iteration (one
/// invocation, eight one-token rows) under the active cost model relative
/// to the default. With a decode cost, the reference uses it.
fn load_scale(cost: CostModel) -> f64 {
    let reference = |c: CostModel| {
        let invocation = match c.decode {
            Some(decode) => decode.per_invocation_micros + 8 * decode.per_row_micros,
            None => c.per_invocation_micros + 8 * c.per_dense_token_micros,
        };
        (c.fixed_micros + invocation) as f64
    };
    reference(cost) / reference(CostModel::default())
}

fn export(args: &[String]) -> Result<(), Box<dyn Error>> {
    let [trace_name, seed, requests, load, out] = args else {
        unreachable!("argument count checked by the caller");
    };
    let nominal = match load.as_str() {
        "light" => 4_000.0,
        "moderate" => 2_000.0,
        "overload" => 1_000.0,
        other => return Err(format!("unknown load `{other}`").into()),
    };
    let scale = load_scale(cost_model()?);
    let parameters = TraceParameters {
        seed: seed.parse()?,
        requests: requests.parse()?,
        mean_interarrival_micros: (nominal * scale).round().max(1.0) as u64,
    };
    let trace = match trace_name.as_str() {
        "overload" => SchedulingTrace::overload(parameters),
        "priority_classes" => SchedulingTrace::priority_classes(parameters),
        other => return Err(format!("unsupported trace `{other}`").into()),
    };
    let mut file = BufWriter::new(File::create(out)?);
    for (index, request) in trace.requests.iter().enumerate() {
        let line = json!({
            "index": index,
            "trace": trace_name,
            "seed": parameters.seed,
            "load": load,
            "arrival_us": request.arrival_micros,
            "prompt_tokens": request.prompt_tokens,
            "max_new_tokens": request.max_new_tokens,
            "priority": request.priority,
            "deadline_us": request
                .deadline_micros
                .map(|deadline| deadline.saturating_sub(request.arrival_micros)),
        });
        writeln!(file, "{line}")?;
    }
    eprintln!(
        "{} requests · mean interarrival {} µs (load scale {scale:.3})",
        trace.requests.len(),
        parameters.mean_interarrival_micros
    );
    Ok(())
}

fn simulate(args: &[String]) -> Result<(), Box<dyn Error>> {
    let [observed, max_active, kv, out] = args else {
        unreachable!("argument count checked by the caller");
    };
    let observed = BufReader::new(File::open(observed)?)
        .lines()
        .map(|line| Ok(serde_json::from_str::<Value>(&line?)?))
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    let first_queued = observed
        .iter()
        .filter_map(|row| row["queued_us"].as_u64())
        .min()
        .ok_or("no observed request has a server queue time")?;
    let field = |row: &Value, name: &str| {
        row[name]
            .as_u64()
            .ok_or_else(|| format!("observed request lacks `{name}`: {row}"))
    };
    let mut requests = Vec::with_capacity(observed.len());
    for row in &observed {
        let max_new_tokens = usize::try_from(field(row, "max_new_tokens")?)?;
        let generated = usize::try_from(field(row, "generated_tokens")?)?;
        // An early end-of-sequence is the last generated token.
        let eos_after = (row["finish_reason"] == "end_of_sequence" && generated < max_new_tokens)
            .then_some(generated);
        requests.push(TraceRequest {
            arrival_micros: field(row, "queued_us")? - first_queued,
            prompt_tokens: usize::try_from(field(row, "prompt_tokens")?)?,
            max_new_tokens,
            priority: i32::try_from(row["priority"].as_i64().unwrap_or(0))?,
            deadline_micros: None,
            eos_after,
            shared_prefix: None,
        });
    }
    // Ties keep the server's queue order.
    let mut order = (0..requests.len()).collect::<Vec<_>>();
    order.sort_by_key(|&index| requests[index].arrival_micros);
    let trace = SchedulingTrace {
        name: "replay",
        parameters: TraceParameters {
            seed: 0,
            requests: requests.len(),
            mean_interarrival_micros: 0,
        },
        requests: order.iter().map(|&index| requests[index].clone()).collect(),
    };
    let [bytes_per_token, tokens_per_block, capacity_bytes] = numbers(kv)?[..] else {
        return Err("kv=bytes_per_token,tokens_per_block,capacity_bytes".into());
    };
    let config = SimulationConfig {
        max_active_requests: max_active.parse()?,
        prefill_chunk_size: None,
        kv: KvGeometry {
            bytes_per_token,
            tokens_per_block: usize::try_from(tokens_per_block)?,
            capacity_bytes,
        },
        cost: cost_model()?,
        ..SimulationConfig::default()
    };
    let policy = ComposedPolicy::default().with_admission(MemoryAwareAdmission);
    let run = run_observed_trial(&trace, policy, &config, &[], &mut |_, _| {})?;

    let mut file = BufWriter::new(File::create(out)?);
    for request in &run.requests {
        let source = &observed[order[request.trace_index]];
        let since_arrival =
            |at: Option<u64>| at.map(|at| at.saturating_sub(request.arrival_micros));
        let line = json!({
            "index": source["index"],
            "outcome": match request.outcome {
                TrialOutcome::Completed => "completed",
                TrialOutcome::Rejected => "rejected",
                TrialOutcome::Failed => "failed",
                TrialOutcome::Cancelled => "cancelled",
                TrialOutcome::Unfinished => "unfinished",
            },
            "arrival_us": request.arrival_micros,
            "generated_tokens": request.output_tokens.len(),
            "sim_ttft_us": since_arrival(request.first_token_micros),
            "sim_completion_us": since_arrival(request.completed_micros),
        });
        writeln!(file, "{line}")?;
    }
    eprintln!(
        "simulated {} requests · makespan {} µs",
        run.requests.len(),
        run.makespan_micros
    );
    Ok(())
}
