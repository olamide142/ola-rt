//! `ola serve`: load a model and serve it over HTTP until Ctrl-C.

use std::{
    fs::OpenOptions,
    io::{BufWriter, Write},
    path::PathBuf,
    sync::{Arc, mpsc},
    thread,
    time::Duration,
};

use ola_core::{ModelDescriptor, MonotonicClock};
use ola_observe::DeviceTelemetry;
use ola_runtime::{ComposedPolicy, Executor, MemoryAwareAdmission, Runtime};
use ola_serve::{Driver, DriverConfig, Instruments, http};
use ola_serve::cluster::ClusterGateway;

use crate::{EXIT_OK, USAGE, load_model, write_line};

struct Options {
    role: String,
    model: PathBuf,
    backend: String,
    name: Option<String>,
    address: String,
    max_active: usize,
    max_in_flight: usize,
    drain: Duration,
    trace: Option<PathBuf>,
    workers: Vec<String>,
}

fn parse(rest: &[&str]) -> Result<Options, String> {
    let mut model = None;
    let mut options = Options {
        role: "standalone".into(),
        model: PathBuf::new(),
        backend: "cpu".into(),
        name: None,
        address: "127.0.0.1:8080".into(),
        max_active: 4,
        max_in_flight: DriverConfig::default().max_in_flight,
        drain: Duration::from_secs(30),
        trace: None,
        workers: Vec::new(),
    };
    let mut iter = rest.iter();
    while let Some(flag) = iter.next() {
        let mut value = || {
            iter.next()
                .map(|value| (*value).to_owned())
                .ok_or_else(|| format!("{flag} requires a value"))
        };
        let number = |text: String| {
            text.parse::<usize>()
                .ok()
                .filter(|value| *value > 0)
                .ok_or_else(|| format!("{flag} expects a positive integer, got `{text}`"))
        };
        match *flag {
            "--role" => options.role = value()?,
            "--model" => model = Some(PathBuf::from(value()?)),
            "--backend" => options.backend = value()?,
            "--name" => options.name = Some(value()?),
            "--addr" => options.address = value()?,
            "--max-active" => options.max_active = number(value()?)?,
            "--max-in-flight" => options.max_in_flight = number(value()?)?,
            "--trace" => options.trace = Some(PathBuf::from(value()?)),
            "--worker" => options.workers.push(value()?),
            "--drain-secs" => {
                let text = value()?;
                let seconds = text
                    .parse::<u64>()
                    .map_err(|_| format!("--drain-secs expects seconds, got `{text}`"))?;
                options.drain = Duration::from_secs(seconds);
            }
            other => return Err(format!("unknown option `{other}`\n\n{USAGE}")),
        }
    }
    if options.role != "gateway" {
        options.model = model.ok_or_else(|| format!("serve needs --model DIR\n\n{USAGE}"))?;
    } else if options.workers.is_empty() {
        return Err(format!("gateway serve needs at least one --worker HOST:PORT\n\n{USAGE}"));
    }
    Ok(options)
}

/// Device memory and utilization for `/health` on accelerator backends;
/// the CPU backend reports none.
fn telemetry(backend: &str) -> Option<Arc<dyn DeviceTelemetry>> {
    match backend {
        #[cfg(feature = "wgpu")]
        "wgpu" => Some(Arc::new(ola_kernel::wgpu_telemetry())),
        #[cfg(feature = "cuda")]
        "cuda" => ola_kernel::cuda_telemetry(0)
            .map(|telemetry| Arc::new(telemetry) as Arc<dyn DeviceTelemetry>),
        _ => None,
    }
}

/// Serves one model until Ctrl-C, then drains: new work is refused and
/// accepted requests finish for up to `--drain-secs` before the rest end.
pub fn serve(rest: &[&str], out: &mut impl Write) -> Result<u8, String> {
    let options = parse(rest)?;
    if options.role == "gateway" {
        return gateway(options, out);
    }
    if options.role != "standalone" {
        return Err(format!("unsupported serve role `{}`", options.role));
    }
    let model = load_model(&options.model, &options.backend)?;
    let name = options.name.clone().unwrap_or_else(|| {
        options
            .model
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("model")
            .to_owned()
    });
    let mut executor = Executor::with_max_active_requests(Runtime::default(), options.max_active)
        .map_err(|error| error.to_string())?
        .with_policy(ComposedPolicy::default().with_admission(MemoryAwareAdmission));
    executor
        .register_model(ModelDescriptor::language(name.as_str(), "llama"), model)
        .map_err(|error| error.to_string())?;
    // Traces of live and recently finished requests are always kept for
    // `GET /v1/requests/{id}/trace`; `--trace` also writes every record.
    let config = DriverConfig {
        max_in_flight: options.max_in_flight,
        trace: true,
        ..DriverConfig::default()
    };
    let trace_sink = match &options.trace {
        Some(path) => {
            let file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)
                .map_err(|error| format!("could not create trace {}: {error}", path.display()))?;
            Some(Box::new(BufWriter::new(file)) as Box<dyn Write + Send>)
        }
        None => None,
    };
    let driver = Driver::spawn_instrumented(
        executor,
        MonotonicClock::start(),
        config,
        Instruments {
            trace_sink,
            telemetry: telemetry(&options.backend),
        },
    )
    .map_err(|error| format!("could not start the driver: {error}"))?;
    let handle = driver.handle();

    // The driver drains on its own thread once the HTTP side stops taking
    // connections, so open streams end as their requests do.
    let (stop, stopped) = mpsc::channel::<()>();
    let grace = options.drain;
    let drainer = thread::spawn(move || {
        let _ = stopped.recv();
        driver.drain(grace)
    });
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|error| error.to_string())?;
    let served = runtime.block_on(async {
        let listener = tokio::net::TcpListener::bind(&options.address)
            .await
            .map_err(|error| format!("could not bind {}: {error}", options.address))?;
        let address = listener.local_addr().map_err(|error| error.to_string())?;
        write_line(
            out,
            format!(
                "serving `{name}` on http://{address} (Ctrl-C drains for up to {}s)",
                grace.as_secs()
            ),
        )?;
        let _ = out.flush();
        http::serve(listener, handle, async move {
            let _ = tokio::signal::ctrl_c().await;
            let _ = stop.send(());
        })
        .await
        .map_err(|error| error.to_string())
    });
    let stopped = drainer
        .join()
        .map_err(|_| "the driver thread panicked".to_owned())?;
    served?;
    stopped.map_err(|error| error.to_string())?;
    write_line(out, "stopped".into())?;
    Ok(EXIT_OK)
}

fn gateway(options: Options, out: &mut impl Write) -> Result<u8, String> {
    let workers = options
        .workers
        .into_iter()
        .map(|address| (address, None))
        .collect::<Vec<_>>();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|error| error.to_string())?;
    runtime.block_on(async {
        let gateway = ClusterGateway::connect(workers)
            .await
            .map_err(|error| format!("could not connect to workers: {error}"))?;
        let listener = tokio::net::TcpListener::bind(&options.address)
            .await
            .map_err(|error| format!("could not bind {}: {error}", options.address))?;
        let address = listener.local_addr().map_err(|error| error.to_string())?;
        write_line(out, format!("serving cluster gateway on http://{address}"))?;
        let _ = out.flush();
        gateway
            .serve(listener, async {
                let _ = tokio::signal::ctrl_c().await;
            })
            .await
            .map_err(|error| error.to_string())
    })?;
    write_line(out, "stopped".into())?;
    Ok(EXIT_OK)
}
