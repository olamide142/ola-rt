//! Standalone OlaRT terminal workbench using the local in-process adapter.

use std::{env, io, path::PathBuf};

use ola_ui::{LocalBackend, LocalDevice};

fn main() -> io::Result<()> {
    match parse_launch()? {
        Launch::Run {
            model_directory,
            device,
        } => ola_ui::run(LocalBackend::with_device(model_directory, device)),
        Launch::Help => {
            println!(
                "usage: ola-ui [--model <artifact-directory>] [--backend cpu|wgpu|cuda]\n\nOLA_MODEL_DIR and OLA_BACKEND may provide the same values. WGPU/CUDA require matching compile features."
            );
            Ok(())
        }
    }
}

enum Launch {
    Run {
        model_directory: Option<PathBuf>,
        device: LocalDevice,
    },
    Help,
}

fn parse_launch() -> io::Result<Launch> {
    let mut arguments = env::args_os().skip(1);
    let mut model_directory = env::var_os("OLA_MODEL_DIR").map(PathBuf::from);
    let mut backend = env::var("OLA_BACKEND").unwrap_or_else(|_| "cpu".into());
    while let Some(argument) = arguments.next() {
        if argument == "--help" || argument == "-h" {
            return Ok(Launch::Help);
        } else if argument == "--model" {
            model_directory = Some(PathBuf::from(arguments.next().ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "--model requires a directory")
            })?));
        } else if argument == "--backend" {
            backend = arguments
                .next()
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "--backend requires a value")
                })?
                .into_string()
                .map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidInput, "backend must be valid Unicode")
                })?;
        } else if argument.to_string_lossy().starts_with('-') || model_directory.is_some() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("unexpected argument `{}`", argument.to_string_lossy()),
            ));
        } else {
            model_directory = Some(PathBuf::from(argument));
        }
    }
    let device = match backend.as_str() {
        "cpu" => LocalDevice::Cpu,
        "wgpu" => LocalDevice::Wgpu,
        "cuda" => LocalDevice::Cuda,
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("unknown backend `{backend}`; expected cpu, wgpu, or cuda"),
            ));
        }
    };
    Ok(Launch::Run {
        model_directory,
        device,
    })
}
