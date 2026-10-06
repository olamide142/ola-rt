//! Standalone OlaRT terminal workbench using the local in-process adapter.

use std::{env, io, path::PathBuf};

use ola_ui::LocalBackend;

fn main() -> io::Result<()> {
    match parse_launch()? {
        Launch::Run(model_directory) => ola_ui::run(LocalBackend::new(model_directory)),
        Launch::Help => {
            println!(
                "usage: ola-ui [--model <artifact-directory>]\n\nOLA_MODEL_DIR may provide the same path."
            );
            Ok(())
        }
    }
}

enum Launch {
    Run(Option<PathBuf>),
    Help,
}

fn parse_launch() -> io::Result<Launch> {
    let mut arguments = env::args_os().skip(1);
    let argument = arguments.next();
    let model = match argument.as_deref() {
        None => env::var_os("OLA_MODEL_DIR").map(PathBuf::from),
        Some(value) if value == "--help" || value == "-h" => return Ok(Launch::Help),
        Some(value) if value == "--model" => {
            Some(PathBuf::from(arguments.next().ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "--model requires a directory")
            })?))
        }
        Some(value) => Some(PathBuf::from(value)),
    };
    if arguments.next().is_some() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "unexpected extra command-line arguments",
        ));
    }
    Ok(Launch::Run(model))
}
