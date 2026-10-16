//! `ola`: run, list, and compare OlaRT experiments from the command line.

use std::{io, process::ExitCode};

fn main() -> ExitCode {
    let code = ola_cli::run(std::env::args().skip(1), &mut io::stdout().lock());
    ExitCode::from(code)
}
