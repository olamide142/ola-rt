//! Captures build identity for experiment reports.

use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=OLA_REVISION");
    // Rebuild when the checked-out commit changes. The path is relative to
    // this crate; a missing repository simply yields `unknown`.
    println!("cargo:rerun-if-changed=../../../.git/HEAD");
    println!("cargo:rerun-if-changed=../../../.git/refs/heads");
    let revision = std::env::var("OLA_REVISION").ok().or_else(git_revision);
    println!(
        "cargo:rustc-env=OLA_BUILD_REVISION={}",
        revision.as_deref().unwrap_or("unknown")
    );
    let features = std::env::vars()
        .filter_map(|(key, _)| key.strip_prefix("CARGO_FEATURE_").map(str::to_lowercase))
        .collect::<Vec<_>>()
        .join(",");
    println!("cargo:rustc-env=OLA_BUILD_FEATURES={features}");
    for key in ["PROFILE", "TARGET"] {
        let value = std::env::var(key).unwrap_or_else(|_| "unknown".into());
        println!("cargo:rustc-env=OLA_BUILD_{key}={value}");
    }
}

fn git_revision() -> Option<String> {
    let output = Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .filter(|revision| !revision.is_empty())
}
