#![doc = include_str!("../README.md")]

mod driver;
mod error;

pub use driver::{
    Driver, DriverConfig, GenerationOutput, Reply, RequestEvent, RequestHandle, RequestSnapshot,
    RuntimeHandle, Submission,
};
pub use error::ServeError;
