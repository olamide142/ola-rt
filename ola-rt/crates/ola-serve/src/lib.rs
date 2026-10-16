#![doc = include_str!("../README.md")]

mod driver;
mod error;
pub mod http;

pub use driver::{
    Driver, DriverConfig, GenerationOutput, Health, ModelStatus, Reply, RequestEvent,
    RequestHandle, RequestSnapshot, RuntimeHandle, ServiceState, Submission,
};
pub use error::ServeError;
