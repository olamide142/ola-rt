//! Small executable specification of the foundation request lifecycle.

use ola_core::{ModelDescriptor, RequestSpec, Timestamp};
use ola_runtime::{RequestStatus, Runtime};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut runtime = Runtime::default();
    runtime.register_model(ModelDescriptor::language("tiny", "tiny-decoder"))?;

    let request = runtime.submit(
        RequestSpec::text("tiny", "Explain continuous batching."),
        Timestamp::from_micros(10),
    )?;

    let session = runtime
        .start_queue_head(Timestamp::from_micros(20))?
        .expect("the submitted request should be queued");
    runtime.mark_first_token(session.request_id, Timestamp::from_micros(30))?;
    runtime.complete(session.request_id, Timestamp::from_micros(100))?;

    let record = runtime
        .state()
        .request(request)
        .expect("completed requests remain observable");
    assert_eq!(record.status, RequestStatus::Completed);

    println!(
        "request={} queue={}us ttft={}us total={}us",
        request,
        session
            .started_at
            .saturating_duration_since(record.timing.queued_at)
            .as_micros(),
        record
            .timing
            .first_token_at
            .expect("first token was recorded")
            .saturating_duration_since(record.timing.queued_at)
            .as_micros(),
        record
            .timing
            .completed_at
            .expect("completion was recorded")
            .saturating_duration_since(record.timing.queued_at)
            .as_micros(),
    );

    Ok(())
}
