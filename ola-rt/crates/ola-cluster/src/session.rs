//! TCP worker-side protocol session.

use std::{error::Error, fmt, time::Duration};

use ola_core::{Clock, MonotonicClock};
use ola_net::{read_message, write_message, TcpPeer, WireBody, WireError, WireMessage};
use ola_runtime::ExecutorError;
use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};
use tokio::sync::mpsc;

use crate::{WorkerCommand, WorkerHost, WorkerMessage};

/// Errors raised while serving one worker connection.
#[derive(Debug)]
pub enum WorkerSessionError {
    /// The wire connection failed or carried an invalid payload.
    Wire(WireError),
    /// The local executor rejected a command or iteration.
    Executor(ExecutorError),
}

impl fmt::Display for WorkerSessionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Wire(error) => error.fmt(formatter),
            Self::Executor(error) => write!(formatter, "worker executor error: {error}"),
        }
    }
}

impl Error for WorkerSessionError {}

impl From<WireError> for WorkerSessionError {
    fn from(error: WireError) -> Self {
        Self::Wire(error)
    }
}

impl From<ExecutorError> for WorkerSessionError {
    fn from(error: ExecutorError) -> Self {
        Self::Executor(error)
    }
}

/// Runs one worker's executor behind a TCP control-plane connection.
pub struct WorkerSession {
    peer: TcpPeer,
    worker: WorkerHost,
    epoch: u64,
    heartbeat_interval: Duration,
    next_correlation: u64,
    clock: MonotonicClock,
}

impl WorkerSession {
    /// Creates a session with a worker-process epoch and heartbeat cadence.
    pub fn new(
        peer: TcpPeer,
        worker: WorkerHost,
        epoch: u64,
        heartbeat_interval: Duration,
    ) -> Self {
        Self {
            peer,
            worker,
            epoch,
            heartbeat_interval,
            next_correlation: 1,
            clock: MonotonicClock::start(),
        }
    }

    /// Runs until the coordinator sends `Goodbye` or closes the connection.
    pub async fn run(mut self) -> Result<WorkerHost, WorkerSessionError> {
        let worker_id = self.worker.id().clone();
        let (reader, mut writer) = self.peer.into_split();
        let (received_tx, mut received_rx) = mpsc::channel(8);
        let reader_task = tokio::spawn(read_frames(reader, received_tx));

        send_body(
            &mut writer,
            self.epoch,
            &mut self.next_correlation,
            WireBody::Hello { worker: worker_id },
        )
        .await?;
        let summary = summary_body(&self.worker, &self.clock)?;
        send_body(
            &mut writer,
            self.epoch,
            &mut self.next_correlation,
            summary,
        )
        .await?;

        let mut heartbeat = tokio::time::interval(self.heartbeat_interval);
        heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                received = received_rx.recv() => {
                    let Some(received) = received else { break };
                    let message = received?;
                    let now = self.clock.now();
                    match message.body {
                        WireBody::Payload { .. } => {
                            let command: WorkerCommand = message.body.decode_json()?;
                            let messages = self.worker.handle(command, now)?;
                            send_messages(&mut writer, self.epoch, &mut self.next_correlation, messages).await?;
                        }
                        WireBody::Heartbeat { sent_at_us } => {
                            send_body(
                                &mut writer,
                                self.epoch,
                                &mut self.next_correlation,
                                WireBody::HeartbeatAck { received_at_us: sent_at_us },
                            ).await?;
                        }
                        WireBody::Goodbye { .. } => break,
                        WireBody::HeartbeatAck { .. } | WireBody::Hello { .. } => {}
                    }
                }
                _ = heartbeat.tick() => {
                    let now = self.clock.now();
                    send_body(
                        &mut writer,
                        self.epoch,
                        &mut self.next_correlation,
                        WireBody::Heartbeat { sent_at_us: now.as_micros() },
                    ).await?;
                    let summary = summary_body(&self.worker, &self.clock)?;
                    send_body(
                        &mut writer,
                        self.epoch,
                        &mut self.next_correlation,
                        summary,
                    ).await?;
                    if let Some((_, messages)) = self.worker.step(now)? {
                        send_messages(&mut writer, self.epoch, &mut self.next_correlation, messages).await?;
                    }
                }
            }
        }
        reader_task.abort();
        Ok(self.worker)
    }
}

async fn read_frames(
    mut reader: OwnedReadHalf,
    sender: mpsc::Sender<Result<WireMessage, WireError>>,
) {
    loop {
        let result = read_message(&mut reader).await;
        let done = result.is_err() || sender.send(result).await.is_err();
        if done {
            return;
        }
    }
}

async fn send_body(
    writer: &mut OwnedWriteHalf,
    epoch: u64,
    next_correlation: &mut u64,
    body: WireBody,
) -> Result<(), WorkerSessionError> {
    let correlation = *next_correlation;
    *next_correlation = next_correlation.saturating_add(1);
    write_message(writer, &WireMessage::new(epoch, correlation, body)).await?;
    Ok(())
}

fn summary_body(
    worker: &WorkerHost,
    clock: &MonotonicClock,
) -> Result<WireBody, WorkerSessionError> {
    Ok(WireBody::json_payload(&WorkerMessage::Summary(worker.summary(clock.now())))?)
}

async fn send_messages(
    writer: &mut OwnedWriteHalf,
    epoch: u64,
    next_correlation: &mut u64,
    messages: Vec<WorkerMessage>,
) -> Result<(), WorkerSessionError> {
    for message in messages {
        send_body(
            writer,
            epoch,
            next_correlation,
            WireBody::json_payload(&message)?,
        )
        .await?;
    }
    Ok(())
}
