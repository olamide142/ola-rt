//! Coordinator-side link to one TCP worker.
//!
//! A link owns the write half of the connection. A dedicated task owns the
//! read half and forwards every frame, tagged with the worker's ID, into a
//! channel shared by all links. Reading therefore never happens inside a
//! caller's future, so dropping that future (a client disconnecting, a
//! `select!` choosing another branch) can never abandon a half-read frame and
//! desynchronize the length-prefixed stream.

use std::{error::Error, fmt};

use ola_core::WorkerId;
use ola_net::{
    BinaryFrame, TcpPeer, WireBody, WireError, WireMessage, read_attached, write_attached,
    write_message,
};
use tokio::{net::tcp::OwnedWriteHalf, sync::mpsc, task::JoinHandle};

use crate::{WorkerCommand, WorkerMessage, WorkerSummary};

/// A message read from one worker with any binary frame attached to it, or
/// the error that ended that worker's reader.
pub type LinkFrame = (
    WorkerId,
    Result<(WireMessage, Option<BinaryFrame>), WireError>,
);

/// Errors raised while connecting to or communicating with one worker.
#[derive(Debug)]
pub enum CoordinatorLinkError {
    /// The wire connection failed or carried an invalid payload.
    Wire(WireError),
    /// The worker announced a different identity than the caller expected.
    UnexpectedWorker {
        /// Identity requested by the caller.
        expected: WorkerId,
        /// Identity announced by the peer.
        received: WorkerId,
    },
    /// The peer sent a message that is not valid in the coordinator link.
    UnexpectedMessage(&'static str),
}

/// An event received from a worker link after transport-level heartbeat
/// acknowledgements have been handled.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CoordinatorLinkEvent {
    /// A domain message from the worker.
    Message(WorkerMessage),
    /// A worker liveness report.
    Heartbeat {
        /// Worker process epoch.
        epoch: u64,
        /// Timestamp in the worker's clock domain.
        sent_at_us: u64,
    },
}

impl fmt::Display for CoordinatorLinkError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Wire(error) => error.fmt(formatter),
            Self::UnexpectedWorker { expected, received } => write!(
                formatter,
                "expected worker `{expected}`, connected worker is `{received}`"
            ),
            Self::UnexpectedMessage(kind) => {
                write!(formatter, "unexpected coordinator-link message `{kind}`")
            }
        }
    }
}

impl Error for CoordinatorLinkError {}

impl From<WireError> for CoordinatorLinkError {
    fn from(error: WireError) -> Self {
        Self::Wire(error)
    }
}

/// One coordinator connection to one worker.
pub struct CoordinatorWorkerLink {
    writer: OwnedWriteHalf,
    worker: WorkerId,
    next_correlation: u64,
    reader: JoinHandle<()>,
}

impl CoordinatorWorkerLink {
    /// Connects, completes the worker hello/summary handshake, and starts
    /// forwarding the worker's frames into `frames`.
    ///
    /// # Errors
    ///
    /// Fails if the connection or handshake fails, or the worker is not
    /// `expected_worker`.
    pub async fn connect(
        address: impl tokio::net::ToSocketAddrs,
        expected_worker: Option<WorkerId>,
        frames: mpsc::UnboundedSender<LinkFrame>,
    ) -> Result<(Self, WorkerSummary), CoordinatorLinkError> {
        let mut peer = TcpPeer::connect(address).await?;
        // The handshake reads inline: nothing else can be waiting on this
        // connection yet, and a failed handshake drops it.
        let hello = peer.receive().await?;
        let worker = match hello.body {
            WireBody::Hello { worker } => worker,
            other => return Err(CoordinatorLinkError::UnexpectedMessage(other.kind_name())),
        };
        if let Some(expected) = expected_worker
            && expected != worker
        {
            return Err(CoordinatorLinkError::UnexpectedWorker {
                expected,
                received: worker,
            });
        }
        let summary = peer.receive().await?;
        let summary: WorkerMessage = summary.body.decode_json()?;
        let WorkerMessage::Summary(summary) = summary else {
            return Err(CoordinatorLinkError::UnexpectedMessage("worker_message"));
        };
        if summary.worker != worker {
            return Err(CoordinatorLinkError::UnexpectedWorker {
                expected: worker,
                received: summary.worker,
            });
        }
        let (mut reader, writer) = peer.into_split();
        let reader_worker = worker.clone();
        let reader = tokio::spawn(async move {
            loop {
                let frame = read_attached(&mut reader).await;
                let done = frame.is_err();
                if frames.send((reader_worker.clone(), frame)).is_err() || done {
                    return;
                }
            }
        });
        Ok((
            Self {
                writer,
                worker,
                next_correlation: 1,
                reader,
            },
            summary,
        ))
    }

    /// The worker identity established during the handshake.
    pub const fn worker(&self) -> &WorkerId {
        &self.worker
    }

    /// Sends a coordinator command to the worker.
    ///
    /// # Errors
    ///
    /// Fails if the command cannot be encoded or written.
    ///
    /// A migrated request's session snapshot travels as an attached binary
    /// frame rather than inside the JSON.
    pub async fn send_command(
        &mut self,
        command: &WorkerCommand,
    ) -> Result<(), CoordinatorLinkError> {
        if let WorkerCommand::AdmitMigrated {
            request,
            state,
            arrival_at,
        } = command
            && state.session_state.is_some()
        {
            let mut state = state.clone();
            if let Some(frame) = crate::transfer::detach_session(&mut state, 0, request.get()) {
                let body = WireBody::json_payload(&WorkerCommand::AdmitMigrated {
                    request: *request,
                    state,
                    arrival_at: *arrival_at,
                })?;
                let message = WireMessage::new(0, self.next_correlation(), body);
                write_attached(&mut self.writer, &message, &frame).await?;
                return Ok(());
            }
        }
        let body = WireBody::json_payload(command)?;
        self.send(body).await
    }

    /// Interprets one frame from this worker, acknowledging heartbeats.
    ///
    /// # Errors
    ///
    /// Fails if the frame is an error, a goodbye, or not valid on a
    /// coordinator link.
    pub async fn accept_frame(
        &mut self,
        frame: Result<(WireMessage, Option<BinaryFrame>), WireError>,
    ) -> Result<CoordinatorLinkEvent, CoordinatorLinkError> {
        let (message, attachment) = frame?;
        match message.body {
            WireBody::Payload { .. } => {
                let mut decoded: WorkerMessage = message.body.decode_json()?;
                match (&mut decoded, attachment) {
                    (WorkerMessage::Migrated { state, .. }, Some(frame)) => {
                        crate::transfer::attach_session(state, &frame);
                    }
                    (_, Some(_)) => {
                        return Err(CoordinatorLinkError::UnexpectedMessage("attachment"));
                    }
                    (_, None) => {}
                }
                Ok(CoordinatorLinkEvent::Message(decoded))
            }
            WireBody::Heartbeat { sent_at_us } => {
                self.send(WireBody::HeartbeatAck {
                    echo_sent_at_us: sent_at_us,
                })
                .await?;
                Ok(CoordinatorLinkEvent::Heartbeat {
                    epoch: message.sender_epoch,
                    sent_at_us,
                })
            }
            WireBody::Goodbye { .. } => Err(CoordinatorLinkError::Wire(WireError::Truncated)),
            other => Err(CoordinatorLinkError::UnexpectedMessage(other.kind_name())),
        }
    }

    /// Asks the worker to close the connection gracefully.
    ///
    /// # Errors
    ///
    /// Fails if the goodbye cannot be written.
    pub async fn close(&mut self) -> Result<(), CoordinatorLinkError> {
        self.send(WireBody::Goodbye {
            reason: Some("coordinator shutdown".into()),
        })
        .await
    }

    async fn send(&mut self, body: WireBody) -> Result<(), CoordinatorLinkError> {
        let correlation = self.next_correlation();
        write_message(&mut self.writer, &WireMessage::new(0, correlation, body)).await?;
        Ok(())
    }

    fn next_correlation(&mut self) -> u64 {
        let correlation = self.next_correlation;
        self.next_correlation = self.next_correlation.saturating_add(1);
        correlation
    }
}

impl Drop for CoordinatorWorkerLink {
    fn drop(&mut self) {
        self.reader.abort();
    }
}
