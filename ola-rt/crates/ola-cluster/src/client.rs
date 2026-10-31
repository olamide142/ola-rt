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
use ola_net::{TcpPeer, WireBody, WireError, WireMessage, read_message, write_message};
use tokio::{net::tcp::OwnedWriteHalf, sync::mpsc, task::JoinHandle};

use crate::{WorkerCommand, WorkerMessage, WorkerSummary};

/// A frame read from one worker, or the error that ended its reader.
pub type LinkFrame = (WorkerId, Result<WireMessage, WireError>);

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
                let frame = read_message(&mut reader).await;
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
    pub async fn send_command(
        &mut self,
        command: &WorkerCommand,
    ) -> Result<(), CoordinatorLinkError> {
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
        frame: Result<WireMessage, WireError>,
    ) -> Result<CoordinatorLinkEvent, CoordinatorLinkError> {
        let message = frame?;
        match message.body {
            WireBody::Payload { .. } => {
                Ok(CoordinatorLinkEvent::Message(message.body.decode_json()?))
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
        let correlation = self.next_correlation;
        self.next_correlation = self.next_correlation.saturating_add(1);
        write_message(&mut self.writer, &WireMessage::new(0, correlation, body)).await?;
        Ok(())
    }
}

impl Drop for CoordinatorWorkerLink {
    fn drop(&mut self) {
        self.reader.abort();
    }
}
