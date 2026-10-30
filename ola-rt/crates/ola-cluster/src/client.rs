//! Coordinator-side link to one TCP worker.

use std::{error::Error, fmt};

use ola_core::WorkerId;
use ola_net::{TcpPeer, WireBody, WireError, WireMessage};

use crate::{WorkerCommand, WorkerMessage, WorkerSummary};

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
    peer: TcpPeer,
    worker: WorkerId,
    next_correlation: u64,
}

impl CoordinatorWorkerLink {
    /// Connects and completes the worker hello/summary handshake.
    pub async fn connect(
        address: impl tokio::net::ToSocketAddrs,
        expected_worker: Option<WorkerId>,
    ) -> Result<(Self, WorkerSummary), CoordinatorLinkError> {
        let mut peer = TcpPeer::connect(address).await?;
        let hello = peer.receive().await?;
        let worker = match hello.body {
            WireBody::Hello { worker } => worker,
            other => return Err(CoordinatorLinkError::UnexpectedMessage(other.kind_name_for_error())),
        };
        if let Some(expected) = expected_worker {
            if expected != worker {
                return Err(CoordinatorLinkError::UnexpectedWorker {
                    expected,
                    received: worker,
                });
            }
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
        Ok((
            Self {
                peer,
                worker,
                next_correlation: 1,
            },
            summary,
        ))
    }

    /// The worker identity established during the handshake.
    pub const fn worker(&self) -> &WorkerId {
        &self.worker
    }

    /// Sends a coordinator command to the worker.
    pub async fn send_command(&mut self, command: &WorkerCommand) -> Result<(), CoordinatorLinkError> {
        let correlation = self.next_correlation;
        self.next_correlation = self.next_correlation.saturating_add(1);
        self.peer
            .send(&WireMessage::new(
                0,
                correlation,
                WireBody::json_payload(command)?,
            ))
            .await?;
        Ok(())
    }

    /// Receives the next worker message, acknowledging worker heartbeats.
    pub async fn receive(&mut self) -> Result<WorkerMessage, CoordinatorLinkError> {
        loop {
            match self.receive_event().await? {
                CoordinatorLinkEvent::Message(message) => return Ok(message),
                CoordinatorLinkEvent::Heartbeat { .. } => {}
            }
        }
    }

    /// Receives the next worker event and acknowledges heartbeats.
    pub async fn receive_event(&mut self) -> Result<CoordinatorLinkEvent, CoordinatorLinkError> {
        let message = self.peer.receive().await?;
        match message.body {
            WireBody::Payload { .. } => {
                Ok(CoordinatorLinkEvent::Message(message.body.decode_json()?))
            }
            WireBody::Heartbeat { sent_at_us } => {
                self.peer
                    .send(&WireMessage::new(
                        0,
                        message.correlation,
                        WireBody::HeartbeatAck {
                            received_at_us: sent_at_us,
                        },
                    ))
                    .await?;
                Ok(CoordinatorLinkEvent::Heartbeat {
                    epoch: message.sender_epoch,
                    sent_at_us,
                })
            }
            WireBody::Goodbye { .. } => Err(CoordinatorLinkError::Wire(WireError::Truncated)),
            other => Err(CoordinatorLinkError::UnexpectedMessage(
                other.kind_name_for_error(),
            )),
        }
    }

    /// Asks the worker to close the connection gracefully.
    pub async fn close(&mut self) -> Result<(), CoordinatorLinkError> {
        self.peer
            .send(&WireMessage::new(
                0,
                self.next_correlation,
                WireBody::Goodbye {
                    reason: Some("coordinator shutdown".into()),
                },
            ))
            .await?;
        self.next_correlation = self.next_correlation.saturating_add(1);
        Ok(())
    }
}

trait WireBodyKind {
    fn kind_name_for_error(&self) -> &'static str;
}

impl WireBodyKind for WireBody {
    fn kind_name_for_error(&self) -> &'static str {
        match self {
            Self::Hello { .. } => "hello",
            Self::Heartbeat { .. } => "heartbeat",
            Self::HeartbeatAck { .. } => "heartbeat_ack",
            Self::Payload { .. } => "payload",
            Self::Goodbye { .. } => "goodbye",
        }
    }
}
