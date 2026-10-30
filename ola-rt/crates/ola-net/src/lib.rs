//! Versioned, length-prefixed control-plane transport primitives.
//!
//! The protocol is deliberately independent of sockets: [`WireMessage`] is
//! serialized by [`write_message`] and decoded by [`read_message`], so the
//! same envelope can be used by TCP, deterministic in-process transports, and
//! the virtual-time simulator.

use std::{error::Error, fmt};

use ola_core::WorkerId;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

/// Current control-plane protocol version.
pub const PROTOCOL_VERSION: u16 = 1;

/// Maximum encoded control-plane message size.
pub const MAX_MESSAGE_BYTES: usize = 16 * 1024 * 1024;

/// A versioned control-plane envelope.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct WireMessage {
    /// Protocol version used to encode the payload.
    pub version: u16,
    /// Sender's process epoch; changes whenever a worker restarts.
    pub sender_epoch: u64,
    /// Correlates requests and acknowledgements.
    pub correlation: u64,
    /// Message body.
    pub body: WireBody,
}

impl WireMessage {
    /// Creates a message using the current protocol version.
    pub fn new(sender_epoch: u64, correlation: u64, body: WireBody) -> Self {
        Self {
            version: PROTOCOL_VERSION,
            sender_epoch,
            correlation,
            body,
        }
    }
}

/// Control-plane messages exchanged by gateway, coordinator, and workers.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WireBody {
    /// Worker announces itself and the address-independent identity used by
    /// the coordinator.
    Hello {
        /// Stable worker identity.
        worker: WorkerId,
    },
    /// Liveness report. `sent_at_us` belongs to the sender's clock domain.
    Heartbeat {
        /// Sender-clock timestamp when the heartbeat was emitted.
        sent_at_us: u64,
    },
    /// Reply to a heartbeat, carrying the receiver's clock reading.
    HeartbeatAck {
        /// Receiver-clock timestamp when the heartbeat was received.
        received_at_us: u64,
    },
    /// Opaque domain payload. Cluster commands/messages are kept out of the
    /// wire crate until their serde representation is part of the stable
    /// protocol.
    Payload {
        /// Domain-level payload media type.
        content_type: String,
        /// Encoded domain payload.
        data: Vec<u8>,
    },
    /// Graceful connection shutdown.
    Goodbye {
        /// Optional human-readable shutdown reason.
        reason: Option<String>,
    },
}

impl WireBody {
    /// Encodes a serializable domain value as a JSON payload.
    pub fn json_payload<T: Serialize>(value: &T) -> Result<Self, WireError> {
        Ok(Self::Payload {
            content_type: "application/json".into(),
            data: serde_json::to_vec(value)?,
        })
    }

    /// Decodes a JSON payload into its domain value.
    pub fn decode_json<T: DeserializeOwned>(&self) -> Result<T, WireError> {
        match self {
            Self::Payload {
                content_type,
                data,
            } if content_type == "application/json" => Ok(serde_json::from_slice(data)?),
            Self::Payload { content_type, .. } => Err(WireError::UnexpectedContentType(
                content_type.clone(),
            )),
            other => Err(WireError::ExpectedPayload(other.kind_name())),
        }
    }

    const fn kind_name(&self) -> &'static str {
        match self {
            Self::Hello { .. } => "hello",
            Self::Heartbeat { .. } => "heartbeat",
            Self::HeartbeatAck { .. } => "heartbeat_ack",
            Self::Payload { .. } => "payload",
            Self::Goodbye { .. } => "goodbye",
        }
    }
}

/// Errors raised while framing or decoding a message.
#[derive(Debug)]
pub enum WireError {
    /// The four-byte length prefix was invalid or exceeded the configured
    /// maximum.
    InvalidLength(u32),
    /// The peer closed the stream before a complete frame arrived.
    Truncated,
    /// JSON encoding or decoding failed.
    Json(serde_json::Error),
    /// A typed payload was expected but another message body was supplied.
    ExpectedPayload(&'static str),
    /// The payload used a content type this helper does not decode.
    UnexpectedContentType(String),
    /// The asynchronous stream failed.
    Io(std::io::Error),
    /// The peer used a protocol version this crate cannot interpret.
    UnsupportedVersion(u16),
}

/// A TCP connection carrying [`WireMessage`] frames.
pub struct TcpPeer {
    stream: TcpStream,
}

impl TcpPeer {
    /// Connects to a TCP transport endpoint.
    pub async fn connect(address: impl tokio::net::ToSocketAddrs) -> Result<Self, WireError> {
        Ok(Self {
            stream: TcpStream::connect(address).await?,
        })
    }

    /// Wraps an accepted TCP stream.
    pub const fn from_stream(stream: TcpStream) -> Self {
        Self { stream }
    }

    /// Accepts one peer from a listener.
    pub async fn accept(listener: &TcpListener) -> Result<(Self, std::net::SocketAddr), WireError> {
        let (stream, address) = listener.accept().await?;
        Ok((Self::from_stream(stream), address))
    }

    /// Sends one framed message and flushes it to the socket.
    pub async fn send(&mut self, message: &WireMessage) -> Result<(), WireError> {
        write_message(&mut self.stream, message).await
    }

    /// Receives one framed message.
    pub async fn receive(&mut self) -> Result<WireMessage, WireError> {
        read_message(&mut self.stream).await
    }

    /// Returns the local socket address.
    pub fn local_addr(&self) -> Result<std::net::SocketAddr, WireError> {
        Ok(self.stream.local_addr()?)
    }

    /// Returns the remote socket address.
    pub fn peer_addr(&self) -> Result<std::net::SocketAddr, WireError> {
        Ok(self.stream.peer_addr()?)
    }
}

impl fmt::Display for WireError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLength(length) => write!(formatter, "invalid wire frame length {length}"),
            Self::Truncated => formatter.write_str("truncated wire frame"),
            Self::Json(error) => write!(formatter, "wire JSON error: {error}"),
            Self::ExpectedPayload(kind) => write!(formatter, "expected payload, got {kind}"),
            Self::UnexpectedContentType(content_type) => {
                write!(formatter, "unexpected wire content type `{content_type}`")
            }
            Self::Io(error) => write!(formatter, "wire I/O error: {error}"),
            Self::UnsupportedVersion(version) => {
                write!(formatter, "unsupported wire protocol version {version}")
            }
        }
    }
}

impl Error for WireError {}

impl From<serde_json::Error> for WireError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

impl From<std::io::Error> for WireError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

/// Writes one big-endian length-prefixed JSON message.
pub async fn write_message<W>(writer: &mut W, message: &WireMessage) -> Result<(), WireError>
where
    W: AsyncWrite + Unpin,
{
    if message.version != PROTOCOL_VERSION {
        return Err(WireError::UnsupportedVersion(message.version));
    }
    let payload = serde_json::to_vec(message)?;
    let length = u32::try_from(payload.len()).map_err(|_| WireError::InvalidLength(u32::MAX))?;
    if payload.len() > MAX_MESSAGE_BYTES {
        return Err(WireError::InvalidLength(length));
    }
    writer.write_u32(length).await?;
    writer.write_all(&payload).await?;
    writer.flush().await?;
    Ok(())
}

/// Reads one big-endian length-prefixed JSON message.
pub async fn read_message<R>(reader: &mut R) -> Result<WireMessage, WireError>
where
    R: AsyncRead + Unpin,
{
    let length = match reader.read_u32().await {
        Ok(length) => length,
        Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => {
            return Err(WireError::Truncated);
        }
        Err(error) => return Err(WireError::Io(error)),
    };
    if usize::try_from(length).unwrap_or(usize::MAX) > MAX_MESSAGE_BYTES {
        return Err(WireError::InvalidLength(length));
    }
    let mut payload = vec![0; length as usize];
    reader.read_exact(&mut payload).await.map_err(|error| {
        if error.kind() == std::io::ErrorKind::UnexpectedEof {
            WireError::Truncated
        } else {
            WireError::Io(error)
        }
    })?;
    let message: WireMessage = serde_json::from_slice(&payload)?;
    if message.version != PROTOCOL_VERSION {
        return Err(WireError::UnsupportedVersion(message.version));
    }
    Ok(message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::duplex;
    use tokio::net::TcpListener;

    #[tokio::test]
    async fn round_trips_a_message_over_a_stream() {
        let (mut left, mut right) = duplex(1024);
        let sent = WireMessage::new(
            7,
            11,
            WireBody::Hello {
                worker: WorkerId::new("worker-a"),
            },
        );
        let expected = sent.clone();
        let write = tokio::spawn(async move { write_message(&mut left, &sent).await });
        let received = read_message(&mut right).await.unwrap();
        write.await.unwrap().unwrap();
        assert_eq!(received, expected);
    }

    #[tokio::test]
    async fn rejects_oversized_frames_before_allocating_payload() {
        let (mut left, mut right) = duplex(16);
        let write = tokio::spawn(async move {
            left.write_u32((MAX_MESSAGE_BYTES as u32) + 1).await.unwrap();
        });
        assert!(matches!(
            read_message(&mut right).await,
            Err(WireError::InvalidLength(_))
        ));
        write.await.unwrap();
    }

    #[tokio::test]
    async fn rejects_an_unknown_version() {
        let (mut left, mut right) = duplex(1024);
        let message = WireMessage {
            version: PROTOCOL_VERSION + 1,
            sender_epoch: 0,
            correlation: 0,
            body: WireBody::Goodbye { reason: None },
        };
        let write = tokio::spawn(async move {
            let payload = serde_json::to_vec(&message).unwrap();
            left.write_u32(payload.len() as u32).await.unwrap();
            left.write_all(&payload).await.unwrap();
        });
        assert!(matches!(
            read_message(&mut right).await,
            Err(WireError::UnsupportedVersion(version)) if version == PROTOCOL_VERSION + 1
        ));
        write.await.unwrap();
    }

    #[tokio::test]
    async fn exchanges_messages_over_a_loopback_tcp_peer() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut peer, _) = TcpPeer::accept(&listener).await.unwrap();
            let message = peer.receive().await.unwrap();
            assert_eq!(message.correlation, 41);
            peer.send(&WireMessage::new(
                9,
                message.correlation,
                WireBody::HeartbeatAck { received_at_us: 123 },
            ))
            .await
            .unwrap();
        });

        let mut client = TcpPeer::connect(address).await.unwrap();
        client
            .send(&WireMessage::new(
                8,
                41,
                WireBody::Heartbeat { sent_at_us: 100 },
            ))
            .await
            .unwrap();
        let reply = client.receive().await.unwrap();
        assert_eq!(
            reply.body,
            WireBody::HeartbeatAck { received_at_us: 123 }
        );
        server.await.unwrap();
    }

    #[test]
    fn encodes_and_decodes_typed_json_payloads() {
        #[derive(Debug, Deserialize, Eq, PartialEq, Serialize)]
        struct Command {
            sequence: u64,
        }

        let body = WireBody::json_payload(&Command { sequence: 3 }).unwrap();
        assert_eq!(body.decode_json::<Command>().unwrap(), Command { sequence: 3 });
        assert!(matches!(
            WireBody::Heartbeat { sent_at_us: 1 }.decode_json::<Command>(),
            Err(WireError::ExpectedPayload("heartbeat"))
        ));
    }
}
