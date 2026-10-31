//! Session handoff cost selection for migration and disaggregation.

use std::{error::Error, fmt};

use ola_model::SessionState;
use ola_net::{BinaryFrame, PROTOCOL_VERSION};
use serde::{Deserialize, Serialize};

/// Media type used for architecture-owned session/KV frames.
pub const SESSION_STATE_CONTENT_TYPE: &str = "application/vnd.ola.session-state";

#[derive(Deserialize, Serialize)]
struct SessionFrameMetadata {
    format: String,
    cursor: usize,
}

/// Failure while converting a binary session frame into a model session.
#[derive(Debug)]
pub enum SessionFrameError {
    /// The frame is not an OlaRT session frame.
    UnexpectedContentType(String),
    /// The frame uses a protocol version this adapter cannot interpret.
    UnsupportedVersion(u16),
    /// The session metadata is malformed.
    InvalidMetadata(serde_json::Error),
}

impl fmt::Display for SessionFrameError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedContentType(content_type) => {
                write!(
                    formatter,
                    "unexpected session frame content type `{content_type}`"
                )
            }
            Self::UnsupportedVersion(version) => {
                write!(formatter, "unsupported session frame version {version}")
            }
            Self::InvalidMetadata(error) => write!(formatter, "invalid session metadata: {error}"),
        }
    }
}

impl Error for SessionFrameError {}

/// Encodes a model-owned session snapshot as a binary data-plane frame.
pub fn session_state_to_frame(
    state: &SessionState,
    sender_epoch: u64,
    correlation: u64,
) -> Result<BinaryFrame, serde_json::Error> {
    Ok(BinaryFrame {
        version: PROTOCOL_VERSION,
        sender_epoch,
        correlation,
        content_type: SESSION_STATE_CONTENT_TYPE.into(),
        metadata: serde_json::to_vec(&SessionFrameMetadata {
            format: state.format.clone(),
            cursor: state.cursor,
        })?,
        data: state.payload.clone(),
    })
}

/// Decodes and validates a binary data-plane session frame.
pub fn session_state_from_frame(frame: &BinaryFrame) -> Result<SessionState, SessionFrameError> {
    if frame.version != PROTOCOL_VERSION {
        return Err(SessionFrameError::UnsupportedVersion(frame.version));
    }
    if frame.content_type != SESSION_STATE_CONTENT_TYPE {
        return Err(SessionFrameError::UnexpectedContentType(
            frame.content_type.clone(),
        ));
    }
    let metadata = serde_json::from_slice::<SessionFrameMetadata>(&frame.metadata)
        .map_err(SessionFrameError::InvalidMetadata)?;
    Ok(SessionState {
        format: metadata.format,
        cursor: metadata.cursor,
        payload: frame.data.clone(),
    })
}

/// Moves a migrated request's session snapshot out into a binary frame, so
/// the JSON control message stays small. Returns `None`, leaving the state
/// inline, when there is no snapshot or it cannot be framed.
pub(crate) fn detach_session(
    state: &mut ola_runtime::MigratedRequest,
    sender_epoch: u64,
    correlation: u64,
) -> Option<BinaryFrame> {
    let session = state.session_state.as_ref()?;
    let frame = session_state_to_frame(session, sender_epoch, correlation).ok()?;
    state.session_state = None;
    Some(frame)
}

/// Restores a snapshot carried by [`detach_session`]. A frame that does not
/// decode leaves the request without one, so its destination recomputes
/// instead: slower, but the output is the same.
pub(crate) fn attach_session(state: &mut ola_runtime::MigratedRequest, frame: &BinaryFrame) {
    state.session_state = session_state_from_frame(frame).ok();
}

/// The two supported ways to resume a request on another worker.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SessionHandoffMode {
    /// Send an architecture-owned session snapshot over the link.
    Transfer,
    /// Send the portable prompt/output log and recompute the session.
    Recompute,
}

/// Measured or calibrated costs used to choose a handoff mode.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HandoffCostModel {
    /// One-way link latency in microseconds.
    pub latency_us: u64,
    /// Sustained link bandwidth in bytes per microsecond; zero means unknown.
    pub bytes_per_us: u64,
    /// Estimated model time per recomputed token in microseconds.
    pub recompute_token_us: u64,
}

/// Returns the serialized session size, including a small framing allowance.
pub fn session_state_bytes(state: &SessionState) -> u64 {
    state.payload.len() as u64 + state.format.len() as u64 + 32
}

/// Chooses transfer only when its calibrated cost is no greater than
/// recomputing the supplied number of tokens. Unknown bandwidth deliberately
/// selects recompute so migration cannot stall on an uncalibrated link.
pub fn choose_handoff(
    state: Option<&SessionState>,
    recompute_tokens: usize,
    costs: HandoffCostModel,
) -> SessionHandoffMode {
    let Some(state) = state else {
        return SessionHandoffMode::Recompute;
    };
    if costs.bytes_per_us == 0 || costs.recompute_token_us == 0 {
        return SessionHandoffMode::Recompute;
    }
    let bytes = session_state_bytes(state);
    let transfer_us = costs.latency_us.saturating_add(
        bytes
            .saturating_add(costs.bytes_per_us - 1)
            .checked_div(costs.bytes_per_us)
            .unwrap_or(u64::MAX),
    );
    let recompute_us = (recompute_tokens as u64).saturating_mul(costs.recompute_token_us);
    if transfer_us <= recompute_us {
        SessionHandoffMode::Transfer
    } else {
        SessionHandoffMode::Recompute
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(bytes: usize) -> SessionState {
        SessionState {
            format: "test/session".into(),
            cursor: 8,
            payload: vec![0; bytes],
        }
    }

    #[test]
    fn session_state_round_trips_through_binary_frame() {
        let original = state(128);
        let frame = session_state_to_frame(&original, 7, 11).unwrap();
        assert_eq!(frame.sender_epoch, 7);
        assert_eq!(frame.correlation, 11);
        assert_eq!(session_state_from_frame(&frame).unwrap(), original);
    }

    #[test]
    fn session_state_frame_rejects_wrong_media_and_version() {
        let original = state(4);
        let mut frame = session_state_to_frame(&original, 0, 0).unwrap();
        frame.content_type = "application/octet-stream".into();
        assert!(matches!(
            session_state_from_frame(&frame),
            Err(SessionFrameError::UnexpectedContentType(_))
        ));
        frame.content_type = SESSION_STATE_CONTENT_TYPE.into();
        frame.version = PROTOCOL_VERSION + 1;
        assert!(matches!(
            session_state_from_frame(&frame),
            Err(SessionFrameError::UnsupportedVersion(_))
        ));
    }

    #[test]
    fn selects_transfer_when_link_beats_recompute() {
        assert_eq!(
            choose_handoff(
                Some(&state(100)),
                100,
                HandoffCostModel {
                    latency_us: 10,
                    bytes_per_us: 100,
                    recompute_token_us: 10,
                },
            ),
            SessionHandoffMode::Transfer
        );
    }

    #[test]
    fn selects_recompute_without_state_or_calibration() {
        let costs = HandoffCostModel {
            latency_us: 1,
            bytes_per_us: 0,
            recompute_token_us: 1,
        };
        assert_eq!(
            choose_handoff(None, 10, costs),
            SessionHandoffMode::Recompute
        );
        assert_eq!(
            choose_handoff(Some(&state(1)), 10, costs),
            SessionHandoffMode::Recompute
        );
    }
}
