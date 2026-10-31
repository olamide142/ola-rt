//! Session handoff cost selection for migration and disaggregation.

use ola_model::SessionState;

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
