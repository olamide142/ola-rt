# ola-net

`ola-net` is the wire layer of a distributed OlaRT deployment
(`../DISTRIBUTED.md`). It frames messages on any tokio byte stream and
provides a TCP peer. It knows nothing about clusters: `ola-cluster` puts its
`WorkerCommand` and `WorkerMessage` values inside opaque payloads.

## Control frames

A control frame is a big-endian `u32` length followed by that many bytes of
JSON, a `WireMessage`:

```json
{"version": 1, "sender_epoch": 1730368800000000, "correlation": 42,
 "body": {"kind": "heartbeat", "sent_at_us": 1500}}
```

- `version` must equal `PROTOCOL_VERSION`; both writing and reading reject
  anything else with `UnsupportedVersion`.
- `sender_epoch` identifies the sending process. A worker uses a larger
  epoch after every restart, so a coordinator can ignore a delayed
  heartbeat from the old process.
- `correlation` numbers the sender's frames; acknowledgements reuse the
  number they answer.
- Frames over `MAX_MESSAGE_BYTES` (16 MiB) are rejected before their payload
  is allocated.

| `body.kind` | Fields | Meaning |
| --- | --- | --- |
| `hello` | `worker` | A worker's first frame: its stable ID. |
| `heartbeat` | `sent_at_us` | Liveness, stamped on the sender's clock. |
| `heartbeat_ack` | `echo_sent_at_us` | Reply that echoes the heartbeat's stamp, so the sender can time the round trip on its own clock. |
| `payload` | `content_type`, `data` | An opaque domain value. `json_payload`/`decode_json` use `application/json`. |
| `goodbye` | `reason` | Graceful close. |

`data` is a byte array, which JSON writes as an array of numbers. This is
simple but costs several bytes per payload byte; large values belong in
binary frames.

A message may also set `"attachment": true` (omitted when false), meaning
one binary frame follows it immediately. `write_attached` sends the pair and
`read_attached` reads it; `write_message` and `read_message` refuse such
messages with `MisplacedAttachment`, so a peer that does not expect a frame
fails loudly instead of misreading the stream.

## Binary frames

A `BinaryFrame` carries bulk data, such as serialized KV blocks, beside the
control stream:

```text
u32 total_len | u32 header_len | header (JSON) | metadata | data
```

The JSON header holds `version`, `sender_epoch`, `correlation`,
`content_type`, and `metadata_len`; `data` is the rest of the frame. Frames
over `MAX_BINARY_FRAME_BYTES` (256 MiB) or with inconsistent lengths are
rejected, and so is any version other than `PROTOCOL_VERSION`.
`ola-cluster`'s `session_state_to_frame` and `session_state_from_frame`
adapt model session snapshots to this format, and its worker sessions and
coordinator links attach them to migration messages.

## Errors

`read_message`, `read_attached`, and `read_binary_frame` report a stream that ends part-way
through a frame as `Truncated`, an oversized or inconsistent length as
`InvalidLength`, and malformed JSON as `Json`.

## Reading safely

These functions read a frame in several steps. If the future reading a
frame is dropped part-way, the stream is left mid-frame and every later read
is garbage. Give each connection one task that owns its read half and does
nothing but read frames into a channel; `ola-cluster`'s `WorkerSession` and
`CoordinatorWorkerLink` both do this.
