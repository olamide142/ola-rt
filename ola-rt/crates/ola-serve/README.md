# ola-serve

`ola-serve` defines the application-facing boundary around `ola-runtime`.

It currently translates a `GenerateRequest` into a validated runtime
`RequestSpec`, exposes health and model discovery, and retains access to the
underlying runtime for an executor loop. It is intentionally transport-neutral:
an Axum/OpenAI-compatible layer can be added later without moving HTTP concepts
into lifecycle or scheduling code.

Arrival timestamps are supplied by the transport. In production they should be
derived from a monotonic process clock; in tests and trace replay they may come
from simulated time.
