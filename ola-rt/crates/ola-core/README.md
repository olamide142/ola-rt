# ola-core

`ola-core` defines the backend-neutral vocabulary shared across OlaRT.

It owns model and architecture identifiers, monotonic timestamps, request
inputs, generation parameters, model descriptors, and the smallest useful
compute-backend boundary. It intentionally contains no queue, scheduler, HTTP,
or Burn types.
Keeping those dependencies out prevents research code from coupling every
experiment to one execution backend.

Important semantics:

- `Timestamp` is relative monotonic time in microseconds, never wall-clock time.
- `RequestSpec` is caller intent; `Request` is the validated, identified form.
- `ModelId` identifies one selectable artifact; `ArchitectureId` identifies the
  implementation family capable of interpreting it.
- Inputs are multimodal from the beginning so the runtime does not assume that
  every request is a single text prompt.
- `ComputeBackend` exposes identity and device ownership only. Tensor execution
  will be added after a real model path demonstrates the correct abstraction.
