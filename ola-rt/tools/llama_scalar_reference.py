#!/usr/bin/env python3
"""Compute small Llama logits without Burn, PyTorch, NumPy, or Transformers.

This script is an intentionally slow conformance oracle. It implements the
Hugging Face Llama equations with Python scalar arithmetic and reads F32 values
directly from a SafeTensors file. It is suitable for tiny fixtures only; its
purpose is to create inspectable expected values for OlaRT tests, not inference.

Keeping the oracle independent of Burn helps detect errors that shape-only
tests cannot see: linear weight orientation, causal masking, grouped-query head
mapping, rotary layout, residual ordering, and RMSNorm placement.

Usage:

    python3 tools/llama_scalar_reference.py ARTIFACT_DIR 1 15043 1789

The output is JSON containing the first 16 logits and argmax at every input
position. Pin the artifact revision and hashes wherever those values are used.
"""

from __future__ import annotations

import json
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Tensor:
    """One flattened F32 tensor and its row-major shape."""

    shape: tuple[int, ...]
    values: tuple[float, ...]

    def row(self, index: int) -> tuple[float, ...]:
        """Return a row from a rank-two tensor."""

        assert len(self.shape) == 2
        width = self.shape[1]
        start = index * width
        return self.values[start : start + width]


def read_safetensors(path: Path) -> dict[str, Tensor]:
    """Read an uncompressed F32 SafeTensors file into host tuples."""

    payload = path.read_bytes()
    header_length = struct.unpack_from("<Q", payload, 0)[0]
    header_start = 8
    data_start = header_start + header_length
    metadata = json.loads(payload[header_start:data_start])
    tensors: dict[str, Tensor] = {}
    for name, descriptor in metadata.items():
        if name == "__metadata__":
            continue
        if descriptor["dtype"] != "F32":
            raise ValueError(f"{name}: expected F32, found {descriptor['dtype']}")
        start, end = descriptor["data_offsets"]
        count = (end - start) // 4
        values = struct.unpack_from(f"<{count}f", payload, data_start + start)
        tensors[name] = Tensor(tuple(descriptor["shape"]), values)
    return tensors


def linear(vector: list[float], weight: Tensor) -> list[float]:
    """Apply a Hugging Face `[output, input]` linear weight."""

    assert weight.shape[1] == len(vector)
    return [math.fsum(x * w for x, w in zip(vector, weight.row(row))) for row in range(weight.shape[0])]


def rms_norm(vector: list[float], weight: Tensor, epsilon: float) -> list[float]:
    """Apply Llama RMSNorm in F64 scalar arithmetic."""

    mean_square = math.fsum(value * value for value in vector) / len(vector)
    inverse_rms = 1.0 / math.sqrt(mean_square + epsilon)
    return [value * inverse_rms * scale for value, scale in zip(vector, weight.values)]


def silu(value: float) -> float:
    """Numerically stable scalar SiLU."""

    if value >= 0.0:
        return value / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return value * exponential / (1.0 + exponential)


def softmax(values: list[float]) -> list[float]:
    """Compute a stable scalar softmax."""

    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    denominator = math.fsum(exponentials)
    return [value / denominator for value in exponentials]


def project_sequence(sequence: list[list[float]], weight: Tensor) -> list[list[float]]:
    """Apply one projection to every sequence position."""

    return [linear(vector, weight) for vector in sequence]


def split_heads(sequence: list[list[float]], heads: int, head_dim: int) -> list[list[list[float]]]:
    """Reshape `[sequence, heads * head_dim]` into `[sequence, heads, head_dim]`."""

    return [
        [vector[head * head_dim : (head + 1) * head_dim] for head in range(heads)]
        for vector in sequence
    ]


def apply_rope(sequence: list[list[list[float]]], head_dim: int, theta: float) -> None:
    """Apply Hugging Face Llama's split-half rotary convention in place."""

    half = head_dim // 2
    inverse_frequencies = [theta ** (-(2 * index) / head_dim) for index in range(half)]
    for position, heads in enumerate(sequence):
        for head in heads:
            original = head.copy()
            for index, inverse_frequency in enumerate(inverse_frequencies):
                angle = position * inverse_frequency
                cosine = math.cos(angle)
                sine = math.sin(angle)
                head[index] = original[index] * cosine - original[index + half] * sine
                head[index + half] = original[index + half] * cosine + original[index] * sine


def causal_attention(
    query: list[list[list[float]]],
    key: list[list[list[float]]],
    value: list[list[list[float]]],
    query_heads: int,
    key_value_heads: int,
    head_dim: int,
) -> list[list[float]]:
    """Compute unbatched grouped-query causal attention."""

    groups = query_heads // key_value_heads
    scale = 1.0 / math.sqrt(head_dim)
    output: list[list[float]] = []
    for query_position in range(len(query)):
        concatenated: list[float] = []
        for query_head in range(query_heads):
            key_value_head = query_head // groups
            scores = [
                math.fsum(
                    query[query_position][query_head][dimension]
                    * key[key_position][key_value_head][dimension]
                    for dimension in range(head_dim)
                )
                * scale
                for key_position in range(query_position + 1)
            ]
            probabilities = softmax(scores)
            concatenated.extend(
                math.fsum(
                    probabilities[key_position]
                    * value[key_position][key_value_head][dimension]
                    for key_position in range(query_position + 1)
                )
                for dimension in range(head_dim)
            )
        output.append(concatenated)
    return output


def add(left: list[float], right: list[float]) -> list[float]:
    """Add equal-length vectors."""

    return [a + b for a, b in zip(left, right)]


def forward(config: dict[str, object], weights: dict[str, Tensor], token_ids: list[int]) -> list[list[float]]:
    """Run an unbatched Llama forward pass and return every position's logits."""

    hidden_size = int(config["hidden_size"])
    query_heads = int(config["num_attention_heads"])
    key_value_heads = int(config["num_key_value_heads"])
    head_dim = hidden_size // query_heads
    epsilon = float(config["rms_norm_eps"])
    theta = float(config.get("rope_theta", 10_000.0))
    hidden = [list(weights["model.embed_tokens.weight"].row(token)) for token in token_ids]

    for layer in range(int(config["num_hidden_layers"])):
        prefix = f"model.layers.{layer}"
        normalized = [
            rms_norm(vector, weights[f"{prefix}.input_layernorm.weight"], epsilon)
            for vector in hidden
        ]
        query = split_heads(
            project_sequence(normalized, weights[f"{prefix}.self_attn.q_proj.weight"]),
            query_heads,
            head_dim,
        )
        key = split_heads(
            project_sequence(normalized, weights[f"{prefix}.self_attn.k_proj.weight"]),
            key_value_heads,
            head_dim,
        )
        value = split_heads(
            project_sequence(normalized, weights[f"{prefix}.self_attn.v_proj.weight"]),
            key_value_heads,
            head_dim,
        )
        apply_rope(query, head_dim, theta)
        apply_rope(key, head_dim, theta)
        attended = causal_attention(
            query, key, value, query_heads, key_value_heads, head_dim
        )
        attention_output = project_sequence(
            attended, weights[f"{prefix}.self_attn.o_proj.weight"]
        )
        hidden = [add(residual, update) for residual, update in zip(hidden, attention_output)]

        normalized = [
            rms_norm(vector, weights[f"{prefix}.post_attention_layernorm.weight"], epsilon)
            for vector in hidden
        ]
        gate = project_sequence(normalized, weights[f"{prefix}.mlp.gate_proj.weight"])
        up = project_sequence(normalized, weights[f"{prefix}.mlp.up_proj.weight"])
        intermediate = [
            [silu(gate_value) * up_value for gate_value, up_value in zip(gate_row, up_row)]
            for gate_row, up_row in zip(gate, up)
        ]
        mlp_output = project_sequence(intermediate, weights[f"{prefix}.mlp.down_proj.weight"])
        hidden = [add(residual, update) for residual, update in zip(hidden, mlp_output)]

    hidden = [rms_norm(vector, weights["model.norm.weight"], epsilon) for vector in hidden]
    return project_sequence(hidden, weights["lm_head.weight"])


def summarize(logits: Iterable[list[float]]) -> list[dict[str, object]]:
    """Select stable, compact observations for a committed fixture."""

    return [
        {
            "first_16": row[:16],
            "argmax": max(range(len(row)), key=row.__getitem__),
            "mean": math.fsum(row) / len(row),
        }
        for row in logits
    ]


def main(arguments: list[str]) -> int:
    """Command-line entry point."""

    if len(arguments) < 3:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    directory = Path(arguments[1])
    token_ids = [int(value) for value in arguments[2:]]
    config = json.loads((directory / "config.json").read_text())
    weights = read_safetensors(directory / "model.safetensors")
    print(
        json.dumps(
            {
                "token_ids": token_ids,
                "positions": summarize(forward(config, weights, token_ids)),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
