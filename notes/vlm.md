Quite easy **if you design ola-rt around a generic execution graph rather than around text-only autoregressive decoding**.

I would explicitly make VLM support a **Year-1 architectural requirement**, even if you don't implement it until later.

The key realization is that a VLM is not an entirely different inference runtime. It adds a **multimodal preprocessing/encoding stage** before largely familiar LLM decoding.

## Conceptually

For an LLM, ola-rt sees:

```text
Text
 │
 ▼
Tokenizer
 │
 ▼
Token IDs
 │
 ▼
Embedding
 │
 ▼
┌─────────────────┐
│ Transformer     │
│                 │
│ Attention       │
│ MLP             │
│ KV Cache        │
└────────┬────────┘
         │
         ▼
      LM Head
         │
         ▼
       Token
```

A typical VLM becomes:

```text
                 ┌───────────────┐
Image ──────────►│ Image         │
                 │ Preprocessor  │
                 └───────┬───────┘
                         ▼
                  ┌─────────────┐
                  │   Vision    │
                  │   Encoder   │
                  │ ViT / etc.  │
                  └──────┬──────┘
                         │
                   visual tokens
                         │
                         ▼
                  ┌─────────────┐
                  │ Projector / │
                  │ Connector   │
                  └──────┬──────┘
                         │
                         ▼
Text ─► Tokenizer ─► Text tokens
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
         visual tokens         text tokens
              │                     │
              └──────────┬──────────┘
                         ▼
                  Multimodal sequence
                         │
                         ▼
                  LLM / Decoder
                         │
                         ▼
                      Output
```

A lot of your existing infrastructure survives unchanged.

### What can remain shared

| ola-rt component        | LLM |              VLM |
| ----------------------- | --: | ---------------: |
| Request scheduler       |   ✓ |                ✓ |
| Execution policy        |   ✓ |                ✓ |
| Executor                |   ✓ |                ✓ |
| Tensor backend          |   ✓ |                ✓ |
| Kernel system           |   ✓ |                ✓ |
| KV cache                |   ✓ |                ✓ |
| PagedAttention          |   ✓ |                ✓ |
| Continuous batching     |   ✓ |                ✓ |
| Quantization            |   ✓ |                ✓ |
| Telemetry               |   ✓ |                ✓ |
| LLM decoder             |   ✓ |                ✓ |
| Vision preprocessing    |     |          **new** |
| Vision encoder          |     |          **new** |
| Multimodal projector    |     |          **new** |
| Visual-token management |     |          **new** |
| Multimodal scheduling   |     | **new/research** |

So I wouldn't create `ola-vlm-rt`.

I'd make VLMs first-class citizens of **ola-rt**.

---

## The architectural decision I'd make now

Don't represent requests like this:

```rust
struct Request {
    prompt: String,
}
```

Instead:

```rust
enum Input {
    Text(TextInput),
    Image(ImageInput),
    // later
    Audio(AudioInput),
    Video(VideoInput),
}

struct Request {
    inputs: Vec<Input>,
    generation: GenerationConfig,
}
```

Then something like:

```rust
let request = Request {
    inputs: vec![
        Input::Image(image),
        Input::Text(
            "What objects are visible in this image?".into()
        ),
    ],
    generation: config,
};
```

Now the runtime isn't inherently an LLM runtime.

It's a **multimodal inference runtime**.

---

# More importantly: make the IR multimodal-agnostic

This is where your earlier ola-IR becomes valuable.

Don't have:

```text
LLMGraph
```

Have:

```text
ExecutionGraph
```

For an LLM:

```text
Tokenize
   ↓
Embedding
   ↓
Transformer
   ↓
LMHead
```

For a VLM:

```text
              ┌─ Resize
Image ───────►├─ Normalize
              └────┬─────
                   ▼
                  ViT
                   │
                Project
                   │
                   ├──────────┐
                              ▼
Text ─► Tokenize ─► Embed ─► Merge
                              │
                              ▼
                         Transformer
                              │
                              ▼
                           LMHead
```

The executor doesn't particularly care that one node happens to be a vision encoder.

It sees something more like:

```rust
Graph {
    nodes: [
        ImagePreprocess,
        VisionEncoder,
        Projection,
        TokenEmbedding,
        MergeEmbeddings,
        Transformer,
        LMHead,
    ]
}
```

That's a much better foundation.

---

# Where it gets interesting for your PhD

Adding VLM inference is relatively straightforward.

**Optimizing VLM inference is considerably more interesting.**

Consider two requests:

```text
Request A
─────────
1 × 512×512 image
20 text tokens

Request B
─────────
8 × 2048×2048 images
1200 text tokens
```

These are radically different workloads.

The runtime now has several phases:

```text
Request
   │
   ▼
┌─────────────────┐
│ Image Decode    │ CPU
└────────┬────────┘
         ▼
┌─────────────────┐
│ Preprocessing   │ CPU/GPU
└────────┬────────┘
         ▼
┌─────────────────┐
│ Vision Encoder  │ GPU
└────────┬────────┘
         ▼
┌─────────────────┐
│ Projector       │ GPU
└────────┬────────┘
         ▼
┌─────────────────┐
│ LLM Prefill     │ GPU
└────────┬────────┘
         ▼
┌─────────────────┐
│ LLM Decode      │ GPU
└─────────────────┘
```

And these phases have **different compute characteristics**.

That gives your execution-policy work considerably more dimensions.

For example, suppose you have:

```text
RTX 5090

Request A ─ vision ─────── LLM prefill ─ decode ─ decode...
Request B ───── vision ─────────── LLM prefill ─ decode...
Request C ─ vision ─ LLM prefill ─ decode...
```

The scheduler now has to answer:

**Should I batch vision encoders together?**

**Should I let an incoming image interrupt decode?**

**Should image preprocessing happen on CPU or GPU?**

**Should I cache image embeddings?**

**Should I batch images by resolution?**

**Should vision and language execute concurrently?**

**How should I allocate GPU memory between visual tokens and KV cache?**

That's excellent territory for ola-rt.

---

## You could model VLM execution explicitly

Your policy state might evolve from:

$$
S_t =
(W_t,H_t,M_t,Q_t)
$$

to something like:

$$
S_t =
(W_t,H_t,M_t,Q_t,V_t)
$$

where \(V_t\) contains multimodal characteristics:

```text
number of images
image resolution
patch count
visual token count
vision encoder
vision-cache state
text token count
expected output length
```

Then:

$$
\pi(S_t) \rightarrow a_t
$$

could choose:

```text
batch images 1, 4 and 7

run ViT now

delay request B

reuse cached visual embeddings

resize image to 768

execute vision on GPU 0

execute language on GPU 1

chunk multimodal prefill

reserve 6 GB for KV cache

use FlashAttention implementation X
```

Now your **inference execution policy formalization applies naturally to multimodal models**.

---

# It also opens up heterogeneous execution

Eventually imagine:

```text
                       ola-rt

                         │
                  Execution Policy
                         │
          ┌──────────────┼───────────────┐
          │              │               │
          ▼              ▼               ▼
         CPU           GPU 0           GPU 1
          │              │               │
 Image decode         Vision           LLM
 Resize               Encoder          Decode
 Tokenizer            Projector
          │              │               │
          └──────────────┼───────────────┘
                         ▼
                      Output
```

Or:

```text
GPU 0: ViT
GPU 1: LLM prefill
GPU 2: LLM decode
```

That's essentially **disaggregated multimodal inference**.

And that's potentially a much richer optimization problem than ordinary LLM serving.

---

# VLM caching becomes interesting too

Suppose Olareport sends the same exam page through multiple prompts:

```text
image.jpg

"Extract the questions"
"Find the diagrams"
"Classify the subject"
"Identify handwritten markings"
```

Naively:

```text
Image → ViT
Image → ViT
Image → ViT
Image → ViT
```

Instead:

```text
                     image.jpg
                         │
                         ▼
                    Vision Encoder
                         │
                         ▼
                 Visual Embeddings
                         │
                   ┌─────┼─────┐
                   ▼     ▼     ▼
                Prompt A B     C ...
```

ola-rt could have a:

```rust
VisionCache {
    image_hash -> VisualEmbedding
}
```

Then your memory manager isn't merely:

```text
weights
KV cache
workspace
```

but:

```text
weights
KV cache
visual embedding cache
prefix cache
workspace
```

And the policy decides what deserves GPU memory.

---

# Video makes the architecture even more valuable

If you get the abstraction right:

```rust
enum Modality {
    Text,
    Image,
    Audio,
    Video,
}
```

you can eventually go:

```text
                       Request
                          │
          ┌───────────────┼────────────────┐
          ▼               ▼                ▼
        Image            Audio            Video
          │               │                │
      Vision Enc       Audio Enc        Frame Enc
          │               │                │
          └───────────────┬┴────────────────┘
                          ▼
                     Projectors
                          │
                          ▼
                    Token/Embedding
                       sequence
                          │
                          ▼
                         LLM
```

The runtime architecture doesn't fundamentally change.

The **execution graph changes**.

That's exactly what you want.

---

## Burn is helpful here too

Initially, Burn can provide the building blocks for:

```text
Conv2D
Linear
LayerNorm
Softmax
Attention
ViT
Tensor operations
GPU execution
```

So you don't have to build a vision framework before researching inference.

Your progression could be:

```text
Phase 1
Text-only small Transformer
        ↓
Phase 2
LLM + KV cache
        ↓
Phase 3
Continuous batching
        ↓
Phase 4
Paged KV / PagedAttention
        ↓
Phase 5
Execution-policy abstraction
        ↓
Phase 6
Vision Transformer encoder
        ↓
Phase 7
VLM
        ↓
Phase 8
Multimodal scheduling
        ↓
Phase 9
Adaptive multimodal execution policies
```

### And this is why I would slightly redefine ola-rt

Rather than:

> **ola-rt — an LLM inference runtime written in Rust**

I'd think about it internally as:

> **ola-rt — an experimental runtime for adaptive AI inference execution.**

Then LLM inference is merely your **first workload**.

```text
                         ola-rt

              Adaptive Execution Runtime
                         │
       ┌─────────────────┼─────────────────┐
       ▼                 ▼                 ▼
      LLM               VLM            Future
       │                 │            Audio/Video
       └─────────────────┼─────────────────┘
                         │
                         ▼
                  Execution Graph
                         │
                         ▼
                  Runtime State Sₜ
                         │
                         ▼
                  Policy π(Sₜ)
                         │
       ┌─────────────────┼──────────────────┐
       ▼                 ▼                  ▼
   Scheduling          Memory             Kernels
       │                 │                  │
       └─────────────────┼──────────────────┘
                         ▼
                     Hardware
```

That architecture gives you room to spend the first year reproducing known LLM inference techniques, and then use the same runtime for genuinely interesting **VLM scheduling, heterogeneous execution, multimodal caching, adaptive resource allocation and optimization research** later in the PhD.

