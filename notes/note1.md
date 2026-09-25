Yes. And because Burn already provides the major Transformer building blocks, we can implement the architecture in your diagram without manually implementing attention mathematics.

I'm targeting the current Burn 0.21 API. Burn provides `Embedding`, `PositionalEncoding`, `TransformerEncoder`, `TransformerDecoder`, and `Linear`; internally its Transformer layers already compose multi-head attention, feed-forward networks, layer norms, residual paths, and dropout. ([Docs.rs][1])

### 1. The model

```rust
use burn::{
    module::Module,
    nn::{
        Embedding,
        EmbeddingConfig,
        Linear,
        LinearConfig,
        PositionalEncoding,
        PositionalEncodingConfig,
    },
    nn::transformer::{
        TransformerDecoder,
        TransformerDecoderConfig,
        TransformerDecoderInput,
        TransformerEncoder,
        TransformerEncoderConfig,
        TransformerEncoderInput,
    },
    tensor::{
        backend::Backend,
        Int,
        Tensor,
    },
};

#[derive(Module, Debug)]
pub struct Transformer<B: Backend> {
    // Bottom-left: Input Embedding
    src_embedding: Embedding<B>,

    // Bottom-right: Output Embedding
    tgt_embedding: Embedding<B>,

    // The ⊕ in the diagram
    src_position: PositionalEncoding<B>,
    tgt_position: PositionalEncoding<B>,

    // Entire left-hand stack
    encoder: TransformerEncoder<B>,

    // Entire right-hand stack
    decoder: TransformerDecoder<B>,

    // Top Linear box
    lm_head: Linear<B>,
}
```

Already, you can map this almost directly onto the diagram:

```text
DIAGRAM                         BURN

Input Embedding          →      Embedding<B>
        +                →      PositionalEncoding<B>

Encoder                  →      TransformerEncoder<B>
  Multi-Head Attention
  Add & Norm
  Feed Forward
  Add & Norm

Output Embedding         →      Embedding<B>
        +
Position                 →      PositionalEncoding<B>

Decoder                  →      TransformerDecoder<B>
  Masked MHA
  Add & Norm
  Cross Attention
  Add & Norm
  Feed Forward
  Add & Norm

Linear                   →      Linear<B>
Softmax                  →      tensor::activation::softmax
```

Burn's `Embedding::forward` specifically transforms `[batch, seq]` integer token IDs into `[batch, seq, d_model]`. ([Docs.rs][2])

### 2. Configuration

Let's build a small version of the original Transformer:

```rust
pub struct TransformerConfig {
    pub src_vocab_size: usize,
    pub tgt_vocab_size: usize,

    pub d_model: usize,
    pub d_ff: usize,

    pub n_heads: usize,
    pub n_layers: usize,

    pub max_seq_len: usize,
}

impl TransformerConfig {
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Transformer<B> {

        let src_embedding =
            EmbeddingConfig::new(
                self.src_vocab_size,
                self.d_model,
            )
            .init(device);

        let tgt_embedding =
            EmbeddingConfig::new(
                self.tgt_vocab_size,
                self.d_model,
            )
            .init(device);

        let src_position =
            PositionalEncodingConfig::new(
                self.d_model,
            )
            .with_max_sequence_size(self.max_seq_len)
            .init(device);

        let tgt_position =
            PositionalEncodingConfig::new(
                self.d_model,
            )
            .with_max_sequence_size(self.max_seq_len)
            .init(device);

        let encoder =
            TransformerEncoderConfig::new(
                self.d_model,
                self.d_ff,
                self.n_heads,
                self.n_layers,
            )
            .init(device);

        let decoder =
            TransformerDecoderConfig::new(
                self.d_model,
                self.d_ff,
                self.n_heads,
                self.n_layers,
            )
            .init(device);

        let lm_head =
            LinearConfig::new(
                self.d_model,
                self.tgt_vocab_size,
            )
            .init(device);

        Transformer {
            src_embedding,
            tgt_embedding,
            src_position,
            tgt_position,
            encoder,
            decoder,
            lm_head,
        }
    }
}
```

For example:

```rust
let config = TransformerConfig {
    src_vocab_size: 32_000,
    tgt_vocab_size: 32_000,

    d_model: 512,
    d_ff: 2048,

    n_heads: 8,
    n_layers: 6,

    max_seq_len: 512,
};
```

Burn's encoder configuration takes exactly the core Transformer hyperparameters `d_model`, `d_ff`, `n_heads`, and `n_layers`; its default FFN activation is GELU. ([Docs.rs][3])

---

# 3. Encoder forward pass

Let's separate this so you can see exactly what happens.

```rust
impl<B: Backend> Transformer<B> {

    pub fn encode(
        &self,
        src_tokens: Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {

        // --------------------------------
        // INPUT EMBEDDING
        // --------------------------------

        let x = self.src_embedding.forward(src_tokens);

        // [batch, src_seq]
        //
        //        ↓ Embedding
        //
        // [batch, src_seq, d_model]


        // --------------------------------
        // POSITIONAL ENCODING
        // --------------------------------

        let x = self.src_position.forward(x);

        // Conceptually:
        //
        // x = token_embedding + positional_encoding


        // --------------------------------
        // ENCODER
        // --------------------------------

        let input = TransformerEncoderInput::new(x);

        self.encoder.forward(input)

        // [batch, src_seq, d_model]
    }
}
```

Burn's positional encoding module implements the sinusoidal positional information from the original Transformer. ([Docs.rs][4])

So:

```text
src_tokens

[batch, seq]
     │
     ▼
Embedding
     │
     ▼
[batch, seq, 512]
     │
     ▼
PositionalEncoding
     │
     ▼
TransformerEncoder
     │
     ▼
encoder_memory
```

---

# 4. What's hidden inside `TransformerEncoder`

This one line:

```rust
self.encoder.forward(input)
```

looks deceptively simple.

Internally, each layer is essentially:

```text
x
│
├──────────────────────────────┐
│                              │
▼                              │
MultiHeadAttention             │
│                              │
▼                              │
Dropout                        │
│                              │
+ ◄────────────────────────────┘
│
▼
LayerNorm
│
├──────────────────────────────┐
│                              │
▼                              │
PositionWiseFeedForward        │
│                              │
▼                              │
Dropout                        │
│                              │
+ ◄────────────────────────────┘
│
▼
LayerNorm
```

Burn's encoder-layer record confirms exactly these constituent modules: `MultiHeadAttention`, `PositionWiseFeedForward`, two `LayerNorm`s, and dropout. ([Docs.rs][5])

And Burn's feed-forward block itself contains:

```text
Linear
  ↓
Activation
  ↓
Dropout
  ↓
Linear
```

mapping:

```text
d_model → d_ff → d_model

512 → 2048 → 512
```

([Docs.rs][6])

So when you write:

```rust
TransformerEncoderConfig::new(
    512,
    2048,
    8,
    6,
)
```

Burn creates roughly:

```text
Encoder

├── Layer 1
│   ├── MultiHeadAttention
│   ├── LayerNorm
│   ├── FeedForward
│   │   ├── Linear(512 → 2048)
│   │   ├── GELU
│   │   └── Linear(2048 → 512)
│   └── LayerNorm
│
├── Layer 2
│   └── ...
│
├── Layer 3
│
├── Layer 4
│
├── Layer 5
│
└── Layer 6
```

That's essentially the entire left side of your image.

---

# 5. Decoder

The decoder needs two things:

```text
target sequence
+
encoder output
```

Conceptually:

```rust
impl<B: Backend> Transformer<B> {

    pub fn decode(
        &self,
        tgt_tokens: Tensor<B, 2, Int>,
        memory: Tensor<B, 3>,
    ) -> Tensor<B, 3> {

        // Output Embedding
        let x = self.tgt_embedding.forward(tgt_tokens);

        // + positional information
        let x = self.tgt_position.forward(x);

        // Decoder receives:
        //
        // 1. decoder representation
        // 2. encoder representation

        let input =
            TransformerDecoderInput::new(x, memory);

        self.decoder.forward(input)
    }
}
```

Conceptually:

```text
                   encoder memory
                         │
                         │
                         ▼
target tokens      ┌───────────────┐
     │              │               │
     ▼              │               │
Embedding           │               │
     │              │               │
Position            │               │
     │              │               │
     ▼              │               │
Masked Self Attn    │               │
     │              │               │
Add & Norm          │               │
     │              │               │
     ▼              │               │
Cross Attention ◄───┘               │
     │
Add & Norm
     │
Feed Forward
     │
Add & Norm
     │
     ▼
decoder output
```

Burn's decoder layer exposes both `self_attn` and `cross_attn`, plus its feed-forward network, three layer norms, and dropout. ([Burn][7])

---

# 6. Full forward

Now combine them:

```rust
impl<B: Backend> Transformer<B> {

    pub fn forward(
        &self,
        src_tokens: Tensor<B, 2, Int>,
        tgt_tokens: Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {

        // ====================================
        // ENCODER
        // ====================================

        let src = self.src_embedding.forward(src_tokens);
        let src = self.src_position.forward(src);

        let memory = self.encoder.forward(
            TransformerEncoderInput::new(src)
        );


        // ====================================
        // DECODER
        // ====================================

        let tgt = self.tgt_embedding.forward(tgt_tokens);
        let tgt = self.tgt_position.forward(tgt);

        let hidden = self.decoder.forward(
            TransformerDecoderInput::new(
                tgt,
                memory,
            )
        );


        // ====================================
        // LM HEAD
        // ====================================

        self.lm_head.forward(hidden)

        // [batch, tgt_seq, vocab_size]
    }
}
```

The final `Linear` works because Burn's linear layer transforms the last dimension from `d_model` to `vocab_size`, preserving the leading dimensions. ([Docs.rs][8])

Therefore:

```text
[batch, seq, 512]

        ↓

Linear(512 → 32,000)

        ↓

[batch, seq, 32,000]
```

---

# 7. Where is Softmax?

I would **not put softmax inside `forward()`**.

Return logits:

```rust
let logits = model.forward(src, tgt);
```

During inference:

```rust
use burn::tensor::activation::softmax;

let probabilities = softmax(logits, 2);
```

because dimension `2` is:

```text
[batch, sequence, vocabulary]
                    ↑
                    2
```

During training you normally feed the logits directly into cross-entropy rather than first explicitly applying softmax.

So the top of your diagram becomes:

```text
decoder
   │
   ▼
Linear
   │
   ▼
logits
   │
   ├────────────── training ─────→ CrossEntropy
   │
   └────────────── inference
                         │
                         ▼
                       Softmax
                         │
                         ▼
                  token probabilities
```

---

# 8. The really interesting part: Burn's MHA

Remember when we manually described:

```text
Q = XWq
K = XWk
V = XWv

       QKᵀ
softmax(────)V
       √d_k
```

Burn's `MultiHeadAttention` literally contains learned query/key/value/output `Linear` projections. Its implementation projects tensors, reshapes them into `[batch, heads, sequence, d_k]`, calculates attention scores, applies masks and softmax, then produces the context tensor. ([Docs.rs][9])

So:

```rust
TransformerEncoder<B>
```

eventually calls:

```rust
MultiHeadAttention<B>
```

which eventually performs the equivalent of:

```text
                X
          ┌─────┼─────┐
          ▼     ▼     ▼
        Linear Linear Linear
          │     │     │
          Q     K     V
          │     │     │
          └──┬──┘     │
             │        │
            QKᵀ       │
             │        │
           /√dk       │
             │        │
          Softmax     │
             │        │
             └────┬───┘
                  │
             Attention × V
                  │
                  ▼
                Linear
                  │
                  ▼
                output
```

Burn even exposes the resulting attention weights as `[batch, n_heads, seq1, seq2]`. ([Docs.rs][10])

---

# 9. Burn is doing more for you than it initially appears

The entire model you showed can therefore be composed approximately as:

```text
Transformer
│
├── Embedding
├── PositionalEncoding
│
├── TransformerEncoder
│    │
│    └── × N
│         ├── MultiHeadAttention
│         │    ├── Linear Q
│         │    ├── Linear K
│         │    ├── Linear V
│         │    └── Linear O
│         │
│         ├── LayerNorm
│         │
│         ├── PositionWiseFeedForward
│         │    ├── Linear
│         │    ├── GELU
│         │    └── Linear
│         │
│         └── LayerNorm
│
├── Embedding
├── PositionalEncoding
│
├── TransformerDecoder
│    │
│    └── × N
│         ├── Masked MultiHeadAttention
│         ├── LayerNorm
│         ├── Cross MultiHeadAttention
│         ├── LayerNorm
│         ├── PositionWiseFeedForward
│         └── LayerNorm
│
└── Linear(d_model → vocab)
```

This is one reason I've been saying **don't throw Burn away while building ola-rt**. Burn already gives you useful neural-network primitives, and the current API even includes attention masking and autoregressive caches intended for token-by-token inference. ([Docs.rs][11])

The interesting progression for your PhD is to start with the high-level version above, then progressively "open the boxes":

```text
Stage 1

TransformerEncoder<B>
TransformerDecoder<B>

              ↓


Stage 2

Build your own EncoderBlock/DecoderBlock using:

MultiHeadAttention<B>
PositionWiseFeedForward<B>
LayerNorm<B>

              ↓


Stage 3

Build your own attention using:

Linear<B>
Tensor<B>
tensor attention primitives

              ↓


Stage 4

Replace attention execution:

normal attention
       ↓
FlashAttention
       ↓
PagedAttention / KV management
       ↓
custom CubeCL kernels

              ↓


Stage 5

OLA-RT

Model
 ↓
Execution Graph / IR
 ↓
Inference Policy
 ↓
Scheduler
 ↓
Continuous Batcher
 ↓
KV Cache Manager
 ↓
Attention Backend
 ↓
Kernel selection
 ↓
GPU
```

**Stage 2 is what I'd implement next**, rather than immediately relying permanently on `TransformerEncoder`/`TransformerDecoder`. Build the exact same architecture using Burn's `MultiHeadAttention`, `LayerNorm`, `PositionWiseFeedForward`, `Embedding`, and `Linear`. That will expose the residual connections, causal mask, Q/K/V flow, and cross-attention directly in your own Rust code—which is precisely the layer of understanding you'll need before implementing KV caching and PagedAttention in ola-rt.

[1]: https://docs.rs/burn/latest/burn/nn/?utm_source=chatgpt.com "burn::nn - Rust"
[2]: https://docs.rs/burn/latest/burn/nn/struct.Embedding.html?utm_source=chatgpt.com "Embedding in burn::nn - Rust"
[3]: https://docs.rs/burn/latest/burn/nn/modules/transformer/struct.TransformerEncoderConfig.html?utm_source=chatgpt.com "TransformerEncoderConfig in burn::nn::modules::transformer - Rust"
[4]: https://docs.rs/burn/latest/burn/nn/modules/struct.PositionalEncoding.html?utm_source=chatgpt.com "PositionalEncoding in burn::nn::modules - Rust"
[5]: https://docs.rs/burn-nn/latest/burn_nn/modules/transformer/struct.TransformerEncoderLayerRecord.html?utm_source=chatgpt.com "TransformerEncoderLayerRecord in burn_nn::modules::transformer - Rust"
[6]: https://docs.rs/burn-nn/latest/burn_nn/modules/transformer/struct.PositionWiseFeedForward.html?utm_source=chatgpt.com "PositionWiseFeedForward in burn_nn::modules::transformer - Rust"
[7]: https://burn.dev/docs/burn/nn/modules/transformer/struct.TransformerDecoderLayer.html?utm_source=chatgpt.com "TransformerDecoderLayer in burn::nn::modules::transformer - Rust"
[8]: https://docs.rs/burn/latest/burn/nn/struct.Linear.html?utm_source=chatgpt.com "Linear in burn::nn - Rust"
[9]: https://docs.rs/burn-nn/latest/src/burn_nn/modules/attention/mha.rs.html?utm_source=chatgpt.com "mha.rs - source"
[10]: https://docs.rs/burn-nn/latest/burn_nn/modules/attention/struct.MhaOutput.html?utm_source=chatgpt.com "MhaOutput in burn_nn::modules::attention - Rust"
[11]: https://docs.rs/burn/latest/burn/nn/modules/attention/index.html?utm_source=chatgpt.com "burn::nn::modules::attention - Rust"





Good. The next version should **not** use Burn's `TransformerEncoder` or `TransformerDecoder`. We will build the architecture ourselves from Burn's lower-level building blocks so you can see exactly where attention, residuals, normalization, FFNs, masks, and cross-attention happen.

Conceptually, we're moving down one abstraction level:

```text
Before:

TransformerEncoder
TransformerDecoder
        ↓

Now:

Embedding
MultiHeadAttention
LayerNorm
Linear
GELU
Dropout
Tensor operations
```

## 1. Feed-forward network

Let's start with the simplest Transformer component.

```rust
use burn::{
    module::Module,
    nn::{
        Dropout,
        DropoutConfig,
        Gelu,
        Linear,
        LinearConfig,
    },
    tensor::{
        backend::Backend,
        Tensor,
    },
};

#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    linear_in: Linear<B>,
    linear_out: Linear<B>,
    dropout: Dropout,
    gelu: Gelu,
}

impl<B: Backend> FeedForward<B> {
    pub fn new(
        d_model: usize,
        d_ff: usize,
        dropout: f64,
        device: &B::Device,
    ) -> Self {
        Self {
            linear_in: LinearConfig::new(d_model, d_ff).init(device),

            linear_out: LinearConfig::new(d_ff, d_model).init(device),

            dropout: DropoutConfig::new(dropout).init(),

            gelu: Gelu::new(),
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let x = self.linear_in.forward(x);

        let x = self.gelu.forward(x);

        let x = self.dropout.forward(x);

        self.linear_out.forward(x)
    }
}
```

For:

```text
d_model = 512
d_ff    = 2048
```

this performs:

```text
[batch, seq, 512]
        │
        ▼
 Linear(512 → 2048)
        │
        ▼
      GELU
        │
        ▼
     Dropout
        │
        ▼
 Linear(2048 → 512)
        │
        ▼
[batch, seq, 512]
```

Notice that sequence length doesn't change.

---

# 2. Encoder block

Now let's build this part of your original diagram ourselves:

```text
         x
         │
    ┌────┴───────────┐
    │                │
    │                ▼
    │        Multi-Head Attention
    │                │
    │                ▼
    └──────────────► +
                     │
                   Norm
                     │
              ┌──────┴─────────┐
              │                │
              │                ▼
              │          Feed Forward
              │                │
              │                ▼
              └──────────────► +
                               │
                             Norm
                               │
                               ▼
                             output
```

In code:

```rust
use burn::{
    module::Module,
    nn::{
        LayerNorm,
        LayerNormConfig,
        attention::{
            MultiHeadAttention,
            MultiHeadAttentionConfig,
            MhaInput,
        },
    },
    tensor::{
        backend::Backend,
        Tensor,
    },
};

#[derive(Module, Debug)]
pub struct EncoderBlock<B: Backend> {
    attention: MultiHeadAttention<B>,

    norm1: LayerNorm<B>,
    norm2: LayerNorm<B>,

    feed_forward: FeedForward<B>,

    dropout: Dropout,
}
```

Initialize it:

```rust
impl<B: Backend> EncoderBlock<B> {
    pub fn new(
        d_model: usize,
        d_ff: usize,
        n_heads: usize,
        dropout: f64,
        device: &B::Device,
    ) -> Self {
        Self {
            attention: MultiHeadAttentionConfig::new(
                d_model,
                n_heads,
            )
            .with_dropout(dropout)
            .init(device),

            norm1: LayerNormConfig::new(d_model)
                .init(device),

            norm2: LayerNormConfig::new(d_model)
                .init(device),

            feed_forward: FeedForward::new(
                d_model,
                d_ff,
                dropout,
                device,
            ),

            dropout: DropoutConfig::new(dropout).init(),
        }
    }
}
```

Now comes the important part.

## 3. Encoder self-attention

For encoder self-attention:

```text
Q = x
K = x
V = x
```

Therefore:

```rust
pub fn forward(
    &self,
    x: Tensor<B, 3>,
) -> Tensor<B, 3> {

    // -----------------------------------
    // SELF ATTENTION
    // -----------------------------------

    let residual = x.clone();

    let attention_input = MhaInput::self_attn(x);

    let attention_output =
        self.attention.forward(attention_input);

    let x = attention_output.context;

    let x = self.dropout.forward(x);

    // -----------------------------------
    // RESIDUAL CONNECTION
    // -----------------------------------

    let x = residual + x;

    // -----------------------------------
    // ADD & NORM
    // -----------------------------------

    let x = self.norm1.forward(x);


    // -----------------------------------
    // FEED FORWARD
    // -----------------------------------

    let residual = x.clone();

    let x = self.feed_forward.forward(x);

    let x = self.dropout.forward(x);

    // -----------------------------------
    // RESIDUAL CONNECTION
    // -----------------------------------

    let x = residual + x;

    // -----------------------------------
    // ADD & NORM
    // -----------------------------------

    self.norm2.forward(x)
}
```

You can now directly map code to the diagram:

```text
Diagram                       Your Rust

Multi-Head Attention    →     attention.forward()

curved arrow            →     residual

Add                     →     residual + x

Norm                    →     norm1.forward()

Feed Forward            →     feed_forward.forward()

curved arrow            →     residual

Add                     →     residual + x

Norm                    →     norm2.forward()
```

That's essentially an encoder Transformer block.

---

# 4. Stack encoder blocks

A Transformer doesn't contain just one.

Suppose:

```text
n_layers = 6
```

We create:

```rust
#[derive(Module, Debug)]
pub struct Encoder<B: Backend> {
    blocks: Vec<EncoderBlock<B>>,
}
```

Initialization:

```rust
impl<B: Backend> Encoder<B> {
    pub fn new(
        n_layers: usize,
        d_model: usize,
        d_ff: usize,
        n_heads: usize,
        dropout: f64,
        device: &B::Device,
    ) -> Self {

        let blocks = (0..n_layers)
            .map(|_| {
                EncoderBlock::new(
                    d_model,
                    d_ff,
                    n_heads,
                    dropout,
                    device,
                )
            })
            .collect();

        Self { blocks }
    }
}
```

Forward:

```rust
pub fn forward(
    &self,
    mut x: Tensor<B, 3>,
) -> Tensor<B, 3> {

    for block in &self.blocks {
        x = block.forward(x);
    }

    x
}
```

So:

```text
Embedding
    │
    ▼
EncoderBlock 1
    │
    ▼
EncoderBlock 2
    │
    ▼
EncoderBlock 3
    │
    ▼
EncoderBlock 4
    │
    ▼
EncoderBlock 5
    │
    ▼
EncoderBlock 6
    │
    ▼
encoder memory
```

---

# 5. Decoder block

The decoder is more interesting.

It contains:

```text
Masked Self Attention
        ↓
Add & Norm
        ↓
Cross Attention  ◄──── encoder
        ↓
Add & Norm
        ↓
Feed Forward
        ↓
Add & Norm
```

Therefore:

```rust
#[derive(Module, Debug)]
pub struct DecoderBlock<B: Backend> {
    self_attention: MultiHeadAttention<B>,

    cross_attention: MultiHeadAttention<B>,

    feed_forward: FeedForward<B>,

    norm1: LayerNorm<B>,
    norm2: LayerNorm<B>,
    norm3: LayerNorm<B>,

    dropout: Dropout,
}
```

Initialize:

```rust
impl<B: Backend> DecoderBlock<B> {
    pub fn new(
        d_model: usize,
        d_ff: usize,
        n_heads: usize,
        dropout: f64,
        device: &B::Device,
    ) -> Self {
        Self {
            self_attention:
                MultiHeadAttentionConfig::new(
                    d_model,
                    n_heads,
                )
                .with_dropout(dropout)
                .init(device),

            cross_attention:
                MultiHeadAttentionConfig::new(
                    d_model,
                    n_heads,
                )
                .with_dropout(dropout)
                .init(device),

            feed_forward:
                FeedForward::new(
                    d_model,
                    d_ff,
                    dropout,
                    device,
                ),

            norm1:
                LayerNormConfig::new(d_model)
                    .init(device),

            norm2:
                LayerNormConfig::new(d_model)
                    .init(device),

            norm3:
                LayerNormConfig::new(d_model)
                    .init(device),

            dropout:
                DropoutConfig::new(dropout).init(),
        }
    }
}
```

---

# 6. Masked self-attention

This is where the decoder differs from the encoder.

Suppose the target is:

```text
<BOS> I love machine learning
```

When processing `"love"`, we cannot allow it to see:

```text
machine learning
```

So attention needs a causal mask:

```text
        K

        1   2   3   4

Q  1   ✓   X   X   X
   2   ✓   ✓   X   X
   3   ✓   ✓   ✓   X
   4   ✓   ✓   ✓   ✓
```

Conceptually:

```rust
let self_attention_input =
    MhaInput::self_attn(x)
        .mask_attn(causal_mask);
```

The exact mask helper/API can vary slightly across Burn releases, but conceptually the MHA receives:

```text
Q = decoder hidden state
K = decoder hidden state
V = decoder hidden state

+

causal attention mask
```

---

# 7. Decoder forward

Now we can write the important logic:

```rust
pub fn forward(
    &self,
    x: Tensor<B, 3>,
    memory: Tensor<B, 3>,
    causal_mask: Tensor<B, 3, Bool>,
) -> Tensor<B, 3> {

    // =========================================
    // 1. MASKED SELF ATTENTION
    // =========================================

    let residual = x.clone();

    let input =
        MhaInput::self_attn(x)
            .mask_attn(causal_mask);

    let output =
        self.self_attention.forward(input);

    let x = self.dropout.forward(output.context);

    let x = residual + x;

    let x = self.norm1.forward(x);


    // =========================================
    // 2. CROSS ATTENTION
    // =========================================

    let residual = x.clone();

    /*
        Query = decoder

        Key   = encoder memory
        Value = encoder memory
    */

    let input = MhaInput::new(
        x,
        memory.clone(),
        memory,
    );

    let output =
        self.cross_attention.forward(input);

    let x = self.dropout.forward(output.context);

    let x = residual + x;

    let x = self.norm2.forward(x);


    // =========================================
    // 3. FEED FORWARD
    // =========================================

    let residual = x.clone();

    let x = self.feed_forward.forward(x);

    let x = self.dropout.forward(x);

    let x = residual + x;

    self.norm3.forward(x)
}
```

This is worth staring at because it exposes one of the fundamental Transformer ideas.

You have two completely different attention operations:

```text
SELF ATTENTION

decoder ───► Q
decoder ───► K
decoder ───► V


CROSS ATTENTION

decoder ───► Q

encoder ───► K
encoder ───► V
```

Same attention algorithm.

Different sources for Q/K/V.

---

# 8. Decoder stack

Exactly like encoder:

```rust
#[derive(Module, Debug)]
pub struct Decoder<B: Backend> {
    blocks: Vec<DecoderBlock<B>>,
}
```

Conceptually:

```rust
pub fn forward(
    &self,
    mut x: Tensor<B, 3>,
    memory: Tensor<B, 3>,
    causal_mask: Tensor<B, 3, Bool>,
) -> Tensor<B, 3> {

    for block in &self.blocks {
        x = block.forward(
            x,
            memory.clone(),
            causal_mask.clone(),
        );
    }

    x
}
```

Now we have:

```text
target
  │
  ▼
DecoderBlock 1 ◄──── encoder memory
  │
  ▼
DecoderBlock 2 ◄──── encoder memory
  │
  ▼
DecoderBlock 3 ◄──── encoder memory
  │
 ...
  ▼
DecoderBlock N
```

---

# 9. Complete Transformer

Now the top-level model becomes very understandable:

```rust
#[derive(Module, Debug)]
pub struct Transformer<B: Backend> {
    src_embedding: Embedding<B>,
    tgt_embedding: Embedding<B>,

    src_position: PositionalEncoding<B>,
    tgt_position: PositionalEncoding<B>,

    encoder: Encoder<B>,
    decoder: Decoder<B>,

    lm_head: Linear<B>,
}
```

And:

```rust
impl<B: Backend> Transformer<B> {
    pub fn forward(
        &self,
        src_tokens: Tensor<B, 2, Int>,
        tgt_tokens: Tensor<B, 2, Int>,
        causal_mask: Tensor<B, 3, Bool>,
    ) -> Tensor<B, 3> {

        // =====================================
        // ENCODER
        // =====================================

        let src =
            self.src_embedding.forward(src_tokens);

        let src =
            self.src_position.forward(src);

        let memory =
            self.encoder.forward(src);


        // =====================================
        // DECODER
        // =====================================

        let tgt =
            self.tgt_embedding.forward(tgt_tokens);

        let tgt =
            self.tgt_position.forward(tgt);

        let hidden =
            self.decoder.forward(
                tgt,
                memory,
                causal_mask,
            );


        // =====================================
        // VOCABULARY PROJECTION
        // =====================================

        self.lm_head.forward(hidden)
    }
}
```

Output:

```text
[batch, target_sequence, vocabulary]
```

Those are logits.

---

# 10. Look at what you've actually built

You started with the intimidating diagram:

```text
                   Softmax
                      ↑
                    Linear
                      ↑
              ┌─────────────┐
              │   Decoder   │
              └──────▲──────┘
                     │
              encoder memory
                     │
              ┌──────┴──────┐
              │   Encoder   │
              └─────────────┘
```

But in Burn it's ultimately composition:

```text
Transformer
│
├── Embedding
│
├── Encoder
│    │
│    ├── EncoderBlock
│    │    ├── MultiHeadAttention
│    │    ├── LayerNorm
│    │    ├── FeedForward
│    │    │    ├── Linear
│    │    │    ├── GELU
│    │    │    └── Linear
│    │    └── LayerNorm
│    │
│    └── × N
│
├── Embedding
│
├── Decoder
│    │
│    ├── DecoderBlock
│    │    ├── Masked MultiHeadAttention
│    │    ├── LayerNorm
│    │    ├── Cross MultiHeadAttention
│    │    ├── LayerNorm
│    │    ├── FeedForward
│    │    └── LayerNorm
│    │
│    └── × N
│
└── Linear → vocabulary
```

And even `MultiHeadAttention` isn't magic. That's the **next box I'd open**.

You would remove Burn's `MultiHeadAttention` and implement:

```text
                         x
                         │
             ┌───────────┼───────────┐
             ▼           ▼           ▼
          Linear       Linear      Linear
             │           │           │
             Q           K           V
             │           │           │
             │           ▼           │
             │       transpose       │
             │           │           │
             └──────► matmul         │
                         │            │
                        /√dₖ          │
                         │            │
                        mask          │
                         │            │
                      softmax         │
                         │            │
                         └── matmul ◄─┘
                              │
                         concatenate
                              │
                            Linear
                              │
                              ▼
                            output
```

Once you've implemented **that** using Burn `Tensor` operations, you'll understand precisely what standard attention costs in memory and computation. Then FlashAttention stops looking like some mysterious algorithm: you can point directly at the `QKᵀ` materialization and ask, **"Why are we creating this enormous intermediate matrix at all?"**

And that's exactly the bridge from **learning Transformers → inference engineering → ola-rt**.

