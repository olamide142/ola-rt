use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig},
    tensor::{
        Device, Int, Tensor,
        activation::silu,
        module::{attention, linear as tensor_linear},
        ops::AttentionModuleOptions,
    },
};
use ola_kernel::{
    PagedAttentionConfig, PagedAttentionInput, PagedAttentionKernel, PagedKvBlocks, PagedKvPage,
};
use ola_runtime::KvBlockError;

use super::{LlamaConfig, kv_cache::LlamaKvCache};

/// Burn module whose field hierarchy mirrors Hugging Face Llama checkpoint keys.
///
/// Keeping `model.embed_tokens`, `model.layers.*`, `model.norm`, and `lm_head`
/// in the module tree is intentional: [`burn_store::PyTorchToBurnAdapter`]
/// can then handle linear transposition and normalization naming while the
/// checkpoint remains recognizable to researchers inspecting it.
#[derive(Module, Debug)]
pub struct LlamaForCausalLM {
    model: LlamaModel,
    lm_head: Linear,
    #[module(skip)]
    tied_embeddings: bool,
}

impl LlamaForCausalLM {
    /// Initializes the module structure before checkpoint parameters are loaded.
    pub fn init(config: &LlamaConfig, device: &Device) -> Self {
        Self {
            model: LlamaModel::init(config, device),
            lm_head: linear(config.hidden_size, config.vocab_size, false, device),
            tied_embeddings: config.tie_word_embeddings,
        }
    }

    fn project_logits(&self, hidden_states: Tensor<3>) -> Tensor<3> {
        if self.tied_embeddings {
            tensor_linear(
                hidden_states,
                self.model.embed_tokens.weight.val().transpose(),
                None,
            )
        } else {
            self.lm_head.forward(hidden_states)
        }
    }

    /// Computes logits for every token position.
    ///
    /// Input shape is `[batch, sequence]`; output shape is
    /// `[batch, sequence, vocabulary]`. The first implementation uses this
    /// full-sequence method for both prefill and decode so correctness can be
    /// established before introducing cache mutation.
    pub fn forward(&self, input_ids: Tensor<2, Int>) -> Tensor<3> {
        self.project_logits(self.model.forward(input_ids))
    }

    /// Processes a complete prompt and returns logits plus its populated cache.
    pub(crate) fn prefill(
        &self,
        input_ids: Tensor<2, Int>,
        cache: &mut LlamaKvCache,
    ) -> Result<Tensor<3>, KvBlockError> {
        let hidden_states = self.model.prefill(input_ids, cache)?;
        Ok(self.project_logits(hidden_states))
    }

    /// Processes a resumable prompt segment against the existing cache.
    pub(crate) fn prefill_chunk(
        &self,
        input_ids: Tensor<2, Int>,
        cache: &mut LlamaKvCache,
        kernel: &dyn PagedAttentionKernel,
    ) -> Result<Tensor<3>, KvBlockError> {
        Ok(self.project_logits(self.model.prefill_chunk(input_ids, cache, kernel)?))
    }

    /// Processes right-padded prompts in one tensor invocation.
    ///
    /// Only each row's real prefix is written to its request-local cache. A
    /// causal mask prevents the padded suffix from influencing real tokens.
    pub(crate) fn prefill_batch(
        &self,
        input_ids: Tensor<2, Int>,
        caches: &mut [&mut LlamaKvCache],
        sequence_lengths: &[usize],
    ) -> Result<Tensor<3>, KvBlockError> {
        Ok(self.project_logits(
            self.model
                .prefill_batch(input_ids, caches, sequence_lengths)?,
        ))
    }

    /// Processes variable-length resumable segments in one tensor path.
    pub(crate) fn prefill_chunk_batch(
        &self,
        input_ids: Tensor<2, Int>,
        caches: &mut [&mut LlamaKvCache],
        sequence_lengths: &[usize],
        kernel: &dyn PagedAttentionKernel,
    ) -> Result<Tensor<3>, KvBlockError> {
        Ok(self.project_logits(self.model.prefill_chunk_batch(
            input_ids,
            caches,
            sequence_lengths,
            kernel,
        )?))
    }

    /// Processes one new token using and extending an existing request cache.
    pub(crate) fn decode(
        &self,
        input_ids: Tensor<2, Int>,
        cache: &mut LlamaKvCache,
        kernel: &dyn PagedAttentionKernel,
    ) -> Result<Tensor<3>, KvBlockError> {
        Ok(self.project_logits(self.model.decode(input_ids, cache, kernel)?))
    }

    /// Processes one token for each request while preserving isolated caches.
    pub(crate) fn decode_batch(
        &self,
        input_ids: Tensor<2, Int>,
        caches: &mut [&mut LlamaKvCache],
        kernel: &dyn PagedAttentionKernel,
    ) -> Result<Tensor<3>, KvBlockError> {
        Ok(self.project_logits(self.model.decode_batch(input_ids, caches, kernel)?))
    }
}

#[derive(Module, Debug)]
struct LlamaModel {
    embed_tokens: Embedding,
    layers: Vec<LlamaDecoderLayer>,
    norm: RmsNorm,
}

impl LlamaModel {
    fn init(config: &LlamaConfig, device: &Device) -> Self {
        Self {
            embed_tokens: EmbeddingConfig::new(config.vocab_size, config.hidden_size).init(device),
            layers: (0..config.num_hidden_layers)
                .map(|_| LlamaDecoderLayer::init(config, device))
                .collect(),
            norm: RmsNormConfig::new(config.hidden_size)
                .with_epsilon(config.rms_norm_eps)
                .init(device),
        }
    }

    fn forward(&self, input_ids: Tensor<2, Int>) -> Tensor<3> {
        let mut hidden_states = self.embed_tokens.forward(input_ids);
        for layer in &self.layers {
            hidden_states = layer.forward(hidden_states);
        }
        self.norm.forward(hidden_states)
    }

    fn prefill(
        &self,
        input_ids: Tensor<2, Int>,
        cache: &mut LlamaKvCache,
    ) -> Result<Tensor<3>, KvBlockError> {
        let sequence_length = input_ids.dims()[1];
        cache.reserve(sequence_length)?;
        let mut hidden_states = self.embed_tokens.forward(input_ids);
        for (layer_index, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.prefill(hidden_states, cache, layer_index);
        }
        Ok(self.norm.forward(hidden_states))
    }

    fn prefill_batch(
        &self,
        input_ids: Tensor<2, Int>,
        caches: &mut [&mut LlamaKvCache],
        sequence_lengths: &[usize],
    ) -> Result<Tensor<3>, KvBlockError> {
        let [batch_size, padded_length] = input_ids.dims();
        debug_assert_eq!(batch_size, caches.len());
        debug_assert_eq!(batch_size, sequence_lengths.len());
        debug_assert!(sequence_lengths.iter().all(|length| *length > 0));
        debug_assert!(
            sequence_lengths
                .iter()
                .all(|length| *length <= padded_length)
        );
        LlamaKvCache::reserve_batch_lengths(caches, sequence_lengths)?;
        let mut hidden_states = self.embed_tokens.forward(input_ids);
        for (layer_index, layer) in self.layers.iter().enumerate() {
            hidden_states =
                layer.prefill_batch(hidden_states, caches, layer_index, sequence_lengths);
        }
        Ok(self.norm.forward(hidden_states))
    }

    fn prefill_chunk(
        &self,
        input_ids: Tensor<2, Int>,
        cache: &mut LlamaKvCache,
        kernel: &dyn PagedAttentionKernel,
    ) -> Result<Tensor<3>, KvBlockError> {
        let [batch_size, sequence_length] = input_ids.dims();
        debug_assert_eq!(batch_size, 1);
        let start_position = cache.tokens();

        // Reserve the complete segment before writing any layer data. A pool
        // capacity failure therefore leaves both block ownership and the
        // request's accepted-token cursor unchanged.
        cache.reserve(sequence_length)?;
        let mut outputs = Vec::with_capacity(sequence_length);
        for offset in 0..sequence_length {
            let position = start_position + offset;
            let token = input_ids.clone().slice([0..1, offset..offset + 1]);
            let mut hidden_states = self.embed_tokens.forward(token);
            for (layer_index, layer) in self.layers.iter().enumerate() {
                hidden_states =
                    layer.prefill_chunk(hidden_states, cache, layer_index, position, kernel);
            }
            outputs.push(self.norm.forward(hidden_states));
        }
        Ok(Tensor::cat(outputs, 1))
    }

    fn prefill_chunk_batch(
        &self,
        input_ids: Tensor<2, Int>,
        caches: &mut [&mut LlamaKvCache],
        sequence_lengths: &[usize],
        kernel: &dyn PagedAttentionKernel,
    ) -> Result<Tensor<3>, KvBlockError> {
        let [batch_size, padded_length] = input_ids.dims();
        debug_assert_eq!(batch_size, caches.len());
        debug_assert_eq!(batch_size, sequence_lengths.len());
        debug_assert!(sequence_lengths.iter().all(|length| *length > 0));
        debug_assert!(
            sequence_lengths
                .iter()
                .all(|length| *length <= padded_length)
        );
        let start_positions = caches
            .iter()
            .map(|cache| cache.tokens())
            .collect::<Vec<_>>();

        // Reserve every segment in one allocator transaction before any
        // request writes K/V state.
        LlamaKvCache::reserve_batch_lengths(caches, sequence_lengths)?;
        let mut final_hidden = (0..batch_size).map(|_| None).collect::<Vec<_>>();
        for offset in 0..padded_length {
            let active_indices = sequence_lengths
                .iter()
                .enumerate()
                .filter_map(|(index, length)| (offset < *length).then_some(index))
                .collect::<Vec<_>>();
            if active_indices.is_empty() {
                break;
            }
            let token_rows = active_indices
                .iter()
                .map(|index| {
                    input_ids
                        .clone()
                        .slice([*index..*index + 1, offset..offset + 1])
                })
                .collect::<Vec<_>>();
            let token_ids = Tensor::cat(token_rows, 0);
            let positions = active_indices
                .iter()
                .map(|index| start_positions[*index] + offset)
                .collect::<Vec<_>>();
            let active_caches = caches
                .iter_mut()
                .enumerate()
                .filter_map(|(index, cache)| {
                    active_indices.contains(&index).then_some(&mut **cache)
                })
                .collect::<Vec<_>>();
            let mut hidden_states = self.embed_tokens.forward(token_ids);
            for (layer_index, layer) in self.layers.iter().enumerate() {
                hidden_states = layer.prefill_chunk_batch(
                    hidden_states,
                    active_caches.as_slice(),
                    layer_index,
                    positions.as_slice(),
                    kernel,
                );
            }
            let hidden_states = self.norm.forward(hidden_states);
            for (row, index) in active_indices.into_iter().enumerate() {
                if offset + 1 == sequence_lengths[index] {
                    final_hidden[index] = Some(hidden_states.clone().slice([
                        row..row + 1,
                        0..1,
                        0..hidden_states.dims()[2],
                    ]));
                }
            }
        }
        Ok(Tensor::cat(
            final_hidden
                .into_iter()
                .map(|row| row.expect("every non-empty chunk has a final hidden row"))
                .collect(),
            0,
        ))
    }

    fn decode(
        &self,
        input_ids: Tensor<2, Int>,
        cache: &mut LlamaKvCache,
        kernel: &dyn PagedAttentionKernel,
    ) -> Result<Tensor<3>, KvBlockError> {
        debug_assert_eq!(input_ids.dims()[1], 1);
        let position = cache.tokens();
        cache.reserve(1)?;
        let mut hidden_states = self.embed_tokens.forward(input_ids);
        for (layer_index, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.decode(hidden_states, cache, layer_index, position, kernel);
        }
        Ok(self.norm.forward(hidden_states))
    }

    fn decode_batch(
        &self,
        input_ids: Tensor<2, Int>,
        caches: &mut [&mut LlamaKvCache],
        kernel: &dyn PagedAttentionKernel,
    ) -> Result<Tensor<3>, KvBlockError> {
        let [batch_size, sequence_length] = input_ids.dims();
        debug_assert_eq!(sequence_length, 1);
        debug_assert_eq!(batch_size, caches.len());
        let positions = caches
            .iter()
            .map(|cache| cache.tokens())
            .collect::<Vec<_>>();
        LlamaKvCache::reserve_batch(caches, 1)?;
        let mut hidden_states = self.embed_tokens.forward(input_ids);
        for (layer_index, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.decode_batch(
                hidden_states,
                caches,
                layer_index,
                positions.as_slice(),
                kernel,
            );
        }
        Ok(self.norm.forward(hidden_states))
    }
}

#[derive(Module, Debug)]
struct LlamaDecoderLayer {
    self_attn: LlamaAttention,
    mlp: LlamaMlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl LlamaDecoderLayer {
    fn init(config: &LlamaConfig, device: &Device) -> Self {
        let norm = || {
            RmsNormConfig::new(config.hidden_size)
                .with_epsilon(config.rms_norm_eps)
                .init(device)
        };
        Self {
            self_attn: LlamaAttention::init(config, device),
            mlp: LlamaMlp::init(config, device),
            input_layernorm: norm(),
            post_attention_layernorm: norm(),
        }
    }

    fn forward(&self, hidden_states: Tensor<3>) -> Tensor<3> {
        let residual = hidden_states.clone();
        let hidden_states = residual
            + self
                .self_attn
                .forward(self.input_layernorm.forward(hidden_states));
        let residual = hidden_states.clone();
        residual
            + self
                .mlp
                .forward(self.post_attention_layernorm.forward(hidden_states))
    }

    fn prefill(&self, hidden_states: Tensor<3>, cache: &LlamaKvCache, layer: usize) -> Tensor<3> {
        let residual = hidden_states.clone();
        let attention =
            self.self_attn
                .prefill(self.input_layernorm.forward(hidden_states), cache, layer);
        let hidden_states = residual + attention;
        let residual = hidden_states.clone();
        residual
            + self
                .mlp
                .forward(self.post_attention_layernorm.forward(hidden_states))
    }

    fn prefill_batch(
        &self,
        hidden_states: Tensor<3>,
        caches: &[&mut LlamaKvCache],
        layer: usize,
        sequence_lengths: &[usize],
    ) -> Tensor<3> {
        let residual = hidden_states.clone();
        let hidden_states = residual
            + self.self_attn.prefill_batch(
                self.input_layernorm.forward(hidden_states),
                caches,
                layer,
                sequence_lengths,
            );
        let residual = hidden_states.clone();
        residual
            + self
                .mlp
                .forward(self.post_attention_layernorm.forward(hidden_states))
    }

    fn prefill_chunk(
        &self,
        hidden_states: Tensor<3>,
        cache: &LlamaKvCache,
        layer: usize,
        position: usize,
        kernel: &dyn PagedAttentionKernel,
    ) -> Tensor<3> {
        let residual = hidden_states.clone();
        let hidden_states = residual
            + self.self_attn.prefill_chunk(
                self.input_layernorm.forward(hidden_states),
                cache,
                layer,
                position,
                kernel,
            );
        let residual = hidden_states.clone();
        residual
            + self
                .mlp
                .forward(self.post_attention_layernorm.forward(hidden_states))
    }

    fn prefill_chunk_batch(
        &self,
        hidden_states: Tensor<3>,
        caches: &[&mut LlamaKvCache],
        layer: usize,
        positions: &[usize],
        kernel: &dyn PagedAttentionKernel,
    ) -> Tensor<3> {
        let residual = hidden_states.clone();
        let hidden_states = residual
            + self.self_attn.prefill_chunk_batch(
                self.input_layernorm.forward(hidden_states),
                caches,
                layer,
                positions,
                kernel,
            );
        let residual = hidden_states.clone();
        residual
            + self
                .mlp
                .forward(self.post_attention_layernorm.forward(hidden_states))
    }

    fn decode(
        &self,
        hidden_states: Tensor<3>,
        cache: &LlamaKvCache,
        layer: usize,
        position: usize,
        kernel: &dyn PagedAttentionKernel,
    ) -> Tensor<3> {
        let residual = hidden_states.clone();
        let hidden_states = residual
            + self.self_attn.decode(
                self.input_layernorm.forward(hidden_states),
                cache,
                layer,
                position,
                kernel,
            );
        let residual = hidden_states.clone();
        residual
            + self
                .mlp
                .forward(self.post_attention_layernorm.forward(hidden_states))
    }

    fn decode_batch(
        &self,
        hidden_states: Tensor<3>,
        caches: &[&mut LlamaKvCache],
        layer: usize,
        positions: &[usize],
        kernel: &dyn PagedAttentionKernel,
    ) -> Tensor<3> {
        let residual = hidden_states.clone();
        let hidden_states = residual
            + self.self_attn.decode_batch(
                self.input_layernorm.forward(hidden_states),
                caches,
                layer,
                positions,
                kernel,
            );
        let residual = hidden_states.clone();
        residual
            + self
                .mlp
                .forward(self.post_attention_layernorm.forward(hidden_states))
    }
}

#[derive(Module, Debug)]
struct LlamaAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    #[module(skip)]
    rotary_embedding: LlamaRotaryEmbedding,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
}

impl LlamaAttention {
    fn init(config: &LlamaConfig, device: &Device) -> Self {
        let head_dim = config.attention_head_dim();
        let query_width = config.num_attention_heads * head_dim;
        let key_value_width = config.num_key_value_heads * head_dim;
        Self {
            q_proj: linear(
                config.hidden_size,
                query_width,
                config.attention_bias,
                device,
            ),
            k_proj: linear(
                config.hidden_size,
                key_value_width,
                config.attention_bias,
                device,
            ),
            v_proj: linear(
                config.hidden_size,
                key_value_width,
                config.attention_bias,
                device,
            ),
            o_proj: linear(
                query_width,
                config.hidden_size,
                config.attention_bias,
                device,
            ),
            rotary_embedding: LlamaRotaryEmbedding {
                head_dim,
                theta: config.rope_theta,
                position_scale: config
                    .rope_scaling_factor()
                    .expect("validated Llama config has valid RoPE scaling")
                    as f32,
            },
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            head_dim,
        }
    }

    fn forward(&self, hidden_states: Tensor<3>) -> Tensor<3> {
        let (query, key, value) = self.project(hidden_states);
        let query = self.rotary_embedding.forward(query, 0);
        let key = self.rotary_embedding.forward(key, 0);
        self.attend(query, key, value, true)
    }

    fn prefill(&self, hidden_states: Tensor<3>, cache: &LlamaKvCache, layer: usize) -> Tensor<3> {
        let (query, key, value) = self.project(hidden_states);
        let query = self.rotary_embedding.forward(query, 0);
        let key = self.rotary_embedding.forward(key, 0);
        let context = self.attend(query, key.clone(), value.clone(), true);
        cache.write_layer(layer, 0, key, value);
        context
    }

    fn prefill_batch(
        &self,
        hidden_states: Tensor<3>,
        caches: &[&mut LlamaKvCache],
        layer: usize,
        sequence_lengths: &[usize],
    ) -> Tensor<3> {
        let (query, key, value) = self.project(hidden_states);
        let query = self.rotary_embedding.forward(query, 0);
        let key = self.rotary_embedding.forward(key, 0);
        let context = self.attend(query, key.clone(), value.clone(), true);
        for (index, (&sequence_length, cache)) in
            sequence_lengths.iter().zip(caches.iter()).enumerate()
        {
            let key_row = key.clone().slice([
                index..index + 1,
                0..self.num_key_value_heads,
                0..sequence_length,
                0..self.head_dim,
            ]);
            let value_row = value.clone().slice([
                index..index + 1,
                0..self.num_key_value_heads,
                0..sequence_length,
                0..self.head_dim,
            ]);
            cache.write_layer(layer, 0, key_row, value_row);
        }
        context
    }

    fn decode(
        &self,
        hidden_states: Tensor<3>,
        cache: &LlamaKvCache,
        layer: usize,
        position: usize,
        kernel: &dyn PagedAttentionKernel,
    ) -> Tensor<3> {
        let (query, key, value) = self.project(hidden_states);
        let query = self.rotary_embedding.forward(query, position);
        let key = self.rotary_embedding.forward(key, position);
        cache.write_layer(layer, position, key, value);
        self.attend_paged_decode(query, cache.read_layer(layer), kernel)
    }

    fn prefill_chunk(
        &self,
        hidden_states: Tensor<3>,
        cache: &LlamaKvCache,
        layer: usize,
        position: usize,
        kernel: &dyn PagedAttentionKernel,
    ) -> Tensor<3> {
        let (query, key, value) = self.project(hidden_states);
        let query = self.rotary_embedding.forward(query, position);
        let key = self.rotary_embedding.forward(key, position);
        cache.write_layer(layer, position, key, value);
        self.attend_paged_decode(query, cache.read_layer_prefix(layer, position + 1), kernel)
    }

    fn prefill_chunk_batch(
        &self,
        hidden_states: Tensor<3>,
        caches: &[&mut LlamaKvCache],
        layer: usize,
        positions: &[usize],
        kernel: &dyn PagedAttentionKernel,
    ) -> Tensor<3> {
        let (query, key, value) = self.project(hidden_states);
        let batch_size = query.dims()[0];
        debug_assert_eq!(batch_size, caches.len());
        debug_assert_eq!(batch_size, positions.len());
        let mut inputs = Vec::with_capacity(batch_size);
        for index in 0..batch_size {
            let query = query.clone().slice([
                index..index + 1,
                0..self.num_attention_heads,
                0..1,
                0..self.head_dim,
            ]);
            let key = key.clone().slice([
                index..index + 1,
                0..self.num_key_value_heads,
                0..1,
                0..self.head_dim,
            ]);
            let value = value.clone().slice([
                index..index + 1,
                0..self.num_key_value_heads,
                0..1,
                0..self.head_dim,
            ]);
            let query = self.rotary_embedding.forward(query, positions[index]);
            let key = self.rotary_embedding.forward(key, positions[index]);
            caches[index].write_layer(layer, positions[index], key, value);
            inputs.push(self.paged_attention_input(
                query,
                caches[index].read_layer_prefix(layer, positions[index] + 1),
            ));
        }
        let contexts = kernel
            .execute_batch(inputs)
            .expect("the planned paged-attention provider must execute batched Llama input");
        self.project_attention_output(Tensor::cat(contexts, 0))
    }

    fn decode_batch(
        &self,
        hidden_states: Tensor<3>,
        caches: &[&mut LlamaKvCache],
        layer: usize,
        positions: &[usize],
        kernel: &dyn PagedAttentionKernel,
    ) -> Tensor<3> {
        let (query, key, value) = self.project(hidden_states);
        let batch_size = query.dims()[0];
        debug_assert_eq!(batch_size, caches.len());
        debug_assert_eq!(batch_size, positions.len());
        let mut inputs = Vec::with_capacity(batch_size);
        for index in 0..batch_size {
            let query = query.clone().slice([
                index..index + 1,
                0..self.num_attention_heads,
                0..1,
                0..self.head_dim,
            ]);
            let key = key.clone().slice([
                index..index + 1,
                0..self.num_key_value_heads,
                0..1,
                0..self.head_dim,
            ]);
            let value = value.clone().slice([
                index..index + 1,
                0..self.num_key_value_heads,
                0..1,
                0..self.head_dim,
            ]);
            let query = self.rotary_embedding.forward(query, positions[index]);
            let key = self.rotary_embedding.forward(key, positions[index]);
            caches[index].write_layer(layer, positions[index], key, value);
            inputs.push(self.paged_attention_input(query, caches[index].read_layer(layer)));
        }
        let contexts = kernel
            .execute_batch(inputs)
            .expect("the planned paged-attention provider must execute batched Llama input");
        self.project_attention_output(Tensor::cat(contexts, 0))
    }

    fn project(&self, hidden_states: Tensor<3>) -> (Tensor<4>, Tensor<4>, Tensor<4>) {
        let [batch_size, sequence_length, _] = hidden_states.dims();
        let query = self
            .q_proj
            .forward(hidden_states.clone())
            .reshape([
                batch_size,
                sequence_length,
                self.num_attention_heads,
                self.head_dim,
            ])
            .swap_dims(1, 2);
        let key = self
            .k_proj
            .forward(hidden_states.clone())
            .reshape([
                batch_size,
                sequence_length,
                self.num_key_value_heads,
                self.head_dim,
            ])
            .swap_dims(1, 2);
        let value = self
            .v_proj
            .forward(hidden_states)
            .reshape([
                batch_size,
                sequence_length,
                self.num_key_value_heads,
                self.head_dim,
            ])
            .swap_dims(1, 2);
        (query, key, value)
    }

    fn attend(
        &self,
        query: Tensor<4>,
        key: Tensor<4>,
        value: Tensor<4>,
        is_causal: bool,
    ) -> Tensor<3> {
        let groups = self.num_attention_heads / self.num_key_value_heads;
        let key = repeat_key_value(key, groups);
        let value = repeat_key_value(value, groups);
        let context = attention(
            query,
            key,
            value,
            None,
            None,
            AttentionModuleOptions {
                is_causal,
                ..Default::default()
            },
        );
        self.project_attention_output(context)
    }

    /// Computes single-token attention directly over logical cache pages.
    ///
    /// A numerically stable online softmax combines one page at a time. K/V
    /// pages are never concatenated into a sequence-sized temporary.
    fn attend_paged_decode(
        &self,
        query: Tensor<4>,
        cache: (Vec<(Tensor<4>, Tensor<4>)>, PagedKvBlocks),
        kernel: &dyn PagedAttentionKernel,
    ) -> Tensor<3> {
        self.project_attention_output(self.attend_paged_decode_context(query, cache, kernel))
    }

    fn attend_paged_decode_context(
        &self,
        query: Tensor<4>,
        cache: (Vec<(Tensor<4>, Tensor<4>)>, PagedKvBlocks),
        kernel: &dyn PagedAttentionKernel,
    ) -> Tensor<4> {
        let input = self.paged_attention_input(query, cache);
        kernel
            .execute(input)
            .expect("the planned paged-attention provider must execute validated Llama input")
    }

    fn paged_attention_input(
        &self,
        query: Tensor<4>,
        cache: (Vec<(Tensor<4>, Tensor<4>)>, PagedKvBlocks),
    ) -> PagedAttentionInput {
        let (pages, blocks) = cache;
        let config = PagedAttentionConfig::new(
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
        )
        .expect("validated Llama dimensions must form a paged-attention configuration");
        let pages = pages
            .into_iter()
            .map(|(key, value)| PagedKvPage::new(key, value))
            .collect();
        let input = PagedAttentionInput::new(query, pages, config)
            .expect("Llama cache pages must satisfy the paged-attention contract")
            .with_blocks(blocks)
            .expect("Llama physical cache must satisfy the paged-attention contract");
        input
    }

    fn project_attention_output(&self, context: Tensor<4>) -> Tensor<3> {
        let [batch_size, _, sequence_length, _] = context.dims();
        let context = context.swap_dims(1, 2).reshape([
            batch_size,
            sequence_length,
            self.num_attention_heads * self.head_dim,
        ]);
        self.o_proj.forward(context)
    }
}

/// Repeats each key/value head contiguously to match the number of query heads.
///
/// Repeating a rank-four tensor directly would tile all heads (`0,1,0,1`).
/// Inserting a group dimension first produces the GQA order expected by Llama
/// (`0,0,1,1`).
fn repeat_key_value(tensor: Tensor<4>, groups: usize) -> Tensor<4> {
    if groups == 1 {
        return tensor;
    }
    let [batch, key_value_heads, sequence, head_dim] = tensor.dims();
    tensor
        .reshape([batch, key_value_heads, 1, sequence, head_dim])
        .repeat_dim(2, groups)
        .reshape([batch, key_value_heads * groups, sequence, head_dim])
}

#[derive(Clone, Debug)]
struct LlamaRotaryEmbedding {
    head_dim: usize,
    theta: f64,
    position_scale: f32,
}

impl LlamaRotaryEmbedding {
    /// Applies Hugging Face Llama's split-half RoPE convention.
    ///
    /// Burn's general `RotaryEncoding` rotates adjacent pairs. Llama instead
    /// pairs the first half of each head with the second half, so substituting
    /// the generic module would preserve shapes while producing wrong logits.
    fn forward(&self, tensor: Tensor<4>, position_offset: usize) -> Tensor<4> {
        let [batch, heads, sequence, head_dim] = tensor.dims();
        debug_assert_eq!(head_dim, self.head_dim);
        let device = tensor.device();
        let half = head_dim / 2;

        let exponent = Tensor::<1, Int>::arange_step(0..head_dim as i64, 2, &device)
            .float()
            .div_scalar(head_dim as f32);
        let inverse_frequency = exponent.mul_scalar(self.theta.ln() as f32).exp().recip();
        let positions = Tensor::<1, Int>::arange(
            position_offset as i64..(position_offset + sequence) as i64,
            &device,
        )
        .float()
        .div_scalar(self.position_scale)
        .reshape([sequence, 1]);
        let frequencies = positions * inverse_frequency.reshape([1, half]);
        let frequencies = Tensor::cat(vec![frequencies.clone(), frequencies], 1);
        let cosine = frequencies
            .clone()
            .cos()
            .reshape([1, 1, sequence, head_dim]);
        let sine = frequencies.sin().reshape([1, 1, sequence, head_dim]);

        let first = tensor
            .clone()
            .slice([0..batch, 0..heads, 0..sequence, 0..half]);
        let second = tensor
            .clone()
            .slice([0..batch, 0..heads, 0..sequence, half..head_dim]);
        let rotated = Tensor::cat(vec![-second, first], 3);
        tensor * cosine + rotated * sine
    }
}

#[derive(Module, Debug)]
struct LlamaMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl LlamaMlp {
    fn init(config: &LlamaConfig, device: &Device) -> Self {
        Self {
            gate_proj: linear(
                config.hidden_size,
                config.intermediate_size,
                config.mlp_bias,
                device,
            ),
            up_proj: linear(
                config.hidden_size,
                config.intermediate_size,
                config.mlp_bias,
                device,
            ),
            down_proj: linear(
                config.intermediate_size,
                config.hidden_size,
                config.mlp_bias,
                device,
            ),
        }
    }

    fn forward(&self, hidden_states: Tensor<3>) -> Tensor<3> {
        let gated = silu(self.gate_proj.forward(hidden_states.clone()));
        self.down_proj
            .forward(gated * self.up_proj.forward(hidden_states))
    }
}

fn linear(input: usize, output: usize, bias: bool, device: &Device) -> Linear {
    LinearConfig::new(input, output)
        .with_bias(bias)
        .init(device)
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use burn::tensor::TensorData;

    use super::*;
    use crate::llama::{
        LlamaKvCacheConfig,
        kv_cache::{LlamaKvCache, LlamaKvPool},
    };

    fn tiny_config() -> LlamaConfig {
        LlamaConfig {
            model_type: "llama".into(),
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 24,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            max_position_embeddings: 64,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            head_dim: None,
            hidden_act: "silu".into(),
            attention_bias: false,
            mlp_bias: false,
            tie_word_embeddings: false,
            rope_scaling: None,
            bos_token_id: Some(1),
            eos_token_id: vec![2],
        }
    }

    #[test]
    fn forward_preserves_batch_and_sequence_and_projects_to_vocabulary() {
        let device = Default::default();
        let model = LlamaForCausalLM::init(&tiny_config(), &device);
        let input = Tensor::from_data(TensorData::new(vec![1_i64, 5, 9], [1, 3]), &device);

        let logits = model.forward(input);

        assert_eq!(logits.dims(), [1, 3, 32]);
    }

    #[test]
    fn tied_output_projection_reuses_input_embedding_matrix() {
        let device = Default::default();
        let mut config = tiny_config();
        config.tie_word_embeddings = true;
        let model = LlamaForCausalLM::init(&config, &device);
        let input = Tensor::from_data(TensorData::new(vec![1_i64, 5, 9], [1, 3]), &device);
        let hidden = model.model.forward(input.clone());

        let expected = tensor_linear(
            hidden,
            model.model.embed_tokens.weight.val().transpose(),
            None,
        );

        assert_eq!(model.forward(input).into_data(), expected.into_data());
    }

    #[test]
    fn linear_rope_scaling_changes_nonzero_rotary_positions() {
        let device = Default::default();
        let mut config = tiny_config();
        config.rope_scaling = Some(serde_json::json!({
            "type": "linear",
            "factor": 2.0,
        }));
        let model = LlamaForCausalLM::init(&config, &device);
        let rotary = &model.model.layers[0].self_attn.rotary_embedding;
        let input = Tensor::from_data(
            TensorData::new(vec![1.0_f32, 2.0, 3.0, 4.0], [1, 1, 1, 4]),
            &device,
        );

        let scaled = rotary.forward(input.clone(), 1).into_data();
        let unscaled = LlamaRotaryEmbedding {
            head_dim: 4,
            theta: config.rope_theta,
            position_scale: 1.0,
        }
        .forward(input, 1)
        .into_data();

        assert_ne!(scaled, unscaled);
    }

    #[test]
    fn configured_projection_biases_are_present_in_attention_and_mlp() {
        let device = Default::default();
        let mut config = tiny_config();
        config.attention_bias = true;
        config.mlp_bias = true;
        let model = LlamaForCausalLM::init(&config, &device);
        let layer = &model.model.layers[0];

        assert!(layer.self_attn.q_proj.bias.is_some());
        assert!(layer.self_attn.k_proj.bias.is_some());
        assert!(layer.self_attn.v_proj.bias.is_some());
        assert!(layer.self_attn.o_proj.bias.is_some());
        assert!(layer.mlp.gate_proj.bias.is_some());
        assert!(layer.mlp.up_proj.bias.is_some());
        assert!(layer.mlp.down_proj.bias.is_some());
    }

    #[test]
    fn grouped_query_expansion_repeats_each_head_contiguously() {
        let device = Default::default();
        let tensor =
            Tensor::<4>::from_data(TensorData::new(vec![1.0_f32, 2.0], [1, 2, 1, 1]), &device);

        let repeated = repeat_key_value(tensor, 2).into_data();

        assert_eq!(
            repeated.try_to_vec::<f32>().unwrap(),
            vec![1.0, 1.0, 2.0, 2.0]
        );
    }

    #[test]
    fn paged_decode_matches_full_sequence_logits_across_page_sizes() {
        let device = Default::default();
        let config = tiny_config();
        let model = LlamaForCausalLM::init(&config, &device);
        let kernel = ola_kernel::PortablePagedAttention::new();
        for tokens_per_block in [1, 2, 4, 16] {
            let pool = LlamaKvPool::new(
                &config,
                LlamaKvCacheConfig::new(config.max_position_embeddings, tokens_per_block),
                device.clone(),
            )
            .unwrap();
            let mut cache = LlamaKvCache::allocate(Arc::new(Mutex::new(pool))).unwrap();
            let mut all_tokens = vec![1_i64, 5, 9];
            let prompt = Tensor::from_data(
                TensorData::new(all_tokens.clone(), [1, all_tokens.len()]),
                &device,
            );

            let prefill_logits = model.prefill(prompt, &mut cache).unwrap();
            let full_prefill = model.forward(Tensor::from_data(
                TensorData::new(all_tokens.clone(), [1, all_tokens.len()]),
                &device,
            ));
            assert_logits_close(prefill_logits, full_prefill, 2e-5);

            for token in [7_i64, 11] {
                all_tokens.push(token);
                let cached = model
                    .decode(
                        Tensor::from_data(TensorData::new(vec![token], [1, 1]), &device),
                        &mut cache,
                        &kernel,
                    )
                    .unwrap();
                let sequence_length = all_tokens.len();
                let full = model
                    .forward(Tensor::from_data(
                        TensorData::new(all_tokens.clone(), [1, sequence_length]),
                        &device,
                    ))
                    .slice([0..1, sequence_length - 1..sequence_length, 0..32]);
                assert_logits_close(cached, full, 2e-5);
                assert_eq!(cache.tokens(), sequence_length);
            }

            // Each physical token slot is 128 bytes across both layers, K/V
            // tensors, two KV heads, four elements, and F32.
            let physical_slots = 5_usize.div_ceil(tokens_per_block) * tokens_per_block;
            assert_eq!(cache.bytes(), physical_slots as u64 * 128);
        }
    }

    fn assert_logits_close(left: Tensor<3>, right: Tensor<3>, tolerance: f32) {
        let left = left.into_data().try_to_vec::<f32>().unwrap();
        let right = right.into_data().try_to_vec::<f32>().unwrap();
        assert_eq!(left.len(), right.len());
        for (index, (left, right)) in left.into_iter().zip(right).enumerate() {
            let difference = (left - right).abs();
            assert!(
                difference <= tolerance,
                "logit {index} differs by {difference}: cached={left}, full={right}"
            );
        }
    }
}
