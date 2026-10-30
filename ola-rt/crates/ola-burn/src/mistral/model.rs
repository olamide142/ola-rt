use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig},
    tensor::{
        Bool, Device, Int, Tensor, activation::silu, module::attention, ops::AttentionModuleOptions,
    },
};
use ola_kernel::{
    PagedAttentionConfig, PagedAttentionInput, PagedAttentionKernel, PagedKvBlocks, PagedKvPage,
};
use ola_runtime::KvBlockError;

use super::{MistralConfig, kv_cache::MistralKvCache};

/// Burn module whose field hierarchy mirrors Hugging Face Mistral checkpoint keys.
///
/// Keeping `model.embed_tokens`, `model.layers.*`, `model.norm`, and `lm_head`
/// in the module tree allows [`burn_store::PyTorchToBurnAdapter`] to handle
/// linear transposition and normalization naming while the checkpoint remains
/// recognizable to researchers inspecting it.
#[derive(Module, Debug)]
pub struct MistralForCausalLM {
    model: MistralModel,
    lm_head: Linear,
}

impl MistralForCausalLM {
    /// Initializes the module structure before checkpoint parameters are loaded.
    pub fn init(config: &MistralConfig, device: &Device) -> Self {
        Self {
            model: MistralModel::init(config, device),
            lm_head: linear(config.hidden_size, config.vocab_size, device),
        }
    }

    /// Computes logits for every token position.
    ///
    /// Input shape is `[batch, sequence]`; output shape is
    /// `[batch, sequence, vocabulary]`.
    pub fn forward(&self, input_ids: Tensor<2, Int>) -> Tensor<3> {
        self.lm_head.forward(self.model.forward(input_ids))
    }

    /// Processes a complete prompt and returns logits plus its populated cache.
    pub(crate) fn prefill(
        &self,
        input_ids: Tensor<2, Int>,
        cache: &mut MistralKvCache,
    ) -> Result<Tensor<3>, KvBlockError> {
        let hidden_states = self.model.prefill(input_ids, cache)?;
        Ok(self.lm_head.forward(hidden_states))
    }

    /// Processes a resumable prompt segment against the existing cache.
    pub(crate) fn prefill_chunk(
        &self,
        input_ids: Tensor<2, Int>,
        cache: &mut MistralKvCache,
        kernel: &dyn PagedAttentionKernel,
    ) -> Result<Tensor<3>, KvBlockError> {
        Ok(self
            .lm_head
            .forward(self.model.prefill_chunk(input_ids, cache, kernel)?))
    }

    /// Processes right-padded prompts in one tensor invocation.
    pub(crate) fn prefill_batch(
        &self,
        input_ids: Tensor<2, Int>,
        caches: &mut [&mut MistralKvCache],
        sequence_lengths: &[usize],
    ) -> Result<Tensor<3>, KvBlockError> {
        Ok(self.lm_head.forward(
            self.model
                .prefill_batch(input_ids, caches, sequence_lengths)?,
        ))
    }

    /// Processes variable-length resumable segments in one tensor path.
    pub(crate) fn prefill_chunk_batch(
        &self,
        input_ids: Tensor<2, Int>,
        caches: &mut [&mut MistralKvCache],
        sequence_lengths: &[usize],
        kernel: &dyn PagedAttentionKernel,
    ) -> Result<Tensor<3>, KvBlockError> {
        Ok(self.lm_head.forward(self.model.prefill_chunk_batch(
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
        cache: &mut MistralKvCache,
        kernel: &dyn PagedAttentionKernel,
    ) -> Result<Tensor<3>, KvBlockError> {
        Ok(self
            .lm_head
            .forward(self.model.decode(input_ids, cache, kernel)?))
    }

    /// Processes one token for each request while preserving isolated caches.
    pub(crate) fn decode_batch(
        &self,
        input_ids: Tensor<2, Int>,
        caches: &mut [&mut MistralKvCache],
        kernel: &dyn PagedAttentionKernel,
    ) -> Result<Tensor<3>, KvBlockError> {
        Ok(self
            .lm_head
            .forward(self.model.decode_batch(input_ids, caches, kernel)?))
    }
}

#[derive(Module, Debug)]
struct MistralModel {
    embed_tokens: Embedding,
    layers: Vec<MistralDecoderLayer>,
    norm: RmsNorm,
}

impl MistralModel {
    fn init(config: &MistralConfig, device: &Device) -> Self {
        Self {
            embed_tokens: EmbeddingConfig::new(config.vocab_size, config.hidden_size).init(device),
            layers: (0..config.num_hidden_layers)
                .map(|_| MistralDecoderLayer::init(config, device))
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
        cache: &mut MistralKvCache,
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
        caches: &mut [&mut MistralKvCache],
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
        MistralKvCache::reserve_batch_lengths(caches, sequence_lengths)?;
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
        cache: &mut MistralKvCache,
        kernel: &dyn PagedAttentionKernel,
    ) -> Result<Tensor<3>, KvBlockError> {
        let [batch_size, sequence_length] = input_ids.dims();
        debug_assert_eq!(batch_size, 1);
        let start_position = cache.tokens();

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
        caches: &mut [&mut MistralKvCache],
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

        MistralKvCache::reserve_batch_lengths(caches, sequence_lengths)?;
        let mut hidden_states = self.embed_tokens.forward(input_ids);
        for (layer_index, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.prefill_chunk_batch(
                hidden_states,
                caches,
                layer_index,
                start_positions.as_slice(),
                sequence_lengths,
                kernel,
            );
        }
        let hidden_states = self.norm.forward(hidden_states);
        Ok(Tensor::cat(
            sequence_lengths
                .iter()
                .enumerate()
                .map(|(index, length)| {
                    hidden_states.clone().slice([
                        index..index + 1,
                        length - 1..*length,
                        0..hidden_states.dims()[2],
                    ])
                })
                .collect(),
            0,
        ))
    }

    fn decode(
        &self,
        input_ids: Tensor<2, Int>,
        cache: &mut MistralKvCache,
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
        caches: &mut [&mut MistralKvCache],
        kernel: &dyn PagedAttentionKernel,
    ) -> Result<Tensor<3>, KvBlockError> {
        let [batch_size, sequence_length] = input_ids.dims();
        debug_assert_eq!(sequence_length, 1);
        debug_assert_eq!(batch_size, caches.len());
        let positions = caches
            .iter()
            .map(|cache| cache.tokens())
            .collect::<Vec<_>>();
        MistralKvCache::reserve_batch(caches, 1)?;
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
struct MistralDecoderLayer {
    self_attn: MistralAttention,
    mlp: MistralMlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl MistralDecoderLayer {
    fn init(config: &MistralConfig, device: &Device) -> Self {
        let norm = || {
            RmsNormConfig::new(config.hidden_size)
                .with_epsilon(config.rms_norm_eps)
                .init(device)
        };
        Self {
            self_attn: MistralAttention::init(config, device),
            mlp: MistralMlp::init(config, device),
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

    fn prefill(&self, hidden_states: Tensor<3>, cache: &MistralKvCache, layer: usize) -> Tensor<3> {
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
        caches: &[&mut MistralKvCache],
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
        cache: &MistralKvCache,
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
        caches: &[&mut MistralKvCache],
        layer: usize,
        start_positions: &[usize],
        sequence_lengths: &[usize],
        kernel: &dyn PagedAttentionKernel,
    ) -> Tensor<3> {
        let residual = hidden_states.clone();
        let hidden_states = residual
            + self.self_attn.prefill_chunk_batch(
                self.input_layernorm.forward(hidden_states),
                caches,
                layer,
                start_positions,
                sequence_lengths,
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
        cache: &MistralKvCache,
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
        caches: &[&mut MistralKvCache],
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
struct MistralAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    #[module(skip)]
    rotary_embedding: MistralRotaryEmbedding,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    sliding_window: Option<usize>,
}

impl MistralAttention {
    fn init(config: &MistralConfig, device: &Device) -> Self {
        let head_dim = config.attention_head_dim();
        let query_width = config.num_attention_heads * head_dim;
        let key_value_width = config.num_key_value_heads * head_dim;
        Self {
            q_proj: linear(config.hidden_size, query_width, device),
            k_proj: linear(config.hidden_size, key_value_width, device),
            v_proj: linear(config.hidden_size, key_value_width, device),
            o_proj: linear(query_width, config.hidden_size, device),
            rotary_embedding: MistralRotaryEmbedding {
                head_dim,
                theta: config.rope_theta,
            },
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            head_dim,
            sliding_window: config.sliding_window,
        }
    }

    fn forward(&self, hidden_states: Tensor<3>) -> Tensor<3> {
        let (query, key, value) = self.project(hidden_states);
        let query = self.rotary_embedding.forward(query, 0);
        let key = self.rotary_embedding.forward(key, 0);
        self.attend(query, key, value, true)
    }

    fn prefill(&self, hidden_states: Tensor<3>, cache: &MistralKvCache, layer: usize) -> Tensor<3> {
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
        caches: &[&mut MistralKvCache],
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

    fn read_layer_for_decode(
        &self,
        cache: &MistralKvCache,
        layer: usize,
        token_count: usize,
    ) -> (Vec<(Tensor<4>, Tensor<4>)>, Option<PagedKvBlocks>) {
        let start = match self.sliding_window {
            Some(window) => token_count.saturating_sub(window),
            None => 0,
        };
        if start == 0 {
            let (pages, blocks) = cache.read_layer_prefix(layer, token_count);
            (pages, Some(blocks))
        } else {
            cache.read_layer_window(layer, start, token_count)
        }
    }

    fn decode(
        &self,
        hidden_states: Tensor<3>,
        cache: &MistralKvCache,
        layer: usize,
        position: usize,
        kernel: &dyn PagedAttentionKernel,
    ) -> Tensor<3> {
        let (query, key, value) = self.project(hidden_states);
        let query = self.rotary_embedding.forward(query, position);
        let key = self.rotary_embedding.forward(key, position);
        cache.write_layer(layer, position, key, value);
        self.attend_paged_decode(query, cache.read_layer(layer, self.sliding_window), kernel)
    }

    fn prefill_chunk(
        &self,
        hidden_states: Tensor<3>,
        cache: &MistralKvCache,
        layer: usize,
        position: usize,
        kernel: &dyn PagedAttentionKernel,
    ) -> Tensor<3> {
        let (query, key, value) = self.project(hidden_states);
        let query = self.rotary_embedding.forward(query, position);
        let key = self.rotary_embedding.forward(key, position);
        cache.write_layer(layer, position, key, value);
        let cache_view = self.read_layer_for_decode(cache, layer, position + 1);
        self.attend_paged_decode(query, cache_view, kernel)
    }

    fn prefill_chunk_batch(
        &self,
        hidden_states: Tensor<3>,
        caches: &[&mut MistralKvCache],
        layer: usize,
        start_positions: &[usize],
        sequence_lengths: &[usize],
        kernel: &dyn PagedAttentionKernel,
    ) -> Tensor<3> {
        let (query, key, value) = self.project(hidden_states);
        let batch_size = query.dims()[0];
        debug_assert_eq!(batch_size, caches.len());
        debug_assert_eq!(batch_size, start_positions.len());
        debug_assert_eq!(batch_size, sequence_lengths.len());
        let mut inputs = Vec::with_capacity(sequence_lengths.iter().sum());
        for index in 0..batch_size {
            let length = sequence_lengths[index];
            let query_row = self.rotary_embedding.forward(
                query.clone().slice([
                    index..index + 1,
                    0..self.num_attention_heads,
                    0..length,
                    0..self.head_dim,
                ]),
                start_positions[index],
            );
            let key_row = self.rotary_embedding.forward(
                key.clone().slice([
                    index..index + 1,
                    0..self.num_key_value_heads,
                    0..length,
                    0..self.head_dim,
                ]),
                start_positions[index],
            );
            let value_row = value.clone().slice([
                index..index + 1,
                0..self.num_key_value_heads,
                0..length,
                0..self.head_dim,
            ]);
            caches[index].write_layer(layer, start_positions[index], key_row, value_row);
            for offset in 0..length {
                let cache_view = self.read_layer_for_decode(
                    caches[index],
                    layer,
                    start_positions[index] + offset + 1,
                );
                inputs.push(self.paged_attention_input(
                    query_row.clone().slice([
                        0..1,
                        0..self.num_attention_heads,
                        offset..offset + 1,
                        0..self.head_dim,
                    ]),
                    cache_view,
                ));
            }
        }
        let contexts = kernel
            .execute_batch(inputs)
            .expect("the planned paged-attention provider must execute batched Mistral input");
        let projected = self.project_attention_output(Tensor::cat(contexts, 0));
        let [_, _, hidden_size] = projected.dims();
        let mut output = Tensor::zeros(
            [
                batch_size,
                sequence_lengths.iter().copied().max().unwrap(),
                hidden_size,
            ],
            &projected.device(),
        );
        let mut row = 0;
        for (index, &length) in sequence_lengths.iter().enumerate() {
            for offset in 0..length {
                output = output.slice_assign(
                    [index..index + 1, offset..offset + 1, 0..hidden_size],
                    projected
                        .clone()
                        .slice([row..row + 1, 0..1, 0..hidden_size]),
                );
                row += 1;
            }
        }
        output
    }

    fn decode_batch(
        &self,
        hidden_states: Tensor<3>,
        caches: &[&mut MistralKvCache],
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
            let cache_view = caches[index].read_layer(layer, self.sliding_window);
            inputs.push(self.paged_attention_input(query, cache_view));
        }
        let contexts = kernel
            .execute_batch(inputs)
            .expect("the planned paged-attention provider must execute batched Mistral input");
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
        let [_, _, sequence_length, _] = query.dims();
        let mask = if is_causal {
            if let Some(window) = self.sliding_window {
                if sequence_length > window {
                    Some(sliding_window_mask(
                        sequence_length,
                        window,
                        &query.device(),
                    ))
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };
        let context = attention(
            query,
            key,
            value,
            mask,
            None,
            AttentionModuleOptions {
                is_causal,
                ..Default::default()
            },
        );
        self.project_attention_output(context)
    }

    fn attend_paged_decode(
        &self,
        query: Tensor<4>,
        cache: (Vec<(Tensor<4>, Tensor<4>)>, Option<PagedKvBlocks>),
        kernel: &dyn PagedAttentionKernel,
    ) -> Tensor<3> {
        self.project_attention_output(self.attend_paged_decode_context(query, cache, kernel))
    }

    fn attend_paged_decode_context(
        &self,
        query: Tensor<4>,
        cache: (Vec<(Tensor<4>, Tensor<4>)>, Option<PagedKvBlocks>),
        kernel: &dyn PagedAttentionKernel,
    ) -> Tensor<4> {
        let input = self.paged_attention_input(query, cache);
        kernel
            .execute(input)
            .expect("the planned paged-attention provider must execute validated Mistral input")
    }

    fn paged_attention_input(
        &self,
        query: Tensor<4>,
        cache: (Vec<(Tensor<4>, Tensor<4>)>, Option<PagedKvBlocks>),
    ) -> PagedAttentionInput {
        let (pages, blocks) = cache;
        let config = PagedAttentionConfig::new(
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
        )
        .expect("validated Mistral dimensions must form a paged-attention configuration");
        let pages = pages
            .into_iter()
            .map(|(key, value)| PagedKvPage::new(key, value))
            .collect();
        let mut input = PagedAttentionInput::new(query, pages, config)
            .expect("Mistral cache pages must satisfy the paged-attention contract");
        if let Some(blocks) = blocks {
            input = input
                .with_blocks(blocks)
                .expect("Mistral physical cache must satisfy the paged-attention contract");
        }
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

/// Creates a boolean mask that masks out tokens outside the sliding window.
///
/// In Burn attention, `true` indicates a masked-out position. Combined with
/// causal masking (`is_causal = true`), token `i` attends to `j` if and only if
/// `j <= i` and `i - j < window`.
fn sliding_window_mask(sequence_length: usize, window: usize, device: &Device) -> Tensor<4, Bool> {
    let rows = Tensor::<1, Int>::arange(0..sequence_length as i64, device).reshape([
        1,
        1,
        sequence_length,
        1,
    ]);
    let cols = Tensor::<1, Int>::arange(0..sequence_length as i64, device).reshape([
        1,
        1,
        1,
        sequence_length,
    ]);
    (rows - cols).greater_equal_elem(window as i64)
}

/// Repeats each key/value head contiguously to match the number of query heads.
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
struct MistralRotaryEmbedding {
    head_dim: usize,
    theta: f64,
}

impl MistralRotaryEmbedding {
    /// Applies Hugging Face Mistral's split-half RoPE convention.
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
struct MistralMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl MistralMlp {
    fn init(config: &MistralConfig, device: &Device) -> Self {
        Self {
            gate_proj: linear(config.hidden_size, config.intermediate_size, device),
            up_proj: linear(config.hidden_size, config.intermediate_size, device),
            down_proj: linear(config.intermediate_size, config.hidden_size, device),
        }
    }

    fn forward(&self, hidden_states: Tensor<3>) -> Tensor<3> {
        let gated = silu(self.gate_proj.forward(hidden_states.clone()));
        self.down_proj
            .forward(gated * self.up_proj.forward(hidden_states))
    }
}

fn linear(input: usize, output: usize, device: &Device) -> Linear {
    LinearConfig::new(input, output)
        .with_bias(false)
        .init(device)
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use burn::tensor::TensorData;

    use super::*;
    use crate::mistral::{
        MistralKvCacheConfig,
        kv_cache::{MistralKvCache, MistralKvPool},
    };

    fn tiny_config(sliding_window: Option<usize>) -> MistralConfig {
        MistralConfig {
            model_type: "mistral".into(),
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 24,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            max_position_embeddings: 64,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            sliding_window,
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
        let model = MistralForCausalLM::init(&tiny_config(None), &device);
        let input = Tensor::from_data(TensorData::new(vec![1_i64, 5, 9], [1, 3]), &device);

        let logits = model.forward(input);

        assert_eq!(logits.dims(), [1, 3, 32]);
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
        let config = tiny_config(None);
        let model = MistralForCausalLM::init(&config, &device);
        let kernel = ola_kernel::PortablePagedAttention::new();
        for tokens_per_block in [1, 2, 4, 16] {
            let pool = MistralKvPool::new(
                &config,
                MistralKvCacheConfig::new(config.max_position_embeddings, tokens_per_block),
                device.clone(),
            )
            .unwrap();
            let mut cache = MistralKvCache::allocate(Arc::new(Mutex::new(pool))).unwrap();
            let mut all_tokens = vec![1_i64, 5, 9];
            let prompt = Tensor::from_data(
                TensorData::new(all_tokens.clone(), [1, all_tokens.len()]),
                &device,
            );
            let prefill_logits = model.prefill(prompt, &mut cache).unwrap();
            let expected_last_logit = prefill_logits
                .slice([
                    0..1,
                    all_tokens.len() - 1..all_tokens.len(),
                    0..config.vocab_size,
                ])
                .into_data()
                .try_to_vec::<f32>()
                .unwrap();

            let full_prompt = Tensor::from_data(
                TensorData::new(all_tokens.clone(), [1, all_tokens.len()]),
                &device,
            );
            let full_logits = model.forward(full_prompt);
            let full_last_logit = full_logits
                .slice([
                    0..1,
                    all_tokens.len() - 1..all_tokens.len(),
                    0..config.vocab_size,
                ])
                .into_data()
                .try_to_vec::<f32>()
                .unwrap();
            for (actual, expected) in expected_last_logit.iter().zip(full_last_logit.iter()) {
                assert!((actual - expected).abs() < 1e-4);
            }

            for next_token in [13_i64, 17, 21, 2] {
                let decode_input =
                    Tensor::from_data(TensorData::new(vec![next_token], [1, 1]), &device);
                let decode_logits = model.decode(decode_input, &mut cache, &kernel).unwrap();
                all_tokens.push(next_token);
                let full_input = Tensor::from_data(
                    TensorData::new(all_tokens.clone(), [1, all_tokens.len()]),
                    &device,
                );
                let full_logits = model.forward(full_input);
                let full_last_logit = full_logits
                    .slice([
                        0..1,
                        all_tokens.len() - 1..all_tokens.len(),
                        0..config.vocab_size,
                    ])
                    .into_data()
                    .try_to_vec::<f32>()
                    .unwrap();
                let actual_decode = decode_logits
                    .slice([0..1, 0..1, 0..config.vocab_size])
                    .into_data()
                    .try_to_vec::<f32>()
                    .unwrap();
                for (actual, expected) in actual_decode.iter().zip(full_last_logit.iter()) {
                    assert!((actual - expected).abs() < 1e-4);
                }
            }
        }
    }

    #[test]
    fn sliding_window_attention_matches_between_forward_and_paged_decode() {
        let device = Default::default();
        // sliding window of size 3
        let config = tiny_config(Some(3));
        let model = MistralForCausalLM::init(&config, &device);
        let kernel = ola_kernel::PortablePagedAttention::new();

        let pool = MistralKvPool::new(
            &config,
            MistralKvCacheConfig::new(config.max_position_embeddings, 2),
            device.clone(),
        )
        .unwrap();
        let mut cache = MistralKvCache::allocate(Arc::new(Mutex::new(pool))).unwrap();
        let mut all_tokens = vec![1_i64, 5];
        let prompt = Tensor::from_data(
            TensorData::new(all_tokens.clone(), [1, all_tokens.len()]),
            &device,
        );
        let _ = model.prefill(prompt, &mut cache).unwrap();

        for next_token in [9_i64, 13, 17, 21, 2] {
            let decode_input =
                Tensor::from_data(TensorData::new(vec![next_token], [1, 1]), &device);
            let decode_logits = model.decode(decode_input, &mut cache, &kernel).unwrap();
            all_tokens.push(next_token);
            let full_input = Tensor::from_data(
                TensorData::new(all_tokens.clone(), [1, all_tokens.len()]),
                &device,
            );
            let full_logits = model.forward(full_input);
            let full_last_logit = full_logits
                .slice([
                    0..1,
                    all_tokens.len() - 1..all_tokens.len(),
                    0..config.vocab_size,
                ])
                .into_data()
                .try_to_vec::<f32>()
                .unwrap();
            let actual_decode = decode_logits
                .slice([0..1, 0..1, 0..config.vocab_size])
                .into_data()
                .try_to_vec::<f32>()
                .unwrap();
            for (index, (actual, expected)) in
                actual_decode.iter().zip(full_last_logit.iter()).enumerate()
            {
                assert!(
                    (actual - expected).abs() < 2e-4,
                    "at token {index}: actual {actual}, expected {expected}"
                );
            }
        }
    }
}
