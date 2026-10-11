use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig},
    tensor::{
        Int, Tensor, activation::silu, backend::Backend, module::attention,
        ops::AttentionModuleOptions,
    },
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
pub struct LlamaForCausalLM<B: Backend> {
    model: LlamaModel<B>,
    lm_head: Linear<B>,
}

impl<B: Backend> LlamaForCausalLM<B> {
    /// Initializes the module structure before checkpoint parameters are loaded.
    pub fn init(config: &LlamaConfig, device: &B::Device) -> Self {
        Self {
            model: LlamaModel::init(config, device),
            lm_head: linear(config.hidden_size, config.vocab_size, device),
        }
    }

    /// Computes logits for every token position.
    ///
    /// Input shape is `[batch, sequence]`; output shape is
    /// `[batch, sequence, vocabulary]`. The first implementation uses this
    /// full-sequence method for both prefill and decode so correctness can be
    /// established before introducing cache mutation.
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.lm_head.forward(self.model.forward(input_ids))
    }

    /// Processes a complete prompt and returns logits plus its populated cache.
    pub(crate) fn prefill(
        &self,
        input_ids: Tensor<B, 2, Int>,
        cache: &mut LlamaKvCache<B>,
    ) -> Result<Tensor<B, 3>, KvBlockError> {
        let hidden_states = self.model.prefill(input_ids, cache)?;
        Ok(self.lm_head.forward(hidden_states))
    }

    /// Processes one new token using and extending an existing request cache.
    pub(crate) fn decode(
        &self,
        input_ids: Tensor<B, 2, Int>,
        cache: &mut LlamaKvCache<B>,
    ) -> Result<Tensor<B, 3>, KvBlockError> {
        Ok(self.lm_head.forward(self.model.decode(input_ids, cache)?))
    }
}

#[derive(Module, Debug)]
struct LlamaModel<B: Backend> {
    embed_tokens: Embedding<B>,
    layers: Vec<LlamaDecoderLayer<B>>,
    norm: RmsNorm<B>,
}

impl<B: Backend> LlamaModel<B> {
    fn init(config: &LlamaConfig, device: &B::Device) -> Self {
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

    fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let mut hidden_states = self.embed_tokens.forward(input_ids);
        for layer in &self.layers {
            hidden_states = layer.forward(hidden_states);
        }
        self.norm.forward(hidden_states)
    }

    fn prefill(
        &self,
        input_ids: Tensor<B, 2, Int>,
        cache: &mut LlamaKvCache<B>,
    ) -> Result<Tensor<B, 3>, KvBlockError> {
        let sequence_length = input_ids.dims()[1];
        cache.reserve(sequence_length)?;
        let mut hidden_states = self.embed_tokens.forward(input_ids);
        for (layer_index, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.prefill(hidden_states, cache, layer_index);
        }
        Ok(self.norm.forward(hidden_states))
    }

    fn decode(
        &self,
        input_ids: Tensor<B, 2, Int>,
        cache: &mut LlamaKvCache<B>,
    ) -> Result<Tensor<B, 3>, KvBlockError> {
        debug_assert_eq!(input_ids.dims()[1], 1);
        let position = cache.tokens();
        cache.reserve(1)?;
        let mut hidden_states = self.embed_tokens.forward(input_ids);
        for (layer_index, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.decode(hidden_states, cache, layer_index, position);
        }
        Ok(self.norm.forward(hidden_states))
    }
}

#[derive(Module, Debug)]
struct LlamaDecoderLayer<B: Backend> {
    self_attn: LlamaAttention<B>,
    mlp: LlamaMlp<B>,
    input_layernorm: RmsNorm<B>,
    post_attention_layernorm: RmsNorm<B>,
}

impl<B: Backend> LlamaDecoderLayer<B> {
    fn init(config: &LlamaConfig, device: &B::Device) -> Self {
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

    fn forward(&self, hidden_states: Tensor<B, 3>) -> Tensor<B, 3> {
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

    fn prefill(
        &self,
        hidden_states: Tensor<B, 3>,
        cache: &LlamaKvCache<B>,
        layer: usize,
    ) -> Tensor<B, 3> {
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

    fn decode(
        &self,
        hidden_states: Tensor<B, 3>,
        cache: &LlamaKvCache<B>,
        layer: usize,
        position: usize,
    ) -> Tensor<B, 3> {
        let residual = hidden_states.clone();
        let hidden_states = residual
            + self.self_attn.decode(
                self.input_layernorm.forward(hidden_states),
                cache,
                layer,
                position,
            );
        let residual = hidden_states.clone();
        residual
            + self
                .mlp
                .forward(self.post_attention_layernorm.forward(hidden_states))
    }
}

#[derive(Module, Debug)]
struct LlamaAttention<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    o_proj: Linear<B>,
    rotary_embedding: LlamaRotaryEmbedding,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
}

impl<B: Backend> LlamaAttention<B> {
    fn init(config: &LlamaConfig, device: &B::Device) -> Self {
        let head_dim = config.attention_head_dim();
        let query_width = config.num_attention_heads * head_dim;
        let key_value_width = config.num_key_value_heads * head_dim;
        Self {
            q_proj: linear(config.hidden_size, query_width, device),
            k_proj: linear(config.hidden_size, key_value_width, device),
            v_proj: linear(config.hidden_size, key_value_width, device),
            o_proj: linear(query_width, config.hidden_size, device),
            rotary_embedding: LlamaRotaryEmbedding {
                head_dim,
                theta: config.rope_theta,
            },
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            head_dim,
        }
    }

    fn forward(&self, hidden_states: Tensor<B, 3>) -> Tensor<B, 3> {
        let (query, key, value) = self.project(hidden_states);
        let query = self.rotary_embedding.forward(query, 0);
        let key = self.rotary_embedding.forward(key, 0);
        self.attend(query, key, value, true)
    }

    fn prefill(
        &self,
        hidden_states: Tensor<B, 3>,
        cache: &LlamaKvCache<B>,
        layer: usize,
    ) -> Tensor<B, 3> {
        let (query, key, value) = self.project(hidden_states);
        let query = self.rotary_embedding.forward(query, 0);
        let key = self.rotary_embedding.forward(key, 0);
        let context = self.attend(query, key.clone(), value.clone(), true);
        cache.write_layer(layer, 0, key, value);
        context
    }

    fn decode(
        &self,
        hidden_states: Tensor<B, 3>,
        cache: &LlamaKvCache<B>,
        layer: usize,
        position: usize,
    ) -> Tensor<B, 3> {
        let (query, key, value) = self.project(hidden_states);
        let query = self.rotary_embedding.forward(query, position);
        let key = self.rotary_embedding.forward(key, position);
        cache.write_layer(layer, position, key, value);
        self.attend_paged_decode(query, cache.read_layer_pages(layer))
    }

    fn project(&self, hidden_states: Tensor<B, 3>) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
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
        query: Tensor<B, 4>,
        key: Tensor<B, 4>,
        value: Tensor<B, 4>,
        is_causal: bool,
    ) -> Tensor<B, 3> {
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
        query: Tensor<B, 4>,
        pages: Vec<(Tensor<B, 4>, Tensor<B, 4>)>,
    ) -> Tensor<B, 3> {
        let [_, _, query_length, _] = query.dims();
        debug_assert_eq!(query_length, 1);
        let groups = self.num_attention_heads / self.num_key_value_heads;
        let scale = (self.head_dim as f32).sqrt();
        let mut pages = pages.into_iter();
        let (first_key, first_value) = pages
            .next()
            .expect("decode attention requires at least one populated cache page");
        let first_key = repeat_key_value(first_key, groups);
        let first_value = repeat_key_value(first_value, groups);
        let scores = query
            .clone()
            .matmul(first_key.swap_dims(2, 3))
            .div_scalar(scale);
        let mut running_max = scores.clone().max_dim(3);
        let weights = (scores - running_max.clone()).exp();
        let mut running_sum = weights.clone().sum_dim(3);
        let mut running_output = weights.matmul(first_value);

        for (key, value) in pages {
            let key = repeat_key_value(key, groups);
            let value = repeat_key_value(value, groups);
            let scores = query.clone().matmul(key.swap_dims(2, 3)).div_scalar(scale);
            let page_max = scores.clone().max_dim(3);
            let new_max = running_max.clone().max_pair(page_max);
            let previous_scale = (running_max - new_max.clone()).exp();
            let weights = (scores - new_max.clone()).exp();
            running_sum = running_sum * previous_scale.clone() + weights.clone().sum_dim(3);
            running_output = running_output * previous_scale + weights.matmul(value);
            running_max = new_max;
        }

        self.project_attention_output(running_output / running_sum)
    }

    fn project_attention_output(&self, context: Tensor<B, 4>) -> Tensor<B, 3> {
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
fn repeat_key_value<B: Backend>(tensor: Tensor<B, 4>, groups: usize) -> Tensor<B, 4> {
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
}

impl LlamaRotaryEmbedding {
    /// Applies Hugging Face Llama's split-half RoPE convention.
    ///
    /// Burn's general `RotaryEncoding` rotates adjacent pairs. Llama instead
    /// pairs the first half of each head with the second half, so substituting
    /// the generic module would preserve shapes while producing wrong logits.
    fn forward<B: Backend>(&self, tensor: Tensor<B, 4>, position_offset: usize) -> Tensor<B, 4> {
        let [batch, heads, sequence, head_dim] = tensor.dims();
        debug_assert_eq!(head_dim, self.head_dim);
        let device = tensor.device();
        let half = head_dim / 2;

        let exponent = Tensor::<B, 1, Int>::arange_step(0..head_dim as i64, 2, &device)
            .float()
            .div_scalar(head_dim as f32);
        let inverse_frequency = exponent.mul_scalar(self.theta.ln() as f32).exp().recip();
        let positions = Tensor::<B, 1, Int>::arange(
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
struct LlamaMlp<B: Backend> {
    gate_proj: Linear<B>,
    up_proj: Linear<B>,
    down_proj: Linear<B>,
}

impl<B: Backend> LlamaMlp<B> {
    fn init(config: &LlamaConfig, device: &B::Device) -> Self {
        Self {
            gate_proj: linear(config.hidden_size, config.intermediate_size, device),
            up_proj: linear(config.hidden_size, config.intermediate_size, device),
            down_proj: linear(config.intermediate_size, config.hidden_size, device),
        }
    }

    fn forward(&self, hidden_states: Tensor<B, 3>) -> Tensor<B, 3> {
        let gated = silu(self.gate_proj.forward(hidden_states.clone()));
        self.down_proj
            .forward(gated * self.up_proj.forward(hidden_states))
    }
}

fn linear<B: Backend>(input: usize, output: usize, device: &B::Device) -> Linear<B> {
    LinearConfig::new(input, output)
        .with_bias(false)
        .init(device)
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use burn::{backend::NdArray, tensor::TensorData};

    use super::*;
    use crate::llama::{
        LlamaKvCacheConfig,
        kv_cache::{LlamaKvCache, LlamaKvPool},
    };

    type TestBackend = NdArray<f32>;

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
        TestBackend::seed(&device, 7);
        let model = LlamaForCausalLM::<TestBackend>::init(&tiny_config(), &device);
        let input = Tensor::from_data(TensorData::new(vec![1_i64, 5, 9], [1, 3]), &device);

        let logits = model.forward(input);

        assert_eq!(logits.dims(), [1, 3, 32]);
    }

    #[test]
    fn grouped_query_expansion_repeats_each_head_contiguously() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 4>::from_data(
            TensorData::new(vec![1.0_f32, 2.0], [1, 2, 1, 1]),
            &device,
        );

        let repeated = repeat_key_value(tensor, 2).into_data();

        assert_eq!(repeated.to_vec::<f32>().unwrap(), vec![1.0, 1.0, 2.0, 2.0]);
    }

    #[test]
    fn paged_decode_matches_full_sequence_logits_across_page_sizes() {
        let device = Default::default();
        TestBackend::seed(&device, 13);
        let config = tiny_config();
        let model = LlamaForCausalLM::<TestBackend>::init(&config, &device);
        for tokens_per_block in [1, 2, 4, 16] {
            let pool = LlamaKvPool::new(
                &config,
                LlamaKvCacheConfig::new(config.max_position_embeddings, tokens_per_block),
                device,
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

    fn assert_logits_close(
        left: Tensor<TestBackend, 3>,
        right: Tensor<TestBackend, 3>,
        tolerance: f32,
    ) {
        let left = left.into_data().to_vec::<f32>().unwrap();
        let right = right.into_data().to_vec::<f32>().unwrap();
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
