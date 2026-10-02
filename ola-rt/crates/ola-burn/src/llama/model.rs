use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig},
    tensor::{
        Int, Tensor, activation::silu, backend::Backend, module::attention,
        ops::AttentionModuleOptions,
    },
};

use super::LlamaConfig;

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

        let query = self.rotary_embedding.forward(query);
        let key = self.rotary_embedding.forward(key);
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
                is_causal: true,
                ..Default::default()
            },
        );
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
    fn forward<B: Backend>(&self, tensor: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch, heads, sequence, head_dim] = tensor.dims();
        debug_assert_eq!(head_dim, self.head_dim);
        let device = tensor.device();
        let half = head_dim / 2;

        let exponent = Tensor::<B, 1, Int>::arange_step(0..head_dim as i64, 2, &device)
            .float()
            .div_scalar(head_dim as f32);
        let inverse_frequency = exponent.mul_scalar(self.theta.ln() as f32).exp().recip();
        let positions = Tensor::<B, 1, Int>::arange(0..sequence as i64, &device)
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
    use burn::{backend::NdArray, tensor::TensorData};

    use super::*;

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
}
