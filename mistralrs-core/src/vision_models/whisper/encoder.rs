#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss, dead_code)]

use candle_core::{Device, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, LayerNorm, Linear};
use mistralrs_quant::ShardedVarBuilder;

use crate::layers::{conv1d, layer_norm, linear, linear_no_bias};

use super::config::WhisperConfig;

const LAYER_NORM_EPS: f64 = 1e-5;
const MLP_EXPANSION: usize = 4;
const CONV_KERNEL_SIZE: usize = 3;
const CONV_STRIDE_1: usize = 1;
const CONV_STRIDE_2: usize = 2;
const CONV_PADDING: usize = 1;

/// Standard OpenAI Whisper encoder. Bidirectional attention, sinusoidal absolute
/// positional embeddings, two-stage Conv1d front-end. See
/// `whisper/__init__.py:AudioEncoder` and `whisper/model.py:AudioEncoder` upstream.
pub struct WhisperEncoder {
    pub conv1: Conv1d,
    pub conv2: Conv1d,
    pub pos_embedding: Tensor,
    pub blocks: Vec<EncoderBlock>,
    pub ln_post: LayerNorm,
}

pub struct EncoderBlock {
    pub attn: MultiHeadAttention,
    pub attn_ln: LayerNorm,
    pub mlp_fc1: Linear,
    pub mlp_fc2: Linear,
    pub mlp_ln: LayerNorm,
}

pub struct MultiHeadAttention {
    pub n_head: usize,
    pub query: Linear,
    pub key: Linear,
    pub value: Linear,
    pub out: Linear,
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, n_head: usize, vb: ShardedVarBuilder) -> Result<Self> {
        // OpenAI Whisper: query/value/out carry bias, key has none. Mirrors
        // `whisper/model.py:MultiHeadAttention.__init__`.
        let query = linear(d_model, d_model, vb.pp("query"))?;
        let key = linear_no_bias(d_model, d_model, vb.pp("key"))?;
        let value = linear(d_model, d_model, vb.pp("value"))?;
        let out = linear(d_model, d_model, vb.pp("out"))?;
        Ok(Self {
            n_head,
            query,
            key,
            value,
            out,
        })
    }

    pub fn forward(&self, _xs: &Tensor) -> Result<Tensor> {
        todo!("WhisperEncoder MultiHeadAttention::forward not implemented yet")
    }
}

impl EncoderBlock {
    pub fn new(d_model: usize, n_head: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let attn = MultiHeadAttention::new(d_model, n_head, vb.pp("attn"))?;
        let attn_ln = layer_norm(d_model, LAYER_NORM_EPS, vb.pp("attn_ln"))?;
        // HF stores the MLP as a Sequential of Linear -> GELU -> Linear,
        // hence keys `.mlp.0` and `.mlp.2`.
        let mlp_hidden = MLP_EXPANSION * d_model;
        let mlp_fc1 = linear(d_model, mlp_hidden, vb.pp("mlp").pp("0"))?;
        let mlp_fc2 = linear(mlp_hidden, d_model, vb.pp("mlp").pp("2"))?;
        let mlp_ln = layer_norm(d_model, LAYER_NORM_EPS, vb.pp("mlp_ln"))?;
        Ok(Self {
            attn,
            attn_ln,
            mlp_fc1,
            mlp_fc2,
            mlp_ln,
        })
    }

    pub fn forward(&self, _xs: &Tensor) -> Result<Tensor> {
        todo!("WhisperEncoder EncoderBlock::forward not implemented yet")
    }
}

impl WhisperEncoder {
    pub fn new(config: &WhisperConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let d_model = config.d_model;
        let n_mels = config.num_mel_bins;
        let n_head = config.encoder_attention_heads;
        let n_layers = config.encoder_layers;
        let max_positions = config.max_source_positions;

        let conv1 = conv1d(
            n_mels,
            d_model,
            CONV_KERNEL_SIZE,
            Conv1dConfig {
                padding: CONV_PADDING,
                stride: CONV_STRIDE_1,
                ..Default::default()
            },
            vb.pp("conv1"),
        )?;
        let conv2 = conv1d(
            d_model,
            d_model,
            CONV_KERNEL_SIZE,
            Conv1dConfig {
                padding: CONV_PADDING,
                stride: CONV_STRIDE_2,
                ..Default::default()
            },
            vb.pp("conv2"),
        )?;

        // Sinusoidal table is not in the checkpoint; OpenAI computes it at init time.
        let pos_embedding = sinusoid_pos_embedding(max_positions, d_model, vb.device())?;

        let vb_blocks = vb.pp("blocks");
        let mut blocks = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            blocks.push(EncoderBlock::new(d_model, n_head, vb_blocks.pp(i))?);
        }

        let ln_post = layer_norm(d_model, LAYER_NORM_EPS, vb.pp("ln_post"))?;

        Ok(Self {
            conv1,
            conv2,
            pos_embedding,
            blocks,
            ln_post,
        })
    }

    pub fn forward(&self, _mel: &Tensor) -> Result<Tensor> {
        todo!("WhisperEncoder::forward not implemented yet")
    }
}

/// Build the absolute sinusoidal positional embedding table used by Whisper.
/// Returns a `[length, channels]` f32 tensor; matches OpenAI's `sinusoids` helper.
fn sinusoid_pos_embedding(
    _length: usize,
    _channels: usize,
    _device: &Device,
) -> Result<Tensor> {
    todo!("sinusoid_pos_embedding not implemented yet")
}
