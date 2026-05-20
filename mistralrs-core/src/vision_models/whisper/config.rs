#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    dead_code
)]

use serde::Deserialize;

use crate::serde_default_fn;

serde_default_fn!(bool, default_true, true);
serde_default_fn!(bool, default_false, false);
serde_default_fn!(usize, default_num_mel_bins, 80);
serde_default_fn!(usize, default_max_source_positions, 1500);
serde_default_fn!(usize, default_max_target_positions, 448);
fn default_activation() -> String {
    "gelu".to_string()
}
serde_default_fn!(f32, default_init_std, 0.02);

/// HuggingFace WhisperConfig schema. Mirrors `transformers.models.whisper.configuration_whisper.WhisperConfig`.
#[derive(Debug, Clone, Deserialize)]
pub struct WhisperConfig {
    pub vocab_size: usize,
    #[serde(default = "default_num_mel_bins")]
    pub num_mel_bins: usize,
    pub encoder_layers: usize,
    pub encoder_attention_heads: usize,
    pub decoder_layers: usize,
    pub decoder_attention_heads: usize,
    pub encoder_ffn_dim: usize,
    pub decoder_ffn_dim: usize,
    pub decoder_start_token_id: u32,
    pub eos_token_id: u32,
    pub pad_token_id: u32,
    pub bos_token_id: u32,
    #[serde(default = "default_max_source_positions")]
    pub max_source_positions: usize,
    #[serde(default = "default_max_target_positions")]
    pub max_target_positions: usize,
    pub d_model: usize,
    #[serde(default = "default_activation")]
    pub activation_function: String,
    #[serde(default = "default_false")]
    pub scale_embedding: bool,
    #[serde(default = "default_init_std")]
    pub init_std: f32,
}
