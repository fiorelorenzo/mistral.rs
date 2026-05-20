#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss, dead_code)]

use anyhow::Result;
use candle_core::{Device, Tensor};
use mistralrs_audio::AudioInput;

use crate::vision_models::voxtral::audio_processing::{
    AudioPaddingPolicy, LogMelFloor, VoxtralAudioProcessor,
};

/// 16 kHz canonical sampling rate for Whisper.
const WHISPER_SAMPLING_RATE: u32 = 16_000;
/// 100 Hz frame rate prior to the 2x conv stride in the encoder.
const WHISPER_FRAME_RATE: f32 = 100.0;
/// Whisper-large uses 128; tiny/base/small/medium use 80.
const WHISPER_NUM_MEL_BINS_DEFAULT: usize = 80;
/// 10 ms hop at 16 kHz.
const WHISPER_HOP_LENGTH: usize = 160;
/// 25 ms window (also the FFT size).
const WHISPER_WINDOW_SIZE: usize = 400;
/// 30 s of audio at 100 Hz mel frame rate.
const WHISPER_TARGET_FRAMES: usize = 3000;

/// Padding policy for the Whisper mel pipeline.
#[derive(Debug, Clone, Copy)]
pub enum WhisperPaddingPolicy {
    /// Whisper: pad or trim to a fixed number of mel frames (default 3000).
    PadOrTrim(usize),
    /// Voxtral-style streaming silence padding.
    SilenceTokens { left: usize, right: usize },
}

/// Log-mel floor policy. `PerSpectrogramMax` matches OpenAI Whisper; `GlobalConstant` matches Voxtral.
#[derive(Debug, Clone, Copy)]
pub enum WhisperLogMelFloor {
    PerSpectrogramMax,
    GlobalConstant(f32),
}

/// Mel spectrogram processor matching OpenAI's Whisper preprocessing.
/// Field shape mirrors `VoxtralAudioProcessor`; STFT and Slaney filterbank are reused.
pub struct WhisperAudioProcessor {
    sampling_rate: u32,
    frame_rate: f32,
    num_mel_bins: usize,
    hop_length: usize,
    window_size: usize,
    padding_policy: WhisperPaddingPolicy,
    log_mel_floor: WhisperLogMelFloor,
    inner: VoxtralAudioProcessor,
}

impl WhisperAudioProcessor {
    /// Build a processor with Whisper defaults (16 kHz, 80 bins, pad-or-trim to 3000 frames).
    pub fn new() -> Self {
        Self::with_num_mel_bins(WHISPER_NUM_MEL_BINS_DEFAULT)
    }

    /// Build a processor with a custom mel bin count (80 for whisper-{tiny,base,small,medium}, 128 for whisper-large-v3).
    pub fn with_num_mel_bins(num_mel_bins: usize) -> Self {
        let inner = VoxtralAudioProcessor::new_raw(
            WHISPER_SAMPLING_RATE,
            WHISPER_FRAME_RATE,
            num_mel_bins,
            WHISPER_HOP_LENGTH,
            WHISPER_WINDOW_SIZE,
            // Unused under PerSpectrogramMax; kept to satisfy the constructor signature.
            0.0,
        );
        Self {
            sampling_rate: WHISPER_SAMPLING_RATE,
            frame_rate: WHISPER_FRAME_RATE,
            num_mel_bins,
            hop_length: WHISPER_HOP_LENGTH,
            window_size: WHISPER_WINDOW_SIZE,
            padding_policy: WhisperPaddingPolicy::PadOrTrim(WHISPER_TARGET_FRAMES),
            log_mel_floor: WhisperLogMelFloor::PerSpectrogramMax,
            inner,
        }
    }

    pub fn sampling_rate(&self) -> u32 {
        self.sampling_rate
    }
    pub fn frame_rate(&self) -> f32 {
        self.frame_rate
    }
    pub fn num_mel_bins(&self) -> usize {
        self.num_mel_bins
    }
    pub fn hop_length(&self) -> usize {
        self.hop_length
    }
    pub fn window_size(&self) -> usize {
        self.window_size
    }
    pub fn padding_policy(&self) -> WhisperPaddingPolicy {
        self.padding_policy
    }
    pub fn log_mel_floor(&self) -> WhisperLogMelFloor {
        self.log_mel_floor
    }

    /// Process an `AudioInput` into a `[1, T, num_mel_bins]` tensor.
    /// `T` is fixed at `target_frames` under `PadOrTrim`; equal to the actual STFT frame count under `SilenceTokens`.
    pub fn process_audio(&self, audio: &AudioInput, device: &Device) -> Result<Tensor> {
        let floor = match self.log_mel_floor {
            WhisperLogMelFloor::PerSpectrogramMax => LogMelFloor::PerSpectrogramMax,
            WhisperLogMelFloor::GlobalConstant(v) => LogMelFloor::GlobalConstant(v),
        };
        let padding = match self.padding_policy {
            WhisperPaddingPolicy::PadOrTrim(n) => AudioPaddingPolicy::PadOrTrim(n),
            WhisperPaddingPolicy::SilenceTokens { left, right } => {
                AudioPaddingPolicy::SilenceTokens { left, right }
            }
        };
        self.inner.process_audio_with(audio, device, floor, padding)
    }
}

impl Default for WhisperAudioProcessor {
    fn default() -> Self {
        Self::new()
    }
}
