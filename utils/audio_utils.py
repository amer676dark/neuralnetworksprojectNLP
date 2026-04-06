"""Audio processing utilities for Arabic ASR."""

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import librosa
from pathlib import Path
from typing import Tuple, Optional, Union


TARGET_SAMPLE_RATE = 16000


def load_audio(
    path: Union[str, Path],
    target_sr: int = TARGET_SAMPLE_RATE,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """Load audio file and resample to target sample rate."""
    waveform, sr = torchaudio.load(str(path))

    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != target_sr:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    return waveform.squeeze().numpy(), target_sr


def extract_mel_spectrogram(
    waveform: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
    n_mels: int = 80,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
    f_min: float = 0.0,
    f_max: float = 8000.0,
) -> np.ndarray:
    """Extract log-mel spectrogram from waveform."""
    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        fmin=f_min,
        fmax=f_max,
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel  # shape: (n_mels, time_frames)


def normalize_audio(waveform: np.ndarray) -> np.ndarray:
    """Normalize audio to [-1, 1]."""
    max_val = np.abs(waveform).max()
    if max_val > 0:
        waveform = waveform / max_val
    return waveform


def pad_or_trim(
    waveform: np.ndarray,
    max_length: int,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> np.ndarray:
    """Pad or trim waveform to exactly max_length seconds."""
    max_samples = int(max_length * sample_rate)
    if len(waveform) > max_samples:
        waveform = waveform[:max_samples]
    elif len(waveform) < max_samples:
        pad_width = max_samples - len(waveform)
        waveform = np.pad(waveform, (0, pad_width), mode="constant")
    return waveform


def get_audio_duration(waveform: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> float:
    """Return duration of audio in seconds."""
    return len(waveform) / sample_rate


def apply_augmentation(
    waveform: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
    noise_factor: float = 0.005,
    time_shift_max: float = 0.1,
    speed_rate: Optional[float] = None,
) -> np.ndarray:
    """Apply data augmentation: additive noise, time shift, speed perturbation."""
    # Additive white noise
    noise = np.random.randn(len(waveform)) * noise_factor
    waveform = waveform + noise

    # Time shift
    shift_samples = int(np.random.uniform(-time_shift_max, time_shift_max) * sample_rate)
    waveform = np.roll(waveform, shift_samples)

    # Speed perturbation
    if speed_rate is not None:
        waveform = librosa.effects.time_stretch(waveform, rate=speed_rate)

    return waveform.astype(np.float32)


def spectrogram_to_tensor(spec: np.ndarray) -> torch.Tensor:
    """Convert numpy spectrogram to torch tensor with channel dim."""
    return torch.tensor(spec, dtype=torch.float32).unsqueeze(0)  # (1, n_mels, time)
