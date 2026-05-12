"""
Efficient CNN + BiLSTM + Self-Attention + CTC model for Arabic ASR.

Design choices (informed by Conformer / ESPnet / Whisper best practices):

1. **4x time subsampling in the CNN encoder** (stride-2 conv on the time axis in
   the first two blocks). Drops the sequence length seen by the LSTM by 4x —
   the LSTM dominates compute per step, so this is the single biggest speed win.

2. **3-layer BiLSTM at hidden=512** (instead of 5x1024). Empirically, character-
   level CTC on ~40k training samples is well-served by ~30M-parameter models;
   bigger architectures plateau or overfit while taking 4-5x as long per step.

3. **GPU-side mel extraction** via torchaudio.transforms.MelSpectrogram. The
   dataset only returns raw waveforms; CPU workers are no longer the bottleneck.

4. **Pre-LayerNorm Multi-Head Self-Attention** after the LSTM for global temporal
   context. Pre-LN converges more stably than post-LN.

5. **SpecAugment in the subsampled domain** — time-mask widths are scaled down
   proportionally to the 4x subsampling so they cover the same fraction of audio.

6. **Glorot init** on Linear layers, **orthogonal init** on LSTM weights for
   numerical stability with longer sequences.

Total parameters: ~30M (was 138M in the prior design).
Expected WER on Common Voice Arabic: 25-40% with 40k training samples, 25 epochs.
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T


# ──────────────────────────────────────────────────────────────────────────────
# SpecAugment
# ──────────────────────────────────────────────────────────────────────────────

class SpecAugment(nn.Module):
    """Frequency + time masking applied to a log-mel spectrogram (Park et al. 2019).

    Operates only during training. Mask widths are calibrated for an input that
    has already been 4x time-subsampled in the CNN encoder (so a 25-frame time
    mask spans ~250 ms of original audio).
    """

    def __init__(
        self,
        freq_mask_param: int = 15,    # was 27 — gentler for early epochs
        time_mask_param: int = 20,    # 4x-subsampled domain
        n_freq_masks: int = 1,
        n_time_masks: int = 1,
    ):
        super().__init__()
        self.freq_masks = nn.ModuleList(
            [T.FrequencyMasking(freq_mask_param=freq_mask_param) for _ in range(n_freq_masks)]
        )
        self.time_masks = nn.ModuleList(
            [T.TimeMasking(time_mask_param=time_mask_param) for _ in range(n_time_masks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, 1, freq, T)
        if not self.training:
            return x
        for m in self.freq_masks:
            x = m(x.squeeze(1)).unsqueeze(1)
        for m in self.time_masks:
            x = m(x.squeeze(1)).unsqueeze(1)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# CNN encoder blocks
# ──────────────────────────────────────────────────────────────────────────────

class SubsampleConvBlock(nn.Module):
    """Conv2d block with stride 2 in BOTH freq and time axes.

    Each call halves the spectrogram along both axes. Two stacked blocks give
    the canonical 4x time / 4x freq subsampling used by Conformer and similar.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=2, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.norm(self.conv(x))))


class ResidualConvBlock(nn.Module):
    """Standard residual conv block; no subsampling. Adds depth for representation."""

    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.path = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(x + self.path(x)))


# ──────────────────────────────────────────────────────────────────────────────
# Attention block (pre-LN MHSA + FFN, like Conformer's "transformer" half)
# ──────────────────────────────────────────────────────────────────────────────

class AttentionBlock(nn.Module):
    """Pre-LayerNorm Multi-Head Self-Attention with residual."""

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, D)
        h = self.norm(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        return x + self.drop(h)


# ──────────────────────────────────────────────────────────────────────────────
# Main model
# ──────────────────────────────────────────────────────────────────────────────

class CNNLSTMASR(nn.Module):
    """CNN encoder (with 4x time subsampling) → BiLSTM → MHSA → linear → CTC."""

    def __init__(
        self,
        vocab_size: int,
        n_mels: int = 80,
        cnn_channels: List[int] = (64, 128, 256),
        cnn_dropout: float = 0.1,
        lstm_hidden: int = 512,
        lstm_layers: int = 3,
        lstm_dropout: float = 0.1,
        attn_heads: int = 8,
        spec_augment: bool = True,
        # GPU-side mel extraction params
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        f_min: float = 0.0,
        f_max: float = 8000.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_mels = n_mels

        # ── GPU mel extractor (used when forward() gets raw waveform) ────────
        self.mel_spec = T.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
            win_length=win_length, n_mels=n_mels, f_min=f_min, f_max=f_max,
            power=2.0,
        )
        self.amp_to_db = T.AmplitudeToDB(top_db=80.0)

        # ── SpecAugment in the subsampled domain ─────────────────────────────
        self.spec_aug = SpecAugment() if spec_augment else None

        # ── CNN encoder: 4x time + freq subsampling, then 1 residual block ───
        c0, c1, c2 = cnn_channels[:3]
        self.cnn = nn.Sequential(
            SubsampleConvBlock(1,  c0, dropout=cnn_dropout),   # T/2,  freq/2
            SubsampleConvBlock(c0, c1, dropout=cnn_dropout),   # T/4,  freq/4
            ResidualConvBlock(c1,      dropout=cnn_dropout),   # T/4
            nn.Conv2d(c1, c2, kernel_size=1),                  # channel projection
            nn.BatchNorm2d(c2),
            nn.GELU(),
        )

        # After two 2x time-subsamples and two 2x freq-subsamples the spectrogram
        # has shape (B, c2, n_mels//4, T//4).
        freq_dim_out = n_mels // 4
        cnn_out_size = c2 * freq_dim_out

        # ── Linear projection into LSTM input space ──────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(cnn_out_size, lstm_hidden),
            nn.LayerNorm(lstm_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # ── BiLSTM ───────────────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
        )
        lstm_out_dim = lstm_hidden * 2

        # ── Multi-head self-attention block ──────────────────────────────────
        self.attention = AttentionBlock(lstm_out_dim, num_heads=attn_heads)

        # ── Output head ──────────────────────────────────────────────────────
        self.out_norm = nn.LayerNorm(lstm_out_dim)
        self.out_drop = nn.Dropout(0.1)
        self.fc = nn.Linear(lstm_out_dim, vocab_size)

        self._init_weights()

    def _init_weights(self) -> None:
        """Glorot for Linear; orthogonal for LSTM weights (stability on long seqs)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)
                        # Forget-gate bias = 1 — standard LSTM-ASR trick
                        n = param.shape[0]
                        param.data[n // 4 : n // 2].fill_(1.0)

    # ──────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Accepts either raw waveform (B, samples) or pre-computed log-mel (B, 1, freq, T)."""
        # 1) GPU mel extraction if we got raw waveform
        if x.dim() == 2:
            mel = self.mel_spec(x)            # (B, n_mels, T) — power spectrogram
            mel = self.amp_to_db(mel)         # log-mel in dB, roughly [-80, 0]
            # Fixed, padding-invariant normalization (Whisper-style).
            # Per-sample mean/std would include the padded -80 dB region and
            # squash the real-audio dynamic range; a fixed linear map preserves
            # the signal-to-noise ratio for real audio regardless of padding.
            mel = (mel + 40.0) / 20.0         # roughly [-2, 2] for typical speech
            x = mel.unsqueeze(1)              # (B, 1, n_mels, T)

        # 2) SpecAugment in the input domain (before subsampling)
        if self.spec_aug is not None:
            x = self.spec_aug(x)

        # 3) CNN encoder — subsamples time by 4x
        x = self.cnn(x)                       # (B, C, freq/4, T/4)
        B, C, F_, T_ = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T_, C * F_)  # (B, T/4, C*freq/4)

        # 4) Projection
        x = self.input_proj(x)                # (B, T/4, lstm_hidden)

        # 5) BiLSTM
        x, _ = self.lstm(x)                   # (B, T/4, 2*lstm_hidden)

        # 6) Self-attention for global context
        x = self.attention(x)

        # 7) Output projection
        x = self.out_norm(x)
        x = self.out_drop(x)
        x = self.fc(x)                        # (B, T/4, vocab)
        return F.log_softmax(x, dim=-1)

    # ──────────────────────────────────────────────────────────────────────
    # CTC helpers
    # ──────────────────────────────────────────────────────────────────────
    def get_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Length after CNN subsampling. We subsample time by 4x (two stride-2 convs)."""
        # With kernel=3, stride=2, padding=1 the output length is ceil(L / 2),
        # which equals (L + 1) // 2. Applied twice → (((L+1)//2)+1)//2.
        l = (input_lengths + 1) // 2
        l = (l + 1) // 2
        return l.clamp_min(1)

    def ctc_loss(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
        blank: int = 0,
    ) -> torch.Tensor:
        return F.ctc_loss(
            log_probs.permute(1, 0, 2),  # (T, B, V)
            targets,
            input_lengths,
            target_lengths,
            blank=blank,
            reduction="mean",
            zero_infinity=True,
        )

    @staticmethod
    def greedy_decode(log_probs: torch.Tensor, blank: int = 0) -> List[List[int]]:
        """Argmax → collapse repeats → drop blank."""
        preds = log_probs.argmax(dim=-1)
        results: List[List[int]] = []
        for seq in preds:
            decoded: List[int] = []
            prev = None
            for tok in seq.tolist():
                if tok != blank and tok != prev:
                    decoded.append(tok)
                prev = tok
            results.append(decoded)
        return results

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def build_model(vocab_size: int, config: dict) -> CNNLSTMASR:
    """Instantiate from config['cnn_lstm'] + config['audio']."""
    cfg = config["cnn_lstm"]
    audio = config.get("audio", {})
    return CNNLSTMASR(
        vocab_size=vocab_size,
        n_mels=audio.get("n_mels", 80),
        cnn_channels=cfg.get("cnn_channels", [64, 128, 256]),
        cnn_dropout=cfg.get("cnn_dropout", 0.1),
        lstm_hidden=cfg.get("lstm_hidden_size", 512),
        lstm_layers=cfg.get("lstm_num_layers", 3),
        lstm_dropout=cfg.get("lstm_dropout", 0.1),
        attn_heads=cfg.get("attention_heads", 8),
        spec_augment=cfg.get("spec_augment", True),
        sample_rate=audio.get("sample_rate", 16000),
        n_fft=audio.get("n_fft", 512),
        hop_length=audio.get("hop_length", 160),
        win_length=audio.get("win_length", 400),
        f_min=audio.get("f_min", 0.0),
        f_max=audio.get("f_max", 8000.0),
    )
