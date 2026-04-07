"""
Advanced CNN + BiLSTM + Multi-Head Attention Arabic ASR Model with CTC loss.

Architecture:
  Input: Log-Mel Spectrogram (B, 1, n_mels, T)
  → SpecAugment (frequency + time masking, training only)
  → Residual CNN Encoder (3 blocks, 32→64→128 channels)
  → Linear projection (1280 → 512)
  → Bidirectional LSTM × 3 (hidden=512)
  → Multi-Head Self-Attention (8 heads)
  → Linear → log-softmax → CTC
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from typing import Tuple, Optional, List


# ── SpecAugment ────────────────────────────────────────────────────────────────

class SpecAugment(nn.Module):
    """
    SpecAugment: frequency and time masking.
    Applied only during training (self.training guard).
    Park et al. (2019) — standard for all modern ASR systems.
    """
    def __init__(self, freq_mask_param: int = 27, time_mask_param: int = 100,
                 n_freq_masks: int = 2, n_time_masks: int = 2):
        super().__init__()
        self.freq_masks = nn.ModuleList([
            T.FrequencyMasking(freq_mask_param=freq_mask_param)
            for _ in range(n_freq_masks)
        ])
        self.time_masks = nn.ModuleList([
            T.TimeMasking(time_mask_param=time_mask_param)
            for _ in range(n_time_masks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, n_mels, T)"""
        if not self.training:
            return x
        for mask in self.freq_masks:
            x = mask(x.squeeze(1)).unsqueeze(1)
        for mask in self.time_masks:
            x = mask(x.squeeze(1)).unsqueeze(1)
        return x


# ── Residual CNN Block ─────────────────────────────────────────────────────────

class ResidualConvBlock(nn.Module):
    """
    Two-layer Conv2d block with 1×1 projection shortcut (residual).
    Uses GELU activation and frequency-axis max pooling.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, dropout: float = 0.15):
        super().__init__()
        pad = kernel_size // 2

        self.conv_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=pad),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=pad),
            nn.BatchNorm2d(out_channels),
        )
        # 1×1 shortcut to match channel dims
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
        ) if in_channels != out_channels else nn.Identity()

        self.pool = nn.MaxPool2d(kernel_size=(2, 1))  # halve freq, keep time
        self.dropout = nn.Dropout2d(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv_path(x)
        out = self.act(out + residual)
        out = self.pool(out)
        out = self.dropout(out)
        return out


# ── Multi-Head Self-Attention Block ───────────────────────────────────────────

class AttentionBlock(nn.Module):
    """
    Pre-LayerNorm multi-head self-attention with residual.
    Placed after BiLSTM to add global temporal context.
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, embed_dim)"""
        normed = self.norm(x)
        attn_out, _ = self.attn(normed, normed, normed)
        return x + self.dropout(attn_out)


# ── Main Model ────────────────────────────────────────────────────────────────

class CNNLSTMASR(nn.Module):
    """
    Advanced Arabic ASR model:
      SpecAugment → Residual CNN → Projection → BiLSTM → Self-Attention → CTC

    Parameters
    ----------
    vocab_size    : size of character vocabulary (including blank=0)
    n_mels        : number of mel bands in input spectrogram
    cnn_channels  : list of output channels per ResidualConvBlock
    lstm_hidden   : BiLSTM hidden size (output is 2×hidden due to bidirectional)
    lstm_layers   : number of stacked BiLSTM layers
    lstm_dropout  : dropout between LSTM layers
    attn_heads    : number of attention heads (must divide 2×lstm_hidden)
    """

    def __init__(
        self,
        vocab_size: int,
        n_mels: int = 80,
        cnn_channels: List[int] = (32, 64, 128),
        cnn_kernel_size: int = 3,
        cnn_dropout: float = 0.15,
        lstm_hidden: int = 512,
        lstm_layers: int = 3,
        lstm_dropout: float = 0.3,
        attn_heads: int = 8,
        spec_augment: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_mels = n_mels

        # SpecAugment
        self.spec_aug = SpecAugment() if spec_augment else None

        # Residual CNN encoder
        channels = [1] + list(cnn_channels)
        self.cnn = nn.Sequential(*[
            ResidualConvBlock(channels[i], channels[i+1], cnn_kernel_size, cnn_dropout)
            for i in range(len(cnn_channels))
        ])

        # Compute flattened CNN output size
        freq_dim = n_mels
        for _ in cnn_channels:
            freq_dim = freq_dim // 2
        cnn_out_size = freq_dim * cnn_channels[-1]

        # Linear projection before LSTM (reduces parameters significantly)
        self.input_proj = nn.Sequential(
            nn.Linear(cnn_out_size, lstm_hidden),
            nn.LayerNorm(lstm_hidden),
            nn.GELU(),
        )

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0,
            bidirectional=True,
        )

        lstm_out_dim = lstm_hidden * 2  # bidirectional

        # Multi-head self-attention (global temporal context)
        self.attention = AttentionBlock(lstm_out_dim, attn_heads)

        # Output
        self.norm = nn.LayerNorm(lstm_out_dim)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(lstm_out_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, n_mels, T)
        returns: (B, T, vocab_size) log-softmax
        """
        # SpecAugment
        if self.spec_aug is not None:
            x = self.spec_aug(x)

        # CNN: (B, 1, n_mels, T) → (B, C, freq', T)
        x = self.cnn(x)

        B, C, freq, T = x.shape
        # Reshape: (B, T, C*freq)
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * freq)

        # Project to LSTM input size
        x = self.input_proj(x)

        # BiLSTM
        x, _ = self.lstm(x)  # (B, T, 2*hidden)

        # Multi-head self-attention
        x = self.attention(x)

        # Output projection
        x = self.norm(x)
        x = self.dropout(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)

    def get_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute CTC input lengths after CNN time-axis processing.
        Our CNN only pools along frequency axis (MaxPool2d(2,1)),
        so time dimension is unchanged. Returns input_lengths as-is.
        """
        return input_lengths

    def ctc_loss(self, log_probs, targets, input_lengths, target_lengths, blank=0):
        """log_probs: (B, T, V) — will be permuted to (T, B, V) for CTC."""
        ctc_lengths = self.get_output_lengths(
            torch.full((log_probs.shape[0],), log_probs.shape[1],
                       dtype=torch.long, device=log_probs.device)
        )
        return F.ctc_loss(
            log_probs.permute(1, 0, 2),
            targets,
            ctc_lengths,
            target_lengths,
            blank=blank,
            reduction="mean",
            zero_infinity=True,
        )

    def greedy_decode(self, log_probs: torch.Tensor, blank: int = 0) -> List[List[int]]:
        """Greedy CTC: argmax, collapse repeats, remove blanks."""
        preds = log_probs.argmax(dim=-1)
        results = []
        for seq in preds:
            decoded, prev = [], None
            for token in seq.tolist():
                if token != blank and token != prev:
                    decoded.append(token)
                prev = token
            results.append(decoded)
        return results

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(vocab_size: int, config: dict) -> CNNLSTMASR:
    """Instantiate model from config. Uses .get() with defaults for all new keys."""
    cfg = config["cnn_lstm"]
    return CNNLSTMASR(
        vocab_size=vocab_size,
        n_mels=config["audio"]["n_mels"],
        cnn_channels=cfg.get("cnn_channels", [32, 64, 128]),
        cnn_kernel_size=cfg.get("cnn_kernel_size", 3),
        cnn_dropout=cfg.get("cnn_dropout", 0.15),
        lstm_hidden=cfg.get("lstm_hidden_size", 512),
        lstm_layers=cfg.get("lstm_num_layers", 3),
        lstm_dropout=cfg.get("lstm_dropout", 0.3),
        attn_heads=cfg.get("attention_heads", 8),
        spec_augment=cfg.get("spec_augment", True),
    )
