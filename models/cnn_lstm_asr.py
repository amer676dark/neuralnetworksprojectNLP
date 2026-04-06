"""
CNN + LSTM Arabic ASR Model with CTC loss.

Architecture:
  Input: Log-Mel Spectrogram (B, 1, n_mels, T)
  → CNN Encoder: extract local spectro-temporal features
  → Reshape: (B, T', C) sequence for LSTM
  → Bidirectional LSTM: capture long-range temporal dependencies
  → Linear projection → CTC output (B, T', vocab_size)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → ReLU → MaxPool block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),  # halve frequency axis, keep time
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNNEncoder(nn.Module):
    """
    Stack of ConvBlocks that compress the mel-frequency axis
    while preserving the time axis for the LSTM.
    """

    def __init__(
        self,
        in_channels: int = 1,
        cnn_channels: Tuple[int, ...] = (32, 64, 128),
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        channels = [in_channels] + list(cnn_channels)
        self.blocks = nn.ModuleList([
            ConvBlock(channels[i], channels[i + 1], kernel_size, dropout)
            for i in range(len(cnn_channels))
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, n_mels, T) → (B, C, n_mels', T)"""
        for block in self.blocks:
            x = block(x)
        return x


class BidirectionalLSTM(nn.Module):
    """Bidirectional LSTM with residual connection (if sizes match)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, input_size) → (B, T, 2*hidden_size)"""
        out, _ = self.lstm(x)
        return out


class CNNLSTMASR(nn.Module):
    """
    Full CNN + LSTM ASR model with CTC decoding.

    Input:  (B, 1, n_mels, T)  — log-mel spectrogram
    Output: (B, T, vocab_size)  — log-softmax over vocabulary per timestep
    """

    def __init__(
        self,
        vocab_size: int,
        n_mels: int = 80,
        cnn_channels: Tuple[int, ...] = (32, 64, 128),
        cnn_kernel_size: int = 3,
        cnn_dropout: float = 0.2,
        lstm_hidden_size: int = 512,
        lstm_num_layers: int = 3,
        lstm_dropout: float = 0.3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_mels = n_mels

        # CNN encoder
        self.cnn = CNNEncoder(
            in_channels=1,
            cnn_channels=cnn_channels,
            kernel_size=cnn_kernel_size,
            dropout=cnn_dropout,
        )

        # Compute CNN output frequency dimension
        freq_dim = n_mels
        for _ in cnn_channels:
            freq_dim = freq_dim // 2  # each MaxPool2d(2,1) halves freq
        lstm_input_size = freq_dim * cnn_channels[-1]

        # Bidirectional LSTM
        self.lstm = BidirectionalLSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=lstm_dropout,
        )

        # Linear projection to vocab
        self.fc = nn.Linear(lstm_hidden_size * 2, vocab_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, n_mels, T)
        returns: (B, T, vocab_size) log-softmax
        """
        # CNN: (B, 1, n_mels, T) → (B, C, freq', T)
        x = self.cnn(x)

        B, C, freq, T = x.shape
        # Reshape: merge C and freq, keep T → (B, T, C*freq)
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * freq)

        # LSTM: (B, T, C*F) → (B, T, 2*hidden)
        x = self.lstm(x)
        x = self.dropout(x)

        # Projection: (B, T, vocab_size)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)

    def ctc_loss(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
        blank: int = 0,
    ) -> torch.Tensor:
        """Compute CTC loss. log_probs: (B, T, V) → expects (T, B, V)."""
        log_probs = log_probs.permute(1, 0, 2)  # (T, B, V)
        return F.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=blank,
            reduction="mean",
            zero_infinity=True,
        )

    def greedy_decode(self, log_probs: torch.Tensor, blank: int = 0) -> list:
        """
        Greedy CTC decoding: argmax at each timestep, collapse repeats, remove blanks.
        log_probs: (B, T, V)
        returns: list of lists of token IDs
        """
        preds = log_probs.argmax(dim=-1)  # (B, T)
        results = []
        for seq in preds:
            decoded = []
            prev = None
            for token in seq.tolist():
                if token != blank and token != prev:
                    decoded.append(token)
                prev = token
            results.append(decoded)
        return results

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(vocab_size: int, config: dict) -> CNNLSTMASR:
    """Instantiate CNN+LSTM model from config dict."""
    cfg = config["cnn_lstm"]
    model = CNNLSTMASR(
        vocab_size=vocab_size,
        n_mels=config["audio"]["n_mels"],
        cnn_channels=cfg["cnn_channels"],
        cnn_kernel_size=cfg["cnn_kernel_size"],
        cnn_dropout=cfg["cnn_dropout"],
        lstm_hidden_size=cfg["lstm_hidden_size"],
        lstm_num_layers=cfg["lstm_num_layers"],
        lstm_dropout=cfg["lstm_dropout"],
    )
    return model
