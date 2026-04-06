"""Visualization utilities: spectrograms, training curves, WER analysis."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import List, Optional, Dict
import librosa.display


def plot_waveform(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    title: str = "Waveform",
    save_path: Optional[str] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 3))
    time = np.arange(len(waveform)) / sample_rate
    ax.plot(time, waveform, linewidth=0.5, color="steelblue")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_mel_spectrogram(
    mel_spec: np.ndarray,
    sample_rate: int = 16000,
    hop_length: int = 160,
    title: str = "Log-Mel Spectrogram",
    save_path: Optional[str] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    img = librosa.display.specshow(
        mel_spec,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        ax=ax,
        cmap="viridis",
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_wers: Optional[List[float]] = None,
    val_wers: Optional[List[float]] = None,
    save_path: Optional[str] = None,
) -> None:
    n_plots = 2 if (train_wers and val_wers) else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    epochs = range(1, len(train_losses) + 1)

    # Loss plot
    axes[0].plot(epochs, train_losses, "b-o", markersize=3, label="Train Loss")
    axes[0].plot(epochs, val_losses, "r-o", markersize=3, label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("CTC Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # WER plot
    if train_wers and val_wers:
        axes[1].plot(epochs, train_wers, "b-o", markersize=3, label="Train WER")
        axes[1].plot(epochs, val_wers, "r-o", markersize=3, label="Val WER")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("WER")
        axes[1].set_title("Word Error Rate")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_model_comparison(
    model_names: List[str],
    wer_scores: List[float],
    cer_scores: Optional[List[float]] = None,
    save_path: Optional[str] = None,
) -> None:
    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2 if cer_scores else x, wer_scores, width, label="WER", color="steelblue", alpha=0.8)

    if cer_scores:
        bars2 = ax.bar(x + width / 2, cer_scores, width, label="CER", color="coral", alpha=0.8)

    ax.set_xlabel("Model")
    ax.set_ylabel("Error Rate")
    ax.set_title("Model Comparison: WER & CER on Arabic ASR")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate bars
    for bar in bars1:
        ax.annotate(
            f"{bar.get_height():.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix_wer(
    references: List[str],
    hypotheses: List[str],
    save_path: Optional[str] = None,
) -> None:
    """Visualize per-sample WER distribution."""
    from utils.metrics import compute_wer

    per_wer = [compute_wer([r], [h]) for r, h in zip(references, hypotheses)]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(per_wer, bins=20, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(np.mean(per_wer), color="red", linestyle="--", label=f"Mean WER: {np.mean(per_wer):.3f}")
    ax.set_xlabel("WER per Sample")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Per-Sample WER")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
