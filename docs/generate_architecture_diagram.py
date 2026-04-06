"""
Generate the system architecture diagram and save it to docs/architecture.png
Run: python docs/generate_architecture_diagram.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent / "architecture.png"


def box(ax, x, y, w, h, label, sublabel="", color="#4A90D9", text_color="white", fontsize=10):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.02", linewidth=1.5,
                          edgecolor="#2c3e50", facecolor=color, zorder=3)
    ax.add_patch(rect)
    ax.text(x, y + (0.06 if sublabel else 0), label,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=text_color, zorder=4)
    if sublabel:
        ax.text(x, y - 0.1, sublabel, ha="center", va="center",
                fontsize=7.5, color=text_color, alpha=0.9, zorder=4)


def arrow(ax, x1, y1, x2, y2, label=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="#2c3e50",
                                lw=1.8, mutation_scale=18), zorder=2)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.05, my, label, fontsize=8, color="#555", va="center")


fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis("off")
fig.patch.set_facecolor("#f8f9fa")
ax.set_facecolor("#f8f9fa")

# ── Title ──────────────────────────────────────────────────────────────────
ax.text(8, 9.5, "Arabic Speech Recognition System — Architecture",
        ha="center", va="center", fontsize=15, fontweight="bold", color="#2c3e50")

# ── INPUT ─────────────────────────────────────────────────────────────────
box(ax, 8, 8.6, 3.2, 0.65, "Arabic Audio Input",
    "WAV / MP3  ·  any sample rate", color="#27ae60")

arrow(ax, 8, 8.27, 8, 7.75)

# ── PREPROCESSING ─────────────────────────────────────────────────────────
box(ax, 8, 7.45, 3.8, 0.55, "Preprocessing",
    "Resample → 16 kHz  ·  Normalize  ·  Log-Mel Spectrogram (80 bands)",
    color="#8e44ad", fontsize=9)

arrow(ax, 8, 7.17, 8, 6.65)

# ── three model branches ───────────────────────────────────────────────────
ax.text(8, 6.45, "Three Parallel Models", ha="center", va="center",
        fontsize=10, color="#555", style="italic")

# branch arrows
ax.annotate("", xy=(2.8, 5.85), xytext=(8, 6.2),
            arrowprops=dict(arrowstyle="-|>", color="#2c3e50", lw=1.5, mutation_scale=14))
ax.annotate("", xy=(8, 5.85), xytext=(8, 6.2),
            arrowprops=dict(arrowstyle="-|>", color="#2c3e50", lw=1.5, mutation_scale=14))
ax.annotate("", xy=(13.2, 5.85), xytext=(8, 6.2),
            arrowprops=dict(arrowstyle="-|>", color="#2c3e50", lw=1.5, mutation_scale=14))

# MODEL 1 — CNN+LSTM
box(ax, 2.8, 5.5, 4.2, 0.65, "Model 1 — CNN + BiLSTM + CTC",
    "Custom · 20M params · trained from scratch", color="#e74c3c")
box(ax, 2.8, 4.6, 3.8, 0.55, "CNN Encoder",
    "3× ConvBlock (32→64→128 ch)  ·  compress frequency", color="#c0392b", fontsize=9)
arrow(ax, 2.8, 4.32, 2.8, 3.88)
box(ax, 2.8, 3.65, 3.8, 0.45, "Bidirectional LSTM",
    "3 layers · 512 hidden · dropout 0.3", color="#c0392b", fontsize=9)
arrow(ax, 2.8, 3.42, 2.8, 2.98)
box(ax, 2.8, 2.75, 3.0, 0.45, "CTC Decoder",
    "Greedy argmax · collapse repeats", color="#c0392b", fontsize=9)
arrow(ax, 2.8, 2.52, 2.8, 2.02)

# MODEL 2 — Whisper
box(ax, 8, 5.5, 4.2, 0.65, "Model 2 — OpenAI Whisper",
    "Pretrained · 769M params · zero-shot Arabic", color="#2980b9")
box(ax, 8, 4.6, 3.8, 0.55, "Audio Encoder",
    "Conv1D stem + Transformer (32 layers)", color="#1a6fa0", fontsize=9)
arrow(ax, 8, 4.32, 8, 3.88)
box(ax, 8, 3.65, 3.8, 0.45, "Text Decoder",
    "Transformer decoder · beam search (size 5)", color="#1a6fa0", fontsize=9)
arrow(ax, 8, 3.42, 8, 2.02)

# MODEL 3 — Wav2Vec
box(ax, 13.2, 5.5, 4.2, 0.65, "Model 3 — Wav2Vec 2.0 XLSR",
    "Pretrained Arabic · 300M params · CTC head", color="#e67e22")
box(ax, 13.2, 4.6, 3.8, 0.55, "Feature Encoder",
    "7-layer CNN · raw waveform → features", color="#ca6f1e", fontsize=9)
arrow(ax, 13.2, 4.32, 13.2, 3.88)
box(ax, 13.2, 3.65, 3.8, 0.45, "Transformer Context",
    "24 layers · self-attention · CTC output", color="#ca6f1e", fontsize=9)
arrow(ax, 13.2, 3.42, 13.2, 2.02)

# ── OUTPUT ────────────────────────────────────────────────────────────────
ax.annotate("", xy=(8, 1.8), xytext=(2.8, 2.0),
            arrowprops=dict(arrowstyle="-|>", color="#2c3e50", lw=1.5, mutation_scale=14))
ax.annotate("", xy=(8, 1.8), xytext=(8, 2.0),
            arrowprops=dict(arrowstyle="-|>", color="#2c3e50", lw=1.5, mutation_scale=14))
ax.annotate("", xy=(8, 1.8), xytext=(13.2, 2.0),
            arrowprops=dict(arrowstyle="-|>", color="#2c3e50", lw=1.5, mutation_scale=14))

box(ax, 8, 1.5, 3.8, 0.55, "Arabic Text Transcript",
    "UTF-8 Arabic string", color="#27ae60")

arrow(ax, 8, 1.22, 8, 0.78)

# ── EVALUATION ────────────────────────────────────────────────────────────
box(ax, 8, 0.5, 5.0, 0.5, "Evaluation: WER · CER · Per-sample analysis",
    "", color="#555", fontsize=9)

# ── DATASET BADGE ─────────────────────────────────────────────────────────
badge = FancyBboxPatch((0.15, 0.1), 3.5, 1.2,
                       boxstyle="round,pad=0.03", linewidth=1,
                       edgecolor="#aaa", facecolor="#ecf0f1", zorder=3)
ax.add_patch(badge)
ax.text(1.9, 1.0, "Dataset", fontsize=9, fontweight="bold", ha="center", color="#2c3e50", zorder=4)
ax.text(1.9, 0.72, "Mozilla Common Voice 13", fontsize=8, ha="center", color="#555", zorder=4)
ax.text(1.9, 0.5, "Arabic · 16 kHz · train/val/test", fontsize=7.5, ha="center", color="#777", zorder=4)
ax.text(1.9, 0.28, "HuggingFace: common_voice_13_0", fontsize=7, ha="center",
        color="#888", style="italic", zorder=4)

# ── LEGEND ────────────────────────────────────────────────────────────────
patches = [
    mpatches.Patch(color="#e74c3c", label="CNN+LSTM (custom, from scratch)"),
    mpatches.Patch(color="#2980b9", label="Whisper (zero-shot, OpenAI)"),
    mpatches.Patch(color="#e67e22", label="Wav2Vec 2.0 (pretrained Arabic)"),
]
ax.legend(handles=patches, loc="lower right", fontsize=8.5,
          framealpha=0.9, edgecolor="#ccc", bbox_to_anchor=(0.99, 0.01))

plt.tight_layout()
plt.savefig(OUT, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {OUT}")
