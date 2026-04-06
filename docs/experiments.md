# Experiments

## Overview

We run three sets of experiments on the Mozilla Common Voice Arabic test set to compare:
1. A **custom CNN+BiLSTM** model trained from scratch
2. **OpenAI Whisper** (zero-shot, no fine-tuning)
3. **Wav2Vec 2.0 XLSR Arabic** (pre-trained, used as-is)

All experiments use the same test split for fair comparison.

---

## Experiment 1 — Baseline: Whisper (Zero-Shot)

**Goal:** Establish a strong upper-bound baseline using a large pre-trained model.

**Setup:**
| Param | Value |
|-------|-------|
| Model | `openai/whisper-medium` |
| Parameters | 769M |
| Language | Arabic (`ar`) |
| Task | Transcribe |
| Beam size | 5 |
| Fine-tuning | None (zero-shot) |
| Test samples | 200 |

**How to run:**
```bash
python evaluation/evaluate_all.py --model whisper --num_samples 200
```

**Expected result:** WER ~15–25%, CER ~10–15%

---

## Experiment 2 — Pre-trained: Wav2Vec 2.0 XLSR Arabic

**Goal:** Compare a self-supervised model fine-tuned on Arabic CTC.

**Setup:**
| Param | Value |
|-------|-------|
| Model | `facebook/wav2vec2-large-xlsr-53-arabic` |
| Parameters | 300M |
| Pre-training | 53-language XLSR |
| Fine-tuning | Arabic CTC (included in checkpoint) |
| Test samples | 200 |

**How to run:**
```bash
python evaluation/evaluate_all.py --model wav2vec --num_samples 200
```

**Expected result:** WER ~20–35%, CER ~12–20%

---

## Experiment 3 — Custom CNN+BiLSTM (From Scratch)

**Goal:** Train a lightweight model end-to-end to understand the fundamental pipeline.

**Architecture:**
```
Input: Log-Mel Spectrogram (B, 1, 80, T)
  ↓
ConvBlock × 3  (32 → 64 → 128 channels, kernel 3×3, MaxPool freq axis)
  ↓
Reshape: (B, T, 1024)
  ↓
BiLSTM × 3  (hidden=512, dropout=0.3)
  ↓
Linear projection → log-softmax
  ↓
CTC Loss / Greedy Decode
```

**Training hyperparameters:**
| Param | Value |
|-------|-------|
| Optimizer | AdamW |
| Learning rate | 3e-4 (OneCycleLR) |
| Batch size | 16 |
| Epochs | 50 |
| Gradient clip | 5.0 |
| Warmup | 10% of steps |
| Train samples | 5,000 (Common Voice Arabic) |
| Augmentation | Noise + time shift |

**How to run:**
```bash
python training/train_cnn_lstm.py --config configs/config.yaml
python evaluation/evaluate_all.py --model cnn_lstm \
  --checkpoint outputs/checkpoints/cnn_lstm/best_model.pt
```

**Expected result:** WER ~40–60%, CER ~25–35%

---

## Experiment 4 — Ablation: Effect of Whisper Model Size

**Goal:** Show WER vs. model size trade-off for Whisper.

| Size | Params | Est. WER | Speed (CPU) |
|------|--------|---------|-------------|
| tiny | 39M | ~40% | Fast (~5s/clip) |
| base | 74M | ~35% | Moderate |
| small | 244M | ~25% | Slow |
| medium | 769M | ~18% | Very slow |
| large-v2 | 1.5B | ~15% | GPU only |

**How to run (change `model_size` in config.yaml or pass via Gradio GUI):**
```yaml
whisper:
  model_size: "tiny"   # change to base / small / medium / large-v2
```

---

## Experiment 5 — All Models Side-by-Side

**Goal:** Single command to evaluate all trained models and produce a comparison chart.

```bash
python evaluation/evaluate_all.py --config configs/config.yaml --num_samples 200
```

Outputs:
- `outputs/results/all_models_evaluation.json` — all metrics
- `outputs/results/model_comparison.png` — bar chart (WER + CER)
- `outputs/results/whisper_predictions.json` — per-sample REF/HYP
- `outputs/results/wav2vec_predictions.json`
- `outputs/results/cnn_lstm_predictions.json`

---

## Evaluation Metric Details

### Word Error Rate (WER)
```
WER = (S + D + I) / N
```
- **S** = substitutions (wrong word)
- **D** = deletions (missed word)
- **I** = insertions (extra word)
- **N** = total words in reference

WER = 0.0 is perfect. WER > 1.0 is possible (too many insertions).

### Character Error Rate (CER)
Same formula but applied at character level. More informative for Arabic because:
- Arabic morphology is complex (one word can have many forms)
- CER is less sensitive to minor spelling differences

### Implementation
Uses `jiwer` library. Arabic text is lowercased and punctuation removed before scoring.

```python
from utils.metrics import compute_wer, compute_cer, compute_batch_metrics
wer = compute_wer(["مرحبا بالعالم"], ["مرحبا عالم"])  # → 0.5
cer = compute_cer(["مرحبا بالعالم"], ["مرحبا عالم"])
```
