# Experiments

This project runs **four ASR models** plus **four optional advanced tasks**.

---

## Models compared

| # | Model | Params | Pretrained | Approach |
|---|---|---|---|---|
| 1 | **CNN+BiLSTM+Attention+CTC** (ours) | ~64M | From scratch | Custom — trained on 20k Kaggle samples |
| 2 | **OpenAI Whisper medium** | 769M | Yes | Encoder-decoder transformer, zero-shot |
| 3 | **Wav2Vec 2.0 XLSR Arabic** | 300M | Yes | Self-supervised + CTC fine-tuned |
| 4 | **SeamlessM4T-v2 large** (replaces DeepSpeech) | 2.3B | Yes | Meta multilingual speech model (2023) |

> **Note on DeepSpeech:** The project requirements list DeepSpeech as an
> example model, but Mozilla DeepSpeech was discontinued in 2020 and has no
> publicly available pretrained Arabic checkpoint. We substituted Meta's
> **SeamlessM4T-v2** (2023) which is a state-of-the-art multilingual speech
> model with native Arabic support. The substitution is explicitly permitted
> by the requirement wording ("such as: ... DeepSpeech").

All four are evaluated on the same 200-sample test slice for fair comparison.

---

## Experiment 1 — Custom CNN+BiLSTM+Attention (from scratch)

**Goal:** Understand the full ASR pipeline by training a model end-to-end.

**Architecture:**
```
Log-Mel Spectrogram (1, 80, T)
  ↓ SpecAugment (2 freq + 2 time masks)
  ↓ Residual CNN × 3 — channels 64 → 128 → 256
  ↓ Linear projection (1280 → 768) + LayerNorm + GELU
  ↓ Bidirectional LSTM × 4 (hidden=768, dropout=0.3)
  ↓ Multi-Head Self-Attention (8 heads, pre-LN, residual)
  ↓ Linear → log-softmax
  ↓ CTC Loss · Greedy Decode
```

**Hyperparameters:**
| Param | Value |
|---|---|
| Optimizer | AdamW (weight decay 1e-5) |
| LR schedule | OneCycleLR, max_lr=5e-4, 10% warmup |
| Batch size | 64 |
| Epochs | 50 |
| Gradient clip | 5.0 |
| Train samples | 20,000 (Kaggle arabic_tts) |
| Augmentation | Noise + time shift + SpecAugment |

**Expected results (RTX Pro 6000, ~45 min training):**

| Metric | Expected |
|---|---|
| Val WER | 30–45% |
| Test WER | 30–45% |
| Test CER | 18–28% |
| Final CTC loss | 0.4–1.2 |

---

## Experiment 2 — Whisper-medium (zero-shot)

**Goal:** Establish a strong upper-bound baseline using a massively pretrained model.

| Param | Value |
|---|---|
| Model | `openai/whisper-medium` |
| Parameters | 769M |
| Language token | `ar` (Arabic) |
| Beam size | 5 |
| Fine-tuning | None (zero-shot) |

**Expected:** WER ~15–22%, CER ~9–14%

---

## Experiment 3 — Wav2Vec 2.0 XLSR Arabic

**Goal:** Compare a self-supervised model that was fine-tuned for Arabic CTC.

| Param | Value |
|---|---|
| Model | `facebook/wav2vec2-large-xlsr-53-arabic` |
| Parameters | 300M |
| Pretraining | XLSR (53 languages) |
| Fine-tuning | Arabic CTC (included in checkpoint) |

**Expected:** WER ~20–35%, CER ~12–20%

---

## Experiment 4 — SeamlessM4T-v2 (state-of-the-art, 2023)

**Goal:** Show modern SOTA — the strongest model in our comparison.

| Param | Value |
|---|---|
| Model | `facebook/seamless-m4t-v2-large` |
| Parameters | 2.3B |
| Language token | `arb` (ISO 639-3, Modern Standard Arabic) |
| Decoding | Greedy (default) |
| Quantization | fp16 on GPU |

**Expected:** WER ~12–20%, CER ~7–12% — typically beats Whisper on Arabic.

---

## Experiment 5 — Side-by-side comparison

Single command evaluates all 4 models on 200 test samples from
the configured dataset:

```bash
python evaluation/evaluate_all.py --config configs/config.yaml --num_samples 200
```

Outputs:
- `outputs/results/all_models_evaluation.json` — WER/CER for each model
- `outputs/results/model_comparison.png` — bar chart of WER + CER
- `outputs/results/{whisper,wav2vec,seamless_m4t,cnn_lstm}_predictions.json` — per-sample REF/HYP
- `outputs/results/cnn_lstm_wer_dist.png` — per-sample WER histogram for our custom model

---

## Optional Advanced Tasks

### A) Keyword Spotting

Detects specific keywords (Arabic or English) in transcribed speech using:
- Arabic-aware normalization (strips tashkeel, unifies alif forms, ta-marbuta)
- Exact substring match
- Levenshtein-fuzzy match (≤1 edit) for single-token keywords

Default keywords: طوارئ (emergency), موعد نهائي (deadline), امتحان (exam),
plus English equivalents.

### B) Speaker Identification (ECAPA-TDNN)

Pipeline:
1. Enroll one or more reference clips per speaker → average 192-dim embedding
2. For a new clip, compute embedding and rank against all enrolled speakers
3. Cosine similarity ≥ threshold → identified, else "unknown"
4. Simple sliding-window diarization: 2-second chunks, 1-second stride, merge adjacent same-speaker segments

Model: `speechbrain/spkrec-ecapa-voxceleb` (pretrained on VoxCeleb).

### C) Emotion Detection

8-class speech emotion classifier:
angry · calm · disgust · fear · happy · neutral · sad · surprise

Model: `harshit345/xlsr-wav2vec-speech-emotion-recognition`
(Wav2Vec2-XLSR backbone, RAVDESS fine-tuned).

### D) Summarize + Search (Speech → Text → Summary → Search)

Three-stage pipeline:

1. **ASR** — Whisper transcribes the audio
2. **Summarize** — `csebuetnlp/mT5_multilingual_XLSum` produces an Arabic summary
3. **Index + Search** — Sentence-transformer
   `paraphrase-multilingual-MiniLM-L12-v2` produces 384-dim embeddings
   stored in an in-memory cosine-similarity index. Queries return top-k.

Demonstrates the full requirement of "Intelligent Audio Analysis System
(Speech → Text → Summary → Search)" mentioned in the project spec.

---

## Evaluation Metrics

### WER
```
WER = (S + D + I) / N
```
where S, D, I are substitutions, deletions, insertions and N is the
reference word count. Range [0, ∞), with 0 = perfect.

### CER
Same formula at the character level. More informative for Arabic because:
- Arabic morphology is rich (one word ↔ many surface forms)
- ASR errors are often a single missing letter, not a whole missing word
- CER is less penalty-heavy for minor agglutination errors

### Per-sample distribution
We additionally plot the per-sample WER histogram for the custom CNN+LSTM
to show how performance varies across the test set.

### Implementation
`jiwer` library, with normalization: lowercase, punctuation strip, whitespace collapse.

---

## Reproducibility

| Setting | Value |
|---|---|
| Random seed | Not pinned (training non-deterministic across CUDA versions anyway) |
| Hardware | RTX Pro 6000 (96 GB) — but any 16+ GB VRAM GPU works |
| Torch | ≥ 2.0 with CUDA 12.1 |
| Transformers | ≥ 4.40 |
| Total compute time | ~2.5 hours on RTX Pro 6000 (training + eval + advanced tasks) |
