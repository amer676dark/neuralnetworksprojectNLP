# Deep Learning Based Arabic Audio Understanding and Retrieval System

Intelligent Arabic Speech-to-Text (ASR) pipeline comparing a custom **CNN+BiLSTM+Attention+CTC** model against three pretrained baselines: **OpenAI Whisper-medium**, **Wav2Vec 2.0 XLSR-Arabic**, and **Meta SeamlessM4T-v2-large**. Plus four optional advanced tasks: keyword spotting, speaker identification, emotion detection, and a Speech → Text → Summary → Search pipeline.

> **For the full process write-up, qualitative examples, and lessons learned, see [RESULTS.md](RESULTS.md).**

---

## Headline results — actual measured WER/CER on 300 test samples

| Model | Params | WER | CER |
|---|---:|---:|---:|
| **Wav2Vec2-XLSR-Arabic** (jonatasgrosman) | 300 M | **27.21%** | **12.39%** |
| **SeamlessM4T-v2-large** (Meta) | 2.3 B | 36.85% | 16.62% |
| **Whisper-medium** (OpenAI) | 769 M | 58.22% | 27.76% |
| **CNN + BiLSTM + Attention + CTC** *(ours, from scratch)* | 66 M | 88.68% | 53.73% |

Test set: 300 samples from Kaggle `arabic_tts` (Common Voice 11 Arabic subset), shuffle seed = 42.
Decoding: greedy/argmax across all models, no LM rescoring, no Arabic-text normalization.

**Headline finding**: smallest task-specialized model (Wav2Vec2-XLSR fine-tuned on CV Arabic CTC) **beats** the 7.7× larger SeamlessM4T-v2 and the 2.6× larger Whisper-medium. Scale isn't everything — domain-matched fine-tuning + the right loss function wins.

Full JSON breakdown: [`outputs/results/all_models_evaluation.json`](outputs/results/all_models_evaluation.json)

---

## System Pipeline

```
Audio Input (Arabic Speech)
        │
        ▼
  Preprocessing (GPU)
  Resample → 16 kHz · Normalize · MelSpectrogram(80) · log dB
        │
        ▼
  Speech Recognition (ASR)
  ┌────────────────────────────────────────────────────────┐
  │ Model 1: CNN + BiLSTM + Attention + CTC (custom, 66M)  │
  │ Model 2: OpenAI Whisper-medium             (769M)      │
  │ Model 3: Wav2Vec 2.0 XLSR Arabic           (300M)      │
  │ Model 4: SeamlessM4T-v2-large              (2.3B)      │
  └────────────────────────────────────────────────────────┘
        │
        ▼
  Arabic Transcript
        │
        ├────► Keyword Spotting   (Tab 4)
        ├────► Speaker ID         (Tab 5)
        ├────► Emotion Detection  (Tab 6)
        └────► Summarize + Search (Tab 7)
        │
        ▼
  Evaluation: WER, CER, per-sample distribution
```

---

## Project Structure

```
NeuralNetworksProject/
├── configs/config.yaml                    # All hyperparameters
├── data/dataset.py                        # Multi-dataset loader (Kaggle / Common Voice / MASC / ASC)
├── models/
│   ├── cnn_lstm_asr.py                    # 4× subsampling CNN + BiLSTM + Attention + CTC
│   ├── whisper_asr.py                     # OpenAI Whisper wrapper
│   ├── wav2vec_asr.py                     # Wav2Vec 2.0 fine-tuning + inference
│   └── seamless_m4t_asr.py                # Meta SeamlessM4T-v2 wrapper
├── advanced/                              # 4 optional advanced tasks
│   ├── keyword_spotting.py                # Arabic-aware fuzzy substring matching
│   ├── speaker_id.py                      # ECAPA-TDNN embeddings + diarization
│   ├── emotion.py                         # Wav2Vec2 emotion classifier
│   └── summarize_search.py                # mT5-XLSum summarizer + MiniLM search index
├── training/train_cnn_lstm.py             # AMP bf16, EMA, OneCycleLR, JSON metric history
├── evaluation/evaluate_all.py             # Unified eval over all 4 ASR models
├── utils/{audio_utils,metrics,visualization}.py
├── notebooks/                             # 01 data · 02 training · 03 evaluation
├── outputs/                               # Results from the actual GPU run
│   ├── checkpoints/cnn_lstm/best_model.pt
│   ├── checkpoints/ecapa/                 # cached speaker-ID weights
│   ├── results/                           # JSON metrics + PNG plots + per-sample predictions
│   └── logs/                              # Full train.log + eval.log
├── docs/                                  # Architecture diagram, dataset description, experiments
├── demo.py                                # 7-tab Gradio demo (run locally)
├── train_evaluate.py                      # Alternative Gradio-driven training UI
├── smoke_test.py                          # Local no-GPU verification
├── VAST_AI_RUNBOOK.md                     # Step-by-step GPU rental playbook
├── RESULTS.md                             # Full findings + iteration narrative
└── requirements.txt
```

---

## Models

### 1. Custom CNN + BiLSTM + Attention + CTC (~66 M params)
**Architecture rationale** (informed by Conformer / ESPnet / Whisper best practices):
- **CNN encoder with 4× time + 4× freq subsampling** — drops LSTM sequence length 4×, the single biggest speed win
- **3–4 layer BiLSTM (hidden 768)** — right-sized for ~40 k training samples
- **Multi-head self-attention** — adds global context across the subsampled time axis
- **GPU-side mel extraction** — `torchaudio.transforms.MelSpectrogram` inside the model `forward`, removes the librosa CPU bottleneck
- **Padding-invariant fixed normalization** (Whisper-style `(mel + 40) / 20`) — critical for stability
- **EMA shadow weights** (decay 0.999) for evaluation and best-checkpoint
- **AMP bf16** on Blackwell tensor cores
- Greedy CTC decoding (no LM rescoring)

**Trained on**: 40,000 Kaggle `arabic_tts` samples · 18 epochs · ~14 min on RTX Pro 6000.

### 2. OpenAI Whisper-medium (769 M params, zero-shot)
- Encoder-Decoder Transformer trained on 680 k hours of multilingual audio
- Beam-search decoding (size 5)

### 3. Wav2Vec 2.0 XLSR Arabic (300 M params)
- Self-supervised CNN + Transformer backbone, fine-tuned with CTC on CV Arabic
- HF: `jonatasgrosman/wav2vec2-large-xlsr-53-arabic`

### 4. SeamlessM4T-v2-large (2.3 B params)
- Meta's 2023 multilingual speech model, native Arabic support (`arb` ISO 639-3)
- Replaces deprecated Mozilla DeepSpeech in our requirements coverage

---

## Datasets

| Dataset | Role | Samples | Auth |
|---|---|---|---|
| **Kaggle `mayarjao/arabic-tts`** (Common Voice 11 Arabic subset) | Primary training set | 78 720 (~77 708 usable) | Kaggle API token |
| **Arabic Speech Corpus** | Eval-only (single-speaker MSA studio) | 1 813 | none — `wget` |
| **MASC** (`hirundo-io/MASC`) | Eval-only (multi-domain) | ~1 000 | none |

Default config uses Kaggle `arabic_tts`; swap via `data.source` in `configs/config.yaml`.

---

## Optional Advanced Tasks (all shipped)

| Task | Implementation | Demo tab |
|---|---|---|
| **Keyword Spotting** | Arabic-normalized substring + Levenshtein-fuzzy match over Whisper transcripts | Tab 4 |
| **Speaker Identification** | SpeechBrain ECAPA-TDNN embeddings + cosine matching + sliding-window diarization | Tab 5 |
| **Emotion Detection** | Wav2Vec2 XLSR speech-emotion classifier | Tab 6 |
| **Summarization + Search** | Whisper → mT5-XLSum summarizer → multilingual MiniLM cosine-similarity index | Tab 7 |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify everything compiles (no GPU / dataset needed)
python smoke_test.py

# 3. Run the demo (loads cached models from outputs/checkpoints/ when present)
python demo.py
# → http://localhost:7861   (7 tabs: Live · Compare4 · Batch · Keywords · Speaker · Emotion · Summarize+Search)
```

### Reproduce the GPU training run

See [`VAST_AI_RUNBOOK.md`](VAST_AI_RUNBOOK.md) for the end-to-end vast.ai playbook (clone → auth → datasets → train → eval → zip → download → destroy). Budget: ~2 hrs on an RTX Pro 6000 ≈ $3.

---

## Evaluation Metrics

| Metric | Formula | Why we report both |
|---|---|---|
| **WER** (Word Error Rate) | `(S + D + I) / N` | Standard ASR metric; **understates Arabic ASR quality** when references include diacritics (tashkeel) the model doesn't generate. |
| **CER** (Character Error Rate) | Same, but per character | More informative for Arabic morphology. Wav2Vec2's 12.4 % CER ≈ "1 in 8 characters off" matches the qualitative samples in [RESULTS.md](RESULTS.md). |

For the per-sample REF/HYP pairs across all 4 models, see `outputs/results/*_predictions.json`.

---

## What changed during this project

The project went through ~12 substantive bug fixes / mis-tunes during the GPU run, each documented in the commit history. Three were root-cause failures of training:

1. **CTC padding bug** — train loss collapsed to "predict-blank-everywhere" because we passed the full padded length to CTC instead of the real audio length (`commit 7a3d6c2`).
2. **Per-sample mel normalization included padding** — squashed real-audio dynamic range (`commit 5736216`).
3. **Speaker-clustered split** — CSV-order slice put entire speakers in disjoint train/val/test buckets, faking overfitting (`commit 0e46ed3`).

Each is dissected in [RESULTS.md § Training process](RESULTS.md#training-process--what-actually-happened). Worth reading as a lessons-learned section before defending the report.

---

## Repository history at submission

```
0e46ed3 Shuffle Kaggle arabic_tts samples before train/val/test split
5736216 Fix mel normalization: padding-invariant + gentler SpecAugment
7a3d6c2 Fix CTC padding bug + bump model to ~55M params
5c956a0 Default torch.compile to off — torchaudio SpecAugment incompatibility
fe0240b Switch torch.compile default to mode='default' (no cudagraphs)
3e71fa0 Redesign CNN+LSTM ASR: 4x time subsampling, EMA, torch.compile
9613703 Add AMP bf16 and cut epochs 60->30 for runtime budget
c879b91 Fix leftover mel.shape references after waveform rename
0db5519 Move mel extraction to GPU (kills CPU bottleneck)
d254329 Scale DataLoader workers to host CPU count (4 -> 24)
5336a96 Speed up vocab build: skip audio I/O when only sentence is needed
cda2019 Fix Kaggle arabic_tts loader: auto-detect column order, strip wavs/ prefix
b070074 Swap deprecated facebook/wav2vec2-large-xlsr-53-arabic to jonatasgrosman/
71d8f57 SeamlessM4T: rename `audios` -> `audio` (transformers 5.x API change)
a622972 Scale up training: 40k samples, 138M-param CNN+BiLSTM, 60 epochs
860ed4d Expanded scope: Kaggle dataset, SeamlessM4T-v2, advanced tasks, vast.ai runbook
```
