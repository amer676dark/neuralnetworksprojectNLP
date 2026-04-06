# Deep Learning Based Arabic Audio Understanding and Retrieval System

Intelligent Arabic Speech-to-Text (ASR) pipeline using CNN+LSTM, OpenAI Whisper, and Wav2Vec 2.0.

---

## System Pipeline

```
Audio Input (Arabic Speech)
        │
        ▼
  Preprocessing
  (Resample → 16kHz, Normalize, Log-Mel Spectrogram)
        │
        ▼
  Speech Recognition (ASR)
  ┌─────────────────────────────────────────────┐
  │  Model 1: CNN + BiLSTM + CTC (custom)       │
  │  Model 2: OpenAI Whisper (medium)           │
  │  Model 3: Wav2Vec 2.0 XLSR Arabic           │
  └─────────────────────────────────────────────┘
        │
        ▼
  Arabic Text Transcript
        │
        ▼
  Evaluation: WER, CER
```

---

## Project Structure

```
NeuralNetworksNLP/
├── configs/
│   └── config.yaml              # All hyperparameters and paths
├── data/
│   └── dataset.py               # Mozilla Common Voice Arabic loader + PyTorch Dataset
├── models/
│   ├── cnn_lstm_asr.py          # Custom CNN+BiLSTM+CTC model
│   ├── whisper_asr.py           # OpenAI Whisper wrapper
│   └── wav2vec_asr.py           # Wav2Vec 2.0 fine-tuning and inference
├── training/
│   └── train_cnn_lstm.py        # CNN+LSTM training loop (CTC loss, OneCycleLR)
├── evaluation/
│   └── evaluate_all.py          # Run all models on test set, compare WER/CER
├── demo/
│   └── app.py                   # Gradio web demo (upload or record audio)
├── utils/
│   ├── audio_utils.py           # Load, resample, mel spectrogram, augmentation
│   ├── metrics.py               # WER, CER, detailed evaluation
│   └── visualization.py        # Waveform, spectrogram, training curves plots
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation.ipynb
├── outputs/
│   ├── checkpoints/             # Saved model weights
│   ├── logs/                    # Training logs
│   └── results/                 # WER/CER JSON + plots
└── requirements.txt
```

---

## Models

### 1. CNN + BiLSTM + CTC (Custom, ~8M params)
- **Input**: Log-Mel Spectrogram (80 mel bands)
- **Encoder**: 3 ConvBlock layers (32→64→128 channels), reduces frequency axis
- **Decoder**: 3-layer Bidirectional LSTM (512 hidden units)
- **Loss**: Connectionist Temporal Classification (CTC)
- **Decoding**: Greedy argmax + repeat collapse
- **Expected WER**: ~40–60% on Common Voice Arabic

### 2. OpenAI Whisper (769M params — medium)
- Encoder-Decoder Transformer trained on 680k hours
- Zero-shot Arabic — no fine-tuning required
- **Expected WER**: ~15–25% on Common Voice Arabic

### 3. Wav2Vec 2.0 XLSR Arabic (300M params)
- Self-supervised CNN + Transformer backbone
- Pre-trained on 53 languages, fine-tuned on Arabic
- **Expected WER**: ~20–35% on Common Voice Arabic

---

## Dataset

**Mozilla Common Voice 13.0 — Arabic**
- Source: `mozilla-foundation/common_voice_13_0` (HuggingFace)
- Language: Arabic (`ar`)
- Format: MP3 → resampled to 16kHz WAV
- Splits: train / validation / test

Other supported datasets (see `configs/config.yaml`):
- Arabic Speech Corpus
- MASC Dataset (HuggingFace: `hirundo-io/MASC`)

---

## Quick Start

### Install dependencies
```bash
pip install -r requirements.txt
```

### Train CNN+LSTM from scratch
```bash
python training/train_cnn_lstm.py --config configs/config.yaml
```

### Evaluate all models
```bash
python evaluation/evaluate_all.py --config configs/config.yaml
```

### Run the Gradio demo
```bash
python demo/app.py
# open http://localhost:7860
```

### Notebooks
```bash
jupyter notebook notebooks/
```

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **WER** | Word Error Rate = (S+D+I)/N — primary metric for ASR |
| **CER** | Character Error Rate — more granular, better for Arabic morphology |

---

## Results (Expected)

| Model | WER | CER | Params | Device |
|-------|-----|-----|--------|--------|
| CNN+LSTM (ours) | ~50% | ~30% | 8M | CPU/GPU |
| Wav2Vec 2.0 XLSR | ~25% | ~15% | 300M | GPU rec. |
| Whisper medium | ~20% | ~12% | 769M | GPU rec. |

---

## Optional Extensions (from requirements)
- **Speaker Identification** — detect different speakers
- **Emotion Detection** — happy, angry, neutral, sad
- **Keyword Spotting** — detect "emergency", "deadline", "exam"
- **Summarization** — Speech → Text → Summary
