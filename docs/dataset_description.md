# Dataset Description

This project uses **three Arabic speech datasets** for training and evaluation.

| Dataset | Role | Samples | Hours | Source |
|---|---|---|---|---|
| **Kaggle arabic_tts** | Primary training set | ~78,700 | ~75 | Kaggle (CV 11.0 Arabic subset) |
| **Arabic Speech Corpus** | Single-speaker eval | 1,813 | ~3 | arabicspeechcorpus.com |
| **MASC** | Cross-domain eval | ~1,000+ | varied | HuggingFace |

---

## 1. Kaggle "arabic_tts" (Primary Dataset)

**Source:** [kaggle.com/datasets/mayarjao/arabic-tts](https://www.kaggle.com/datasets/mayarjao/arabic-tts)
**Origin:** Arabic subset of Mozilla Common Voice 11.0
**License:** CC0 1.0 (Public Domain)

| Property | Value |
|---|---|
| Format | `.wav` files + `metadata.csv` (LJSpeech-style: `filename | transcript`) |
| Total clips | ~78,700 |
| Sample rate | 16 kHz (already resampled in Kaggle release) |
| Language | Modern Standard Arabic (MSA) |
| Total audio | ~75 hours |
| Layout | `wavs/` folder + `metadata.csv` + `metadata-wav.csv` |

### How to download (on the GPU box)

```bash
kaggle datasets download mayarjao/arabic-tts -p data/arabic_tts --unzip
```

Requires a Kaggle API token at `~/.kaggle/kaggle.json`.

### Why this dataset

- **Largest open Arabic ASR corpus** with a permissive license
- **Already preprocessed** to 16 kHz wavs — no MP3 → WAV conversion needed
- **LJSpeech-style layout** — simple loader, no HuggingFace gating
- **Diverse speakers** from the original Common Voice contributors

### Training/eval split

We split 90/5/5 (train/val/test) deterministically by line order in
`metadata.csv`. For training we use the first 20,000 train samples
(~20 hours of audio) — empirically the best WER/cost trade-off on a
single GPU rental.

---

## 2. Arabic Speech Corpus

**Source:** [arabicspeechcorpus.com](https://en.arabicspeechcorpus.com/)
**Origin:** PhD thesis corpus (Halabi, Edinburgh)

| Property | Value |
|---|---|
| Total clips | 1,813 (+ 18 min held-out eval set) |
| Speaker | Single male speaker |
| Sample rate | 48 kHz (resampled to 16 kHz) |
| Annotations | `.lab` (orthographic) + `.TextGrid` (phoneme alignments) |
| Format | `wav/` + `orthographic-transcript.txt` (tab-separated) |

### How to download

```bash
wget https://en.arabicspeechcorpus.com/arabic-speech-corpus.zip
unzip arabic-speech-corpus.zip -d data/arabic_speech_corpus
```

No authentication required.

### Role in this project

- **Eval-only dataset** — used to measure how well models trained on
  crowd-sourced Common Voice generalize to a clean studio recording.
- Phoneme-level alignment makes it useful for analyzing where ASR fails.

---

## 3. MASC (HuggingFace)

**Source:** [huggingface.co/datasets/hirundo-io/MASC](https://huggingface.co/datasets/hirundo-io/MASC)
**License:** CC-BY-4.0

| Property | Value |
|---|---|
| Samples | ~1,000+ |
| Domain | Multi-domain Arabic speech |
| Format | HuggingFace `audio` column + `text` column |
| Authentication | None |
| Splits | Single 'train' split — we manually 80/10/10 |

### How to download

Automatic via the HuggingFace `datasets` library — no manual step.

### Role in this project

- **Cross-domain evaluation** — different recording conditions than CV
- Tests robustness of all models when domain shifts

---

## Preprocessing (applied to all datasets, all models)

1. **Resample** — to 16 kHz mono float32
2. **Normalize** — peak-normalize to [-1, 1]
3. **Pad/Trim** — to `max_audio_length` seconds (default 10)
4. **Log-Mel Spectrogram** — 80 mel bands, FFT 512, hop 160, window 400
   *(CNN+LSTM only; Whisper/Wav2Vec2/SeamlessM4T consume raw waveform)*

## Data augmentation (CNN+LSTM training only)

| Technique | Parameters |
|---|---|
| Additive Gaussian noise | std = 0.005 |
| Time shift | ±10% of clip length |
| SpecAugment | 2 freq masks (≤27 bins), 2 time masks (≤100 frames) |

---

## Vocabulary

Built dynamically from the training set transcripts. Typical Arabic
vocabulary covers ~60–80 characters:
- 28 base Arabic letters + variants (alif forms, ta-marbuta, etc.)
- 10 digits
- Punctuation, space
- Special tokens: `<pad>`, `<unk>`, `<sos>`, `<eos>` (indices 0–3, with 0 = CTC blank)
