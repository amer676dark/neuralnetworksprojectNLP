# Dataset Description

## Primary Dataset: Mozilla Common Voice 13.0 — Arabic

| Property | Value |
|----------|-------|
| **Source** | Mozilla Foundation via HuggingFace |
| **HuggingFace ID** | `mozilla-foundation/common_voice_13_0` (language: `ar`) |
| **Language** | Modern Standard Arabic (MSA) |
| **Format** | MP3 audio + UTF-8 text transcriptions |
| **Sample Rate** | Resampled to 16 kHz for all models |
| **License** | CC0 (public domain) |

### Splits

| Split | Approx. Samples | Use |
|-------|----------------|-----|
| `train` | ~17,000 | Model training |
| `validation` | ~3,000 | Hyperparameter tuning |
| `test` | ~3,000 | Final evaluation (never seen during training) |

### Audio Properties

| Property | Value |
|----------|-------|
| Average duration | ~4–6 seconds per clip |
| Total hours (train) | ~25 hours |
| Speakers | Multiple volunteer speakers (varied accents) |
| Recording environment | Crowdsourced (home/office, variable quality) |

### Transcript Properties

| Property | Value |
|----------|-------|
| Average words per sentence | 5–8 words |
| Script | Arabic (right-to-left) |
| Diacritics | Mostly undiacritized (common in MSA text) |
| Vocabulary size (chars) | ~70 unique characters including Arabic letters, digits, spaces |

### Why Mozilla Common Voice?

- **Free & open** — no license restrictions for academic use
- **Largest open Arabic ASR corpus** available on HuggingFace
- **Diverse speakers** — not recorded by a single speaker, so models must generalise
- **Clean transcriptions** — community-validated text labels
- **Easy to load** — one line via `datasets` library

### How to Download

The dataset is downloaded automatically on first use:

```python
from datasets import load_dataset, Audio

dataset = load_dataset(
    "mozilla-foundation/common_voice_13_0",
    "ar",
    split="train",
    trust_remote_code=True,
)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
```

First download is ~2 GB. Cached to `~/.cache/huggingface/datasets/` automatically.

---

## Other Available Datasets (not used by default)

| Dataset | Size | Notes |
|---------|------|-------|
| Arabic Speech Corpus | ~1.5h | Single speaker, MSA, diacritized |
| MASC (`hirundo-io/MASC`) | ~1,000h | Large, multi-domain, HuggingFace |
| Arabic Broadcast News (LDC2006S46) | ~20h | News domain, requires LDC license |
| EJUST Dataset | ~? | Provided by instructor — do not share publicly |

To switch dataset, change `data.dataset_name` in `configs/config.yaml`.

---

## Preprocessing Steps (applied to all models)

1. **Resample** — convert any sample rate to 16,000 Hz (required by all three models)
2. **Mono** — convert stereo to mono by averaging channels
3. **Normalize** — scale waveform to [−1, 1]
4. **Trim/Pad** — clip or zero-pad to `max_audio_length` seconds (default: 10s)
5. **Log-Mel Spectrogram** — 80 mel bands, FFT size 512, hop 160, window 400 (for CNN+LSTM only; Whisper and Wav2Vec take raw waveform)

## Data Augmentation (CNN+LSTM training only)

Applied randomly during training to improve generalisation:

| Technique | Parameters |
|-----------|-----------|
| Additive white noise | factor = 0.005 |
| Time shift | ±10% of clip length |
