# Arabic ASR Project — Full Usage Guide

## Two main files

| File | Purpose | Port |
|------|---------|------|
| `python train_evaluate.py` | Download dataset · Train CNN+LSTM · Evaluate all models · Generate results | 7860 |
| `python demo.py` | Live transcription · Side-by-side model comparison · Batch files | 7861 |

---

## Datasets — what to download

| Dataset | Size | Auth | How to get |
|---------|------|------|-----------|
| **Mozilla Common Voice Arabic** | ~14 GB | HF login + accept terms | See Step A below |
| **MASC** | auto-download | None | Loads automatically — no action needed |
| **Arabic Speech Corpus** | ~1.5 GB | Free registration | See Step B below |
| **EJUST** | varies | Instructor | Google Drive link from instructor |

### Step A — Mozilla Common Voice Arabic
1. Create a free account at https://huggingface.co
2. Go to https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0
3. Click **"Access repository"** and accept the terms
4. Get your token from https://huggingface.co/settings/tokens (click "New token", type: Read)
5. Paste the token into `train_evaluate.py` → Tab 1 → HuggingFace token field
   OR set it once in terminal:
   ```
   huggingface-cli login
   ```

### Step B — Arabic Speech Corpus
1. Go to https://en.arabicspeechcorpus.com/
2. Register (free) and download the corpus ZIP
3. Extract to a folder, e.g. `/Downloads/arabic-speech-corpus/`
4. In `train_evaluate.py` → Tab 1 → set source = `arabic_speech_corpus` and paste the folder path

### EJUST Dataset
1. Download from the Google Drive link your instructor gave you
2. Extract to a local folder
3. In `train_evaluate.py` → Tab 1 → set source = `ejust` and paste the folder path

---

## Workflow — Windows PC (training)

### 1. Install Python + dependencies
```bash
# Install Python 3.10 from python.org (check "Add to PATH")

# Install PyTorch WITH CUDA (do this first, before requirements.txt)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install everything else
pip install -r requirements.txt
```

### 2. Verify GPU
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Should print: True   NVIDIA GeForce RTX 2060 SUPER
```

### 3. Run train_evaluate.py
```bash
python train_evaluate.py
# Opens at http://localhost:7860
```

### 4. Tab 1 — Dataset Setup
- Select your dataset source
- Paste local folder path (for ASC / EJUST) or HF token (for Common Voice)
- Set max train samples (start with 2000 to test, then increase to 5000+)
- Click **Verify Dataset** — confirm 3 samples appear with Arabic text
- Click **Save Config**

### 5. Tab 2 — Train CNN+LSTM
Recommended settings for RTX 2060S:
- Epochs: 50
- Batch size: 32
- Learning rate: 0.0003
- LSTM hidden: 512, LSTM layers: 3
- SpecAugment: ON
- Click **Start Training**

Training log streams live. Best checkpoint saved automatically to:
`outputs/checkpoints/cnn_lstm/best_model.pt`

Expected time: ~25–35 minutes on RTX 2060S

### 6. Tab 3 — Evaluate Models
- Check Whisper (tiny for speed), Wav2Vec2, CNN+LSTM
- Set test samples to 200
- Click **Run Evaluation**
- WER/CER results appear + comparison chart saved to `outputs/results/`

### 7. Push results to GitHub
```bash
git add outputs/results/ outputs/checkpoints/cnn_lstm/best_model.pt
git commit -m "Training results and model checkpoint"
git push
```

---

## Workflow — Mac (presentation)

### 1. Pull from GitHub
```bash
cd /Users/amerabdelhamid/SoftwareProjects/NeuralNetworksNLP
git pull
```

### 2. Run the demo
```bash
python demo.py
# Opens at http://localhost:7861
```

### Tab 1 — Live Transcribe (show this first)
- Select **Whisper tiny** (fastest on M2 Air)
- Click the microphone, speak Arabic, click stop
- Hit Transcribe — transcript appears in ~1–2 seconds
- Switch to **CNN+LSTM** and repeat — show your custom model

### Tab 2 — Compare Models (best for demonstration)
- Upload one Arabic audio clip
- Paste the correct transcript in the Reference field
- Click **Run All Three**
- Three transcripts appear side by side with individual WER
- Bar chart shows which model is most accurate
- This tab directly answers "how do the models compare?"

### Tab 3 — Batch Files
- Upload several .wav files at once
- Download CSV with all transcriptions

### Also show during discussion
- Open `docs/architecture.png` in Preview — walk through the pipeline
- Open `outputs/results/model_comparison.png` — explain WER numbers
- Open `outputs/results/training_curves.png` — show training progress
- Open `docs/experiments.md` — explain the 5 experiments

---

## Running both files at once (optional)
```bash
# Terminal 1
python train_evaluate.py --port 7860

# Terminal 2
python demo.py --port 7861
```

---

## Common errors

| Error | Fix |
|-------|-----|
| `HF_TOKEN` / auth error | Complete Step A above, paste token in Tab 1 |
| `CNN+LSTM checkpoint not found` | Train first in train_evaluate.py → Tab 2 |
| `CUDA not available` | Run `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121` |
| Dataset folder not found | Check the path in Tab 1, make sure you extracted the ZIP |
| Port already in use | Add `--port 7862` to the run command |
| Whisper download slow | First run downloads model weights — wait once, cached after |

---

## Project deliverables checklist

| # | Item | Location |
|---|------|---------|
| 1 | Source code | All `.py` files |
| 2 | Dataset description | `docs/dataset_description.md` |
| 3 | System architecture diagram | `docs/architecture.png` |
| 4 | Experiments | `docs/experiments.md` |
| 5 | Evaluation results | `outputs/results/` (after running evaluation) |
| 6 | Demo interface | `python demo.py` |

---

## Architecture summary (for discussion)

```
Arabic Audio (any sample rate)
        ↓
Preprocessing: resample 16kHz · normalize · log-mel spectrogram (80 bands)
        ↓
┌────────────────────────────────────────────────────────────────┐
│  Model 1: CNN+BiLSTM+Attention (our custom model, 22M params)  │
│    SpecAugment → 3× Residual CNN → BiLSTM×3 → MultiHeadAttn   │
│    Loss: CTC · Decoding: greedy argmax                          │
├────────────────────────────────────────────────────────────────┤
│  Model 2: Whisper medium (OpenAI, 769M params, zero-shot)      │
│    Transformer encoder-decoder · beam search (size 5)           │
├────────────────────────────────────────────────────────────────┤
│  Model 3: Wav2Vec 2.0 XLSR Arabic (Facebook, 300M params)      │
│    Self-supervised CNN+Transformer · CTC head                   │
└────────────────────────────────────────────────────────────────┘
        ↓
Arabic text transcript
        ↓
Evaluation: WER (word error rate) · CER (character error rate)
```

**Why CNN+LSTM gets higher WER than Whisper/Wav2Vec2:**
- Trained on ~5000 samples vs hundreds of thousands
- 22M params vs 300M–769M
- No pre-training on massive datasets
- But it proves you understand the fundamentals end-to-end
