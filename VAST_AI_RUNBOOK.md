# vast.ai Runbook — Arabic ASR Full Project

End-to-end playbook for running the **expanded scope** on a rented GPU
(RTX Pro 6000 96GB recommended) in ~3 hours.

**What you'll produce in this single run:**
- Trained custom CNN+LSTM (64M params, 20k training samples)
- Evaluation of 3 pretrained ASR baselines: Whisper, Wav2Vec2, SeamlessM4T-v2
- All 4 optional advanced tasks: keyword spotting, speaker ID, emotion, summarize+search
- Three datasets supported: Kaggle Arabic TTS (Common Voice 11), Arabic Speech Corpus, MASC
- All result artifacts ready to download and demo locally

---

## 0 · Before you rent — 1 minute on your Mac

```bash
cd /Users/amerabdelhamid/NeuralNetworksProject
python smoke_test.py   # ensures imports + model build OK locally
git status
git add -A
git commit -m "Expanded scope: Kaggle dataset, SeamlessM4T, 4 advanced tasks"
git push
```

You will also need three credentials:

| Credential | Why | Where |
|---|---|---|
| **HuggingFace token** (Read scope) | Wav2Vec2-XLSR, SeamlessM4T-v2, mT5-XLSum, MASC, sentence-transformers | https://huggingface.co/settings/tokens |
| **Kaggle API token** (kaggle.json) | Downloading the arabic_tts dataset | https://www.kaggle.com/settings → API → "Create New Token" |
| **Accept HF terms** for SeamlessM4T-v2 | Gated model | https://huggingface.co/facebook/seamless-m4t-v2-large |

Have all three ready before renting.

---

## 1 · Pick a vast.ai instance

| Filter | Value |
|---|---|
| GPU | **RTX Pro 6000 (96 GB)** — type #33665865 etc. ~$1.34/hr |
| Disk | ≥ 80 GB |
| Image | `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime` or similar CUDA 12 + PyTorch image |
| Bandwidth | "Internet speed" sort high — you'll pull ~15 GB of datasets + models |
| Budget | ~3 hours × $1.34 ≈ **$4 total** |

Click **Rent**, then **Connect** (SSH or web terminal).

---

## 2 · One-time setup on the GPU box (≈ 10 min)

```bash
# 2.1 Clone repo
cd /workspace
git clone https://github.com/amer676dark/neuralnetworksprojectNLP.git
cd neuralnetworksprojectNLP

# 2.2 Verify GPU is visible to PyTorch
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 2.3 Install Python deps
pip install -r requirements.txt

# 2.4 Smoke test — verifies imports + model build (no dataset)
python smoke_test.py
```

Expect: `SMOKE TEST PASSED — safe to rent a GPU.` (already passed before
you got here, but worth re-running once the box is set up.)

---

## 3 · Authenticate (≈ 2 min)

### HuggingFace
```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
huggingface-cli login --token $HF_TOKEN
```

### Kaggle
```bash
# Either upload your kaggle.json (from your local ~/.kaggle/kaggle.json)
mkdir -p ~/.kaggle
nano ~/.kaggle/kaggle.json   # paste the JSON from kaggle.com → Settings → API
chmod 600 ~/.kaggle/kaggle.json

# Or set via env vars
# export KAGGLE_USERNAME=...
# export KAGGLE_KEY=...
```

---

## 4 · Download all three datasets (≈ 15–20 min)

```bash
mkdir -p data

# 4.1 Kaggle arabic_tts (~6 GB, the main training dataset)
kaggle datasets download mayarjao/arabic-tts -p data/arabic_tts --unzip

# Verify
ls data/arabic_tts/wavs | wc -l       # ~78,700
ls data/arabic_tts/metadata*.csv

# 4.2 Arabic Speech Corpus (~1.5 GB)
mkdir -p data/arabic_speech_corpus
wget -O /tmp/asc.zip https://en.arabicspeechcorpus.com/arabic-speech-corpus.zip
unzip -q /tmp/asc.zip -d data/arabic_speech_corpus
rm /tmp/asc.zip

# MASC auto-downloads from HuggingFace when first accessed — nothing to do here.
```

---

## 5 · Train CNN+LSTM (≈ 50–70 min)

Config defaults (already in `configs/config.yaml`) reflect the v2 architecture:
- Dataset: `kaggle_arabic_tts` · 40,000 train, 1,000 val/test
- Model: ~30M params · CNN with 4× time subsampling · 3-layer BiLSTM (hidden 512) · 8-head self-attention
- Batch 96 · 25 epochs · LR 7e-4 with OneCycleLR · AMP bf16 · torch.compile · EMA decay 0.999
- GPU-side mel extraction (no librosa on CPU)

```bash
python training/train_cnn_lstm.py --config configs/config.yaml 2>&1 | tee outputs/logs/train.log
```

Expect:
- "Loading Kaggle arabic_tts from data/arabic_tts (train)..."
- "Vocab size: ~110–130"
- "Model parameters: ~30,000,000"
- "EMA decay: 0.999"
- "torch.compile: reduce-overhead (compilation happens on first batch)"
- First batch is slow (~30–60 s while torch.compile traces the graph)
- Steady state: **≥ 5 it/s** (batch=96 → ~3.5 min/epoch)
- Per-epoch summary lines `Train Loss / Val Loss / Val WER / elapsed`
- Best checkpoint: `outputs/checkpoints/cnn_lstm/best_model.pt`
- Metric history streamed to `outputs/results/cnn_lstm_history.json` every epoch

**Mid-run OOM?** Drop `batch_size` to 64 in `configs/config.yaml` and resume:
```bash
python training/train_cnn_lstm.py --config configs/config.yaml \
  --resume outputs/checkpoints/cnn_lstm/best_model.pt
```

---

## 6 · Evaluate all four models (≈ 25–35 min)

```bash
python evaluation/evaluate_all.py --config configs/config.yaml --num_samples 200 2>&1 | tee outputs/logs/eval.log
```

Runs Whisper-medium, Wav2Vec2-XLSR-Arabic, SeamlessM4T-v2-large, and your trained CNN+LSTM.

First-time HuggingFace downloads:
- Whisper-medium: ~3 GB
- Wav2Vec2-XLSR-arabic: ~1.2 GB
- SeamlessM4T-v2-large: ~9 GB
- mT5-XLSum (used in pipeline tab): ~2 GB
- Sentence-transformers MiniLM: ~500 MB
- ECAPA-TDNN: ~80 MB
- Emotion model: ~1.2 GB

Outputs:
- `outputs/results/all_models_evaluation.json`
- `outputs/results/whisper_predictions.json`
- `outputs/results/wav2vec_predictions.json`
- `outputs/results/seamless_m4t_predictions.json`
- `outputs/results/cnn_lstm_predictions.json`
- `outputs/results/model_comparison.png`
- `outputs/results/cnn_lstm_wer_dist.png`

---

## 7 · Pre-warm the 4 advanced task models (≈ 5 min)

Pre-pull the smaller models so the demo loads fast when you show it.
(Optional — they'll download on first use otherwise.)

```bash
python -c "
from advanced.keyword_spotting import spot_keywords
print('keyword_spotting OK:', len(spot_keywords('اجتماع طوارئ غدا', ['طوارئ'])))

from advanced.speaker_id import SpeakerIdentifier
SpeakerIdentifier(device='auto')

from advanced.emotion import EmotionRecognizer
EmotionRecognizer(device='auto')

from advanced.summarize_search import ArabicSummarizer, TranscriptSearch
ArabicSummarizer(device='auto').summarize('هذا نص اختباري لتجربة التلخيص العربي.')
TranscriptSearch(device='auto')

print('All 4 advanced task models pre-loaded.')
"
```

---

## 8 · Quick smoke run of each advanced task (≈ 5 min)

```bash
python - <<'PY'
import torch, numpy as np
sr = 16000
wav = (np.random.randn(sr * 4) * 0.1).astype(np.float32)

# Keyword spotting (text-based, doesn't need audio)
from advanced.keyword_spotting import spot_keywords
print("KEYWORDS:", spot_keywords("اجتماع طوارئ غدا قبل الامتحان النهائي",
                                  ["طوارئ", "امتحان"]))

# Emotion
from advanced.emotion import EmotionRecognizer
em = EmotionRecognizer(device="auto")
print("EMOTION:", em.predict(wav))

# Speaker ID
from advanced.speaker_id import SpeakerIdentifier
sp = SpeakerIdentifier(device="auto")
sp.enroll("speaker_a", wav)
print("SPEAKER:", sp.identify(wav))

# Summarize + Search
from advanced.summarize_search import ArabicSummarizer, TranscriptSearch
summ = ArabicSummarizer(device="auto")
print("SUMMARY:", summ.summarize("الذكاء الاصطناعي مجال متطور ويشمل التعلم الآلي والتعلم العميق."))

idx = TranscriptSearch(device="auto")
idx.add("doc1", "اجتماع الميزانية يوم الإثنين الساعة العاشرة صباحا")
idx.add("doc2", "موعد الامتحان النهائي يوم الأربعاء")
print("SEARCH:", [(h['id'], h['score']) for h in idx.query("ميزانية", top_k=2)])
PY
```

Each should print a result without crashing.

---

## 9 · Pull results back to your Mac

From the GPU box:

```bash
cd /workspace/neuralnetworksprojectNLP
# Keep only best_model.pt, drop per-epoch checkpoints
zip -r outputs.zip outputs/ -x "outputs/checkpoints/cnn_lstm/checkpoint_epoch*.pt"
ls -lh outputs.zip
```

From your Mac (vast.ai shows the scp port/host in the "Connect" tab):

```bash
cd /Users/amerabdelhamid/NeuralNetworksProject
scp -P <SSH_PORT> root@<VAST_HOST>:/workspace/neuralnetworksprojectNLP/outputs.zip .
unzip outputs.zip
ls outputs/checkpoints/cnn_lstm/best_model.pt
cat outputs/results/all_models_evaluation.json
```

---

## 10 · Stop the GPU instance

Important — destroy or stop on vast.ai dashboard, or you keep paying.

---

## 11 · Demo locally (no GPU needed)

```bash
python demo.py
# http://localhost:7861
```

Seven tabs:
1. **Live Transcribe** — Whisper / Wav2Vec2 / SeamlessM4T / CNN+LSTM
2. **Compare Models** — same audio through all 4, side-by-side with WER
3. **Batch Files** — upload many wavs, get a CSV
4. **Keyword Spotting** — find specific words in transcribed Arabic
5. **Speaker ID** — enroll speakers, identify + diarize
6. **Emotion** — classify spoken emotion
7. **Summarize + Search** — Speech → Text → Summary, with a semantic search index

For the presentation, the highest-impact demos are **Tab 2** (model
comparison) and **Tab 7** (the full Speech → Text → Summary → Search pipeline).

---

## Deliverables checklist

| # | Item | Where |
|---|---|---|
| 1 | Source code | The repo (cleanly organized) |
| 2 | Dataset description | `docs/dataset_description.md` |
| 3 | System architecture diagram | `docs/architecture.png` |
| 4 | Experiments | `docs/experiments.md` |
| 5 | Evaluation results | `outputs/results/` (JSONs + 2 PNG plots) |
| 6 | Demo interface | `demo.py` (7 tabs) + `train_evaluate.py` (training UI) |

**Optional advanced tasks completed:**
- Keyword Spotting (Arabic-normalized + fuzzy matching)
- Speaker Identification (ECAPA-TDNN + simple diarization)
- Emotion Detection (Wav2Vec2 XLSR, 8 classes)
- Summarization + Search (Speech → mT5-XLSum → MiniLM index)

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `torch.cuda.is_available()` is `False` | Picked a non-GPU instance. Destroy + re-rent. |
| Kaggle 401 / 403 | Check `~/.kaggle/kaggle.json` permissions (`chmod 600`) |
| HF "gated dataset" | Accept terms on huggingface.co while logged in |
| SeamlessM4T OOM | Switch `seamless_m4t.model_name` to `facebook/hf-seamless-m4t-medium` |
| Training CTC `inf` for many batches | Lower LR to 1e-4, or shrink `max_audio_length` to 8 |
| Slow Kaggle download | Try a different vast.ai datacenter (some peer poorly with Kaggle) |
| `speechbrain` import fails | `pip install speechbrain` — check version ≥1.0 |
| Sentence-transformer 401 | `huggingface-cli login` |
