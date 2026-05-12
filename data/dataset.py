"""
Multi-dataset loader for Arabic ASR.

Supported sources:
  - common_voice  : Mozilla Common Voice Arabic (HuggingFace, needs HF login)
  - masc          : MASC Arabic Speech Dataset  (HuggingFace, hirundo-io/MASC)
  - arabic_speech_corpus : local folder download from arabicspeechcorpus.com
  - ejust         : local folder provided by instructor
  - combined      : mix any of the above (set combine_sources in config)
"""

import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Audio
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from utils.audio_utils import (
    extract_mel_spectrogram,
    normalize_audio,
    pad_or_trim,
    apply_augmentation,
    TARGET_SAMPLE_RATE,
)


# ══════════════════════════════════════════════════════════════════
# Per-source loaders
# Each returns an object that supports len() and index access
# with items having {"audio": {"array": ..., "sampling_rate": int}, "sentence": str}
# ══════════════════════════════════════════════════════════════════

def load_common_voice_arabic(
    split: str = "train",
    max_samples: Optional[int] = None,
    hf_token: Optional[str] = None,
) -> object:
    """
    Mozilla Common Voice Arabic.
    Requires free HuggingFace account + accepting dataset terms at:
      https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0
    Then run:  huggingface-cli login
    """
    token = hf_token or os.environ.get("HF_TOKEN")
    print(f"Loading Mozilla Common Voice 17 Arabic ({split})...")
    dataset = load_dataset(
        "mozilla-foundation/common_voice_17_0",
        "ar",
        split=split,
        token=token,
    )
    dataset = dataset.cast_column("audio", Audio(sampling_rate=TARGET_SAMPLE_RATE))
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    print(f"  {len(dataset)} samples loaded.")
    return dataset


def load_masc(
    split: str = "train",
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """
    MASC Arabic Speech Dataset from HuggingFace.
    No authentication required.
    https://huggingface.co/datasets/hirundo-io/MASC
    Note: MASC only has a 'train' split — val/test are created by slicing.
    """
    print(f"Loading MASC Arabic ({split})...")
    dataset = load_dataset("hirundo-io/MASC", split="train", streaming=False)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=TARGET_SAMPLE_RATE))

    # MASC uses 'text' column — rename to 'sentence'
    if "text" in dataset.column_names and "sentence" not in dataset.column_names:
        dataset = dataset.rename_column("text", "sentence")
    elif "transcription" in dataset.column_names and "sentence" not in dataset.column_names:
        dataset = dataset.rename_column("transcription", "sentence")

    # Manual split: 80/10/10
    n = len(dataset)
    train_end = int(n * 0.8)
    val_end   = int(n * 0.9)
    if split == "train":
        dataset = dataset.select(range(train_end))
    elif split == "validation":
        dataset = dataset.select(range(train_end, val_end))
    else:
        dataset = dataset.select(range(val_end, n))

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    print(f"  {len(dataset)} samples loaded.")
    return dataset


def load_arabic_speech_corpus(
    data_dir: str,
    split: str = "train",
    max_samples: Optional[int] = None,
    split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> List[Dict]:
    """
    Arabic Speech Corpus — local folder download from:
      https://en.arabicspeechcorpus.com/

    Expected folder layout:
      data_dir/
        wav/          ← all .wav audio files
        orthographic-transcript.txt   ← tab-separated: filename \\t transcript
        OR
        lab/          ← .lab files with same name as wav (one transcript per file)
    """
    data_dir = Path(data_dir)
    print(f"Loading Arabic Speech Corpus from {data_dir} ({split})...")

    # Find transcript file
    samples = []
    transcript_file = data_dir / "orthographic-transcript.txt"
    wav_dir = data_dir / "wav"

    if transcript_file.exists() and wav_dir.exists():
        # Tab-separated transcript file
        with open(transcript_file, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    fname, text = parts[0], parts[1]
                    wav_path = wav_dir / fname
                    if not wav_path.suffix:
                        wav_path = wav_path.with_suffix(".wav")
                    if wav_path.exists():
                        samples.append({"path": str(wav_path), "sentence": text})
    else:
        # Fallback: scan for wav + matching .lab files
        wav_files = sorted(data_dir.rglob("*.wav"))
        for wav_path in wav_files:
            lab_path = wav_path.with_suffix(".lab")
            txt_path = wav_path.with_suffix(".txt")
            text = ""
            for tp in [lab_path, txt_path]:
                if tp.exists():
                    text = tp.read_text(encoding="utf-8").strip()
                    break
            if text:
                samples.append({"path": str(wav_path), "sentence": text})

    if not samples:
        raise ValueError(
            f"No audio+transcript pairs found in {data_dir}.\n"
            "Expected: wav/ folder + orthographic-transcript.txt\n"
            "OR: wav files with matching .lab or .txt files alongside."
        )

    # Deterministic split
    n = len(samples)
    train_end = int(n * split_ratio[0])
    val_end   = int(n * (split_ratio[0] + split_ratio[1]))
    if split == "train":
        samples = samples[:train_end]
    elif split == "validation":
        samples = samples[train_end:val_end]
    else:
        samples = samples[val_end:]

    if max_samples:
        samples = samples[:max_samples]

    # Load audio lazily (return paths, ArabicASRDataset will load on __getitem__)
    print(f"  {len(samples)} samples found.")
    return _LocalFileSampleList(samples)


def load_kaggle_arabic_tts(
    data_dir: str,
    split: str = "train",
    max_samples: Optional[int] = None,
    split_ratio: Tuple[float, float, float] = (0.9, 0.05, 0.05),
) -> List[Dict]:
    """
    Kaggle 'arabic_tts' dataset (Common Voice 11.0 Arabic subset).
      https://www.kaggle.com/datasets/mayarjao/arabic-tts

    Expected layout after Kaggle download + unzip:
      data_dir/
        wavs/                    ← 78.7k .wav files
        metadata.csv             ← filename|transcript[|speaker] (LJSpeech-style)
        metadata-wav.csv         ← alt. variant with explicit .wav extension
    """
    data_dir = Path(data_dir)
    print(f"Loading Kaggle arabic_tts from {data_dir} ({split})...")

    wav_dir = data_dir / "wavs"
    if not wav_dir.exists():
        # Fall back — sometimes the zip nests an extra folder
        nested = next(data_dir.glob("*/wavs"), None)
        if nested:
            wav_dir = nested
            data_dir = nested.parent
        else:
            raise FileNotFoundError(f"No wavs/ folder under {data_dir}")

    # Prefer metadata-wav.csv (filenames have .wav), then metadata.csv
    meta_path = data_dir / "metadata-wav.csv"
    if not meta_path.exists():
        meta_path = data_dir / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata.csv under {data_dir}")

    samples = []
    # LJSpeech-style separators can be | or , — try both.
    # Some Kaggle releases of arabic_tts put the transcript FIRST and filename SECOND,
    # others do the opposite. We auto-detect which column is the filename.
    def _looks_like_wav(s: str) -> bool:
        s = s.strip().strip('"').strip("'")
        return s.endswith(".wav") or s.startswith("common_voice") or (len(s) < 80 and " " not in s)

    skipped_missing = 0
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Try | first (LJSpeech standard), fall back to comma
            if "|" in line:
                parts = [p.strip() for p in line.split("|")]
            else:
                parts = [p.strip() for p in line.split(",", 1)]
            if len(parts) < 2:
                continue
            a, b = parts[0], parts[1]
            # Auto-detect column order — filename is the one that "looks like a wav"
            if _looks_like_wav(a) and not _looks_like_wav(b):
                fname, text = a, b
            elif _looks_like_wav(b) and not _looks_like_wav(a):
                fname, text = b, a
            else:
                # Default: assume LJSpeech order (filename, text)
                fname, text = a, b
            # Strip wrapping quotes from both fields
            fname = fname.strip('"').strip("'")
            text  = text.strip('"').strip("'")
            # Some variants of metadata-wav.csv prefix the filename with "wavs/"
            if fname.startswith("wavs/") or fname.startswith("wavs\\"):
                fname = fname.split("/", 1)[-1].split("\\", 1)[-1]
            if not fname or not text:
                continue
            wav_path = wav_dir / fname
            if not wav_path.suffix:
                wav_path = wav_path.with_suffix(".wav")
            try:
                if wav_path.exists():
                    samples.append({"path": str(wav_path), "sentence": text})
                else:
                    skipped_missing += 1
            except OSError:
                # File-name-too-long etc. — skip the row
                skipped_missing += 1
    if skipped_missing:
        print(f"  (skipped {skipped_missing} rows whose wav file was not found)")

    if not samples:
        raise ValueError(
            f"No (wav, transcript) pairs found.\n"
            f"  metadata: {meta_path}\n"
            f"  wavs dir: {wav_dir}\n"
            "Inspect the first few lines of metadata.csv and confirm the separator."
        )

    # Deterministically shuffle so train/val/test sample the same speaker
    # distribution. Without this, Common Voice's CSV-order clusters clips by
    # speaker; the slice-based split puts entire speakers in only one of
    # train/val/test, causing severe distribution mismatch (val WER stuck
    # near 1.0 even when training perfectly fits the training set).
    import random as _random
    rng = _random.Random(42)
    rng.shuffle(samples)

    n = len(samples)
    train_end = int(n * split_ratio[0])
    val_end   = int(n * (split_ratio[0] + split_ratio[1]))
    if split == "train":
        samples = samples[:train_end]
    elif split == "validation":
        samples = samples[train_end:val_end]
    else:
        samples = samples[val_end:]

    if max_samples:
        samples = samples[:max_samples]

    print(f"  {len(samples)} samples ({split}) out of {n} total (shuffled, seed=42).")
    return _LocalFileSampleList(samples)


def load_ejust(
    data_dir: str,
    split: str = "train",
    max_samples: Optional[int] = None,
    split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> List[Dict]:
    """
    EJUST Arabic Speech Dataset — local folder from Google Drive (instructor provided).
    DO NOT share online.

    Expected folder layout (two common variants):
      Variant A:
        data_dir/
          audio/   ← .wav files
          transcripts/  ← .txt files with same base name as audio

      Variant B:
        data_dir/
          *.wav   ← wav files directly
          *.txt   ← transcript files alongside
    """
    data_dir = Path(data_dir)
    print(f"Loading EJUST dataset from {data_dir} ({split})...")

    samples = []

    # Variant A
    audio_dir = data_dir / "audio"
    trans_dir = data_dir / "transcripts"
    if audio_dir.exists():
        for wav_path in sorted(audio_dir.glob("*.wav")):
            txt_path = trans_dir / (wav_path.stem + ".txt")
            if not txt_path.exists():
                txt_path = wav_path.with_suffix(".txt")
            if txt_path.exists():
                text = txt_path.read_text(encoding="utf-8").strip()
                samples.append({"path": str(wav_path), "sentence": text})
    else:
        # Variant B — flat folder
        for wav_path in sorted(data_dir.rglob("*.wav")):
            for ext in [".txt", ".lab"]:
                tp = wav_path.with_suffix(ext)
                if tp.exists():
                    text = tp.read_text(encoding="utf-8").strip()
                    samples.append({"path": str(wav_path), "sentence": text})
                    break

    if not samples:
        raise ValueError(
            f"No samples found in {data_dir}.\n"
            "Expected: audio/*.wav + transcripts/*.txt\n"
            "OR: *.wav + *.txt in same folder."
        )

    n = len(samples)
    train_end = int(n * split_ratio[0])
    val_end   = int(n * (split_ratio[0] + split_ratio[1]))
    if split == "train":
        samples = samples[:train_end]
    elif split == "validation":
        samples = samples[train_end:val_end]
    else:
        samples = samples[val_end:]

    if max_samples:
        samples = samples[:max_samples]

    print(f"  {len(samples)} samples found.")
    return _LocalFileSampleList(samples)


class _LocalFileSampleList:
    """
    Thin wrapper around a list of {"path": ..., "sentence": ...} dicts
    that loads audio lazily on index access.
    Presents the same interface as a HuggingFace dataset slice
    (supports len() and __getitem__ returning {"audio": {...}, "sentence": str}).
    """
    def __init__(self, samples: List[Dict]):
        self._samples = samples

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        import torchaudio
        s = self._samples[idx]
        waveform, sr = torchaudio.load(s["path"])
        if sr != TARGET_SAMPLE_RATE:
            import torchaudio.functional as AF
            waveform = AF.resample(waveform, sr, TARGET_SAMPLE_RATE)
        waveform = waveform.mean(0).numpy()  # mono
        return {
            "audio": {"array": waveform, "sampling_rate": TARGET_SAMPLE_RATE},
            "sentence": s["sentence"],
        }

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


# ══════════════════════════════════════════════════════════════════
# Unified loader
# ══════════════════════════════════════════════════════════════════

def load_dataset_by_name(
    name: str,
    split: str = "train",
    data_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
    hf_token: Optional[str] = None,
):
    """
    Route to the correct per-source loader.

    name: "common_voice" | "masc" | "arabic_speech_corpus" | "ejust"
    """
    name = name.lower().strip()
    if name == "common_voice":
        return load_common_voice_arabic(split, max_samples, hf_token)
    elif name == "masc":
        return load_masc(split, max_samples)
    elif name in ("arabic_speech_corpus", "asc"):
        if not data_dir:
            raise ValueError("data_dir is required for Arabic Speech Corpus")
        return load_arabic_speech_corpus(data_dir, split, max_samples)
    elif name in ("kaggle_arabic_tts", "arabic_tts", "kaggle"):
        if not data_dir:
            raise ValueError("data_dir is required for Kaggle arabic_tts dataset")
        return load_kaggle_arabic_tts(data_dir, split, max_samples)
    elif name == "ejust":
        if not data_dir:
            raise ValueError("data_dir is required for EJUST dataset")
        return load_ejust(data_dir, split, max_samples)
    else:
        raise ValueError(f"Unknown dataset: '{name}'. "
                         f"Choose from: common_voice, masc, arabic_speech_corpus, kaggle_arabic_tts, ejust")


# ══════════════════════════════════════════════════════════════════
# PyTorch Dataset
# ══════════════════════════════════════════════════════════════════

class ArabicASRDataset(Dataset):
    """
    PyTorch Dataset — wraps any source returned by load_dataset_by_name().
    Produces (mel_spectrogram, transcript_tokens) pairs for CNN+LSTM training.
    """
    def __init__(
        self,
        source,                        # HF dataset or _LocalFileSampleList
        n_mels: int = 80,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        max_duration: float = 10.0,
        augment: bool = False,
        vocab: Optional[Dict[str, int]] = None,
        return_mode: str = "waveform",  # "waveform" (GPU mel later) | "mel" (legacy CPU mel)
    ):
        self.dataset = source
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.max_duration = max_duration
        self.augment = augment
        self.return_mode = return_mode
        self.vocab = vocab if vocab is not None else self._build_vocab()
        self.idx2char = {v: k for k, v in self.vocab.items()}

    def _build_vocab(self) -> Dict[str, int]:
        """
        Build the character vocabulary by scanning training transcripts.
        Fast path: avoid loading audio files when only the sentence is needed.
        """
        chars = set()
        # Fast path A — _LocalFileSampleList keeps sentences in memory already
        if hasattr(self.dataset, "_samples") and isinstance(self.dataset._samples, list):
            for s in tqdm(self.dataset._samples, desc="Building vocabulary"):
                chars.update(s["sentence"])
        # Fast path B — HuggingFace dataset: project to text column only
        elif hasattr(self.dataset, "select_columns"):
            try:
                txt_only = self.dataset.select_columns(["sentence"])
                for s in tqdm(txt_only, desc="Building vocabulary"):
                    chars.update(s["sentence"])
            except Exception:
                for s in tqdm(self.dataset, desc="Building vocabulary"):
                    chars.update(s["sentence"])
        # Slow fallback — iterate the source as-is
        else:
            for s in tqdm(self.dataset, desc="Building vocabulary"):
                chars.update(s["sentence"])
        vocab = {"<pad>": 0, "<unk>": 1, "<sos>": 2, "<eos>": 3, " ": 4}
        for i, ch in enumerate(sorted(chars - {" "}), start=5):
            vocab[ch] = i
        return vocab

    def encode_text(self, text: str) -> List[int]:
        return [self.vocab.get(c, self.vocab["<unk>"]) for c in text]

    def decode_ids(self, ids: List[int]) -> str:
        return "".join(self.idx2char.get(i, "") for i in ids if i not in {0, 2, 3})

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        waveform = sample["audio"]["array"].astype(np.float32)
        transcript = sample["sentence"]

        waveform = normalize_audio(waveform)

        # IMPORTANT — compute the real audio frame count BEFORE padding so CTC
        # only scores the unpadded portion. Otherwise CTC trivially learns to
        # predict blank everywhere and gets stuck.
        sr = 16000
        max_samples = int(self.max_duration * sr)
        real_samples = min(len(waveform), max_samples)
        real_pre_subsample_frames = max(
            1, (real_samples - self.win_length) // self.hop_length + 1
        )

        waveform = pad_or_trim(waveform, self.max_duration)
        if self.augment and self.return_mode == "mel":
            # Waveform-time augmentation only happens here in legacy mel mode.
            # In waveform mode, the model applies SpecAugment on GPU instead.
            waveform = apply_augmentation(waveform)

        tokens = self.encode_text(transcript)
        target_length = len(tokens)
        tokens_t = torch.tensor(tokens, dtype=torch.long)

        if self.return_mode == "waveform":
            # GPU-side mel extraction path (fast)
            wav_t = torch.tensor(waveform, dtype=torch.float32)
            return {
                "waveform": wav_t,
                "tokens": tokens_t,
                "transcript": transcript,
                # PRE-subsampling frame count. The training loop converts to
                # post-subsampling using model.get_output_lengths().
                "input_length": real_pre_subsample_frames,
                "target_length": target_length,
            }

        # ── Legacy CPU-mel path ──
        mel = extract_mel_spectrogram(
            waveform, n_mels=self.n_mels, n_fft=self.n_fft,
            hop_length=self.hop_length, win_length=self.win_length,
        )
        mel_t = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        mel_t = (mel_t - mel_t.mean()) / (mel_t.std() + 1e-8)
        return {
            "mel": mel_t,
            "tokens": tokens_t,
            "transcript": transcript,
            "input_length": mel_t.shape[-1],
            "target_length": target_length,
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Handles both return modes:
      - waveform mode: stacks raw waveforms (all already pad_or_trim'd to same length)
      - mel mode:      pads variable-length mels along the time axis
    """
    tokens  = [b["tokens"] for b in batch]
    max_tok = max(len(t) for t in tokens)
    padded_tokens = torch.zeros(len(tokens), max_tok, dtype=torch.long)
    for i, t in enumerate(tokens):
        padded_tokens[i, :len(t)] = t

    out = {
        "tokens":         padded_tokens,
        "transcripts":    [b["transcript"] for b in batch],
        "input_lengths":  torch.tensor([b["input_length"]  for b in batch], dtype=torch.long),
        "target_lengths": torch.tensor([b["target_length"] for b in batch], dtype=torch.long),
    }

    if "waveform" in batch[0]:
        out["waveform"] = torch.stack([b["waveform"] for b in batch], dim=0)  # (B, samples)
    else:
        mels = [b["mel"] for b in batch]
        max_mel = max(m.shape[-1] for m in mels)
        padded_mels = torch.zeros(len(mels), 1, mels[0].shape[1], max_mel)
        for i, m in enumerate(mels):
            padded_mels[i, :, :, :m.shape[-1]] = m
        out["mel"] = padded_mels
    return out


# ══════════════════════════════════════════════════════════════════
# DataLoader factory
# ══════════════════════════════════════════════════════════════════

def get_dataloaders(
    config: Dict,
    vocab: Optional[Dict] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Build train/val/test DataLoaders from config.
    Reads config["data"]["source"] to pick dataset.
    """
    src        = config["data"].get("source", "common_voice")
    data_dir   = config["data"].get("data_dir")
    hf_token   = config["data"].get("hf_token") or os.environ.get("HF_TOKEN")
    max_train  = config["data"].get("max_train_samples")
    max_eval   = config["data"].get("max_eval_samples")
    audio_cfg  = config["audio"]
    max_dur    = config["data"].get("max_duration", audio_cfg.get("max_audio_length", 10))

    if src == "combined":
        sources = config["data"].get("combine_sources", ["common_voice"])
        train_sources = [load_dataset_by_name(s, "train",      data_dir, max_train, hf_token) for s in sources]
        val_sources   = [load_dataset_by_name(s, "validation", data_dir, max_eval,  hf_token) for s in sources]
        test_sources  = [load_dataset_by_name(s, "test",       data_dir, max_eval,  hf_token) for s in sources]
        train_raw = _ConcatSource(train_sources)
        val_raw   = _ConcatSource(val_sources)
        test_raw  = _ConcatSource(test_sources)
    else:
        train_raw = load_dataset_by_name(src, "train",      data_dir, max_train, hf_token)
        val_raw   = load_dataset_by_name(src, "validation", data_dir, max_eval,  hf_token)
        test_raw  = load_dataset_by_name(src, "test",       data_dir, max_eval,  hf_token)

    # Default to waveform mode (GPU mel extraction in model — much faster on big runs)
    return_mode = config["cnn_lstm"].get("return_mode", "waveform")
    train_ds = ArabicASRDataset(train_raw,
        n_mels=audio_cfg["n_mels"], n_fft=audio_cfg["n_fft"],
        hop_length=audio_cfg["hop_length"], win_length=audio_cfg["win_length"],
        max_duration=max_dur, augment=True, vocab=vocab, return_mode=return_mode)
    val_ds   = ArabicASRDataset(val_raw,
        n_mels=audio_cfg["n_mels"], n_fft=audio_cfg["n_fft"],
        hop_length=audio_cfg["hop_length"], win_length=audio_cfg["win_length"],
        max_duration=max_dur, augment=False, vocab=train_ds.vocab, return_mode=return_mode)
    test_ds  = ArabicASRDataset(test_raw,
        n_mels=audio_cfg["n_mels"], n_fft=audio_cfg["n_fft"],
        hop_length=audio_cfg["hop_length"], win_length=audio_cfg["win_length"],
        max_duration=max_dur, augment=False, vocab=train_ds.vocab, return_mode=return_mode)

    bs = config["cnn_lstm"]["batch_size"]
    # Audio loading + librosa mel extraction is CPU-bound; scale workers to host.
    # Allow override via config: cnn_lstm.num_workers
    try:
        n_cpu = max(2, (os.cpu_count() or 8))
    except Exception:
        n_cpu = 8
    train_workers = config["cnn_lstm"].get("num_workers", min(16, n_cpu - 2))
    eval_workers  = max(2, train_workers // 2)
    return (
        DataLoader(
            train_ds, batch_size=bs, shuffle=True, collate_fn=collate_fn,
            num_workers=train_workers, pin_memory=True,
            persistent_workers=(train_workers > 0),
            prefetch_factor=4 if train_workers > 0 else None,
        ),
        DataLoader(
            val_ds, batch_size=bs, shuffle=False, collate_fn=collate_fn,
            num_workers=eval_workers,
            persistent_workers=(eval_workers > 0),
            prefetch_factor=2 if eval_workers > 0 else None,
        ),
        DataLoader(
            test_ds, batch_size=bs, shuffle=False, collate_fn=collate_fn,
            num_workers=eval_workers,
            persistent_workers=(eval_workers > 0),
            prefetch_factor=2 if eval_workers > 0 else None,
        ),
        train_ds.vocab,
    )


class _ConcatSource:
    """Concatenate multiple dataset sources."""
    def __init__(self, sources):
        self._sources = sources
        self._lengths = [len(s) for s in sources]
        self._total   = sum(self._lengths)

    def __len__(self):
        return self._total

    def __getitem__(self, idx):
        for src, length in zip(self._sources, self._lengths):
            if idx < length:
                return src[idx]
            idx -= length
        raise IndexError(idx)

    def __iter__(self):
        for src in self._sources:
            yield from src
