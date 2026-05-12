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
    # LJSpeech-style separators can be | or , — try both
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|") if "|" in line else line.split(",", 1)
            if len(parts) < 2:
                continue
            fname, text = parts[0].strip(), parts[1].strip()
            # Strip wrapping quotes
            text = text.strip('"').strip("'")
            wav_path = wav_dir / fname
            if not wav_path.suffix:
                wav_path = wav_path.with_suffix(".wav")
            if wav_path.exists() and text:
                samples.append({"path": str(wav_path), "sentence": text})

    if not samples:
        raise ValueError(
            f"No (wav, transcript) pairs found.\n"
            f"  metadata: {meta_path}\n"
            f"  wavs dir: {wav_dir}\n"
            "Inspect the first few lines of metadata.csv and confirm the separator."
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

    print(f"  {len(samples)} samples ({split}) out of {n} total.")
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
    ):
        self.dataset = source
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.max_duration = max_duration
        self.augment = augment
        self.vocab = vocab if vocab is not None else self._build_vocab()
        self.idx2char = {v: k for k, v in self.vocab.items()}

    def _build_vocab(self) -> Dict[str, int]:
        chars = set()
        for sample in tqdm(self.dataset, desc="Building vocabulary"):
            chars.update(sample["sentence"])
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
        waveform = pad_or_trim(waveform, self.max_duration)
        if self.augment:
            waveform = apply_augmentation(waveform)

        mel = extract_mel_spectrogram(
            waveform, n_mels=self.n_mels, n_fft=self.n_fft,
            hop_length=self.hop_length, win_length=self.win_length,
        )
        mel_t = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        mel_t = (mel_t - mel_t.mean()) / (mel_t.std() + 1e-8)

        tokens = self.encode_text(transcript)
        return {
            "mel": mel_t,
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "transcript": transcript,
            "input_length": mel_t.shape[-1],
            "target_length": len(tokens),
        }


def collate_fn(batch: List[Dict]) -> Dict:
    mels    = [b["mel"]    for b in batch]
    tokens  = [b["tokens"] for b in batch]
    max_mel = max(m.shape[-1] for m in mels)
    max_tok = max(len(t)      for t in tokens)

    padded_mels   = torch.zeros(len(mels),   1, mels[0].shape[1], max_mel)
    padded_tokens = torch.zeros(len(tokens), max_tok, dtype=torch.long)

    for i, (m, t) in enumerate(zip(mels, tokens)):
        padded_mels[i, :, :, :m.shape[-1]] = m
        padded_tokens[i, :len(t)] = t

    return {
        "mel":            padded_mels,
        "tokens":         padded_tokens,
        "transcripts":    [b["transcript"] for b in batch],
        "input_lengths":  torch.tensor([b["input_length"]  for b in batch], dtype=torch.long),
        "target_lengths": torch.tensor([b["target_length"] for b in batch], dtype=torch.long),
    }


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

    train_ds = ArabicASRDataset(train_raw,
        n_mels=audio_cfg["n_mels"], n_fft=audio_cfg["n_fft"],
        hop_length=audio_cfg["hop_length"], win_length=audio_cfg["win_length"],
        max_duration=max_dur, augment=True, vocab=vocab)
    val_ds   = ArabicASRDataset(val_raw,
        n_mels=audio_cfg["n_mels"], n_fft=audio_cfg["n_fft"],
        hop_length=audio_cfg["hop_length"], win_length=audio_cfg["win_length"],
        max_duration=max_dur, augment=False, vocab=train_ds.vocab)
    test_ds  = ArabicASRDataset(test_raw,
        n_mels=audio_cfg["n_mels"], n_fft=audio_cfg["n_fft"],
        hop_length=audio_cfg["hop_length"], win_length=audio_cfg["win_length"],
        max_duration=max_dur, augment=False, vocab=train_ds.vocab)

    bs = config["cnn_lstm"]["batch_size"]
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True,  collate_fn=collate_fn, num_workers=4, pin_memory=True),
        DataLoader(val_ds,   batch_size=bs, shuffle=False, collate_fn=collate_fn, num_workers=2),
        DataLoader(test_ds,  batch_size=bs, shuffle=False, collate_fn=collate_fn, num_workers=2),
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
