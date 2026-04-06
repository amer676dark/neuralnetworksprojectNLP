"""Dataset loading and preprocessing for Arabic ASR using Mozilla Common Voice."""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Audio
from transformers import Wav2Vec2Processor
import yaml
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


def load_common_voice_arabic(
    split: str = "train",
    max_samples: Optional[int] = None,
    streaming: bool = False,
) -> object:
    """
    Load Mozilla Common Voice Arabic dataset from HuggingFace.
    split: 'train', 'validation', or 'test'
    """
    print(f"Loading Mozilla Common Voice Arabic ({split} split)...")
    dataset = load_dataset(
        "mozilla-foundation/common_voice_13_0",
        "ar",
        split=split,
        streaming=streaming,
        trust_remote_code=True,
    )

    # Cast audio column to 16kHz
    if not streaming:
        dataset = dataset.cast_column("audio", Audio(sampling_rate=TARGET_SAMPLE_RATE))
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

    print(f"  Loaded {len(dataset) if not streaming else 'streaming'} samples")
    return dataset


class ArabicASRDataset(Dataset):
    """
    PyTorch Dataset for Arabic ASR — produces (mel_spectrogram, transcript) pairs.
    Used for training the CNN+LSTM model.
    """

    def __init__(
        self,
        hf_dataset,
        n_mels: int = 80,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        max_duration: float = 10.0,
        augment: bool = False,
        vocab: Optional[Dict[str, int]] = None,
    ):
        self.dataset = hf_dataset
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.max_duration = max_duration
        self.augment = augment

        # Build or use provided vocabulary
        if vocab is None:
            self.vocab = self._build_vocab()
        else:
            self.vocab = vocab
        self.idx2char = {v: k for k, v in self.vocab.items()}

    def _build_vocab(self) -> Dict[str, int]:
        """Build character-level vocabulary from all transcripts."""
        chars = set()
        for sample in tqdm(self.dataset, desc="Building vocab"):
            chars.update(sample["sentence"])
        vocab = {"<pad>": 0, "<unk>": 1, "<sos>": 2, "<eos>": 3, " ": 4}
        for i, char in enumerate(sorted(chars - {" "}), start=5):
            vocab[char] = i
        return vocab

    def encode_text(self, text: str) -> List[int]:
        """Encode text string to list of token IDs."""
        return [self.vocab.get(c, self.vocab["<unk>"]) for c in text]

    def decode_ids(self, ids: List[int]) -> str:
        """Decode list of token IDs back to text string."""
        return "".join(self.idx2char.get(i, "") for i in ids if i not in {0, 2, 3})

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.dataset[idx]
        waveform = sample["audio"]["array"].astype(np.float32)
        transcript = sample["sentence"]

        # Normalize & trim
        waveform = normalize_audio(waveform)
        waveform = pad_or_trim(waveform, self.max_duration)

        # Augmentation
        if self.augment:
            waveform = apply_augmentation(waveform)

        # Mel spectrogram: (1, n_mels, T)
        mel = extract_mel_spectrogram(
            waveform,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )
        mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)

        # Normalize mel
        mel_tensor = (mel_tensor - mel_tensor.mean()) / (mel_tensor.std() + 1e-8)

        # Encode transcript
        tokens = self.encode_text(transcript)

        return {
            "mel": mel_tensor,
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "transcript": transcript,
            "input_length": mel_tensor.shape[-1],
            "target_length": len(tokens),
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Pad mel spectrograms and token sequences to batch max length."""
    mels = [b["mel"] for b in batch]
    tokens = [b["tokens"] for b in batch]
    transcripts = [b["transcript"] for b in batch]

    max_mel_len = max(m.shape[-1] for m in mels)
    max_tok_len = max(len(t) for t in tokens)

    padded_mels = torch.zeros(len(mels), 1, mels[0].shape[1], max_mel_len)
    padded_tokens = torch.zeros(len(tokens), max_tok_len, dtype=torch.long)

    for i, (m, t) in enumerate(zip(mels, tokens)):
        padded_mels[i, :, :, : m.shape[-1]] = m
        padded_tokens[i, : len(t)] = t

    input_lengths = torch.tensor([b["input_length"] for b in batch], dtype=torch.long)
    target_lengths = torch.tensor([b["target_length"] for b in batch], dtype=torch.long)

    return {
        "mel": padded_mels,
        "tokens": padded_tokens,
        "transcripts": transcripts,
        "input_lengths": input_lengths,
        "target_lengths": target_lengths,
    }


def get_dataloaders(
    config: Dict,
    vocab: Optional[Dict] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders from config."""
    train_ds = load_common_voice_arabic("train", config["data"]["max_train_samples"])
    val_ds = load_common_voice_arabic("validation", config["data"]["max_eval_samples"])
    test_ds = load_common_voice_arabic("test", config["data"]["max_eval_samples"])

    train_dataset = ArabicASRDataset(
        train_ds,
        n_mels=config["audio"]["n_mels"],
        hop_length=config["audio"]["hop_length"],
        max_duration=config["audio"]["max_audio_length"],
        augment=True,
        vocab=vocab,
    )
    val_dataset = ArabicASRDataset(
        val_ds,
        n_mels=config["audio"]["n_mels"],
        hop_length=config["audio"]["hop_length"],
        max_duration=config["audio"]["max_audio_length"],
        augment=False,
        vocab=train_dataset.vocab,
    )
    test_dataset = ArabicASRDataset(
        test_ds,
        n_mels=config["audio"]["n_mels"],
        hop_length=config["audio"]["hop_length"],
        max_duration=config["audio"]["max_audio_length"],
        augment=False,
        vocab=train_dataset.vocab,
    )

    batch_size = config["cnn_lstm"]["batch_size"]
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2
    )

    return train_loader, val_loader, test_loader, train_dataset.vocab
