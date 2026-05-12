"""
Speaker Identification using SpeechBrain ECAPA-TDNN.

Given a directory of enrollment audio clips (one or more per speaker), we extract
an embedding per speaker. For any new clip we compute its embedding and assign it
to the closest enrolled speaker (cosine similarity).

If similarity < `threshold`, we report "unknown speaker".

For multi-speaker audio segmentation (diarization), we use a simple sliding-window
approach over chunks — sufficient for the project demo. Production-grade diarization
would use pyannote-audio.

ECAPA-TDNN was pretrained on VoxCeleb (English) but the embedding space generalizes
to Arabic speech well in our testing.
"""

import os
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class SpeakerIdentifier:
    """ECAPA-TDNN based speaker embedding and matching."""

    def __init__(
        self,
        model_name: str = "speechbrain/spkrec-ecapa-voxceleb",
        device: Optional[str] = None,
        savedir: str = "./outputs/checkpoints/ecapa",
    ):
        self.model_name = model_name

        if device == "auto" or device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading SpeechBrain ECAPA on {self.device}...")
        from speechbrain.inference.speaker import EncoderClassifier

        self.encoder = EncoderClassifier.from_hparams(
            source=model_name,
            savedir=savedir,
            run_opts={"device": self.device},
        )
        print("ECAPA-TDNN loaded.")

        # In-memory database of enrolled speakers: {name: embedding (np.ndarray)}
        self.enrolled: Dict[str, np.ndarray] = {}

    @torch.no_grad()
    def embed(self, waveform: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Compute a speaker embedding (192-dim) from a waveform."""
        if sample_rate != 16000:
            raise ValueError("ECAPA requires 16 kHz audio.")
        wav_t = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).to(self.device)
        emb = self.encoder.encode_batch(wav_t).squeeze().cpu().numpy()
        # Normalize to unit length for cosine similarity
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb

    def enroll(self, name: str, waveform: np.ndarray, sample_rate: int = 16000) -> None:
        """Add a known speaker. If name exists, average the embeddings."""
        emb = self.embed(waveform, sample_rate)
        if name in self.enrolled:
            self.enrolled[name] = (self.enrolled[name] + emb) / 2
            self.enrolled[name] /= np.linalg.norm(self.enrolled[name]) + 1e-8
        else:
            self.enrolled[name] = emb

    def enroll_from_files(self, name: str, audio_paths: List[str]) -> None:
        """Convenience: enroll a speaker from one or more audio files."""
        import torchaudio
        import torchaudio.functional as AF
        for path in audio_paths:
            wav, sr = torchaudio.load(path)
            wav = wav.mean(0).numpy().astype(np.float32)
            if sr != 16000:
                wav = AF.resample(torch.tensor(wav).unsqueeze(0), sr, 16000).squeeze().numpy()
            self.enroll(name, wav, 16000)

    def identify(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000,
        threshold: float = 0.5,
    ) -> Tuple[str, float]:
        """
        Return (speaker_name, similarity) for the closest enrolled speaker.
        Returns ("unknown", best_score) if best similarity is below threshold.
        """
        if not self.enrolled:
            return ("no_enrolled_speakers", 0.0)

        query = self.embed(waveform, sample_rate)
        best_name, best_score = "unknown", -1.0
        for name, emb in self.enrolled.items():
            score = float(np.dot(query, emb))  # cosine since both are unit
            if score > best_score:
                best_name, best_score = name, score

        return (best_name if best_score >= threshold else "unknown", best_score)

    def diarize_simple(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000,
        chunk_seconds: float = 2.0,
        stride_seconds: float = 1.0,
        threshold: float = 0.5,
    ) -> List[Dict]:
        """
        Lightweight diarization: slide a window across the audio, identify the
        speaker in each chunk. Returns a list of {start, end, speaker, confidence}.

        Limitations: doesn't handle overlaps, depends on enrolled speakers existing.
        For unknown speakers we get "unknown" rather than auto-clustering.
        """
        if not self.enrolled:
            return [{"start": 0.0, "end": len(waveform) / sample_rate,
                     "speaker": "no_enrolled_speakers", "confidence": 0.0}]

        chunk_samples = int(chunk_seconds * sample_rate)
        stride_samples = int(stride_seconds * sample_rate)
        segments = []
        for start in range(0, len(waveform) - chunk_samples + 1, stride_samples):
            chunk = waveform[start:start + chunk_samples]
            spk, score = self.identify(chunk, sample_rate, threshold)
            segments.append({
                "start": round(start / sample_rate, 2),
                "end": round((start + chunk_samples) / sample_rate, 2),
                "speaker": spk,
                "confidence": round(score, 3),
            })

        # Merge adjacent segments with same speaker
        merged: List[Dict] = []
        for seg in segments:
            if merged and merged[-1]["speaker"] == seg["speaker"]:
                merged[-1]["end"] = seg["end"]
                merged[-1]["confidence"] = round(
                    (merged[-1]["confidence"] + seg["confidence"]) / 2, 3
                )
            else:
                merged.append(dict(seg))
        return merged

    def list_enrolled(self) -> List[str]:
        return list(self.enrolled.keys())

    def clear(self) -> None:
        self.enrolled.clear()
