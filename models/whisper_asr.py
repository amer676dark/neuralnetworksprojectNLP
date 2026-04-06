"""
OpenAI Whisper wrapper for Arabic ASR.

Whisper is a state-of-the-art multilingual ASR model trained on 680k hours of audio.
It natively supports Arabic and is our strongest baseline.

Model sizes (WER on Arabic roughly improves with size):
  tiny   ~39M params  — fast, lower accuracy
  base   ~74M params
  small  ~244M params
  medium ~769M params  ← recommended
  large-v2/v3 ~1.5B   — best accuracy, requires GPU
"""

import torch
import numpy as np
import whisper
from pathlib import Path
from typing import Union, Optional, List, Dict
import time


class WhisperASR:
    """Wrapper around OpenAI Whisper for Arabic transcription."""

    def __init__(
        self,
        model_size: str = "medium",
        language: str = "ar",
        device: Optional[str] = None,
    ):
        self.model_size = model_size
        self.language = language

        # Device selection
        if device == "auto" or device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"Loading Whisper {model_size} on {self.device}...")
        self.model = whisper.load_model(model_size, device=self.device)
        print("Whisper model loaded.")

    def transcribe_file(
        self,
        audio_path: Union[str, Path],
        beam_size: int = 5,
        temperature: float = 0.0,
        verbose: bool = False,
    ) -> Dict:
        """
        Transcribe an audio file.
        Returns dict with 'text', 'segments', 'language'.
        """
        result = self.model.transcribe(
            str(audio_path),
            language=self.language,
            task="transcribe",
            beam_size=beam_size,
            temperature=temperature,
            verbose=verbose,
            fp16=(self.device == "cuda"),
        )
        return result

    def transcribe_array(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000,
        beam_size: int = 5,
    ) -> str:
        """Transcribe a numpy audio array (float32, mono, 16kHz)."""
        # Whisper expects 16kHz float32
        if sample_rate != 16000:
            raise ValueError("Whisper requires 16kHz audio. Please resample first.")
        waveform = waveform.astype(np.float32)
        result = self.model.transcribe(
            waveform,
            language=self.language,
            task="transcribe",
            beam_size=beam_size,
            fp16=(self.device == "cuda"),
        )
        return result["text"].strip()

    def transcribe_batch(
        self,
        audio_paths: List[Union[str, Path]],
        verbose: bool = True,
    ) -> List[Dict]:
        """Transcribe a list of audio files."""
        results = []
        for i, path in enumerate(audio_paths):
            if verbose:
                print(f"  [{i+1}/{len(audio_paths)}] {Path(path).name}")
            t0 = time.time()
            result = self.transcribe_file(path)
            result["duration"] = time.time() - t0
            results.append(result)
        return results

    def evaluate(
        self,
        audio_paths: List[Union[str, Path]],
        references: List[str],
    ) -> Dict:
        """Transcribe files and compute WER/CER against references."""
        from utils.metrics import compute_batch_metrics, format_metrics_report

        hypotheses = []
        for path in audio_paths:
            result = self.transcribe_file(path)
            hypotheses.append(result["text"].strip())

        metrics = compute_batch_metrics(references, hypotheses)
        print(format_metrics_report(metrics))
        return metrics, hypotheses

    def get_model_info(self) -> Dict:
        return {
            "model_size": self.model_size,
            "language": self.language,
            "device": self.device,
            "parameters": sum(p.numel() for p in self.model.parameters()),
        }
