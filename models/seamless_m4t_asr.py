"""
Meta SeamlessM4T-v2 Arabic ASR wrapper (replacement for the deprecated DeepSpeech).

SeamlessM4T-v2 (2023) is Meta's state-of-the-art multilingual speech model:
  - Speech-to-text in 100+ languages including Arabic
  - 2.3B parameters (large) — needs GPU with ~10 GB VRAM in fp16
  - Single model handles ASR, translation, and TTS
  - Beats Whisper on many language pairs

Pretrained checkpoint:
  facebook/seamless-m4t-v2-large  (~9 GB download, gated — accept HF terms)

Smaller alternative if VRAM is tight:
  facebook/hf-seamless-m4t-medium  (~2.5 GB, slightly lower quality)
"""

import torch
import numpy as np
from typing import Optional, List, Dict


class SeamlessM4TASR:
    """Wrapper around HuggingFace SeamlessM4T-v2 for Arabic ASR."""

    def __init__(
        self,
        model_name: str = "facebook/seamless-m4t-v2-large",
        language: str = "arb",  # SeamlessM4T uses ISO 639-3: 'arb' for Standard Arabic
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.language = language

        if device == "auto" or device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"Loading SeamlessM4T ({model_name}) on {self.device}...")
        # Lazy import — transformers >=4.35 has SeamlessM4T support
        from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText

        self.processor = AutoProcessor.from_pretrained(model_name)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
            model_name, torch_dtype=dtype
        ).to(self.device)
        self.model.eval()
        print("SeamlessM4T-v2 loaded.")

    @torch.no_grad()
    def transcribe_array(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000,
    ) -> str:
        """Transcribe a numpy audio array (float32, mono, 16 kHz)."""
        if sample_rate != 16000:
            raise ValueError("SeamlessM4T requires 16 kHz audio.")
        wav = waveform.astype(np.float32)
        inputs = self.processor(
            audios=wav,
            sampling_rate=sample_rate,
            return_tensors="pt",
        ).to(self.device)
        if self.device == "cuda":
            inputs = {k: (v.half() if v.dtype == torch.float32 else v)
                      for k, v in inputs.items()}
        output_tokens = self.model.generate(
            **inputs,
            tgt_lang=self.language,
        )
        text = self.processor.decode(output_tokens[0].tolist(), skip_special_tokens=True)
        return text.strip()

    def transcribe_batch(
        self,
        waveforms: List[np.ndarray],
        sample_rate: int = 16000,
    ) -> List[str]:
        """Batch transcription. SeamlessM4T handles batches natively but uses a lot of VRAM."""
        return [self.transcribe_array(w, sample_rate) for w in waveforms]

    def get_model_info(self) -> Dict:
        return {
            "model_name": self.model_name,
            "language": self.language,
            "device": self.device,
            "parameters": sum(p.numel() for p in self.model.parameters()),
        }
