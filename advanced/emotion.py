"""
Speech Emotion Recognition (SER).

Uses a multilingual Wav2Vec2-based emotion classifier from HuggingFace.
Default model: harshit345/xlsr-wav2vec-speech-emotion-recognition
  - Backbone: facebook/wav2vec2-large-xlsr-53
  - 8 classes: angry, calm, disgust, fear, happy, neutral, sad, surprise
  - Trained on RAVDESS — primarily English/Italian but the acoustic features
    transfer to Arabic prosody reasonably well for a project demo.

If you want an Arabic-specific model, swap to:
  Rajaram1996/arabic-speech-emotion-detection-w2v
"""

import torch
import numpy as np
from typing import Dict, List, Optional


DEFAULT_CLASSES = ["angry", "calm", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


class EmotionRecognizer:
    """Wav2Vec2-based speech emotion recognition."""

    def __init__(
        self,
        model_name: str = "harshit345/xlsr-wav2vec-speech-emotion-recognition",
        device: Optional[str] = None,
    ):
        self.model_name = model_name

        if device == "auto" or device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading emotion recognizer ({model_name}) on {self.device}...")
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModelForAudioClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Prefer the model's own label mapping if available
        id2label = getattr(self.model.config, "id2label", None)
        if id2label:
            self.classes = [id2label[i] for i in sorted(id2label)]
        else:
            self.classes = DEFAULT_CLASSES

        print(f"Emotion classes: {self.classes}")

    @torch.no_grad()
    def predict(self, waveform: np.ndarray, sample_rate: int = 16000) -> Dict:
        """
        Returns {"emotion": top_class, "confidence": score, "all_scores": {...}}.
        """
        if sample_rate != 16000:
            raise ValueError("Emotion model expects 16 kHz audio.")
        wav = waveform.astype(np.float32)
        inputs = self.feature_extractor(
            wav, sampling_rate=sample_rate, return_tensors="pt"
        ).to(self.device)
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

        top_idx = int(np.argmax(probs))
        return {
            "emotion": self.classes[top_idx],
            "confidence": float(probs[top_idx]),
            "all_scores": {self.classes[i]: float(probs[i]) for i in range(len(self.classes))},
        }


def format_emotion_report(result: Dict) -> str:
    """Pretty-print emotion result as Markdown."""
    lines = [
        f"**Predicted emotion:** {result['emotion']}  (confidence: {result['confidence']*100:.1f}%)",
        "",
        "| Class | Score |",
        "|---|---|",
    ]
    for cls, score in sorted(result["all_scores"].items(), key=lambda x: -x[1]):
        lines.append(f"| {cls} | {score*100:.1f}% |")
    return "\n".join(lines)
