"""
Arabic ASR Demo — Gradio Interface

Supports:
  1. Upload audio file
  2. Record microphone live
  3. Choose model: Whisper / Wav2Vec2

Usage:
    python demo/app.py
    python demo/app.py --model whisper --size medium
    python demo/app.py --model wav2vec
    python demo/app.py --share  # create public URL
"""

import sys
import os
import argparse
import numpy as np
import gradio as gr
import torch
import tempfile
import soundfile as sf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Model loaders (lazy, cached) ─────────────────────────────────────────────

_whisper_model = None
_wav2vec_model = None


def get_whisper(size: str = "medium"):
    global _whisper_model
    if _whisper_model is None:
        from models.whisper_asr import WhisperASR
        _whisper_model = WhisperASR(model_size=size, language="ar", device="auto")
    return _whisper_model


def get_wav2vec():
    global _wav2vec_model
    if _wav2vec_model is None:
        from models.wav2vec_asr import Wav2Vec2ASR
        _wav2vec_model = Wav2Vec2ASR(device="auto")
    return _wav2vec_model


# ── Transcription functions ───────────────────────────────────────────────────

def transcribe_whisper(audio, model_size: str = "medium") -> tuple:
    """Handle Gradio audio input → transcription."""
    if audio is None:
        return "Please provide an audio file or recording.", ""

    sample_rate, waveform = audio

    # Convert to float32 mono 16kHz
    if waveform.dtype != np.float32:
        waveform = waveform.astype(np.float32)
        if waveform.max() > 1.0:
            waveform /= 32768.0  # int16 → float32

    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)  # stereo → mono

    # Resample to 16kHz if needed
    if sample_rate != 16000:
        import torchaudio.functional as AF
        wt = torch.tensor(waveform).unsqueeze(0)
        wt = AF.resample(wt, sample_rate, 16000)
        waveform = wt.squeeze().numpy()
        sample_rate = 16000

    model = get_whisper(model_size)
    text = model.transcribe_array(waveform)

    duration = len(waveform) / sample_rate
    info = f"Duration: {duration:.1f}s | Model: Whisper-{model_size} | Device: {model.device}"
    return text, info


def transcribe_wav2vec(audio) -> tuple:
    """Handle Gradio audio input → Wav2Vec2 transcription."""
    if audio is None:
        return "Please provide an audio file or recording.", ""

    sample_rate, waveform = audio

    if waveform.dtype != np.float32:
        waveform = waveform.astype(np.float32)
        if waveform.max() > 1.0:
            waveform /= 32768.0

    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)

    if sample_rate != 16000:
        import torchaudio.functional as AF
        wt = torch.tensor(waveform).unsqueeze(0)
        wt = AF.resample(wt, sample_rate, 16000)
        waveform = wt.squeeze().numpy()
        sample_rate = 16000

    model = get_wav2vec()
    text = model.transcribe_array(waveform)

    duration = len(waveform) / sample_rate
    info = f"Duration: {duration:.1f}s | Model: Wav2Vec2-XLSR-Arabic | Device: {model.device}"
    return text, info


def transcribe_auto(audio, model_choice: str, whisper_size: str) -> tuple:
    """Route to correct model based on user selection."""
    if model_choice == "Whisper":
        return transcribe_whisper(audio, whisper_size)
    else:
        return transcribe_wav2vec(audio)


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_interface():
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="indigo",
    )

    with gr.Blocks(theme=theme, title="Arabic ASR System") as demo:
        gr.Markdown("""
        # Arabic Speech Recognition System
        ### Deep Learning Based Arabic Audio Understanding
        Convert spoken Arabic audio to text using state-of-the-art neural networks.

        **Supported Models:**
        - **Whisper** (OpenAI) — Transformer-based, best accuracy, multilingual
        - **Wav2Vec 2.0** (Facebook) — Self-supervised, fine-tuned on Arabic
        """)

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Input Audio")
                audio_input = gr.Audio(
                    label="Upload or Record Arabic Speech",
                    sources=["upload", "microphone"],
                    type="numpy",
                )

                with gr.Row():
                    model_choice = gr.Radio(
                        choices=["Whisper", "Wav2Vec 2.0"],
                        value="Whisper",
                        label="Model",
                    )
                    whisper_size = gr.Dropdown(
                        choices=["tiny", "base", "small", "medium", "large-v2"],
                        value="medium",
                        label="Whisper Size",
                        visible=True,
                    )

                transcribe_btn = gr.Button("Transcribe", variant="primary", size="lg")

            with gr.Column(scale=2):
                gr.Markdown("### Transcription Output")
                output_text = gr.Textbox(
                    label="Arabic Transcription",
                    lines=8,
                    placeholder="Transcribed text will appear here...",
                    rtl=True,
                    text_align="right",
                )
                info_text = gr.Textbox(
                    label="Model Info",
                    lines=1,
                    interactive=False,
                )

        # Show/hide whisper size based on model selection
        def toggle_whisper_size(choice):
            return gr.update(visible=(choice == "Whisper"))

        model_choice.change(toggle_whisper_size, inputs=[model_choice], outputs=[whisper_size])

        transcribe_btn.click(
            fn=transcribe_auto,
            inputs=[audio_input, model_choice, whisper_size],
            outputs=[output_text, info_text],
        )

        gr.Markdown("""
        ---
        ### About This System
        | Model | Architecture | Parameters | Arabic WER (est.) |
        |-------|-------------|-----------|------------------|
        | Whisper medium | Encoder-Decoder Transformer | 769M | ~15-25% |
        | Wav2Vec 2.0 XLSR | CNN + Transformer | 300M | ~20-35% |
        | CNN+LSTM (custom) | CNN + BiLSTM + CTC | ~8M | ~40-60% |

        **Metrics**: WER = Word Error Rate (lower is better)

        **Dataset**: Trained/evaluated on Mozilla Common Voice Arabic
        """)

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arabic ASR Demo")
    parser.add_argument("--share", action="store_true", help="Create public URL")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    demo = build_interface()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )
