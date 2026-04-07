"""
FILE 2 — Demo & Inference
=========================
Run:  python demo.py
Opens a Gradio GUI at http://localhost:7861

Tabs:
  1. Live Transcribe   — upload or record, choose any model
  2. Compare Models    — run same audio through all 3 side by side
  3. Batch Files       — transcribe multiple audio files at once
"""

import sys, os, json, time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
import gradio as gr

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

CONFIG_PATH = ROOT / "configs" / "config.yaml"
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")

DEFAULT_CKPT = str(ROOT / "outputs" / "checkpoints" / "cnn_lstm" / "best_model.pt")

# ── model cache ───────────────────────────────────────────────────────────────
_models = {}

def _load_whisper(size):
    key = f"whisper_{size}"
    if key not in _models:
        from models.whisper_asr import WhisperASR
        _models[key] = WhisperASR(model_size=size, language="ar", device="auto")
    return _models[key]

def _load_wav2vec():
    if "wav2vec" not in _models:
        from models.wav2vec_asr import Wav2Vec2ASR
        _models["wav2vec"] = Wav2Vec2ASR(device="auto")
    return _models["wav2vec"]

def _load_cnn_lstm(ckpt_path):
    key = f"cnn_lstm_{ckpt_path}"
    if key not in _models:
        from models.cnn_lstm_asr import build_model
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(
                f"CNN+LSTM checkpoint not found: {ckpt_path}\n"
                "Train it first using train_evaluate.py → Tab 2."
            )
        state   = torch.load(ckpt_path, map_location="cpu")
        vocab   = state["vocab"]
        dev     = torch.device(DEVICE)
        model   = build_model(len(vocab), CONFIG).to(dev)
        model.load_state_dict(state["model_state"])
        model.eval()
        _models[key] = (model, vocab, dev)
    return _models[key]

# ── audio preprocessing ───────────────────────────────────────────────────────
def _prep_audio(audio):
    """Gradio (sr, array) → float32 mono 16 kHz numpy."""
    if audio is None:
        return None, None
    sr, wav = audio
    wav = wav.astype(np.float32)
    if wav.max() > 1.0:
        wav /= 32768.0
    if wav.ndim == 2:
        wav = wav.mean(1)
    if sr != 16000:
        import torchaudio.functional as AF
        wav = AF.resample(torch.tensor(wav).unsqueeze(0), sr, 16000).squeeze().numpy()
        sr  = 16000
    return wav, sr

def _run_whisper(wav, size):
    t0 = time.time()
    m  = _load_whisper(size)
    text = m.transcribe_array(wav)
    return text, round(time.time()-t0, 2), m.device

def _run_wav2vec(wav):
    t0 = time.time()
    m  = _load_wav2vec()
    text = m.transcribe_array(wav)
    return text, round(time.time()-t0, 2), m.device

def _run_cnn_lstm(wav, ckpt):
    from utils.audio_utils import extract_mel_spectrogram, normalize_audio, pad_or_trim
    t0 = time.time()
    model, vocab, dev = _load_cnn_lstm(ckpt)
    idx2ch = {v: k for k, v in vocab.items()}
    w = normalize_audio(wav)
    w = pad_or_trim(w, CONFIG["audio"]["max_audio_length"])
    mel = extract_mel_spectrogram(w, n_mels=CONFIG["audio"]["n_mels"],
                                   n_fft=CONFIG["audio"]["n_fft"],
                                   hop_length=CONFIG["audio"]["hop_length"])
    mt = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(dev)
    mt = (mt - mt.mean()) / (mt.std() + 1e-8)
    with torch.no_grad():
        lp = model(mt)
    ids  = model.greedy_decode(lp)[0]
    text = "".join(idx2ch.get(i,"") for i in ids if i not in {0,2,3})
    return text, round(time.time()-t0, 2), str(dev)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE TRANSCRIBE
# ══════════════════════════════════════════════════════════════════════════════

def transcribe_live(audio, model_choice, whisper_size, cnn_ckpt):
    wav, sr = _prep_audio(audio)
    if wav is None:
        return "Please upload or record audio.", ""
    dur = len(wav) / sr
    try:
        if model_choice == "Whisper":
            text, secs, dev = _run_whisper(wav, whisper_size)
        elif model_choice == "Wav2Vec 2.0":
            text, secs, dev = _run_wav2vec(wav)
        else:
            text, secs, dev = _run_cnn_lstm(wav, cnn_ckpt)
        info = f"Model: {model_choice} | Audio: {dur:.1f}s | Inference: {secs}s | Device: {dev}"
        return text, info
    except Exception as e:
        return f"Error: {e}", ""

def build_transcribe_tab():
    with gr.Tab("1 · Live Transcribe"):
        gr.Markdown("""
        ### Arabic Speech → Text
        Upload an audio file **or** click the microphone to record live Arabic speech.
        """)
        with gr.Row():
            with gr.Column():
                audio_in    = gr.Audio(label="Arabic Audio",
                                       sources=["upload","microphone"], type="numpy")
                model_radio = gr.Radio(["Whisper","Wav2Vec 2.0","CNN+LSTM (our model)"],
                                       value="Whisper", label="Model")
                w_size      = gr.Dropdown(["tiny","base","small","medium","large-v2"],
                                          value="medium", label="Whisper size", visible=True)
                cnn_ckpt    = gr.Textbox(label="CNN+LSTM checkpoint",
                                         value=DEFAULT_CKPT, visible=False)

                def _toggle(m):
                    return (gr.update(visible=m=="Whisper"),
                            gr.update(visible=m=="CNN+LSTM (our model)"))
                model_radio.change(_toggle, model_radio, [w_size, cnn_ckpt])

                btn = gr.Button("Transcribe ▶", variant="primary")

            with gr.Column():
                out_text = gr.Textbox(label="Transcript (Arabic)", lines=8,
                                      rtl=True, text_align="right")
                out_info = gr.Textbox(label="Info", lines=2, interactive=False)

        btn.click(transcribe_live,
                  [audio_in, model_radio, w_size, cnn_ckpt],
                  [out_text, out_info])

        gr.Markdown("""
        **Model guide for the presentation:**
        - Use **Whisper medium** for best accuracy
        - Use **Whisper tiny** for fastest response on Mac
        - Use **CNN+LSTM** to show your custom model working
        """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — COMPARE ALL MODELS SIDE BY SIDE
# ══════════════════════════════════════════════════════════════════════════════

def compare_all(audio, whisper_size, cnn_ckpt, reference):
    wav, sr = _prep_audio(audio)
    if wav is None:
        return "No audio", "", "No audio", "", "No audio", "", None

    from utils.metrics import compute_wer

    results = {}

    # Whisper
    try:
        text, secs, dev = _run_whisper(wav, whisper_size)
        wer = f"WER: {compute_wer([reference],[text]):.3f}" if reference.strip() else ""
        results["whisper"] = (text, f"Whisper-{whisper_size} | {secs}s | {dev} | {wer}")
    except Exception as e:
        results["whisper"] = (f"Error: {e}", "")

    # Wav2Vec2
    try:
        text, secs, dev = _run_wav2vec(wav)
        wer = f"WER: {compute_wer([reference],[text]):.3f}" if reference.strip() else ""
        results["wav2vec"] = (text, f"Wav2Vec2-XLSR | {secs}s | {dev} | {wer}")
    except Exception as e:
        results["wav2vec"] = (f"Error: {e}", "")

    # CNN+LSTM
    try:
        text, secs, dev = _run_cnn_lstm(wav, cnn_ckpt)
        wer = f"WER: {compute_wer([reference],[text]):.3f}" if reference.strip() else ""
        results["cnn_lstm"] = (text, f"CNN+LSTM | {secs}s | {dev} | {wer}")
    except Exception as e:
        results["cnn_lstm"] = (f"Error: {e}", "")

    # Bar chart of inference times (if all succeeded)
    plot = None
    try:
        names = ["Whisper", "Wav2Vec2", "CNN+LSTM"]
        wers_val = []
        for key in ["whisper","wav2vec","cnn_lstm"]:
            txt = results[key][0]
            if reference.strip() and not txt.startswith("Error"):
                wers_val.append(compute_wer([reference],[txt]))
            else:
                wers_val.append(0)
        if reference.strip() and any(w > 0 for w in wers_val):
            fig, ax = plt.subplots(figsize=(7,3))
            colors = ["#2980b9","#e67e22","#e74c3c"]
            ax.bar(names, wers_val, color=colors, alpha=0.85)
            for i,(n,v) in enumerate(zip(names,wers_val)):
                ax.text(i, v+0.01, f"{v:.3f}", ha="center", fontsize=10)
            ax.set_ylabel("WER (lower=better)")
            ax.set_title("Live Comparison — WER")
            ax.set_ylim(0,1.1); ax.grid(axis="y",alpha=0.3)
            plt.tight_layout()
            path = str(ROOT / "outputs" / "results" / "live_comparison.png")
            fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
            plot = path
    except Exception:
        pass

    return (results["whisper"][0],  results["whisper"][1],
            results["wav2vec"][0],  results["wav2vec"][1],
            results["cnn_lstm"][0], results["cnn_lstm"][1],
            plot)

def build_compare_tab():
    with gr.Tab("2 · Compare Models"):
        gr.Markdown("""
        ### Run the same audio through all three models simultaneously.
        Optionally provide the correct reference text to see WER for each model.
        """)
        with gr.Row():
            with gr.Column(scale=1):
                audio_in   = gr.Audio(label="Input Audio",
                                      sources=["upload","microphone"], type="numpy")
                reference  = gr.Textbox(label="Reference text (optional — for WER)",
                                        placeholder="اكتب النص الصحيح هنا", rtl=True)
                w_size     = gr.Dropdown(["tiny","base","small","medium","large-v2"],
                                         value="tiny", label="Whisper size")
                cnn_ckpt   = gr.Textbox(label="CNN+LSTM checkpoint", value=DEFAULT_CKPT)
                run_btn    = gr.Button("▶  Run All Three", variant="primary")

            with gr.Column(scale=2):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Whisper")
                        w_out   = gr.Textbox(label="Transcript", lines=4, rtl=True)
                        w_info  = gr.Textbox(label="Info", lines=1, interactive=False)
                    with gr.Column():
                        gr.Markdown("#### Wav2Vec 2.0")
                        wv_out  = gr.Textbox(label="Transcript", lines=4, rtl=True)
                        wv_info = gr.Textbox(label="Info", lines=1, interactive=False)
                    with gr.Column():
                        gr.Markdown("#### CNN+LSTM (our model)")
                        cl_out  = gr.Textbox(label="Transcript", lines=4, rtl=True)
                        cl_info = gr.Textbox(label="Info", lines=1, interactive=False)
                cmp_plot = gr.Image(label="WER Comparison", type="filepath")

        run_btn.click(compare_all,
                      [audio_in, w_size, cnn_ckpt, reference],
                      [w_out, w_info, wv_out, wv_info, cl_out, cl_info, cmp_plot])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — BATCH FILES
# ══════════════════════════════════════════════════════════════════════════════

def batch_transcribe(files, model_choice, whisper_size, cnn_ckpt):
    if not files:
        return "No files uploaded.", None

    rows = [["File", "Transcript", "Duration (s)", "Time (s)"]]
    for f in files:
        try:
            import torchaudio
            wav, sr = torchaudio.load(f.name)
            wav = wav.mean(0).numpy().astype(np.float32)
            if sr != 16000:
                import torchaudio.functional as AF
                wav = AF.resample(torch.tensor(wav).unsqueeze(0), sr, 16000).squeeze().numpy()
            dur = len(wav) / 16000

            if model_choice == "Whisper":
                text, secs, _ = _run_whisper(wav, whisper_size)
            elif model_choice == "Wav2Vec 2.0":
                text, secs, _ = _run_wav2vec(wav)
            else:
                text, secs, _ = _run_cnn_lstm(wav, cnn_ckpt)

            rows.append([Path(f.name).name, text, f"{dur:.1f}", f"{secs}"])
        except Exception as e:
            rows.append([Path(f.name).name, f"Error: {e}", "-", "-"])

    # Save CSV
    import csv
    csv_path = str(ROOT / "outputs" / "results" / "batch_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csvf:
        csv.writer(csvf).writerows(rows)

    # Format as markdown table
    header = "| " + " | ".join(rows[0]) + " |"
    sep    = "| " + " | ".join(["---"]*len(rows[0])) + " |"
    body   = "\n".join("| " + " | ".join(r) + " |" for r in rows[1:])
    return f"{header}\n{sep}\n{body}", csv_path

def build_batch_tab():
    with gr.Tab("3 · Batch Files"):
        gr.Markdown("""
        ### Transcribe multiple audio files at once.
        Upload a folder of recordings — results saved to `outputs/results/batch_results.csv`.
        """)
        with gr.Row():
            with gr.Column(scale=1):
                files_in    = gr.File(label="Upload audio files (.wav/.mp3/.flac)",
                                      file_count="multiple", file_types=["audio"])
                model_radio = gr.Radio(["Whisper","Wav2Vec 2.0","CNN+LSTM (our model)"],
                                       value="Whisper", label="Model")
                w_size      = gr.Dropdown(["tiny","base","small","medium"],
                                          value="tiny", label="Whisper size")
                cnn_ckpt    = gr.Textbox(label="CNN+LSTM checkpoint", value=DEFAULT_CKPT)
                run_btn     = gr.Button("▶  Transcribe All", variant="primary")

            with gr.Column(scale=2):
                results_md  = gr.Markdown(label="Results")
                csv_dl      = gr.File(label="Download CSV")

        run_btn.click(batch_transcribe,
                      [files_in, model_radio, w_size, cnn_ckpt],
                      [results_md, csv_dl])


# ══════════════════════════════════════════════════════════════════════════════
# ASSEMBLE
# ══════════════════════════════════════════════════════════════════════════════

def build_app():
    ckpt_exists = Path(DEFAULT_CKPT).exists()
    with gr.Blocks(title="Arabic ASR — Demo") as app:
        gr.Markdown("""
        <div style='text-align:center;padding:8px 0'>
        <h1 style='margin:0'>Arabic Speech Recognition — Demo</h1>
        <p style='color:#666'>Whisper · Wav2Vec 2.0 · CNN+LSTM (custom model)</p>
        <p style='color:#888;font-size:0.85rem'>Device: <b>{}</b> &nbsp;|&nbsp; CNN+LSTM checkpoint: <b>{}</b></p>
        </div>
        """.format(DEVICE, "found ✓" if ckpt_exists else "not found — train first"))
        build_transcribe_tab()
        build_compare_tab()
        build_batch_tab()
    return app

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--port",  type=int, default=7861)
    p.add_argument("--share", action="store_true")
    args = p.parse_args()
    build_app().launch(
        server_port=args.port,
        share=args.share,
        inbrowser=True,
        theme=gr.themes.Soft(primary_hue="indigo"),
    )
