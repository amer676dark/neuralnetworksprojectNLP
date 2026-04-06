"""
Arabic ASR — Full GUI
Tabs: Transcribe · Train · Evaluate · Data Explorer · Results

Run:  python demo/app.py
"""

import sys, os, json, threading, subprocess, queue, time
from pathlib import Path
import numpy as np
import yaml
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

CONFIG_PATH = ROOT / "configs" / "config.yaml"
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

DEVICE_LABEL = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# ── shared model cache ────────────────────────────────────────────────────────
_models: dict = {}

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

# ── audio normalisation helper ────────────────────────────────────────────────
def _prep_audio(audio):
    """Gradio (sr, array) → float32 mono 16 kHz numpy array."""
    if audio is None:
        return None, None
    sr, wav = audio
    wav = wav.astype(np.float32)
    if wav.max() > 1.0:
        wav /= 32768.0
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != 16000:
        import torchaudio.functional as AF
        t = torch.tensor(wav).unsqueeze(0)
        wav = AF.resample(t, sr, 16000).squeeze().numpy()
        sr = 16000
    return wav, sr


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — TRANSCRIBE
# ═══════════════════════════════════════════════════════════════════════════════

def transcribe(audio, model_choice, whisper_size):
    wav, sr = _prep_audio(audio)
    if wav is None:
        return "Please upload or record audio.", ""
    try:
        if model_choice == "Whisper":
            m = _load_whisper(whisper_size)
            text = m.transcribe_array(wav)
            info = f"Whisper-{whisper_size} | {len(wav)/sr:.1f}s | device: {m.device}"
        else:
            m = _load_wav2vec()
            text = m.transcribe_array(wav)
            info = f"Wav2Vec2-XLSR-Arabic | {len(wav)/sr:.1f}s | device: {m.device}"
        return text, info
    except Exception as e:
        return f"Error: {e}", ""

def build_transcribe_tab():
    with gr.Tab("🎤 Transcribe"):
        gr.Markdown("### Arabic Speech → Text\nUpload a file **or** click the microphone to record live.")
        with gr.Row():
            with gr.Column():
                audio_in = gr.Audio(label="Arabic Audio", sources=["upload", "microphone"], type="numpy")
                with gr.Row():
                    model_radio = gr.Radio(["Whisper", "Wav2Vec 2.0"], value="Whisper", label="Model")
                    w_size = gr.Dropdown(
                        ["tiny", "base", "small", "medium", "large-v2"],
                        value="medium", label="Whisper size", visible=True
                    )
                model_radio.change(lambda c: gr.update(visible=c == "Whisper"), model_radio, w_size)
                btn = gr.Button("Transcribe ▶", variant="primary")

            with gr.Column():
                out_text = gr.Textbox(label="Arabic Transcript", lines=8, rtl=True, text_align="right")
                out_info = gr.Textbox(label="Info", lines=1, interactive=False)

        btn.click(transcribe, [audio_in, model_radio, w_size], [out_text, out_info])

        gr.Markdown("""
        | Model | Architecture | Params | Est. WER |
        |-------|-------------|--------|---------|
        | Whisper medium | Encoder-Decoder Transformer | 769M | ~15-25% |
        | Wav2Vec 2.0 XLSR | CNN + Transformer | 300M | ~20-35% |
        | CNN+LSTM (custom) | CNN + BiLSTM + CTC | 20M | ~40-60% |
        """)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TRAIN
# ═══════════════════════════════════════════════════════════════════════════════

_train_proc = None
_train_log_q: queue.Queue = queue.Queue()

def _stream_proc(proc, q):
    for line in iter(proc.stdout.readline, ""):
        q.put(line)
    proc.stdout.close()

def start_training(epochs, batch_size, lr, max_samples, max_eval):
    global _train_proc
    if _train_proc and _train_proc.poll() is None:
        return "Training is already running. Stop it first.", gr.update(interactive=True)

    # patch config on-the-fly via env override
    env = os.environ.copy()
    env["ASR_EPOCHS"]      = str(int(epochs))
    env["ASR_BATCH"]       = str(int(batch_size))
    env["ASR_LR"]          = str(float(lr))
    env["ASR_TRAIN_N"]     = str(int(max_samples))
    env["ASR_EVAL_N"]      = str(int(max_eval))

    cmd = [sys.executable, str(ROOT / "training" / "train_cnn_lstm.py"),
           "--config", str(CONFIG_PATH)]

    _train_proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env, cwd=str(ROOT)
    )
    threading.Thread(target=_stream_proc, args=(_train_proc, _train_log_q), daemon=True).start()
    return "Training started...\n", gr.update(interactive=True)

def stop_training():
    global _train_proc
    if _train_proc and _train_proc.poll() is None:
        _train_proc.terminate()
        return "Training stopped."
    return "No training process running."

def poll_log(current):
    lines = []
    try:
        while True:
            lines.append(_train_log_q.get_nowait())
    except queue.Empty:
        pass
    return (current or "") + "".join(lines)

def build_train_tab():
    with gr.Tab("🏋️ Train CNN+LSTM"):
        gr.Markdown("### Train the custom CNN+BiLSTM model from scratch on Mozilla Common Voice Arabic.")
        with gr.Row():
            with gr.Column():
                gr.Markdown("**Hyperparameters**")
                epochs     = gr.Slider(1, 100, value=10,   step=1,    label="Epochs")
                batch_size = gr.Slider(4, 64,  value=16,   step=4,    label="Batch Size")
                lr         = gr.Number(value=0.0003, label="Learning Rate", precision=6)
                max_train  = gr.Slider(100, 10000, value=1000, step=100, label="Max Train Samples")
                max_eval   = gr.Slider(50,  2000,  value=200,  step=50,  label="Max Eval Samples")
                with gr.Row():
                    start_btn = gr.Button("▶ Start Training", variant="primary")
                    stop_btn  = gr.Button("⏹ Stop",           variant="stop")
                status = gr.Textbox(label="Status", lines=1, interactive=False, value="Idle")

            with gr.Column():
                gr.Markdown("**Training Log**")
                log_box = gr.Textbox(label="", lines=22, max_lines=22,
                                     interactive=False, autoscroll=True,
                                     placeholder="Logs will appear here once training starts...")
                refresh_btn = gr.Button("↻ Refresh Log")

        start_btn.click(start_training,
                        [epochs, batch_size, lr, max_train, max_eval],
                        [log_box, start_btn])
        stop_btn.click(stop_training, [], status)
        refresh_btn.click(poll_log, log_box, log_box)

        gr.Markdown("""
        **Tips**
        - First run downloads Common Voice Arabic (~2 GB). Keep network stable.
        - `tiny` Whisper or `base` Wav2Vec work fine on CPU for a quick demo.
        - Best model checkpoint saved to `outputs/checkpoints/cnn_lstm/best_model.pt`.
        - Training curves saved to `outputs/results/training_curves.png` after completion.
        """)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — EVALUATE
# ═══════════════════════════════════════════════════════════════════════════════

def run_evaluation(model_name, whisper_size_eval, num_eval_samples):
    try:
        from data.dataset import load_common_voice_arabic
        from utils.metrics import compute_batch_metrics, format_metrics_report

        n = int(num_eval_samples)
        yield f"Loading {n} test samples from Common Voice Arabic...\n", None

        test_ds = load_common_voice_arabic("test", max_samples=n)
        samples = list(test_ds)
        refs = [s["sentence"] for s in samples]

        yield f"Loaded {len(samples)} samples. Running {model_name}...\n", None

        hyps = []
        if model_name == "Whisper":
            m = _load_whisper(whisper_size_eval)
            for i, s in enumerate(samples):
                wav = s["audio"]["array"].astype(np.float32)
                hyps.append(m.transcribe_array(wav))
                if (i + 1) % 10 == 0:
                    yield f"  Transcribed {i+1}/{len(samples)}...\n", None
        elif model_name == "Wav2Vec 2.0":
            m = _load_wav2vec()
            for i, s in enumerate(samples):
                wav = s["audio"]["array"].astype(np.float32)
                hyps.append(m.transcribe_array(wav))
                if (i + 1) % 10 == 0:
                    yield f"  Transcribed {i+1}/{len(samples)}...\n", None

        metrics = compute_batch_metrics(refs, hyps)
        report = format_metrics_report(metrics)

        # Side-by-side table (first 10)
        table_lines = ["| # | Reference | Hypothesis | WER |", "|---|-----------|------------|-----|"]
        from utils.metrics import compute_wer
        for i, (r, h) in enumerate(zip(refs[:10], hyps[:10])):
            w = compute_wer([r], [h])
            table_lines.append(f"| {i+1} | {r[:40]} | {h[:40]} | {w:.2f} |")
        table_md = "\n".join(table_lines)

        # Comparison plot
        fig = _make_eval_plot(metrics, model_name)

        yield report + "\n\n" + table_md, fig

    except Exception as e:
        import traceback
        yield f"Error: {e}\n{traceback.format_exc()}", None

def _make_eval_plot(metrics, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # Bar chart
    bars = axes[0].bar(
        ["WER", "CER"],
        [metrics["wer"], metrics["cer"]],
        color=["steelblue", "coral"], alpha=0.85, width=0.5
    )
    for b in bars:
        axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                     f"{b.get_height():.3f}", ha="center", fontsize=12, fontweight="bold")
    axes[0].set_ylim(0, 1.1)
    axes[0].set_title(f"{model_name} — WER & CER", fontsize=13)
    axes[0].set_ylabel("Error Rate (lower is better)")
    axes[0].grid(axis="y", alpha=0.3)

    # Stats table
    axes[1].axis("off")
    rows = [
        ["Metric", "Value"],
        ["WER", f"{metrics['wer']:.4f}  ({metrics['wer']*100:.1f}%)"],
        ["CER", f"{metrics['cer']:.4f}  ({metrics['cer']*100:.1f}%)"],
        ["Mean sample WER", f"{metrics['mean_sample_wer']:.4f}"],
        ["Std sample WER",  f"{metrics['std_sample_wer']:.4f}"],
        ["Best sample WER", f"{metrics['min_wer']:.4f}"],
        ["Worst sample WER",f"{metrics['max_wer']:.4f}"],
    ]
    tbl = axes[1].table(cellText=rows[1:], colLabels=rows[0],
                        cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_facecolor("#4A90D9")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f0f4ff")

    plt.tight_layout()
    return fig

def build_evaluate_tab():
    with gr.Tab("📊 Evaluate"):
        gr.Markdown("### Run evaluation on Mozilla Common Voice Arabic test set and compute WER / CER.")
        with gr.Row():
            with gr.Column(scale=1):
                eval_model  = gr.Radio(["Whisper", "Wav2Vec 2.0"], value="Whisper", label="Model")
                eval_w_size = gr.Dropdown(["tiny","base","small","medium","large-v2"],
                                          value="tiny", label="Whisper size (tiny is faster)")
                eval_model.change(lambda c: gr.update(visible=c=="Whisper"), eval_model, eval_w_size)
                n_samples   = gr.Slider(10, 500, value=50, step=10, label="Number of test samples")
                eval_btn    = gr.Button("▶ Run Evaluation", variant="primary")

            with gr.Column(scale=2):
                eval_log  = gr.Textbox(label="Results & Predictions", lines=18, interactive=False)
                eval_plot = gr.Plot(label="Metrics Chart")

        eval_btn.click(run_evaluation, [eval_model, eval_w_size, n_samples],
                       [eval_log, eval_plot])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — DATA EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════

_explorer_cache = {}

def explore_sample(split, index):
    try:
        from data.dataset import load_common_voice_arabic
        from utils.audio_utils import extract_mel_spectrogram, normalize_audio

        key = f"{split}_50"
        if key not in _explorer_cache:
            ds = load_common_voice_arabic(split, max_samples=50)
            _explorer_cache[key] = list(ds)
        samples = _explorer_cache[key]

        idx = int(index) % len(samples)
        s = samples[idx]
        wav = s["audio"]["array"].astype(np.float32)
        sr  = s["audio"]["sampling_rate"]
        transcript = s["sentence"]

        duration = len(wav) / sr
        info = f"Sample #{idx+1}/{len(samples)}  |  Duration: {duration:.2f}s  |  SR: {sr} Hz"

        # Waveform + mel spectrogram side by side
        fig, axes = plt.subplots(1, 2, figsize=(12, 3))

        # Waveform
        t = np.linspace(0, duration, len(wav))
        axes[0].plot(t, wav, linewidth=0.4, color="steelblue")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Amplitude")
        axes[0].set_title("Waveform")
        axes[0].grid(alpha=0.3)

        # Mel spectrogram
        mel = extract_mel_spectrogram(normalize_audio(wav), sample_rate=sr)
        im = axes[1].imshow(mel, aspect="auto", origin="lower",
                            extent=[0, duration, 0, 80], cmap="viridis")
        fig.colorbar(im, ax=axes[1], format="%+2.0f dB")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Mel bin")
        axes[1].set_title("Log-Mel Spectrogram")

        plt.suptitle(f'"{transcript}"', fontsize=10, y=1.02)
        plt.tight_layout()

        return (sr, wav), transcript, info, fig

    except Exception as e:
        import traceback
        return None, f"Error: {e}", traceback.format_exc(), None

def build_explorer_tab():
    with gr.Tab("🔍 Data Explorer"):
        gr.Markdown("### Browse Mozilla Common Voice Arabic samples. Loads 50 samples per split on first use.")
        with gr.Row():
            split_dd = gr.Dropdown(["train","validation","test"], value="train", label="Split")
            idx_sl   = gr.Slider(0, 49, value=0, step=1, label="Sample index")
            load_btn = gr.Button("Load Sample", variant="primary")

        with gr.Row():
            audio_out   = gr.Audio(label="Audio", interactive=False)
            with gr.Column():
                trans_out = gr.Textbox(label="Transcript (Arabic)", lines=3, rtl=True, text_align="right")
                info_out  = gr.Textbox(label="Metadata", lines=2, interactive=False)

        plot_out = gr.Plot(label="Waveform & Mel Spectrogram")

        load_btn.click(explore_sample, [split_dd, idx_sl],
                       [audio_out, trans_out, info_out, plot_out])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

def load_results():
    results_dir = ROOT / "outputs" / "results"
    if not results_dir.exists():
        return "No results yet. Run training or evaluation first.", None, None, None

    text_parts = []

    # JSON files
    for jf in sorted(results_dir.glob("*.json")):
        try:
            with open(jf) as f:
                data = json.load(f)
            text_parts.append(f"=== {jf.name} ===")
            for k, v in data.items():
                if isinstance(v, float):
                    text_parts.append(f"  {k}: {v:.4f}")
                elif not isinstance(v, list):
                    text_parts.append(f"  {k}: {v}")
            text_parts.append("")
        except Exception:
            pass

    summary = "\n".join(text_parts) if text_parts else "No JSON results found."

    def _load_img(name):
        p = results_dir / name
        return str(p) if p.exists() else None

    return (
        summary,
        _load_img("training_curves.png"),
        _load_img("model_comparison.png"),
        _load_img("cnn_lstm_wer_dist.png"),
    )

def build_results_tab():
    with gr.Tab("📈 Results"):
        gr.Markdown("### Saved experiment results and plots.")
        refresh_btn = gr.Button("↻ Refresh", variant="secondary")

        summary_box = gr.Textbox(label="Metrics Summary", lines=15, interactive=False)
        with gr.Row():
            img_train  = gr.Image(label="Training Curves",      type="filepath")
            img_cmp    = gr.Image(label="Model Comparison",      type="filepath")
            img_wer    = gr.Image(label="WER Distribution",      type="filepath")

        refresh_btn.click(load_results, [], [summary_box, img_train, img_cmp, img_wer])


# ═══════════════════════════════════════════════════════════════════════════════
# ASSEMBLE
# ═══════════════════════════════════════════════════════════════════════════════

def build_app():
    with gr.Blocks(title="Arabic ASR System") as app:

        gr.Markdown("""
        <div style="text-align:center; padding: 10px 0 4px 0">
            <h1 style="font-size:2rem; margin:0">🎙️ Arabic Speech Recognition System</h1>
            <p style="color:#555; margin:4px 0 0 0">Deep Learning Pipeline · CNN+LSTM · Whisper · Wav2Vec 2.0</p>
            <p style="color:#888; font-size:0.85rem">Device: <b>{device}</b></p>
        </div>
        """.format(device=DEVICE_LABEL))

        build_transcribe_tab()
        build_train_tab()
        build_evaluate_tab()
        build_explorer_tab()
        build_results_tab()

    return app


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--share",  action="store_true")
    parser.add_argument("--port",   type=int, default=7860)
    parser.add_argument("--host",   default="0.0.0.0")
    args = parser.parse_args()

    app = build_app()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=True,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo"),
        css=".gradio-container { max-width: 1200px !important; }",
    )
