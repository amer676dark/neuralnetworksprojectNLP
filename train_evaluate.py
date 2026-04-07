"""
FILE 1 — Train & Evaluate
=========================
Run:  python train_evaluate.py
Opens a Gradio GUI at http://localhost:7860

Tabs:
  1. Dataset Setup    — choose dataset, verify it loads, set HF token
  2. Train CNN+LSTM   — configure hyperparameters, train, live log
  3. Evaluate Models  — run Whisper / Wav2Vec2 / CNN+LSTM on test set
  4. Results          — view all charts and metrics
"""

import sys, os, json, subprocess, threading, queue, time
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

RESULTS_DIR = ROOT / "outputs" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")

# ── helpers ───────────────────────────────────────────────────────────────────

def _save_config(cfg: dict):
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)

def _reload_config():
    global CONFIG
    with open(CONFIG_PATH) as f:
        CONFIG = yaml.safe_load(f)

def _load_results_plots():
    def _img(name):
        p = RESULTS_DIR / name
        return str(p) if p.exists() else None
    summary = ""
    for jf in sorted(RESULTS_DIR.glob("*.json")):
        try:
            data = json.load(open(jf))
            summary += f"=== {jf.name} ===\n"
            for k, v in data.items():
                if isinstance(v, float):
                    summary += f"  {k}: {v:.4f}  ({v*100:.1f}%)\n"
                elif not isinstance(v, list):
                    summary += f"  {k}: {v}\n"
            summary += "\n"
        except Exception:
            pass
    if not summary:
        summary = "No results yet. Complete training and evaluation first."
    return (summary,
            _img("training_curves.png"),
            _img("model_comparison.png"),
            _img("cnn_lstm_wer_dist.png"))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATASET SETUP
# ══════════════════════════════════════════════════════════════════════════════

def verify_dataset(source, data_dir, hf_token, max_samples):
    """Load 5 samples and show preview."""
    try:
        if hf_token.strip():
            import huggingface_hub
            huggingface_hub.login(token=hf_token.strip(), add_to_git_credential=False)
            os.environ["HF_TOKEN"] = hf_token.strip()

        from data.dataset import load_dataset_by_name
        ds = load_dataset_by_name(
            source,
            split="train",
            data_dir=data_dir.strip() if data_dir.strip() else None,
            max_samples=5,
            hf_token=hf_token.strip() or None,
        )
        samples = list(ds)[:3]
        info = f"Dataset: {source} | Loaded {len(ds)} sample(s) for preview\n\n"
        for i, s in enumerate(samples):
            dur = len(s["audio"]["array"]) / s["audio"]["sampling_rate"]
            info += f"Sample {i+1}: \"{s['sentence'][:60]}\"  ({dur:.1f}s)\n"

        wav = samples[0]["audio"]["array"].astype(np.float32)
        sr  = samples[0]["audio"]["sampling_rate"]
        return info, (sr, wav)
    except Exception as e:
        import traceback
        return f"ERROR: {e}\n\n{traceback.format_exc()}", None

def apply_dataset_config(source, data_dir, hf_token, max_train, max_eval, max_dur):
    _reload_config()
    CONFIG["data"]["source"]           = source
    CONFIG["data"]["data_dir"]         = data_dir.strip() or None
    CONFIG["data"]["hf_token"]         = hf_token.strip() or None
    CONFIG["data"]["max_train_samples"] = int(max_train)
    CONFIG["data"]["max_eval_samples"]  = int(max_eval)
    CONFIG["data"]["max_duration"]      = float(max_dur)
    CONFIG["audio"]["max_audio_length"] = float(max_dur)
    _save_config(CONFIG)
    if hf_token.strip():
        os.environ["HF_TOKEN"] = hf_token.strip()
    return f"Config saved.\nDataset: {source} | Train samples: {max_train} | Eval: {max_eval} | Max duration: {max_dur}s"

def build_dataset_tab():
    with gr.Tab("1 · Dataset Setup"):
        gr.Markdown("""
        ### Step 1 — Configure your dataset
        Choose which dataset to train and evaluate on.
        **Verify** before training to confirm the path / token is correct.
        """)
        with gr.Row():
            with gr.Column():
                source_dd = gr.Dropdown(
                    ["common_voice", "masc", "arabic_speech_corpus", "ejust", "combined"],
                    value="common_voice", label="Dataset source"
                )
                data_dir_tb = gr.Textbox(label="Local folder path (for ASC / EJUST)",
                                         placeholder="/path/to/downloaded/dataset",
                                         visible=False)
                hf_token_tb = gr.Textbox(label="HuggingFace token (for Common Voice)",
                                         placeholder="hf_...", type="password",
                                         info="Get from huggingface.co/settings/tokens")

                def toggle_dir(src):
                    return gr.update(visible=src in ("arabic_speech_corpus","ejust","combined"))
                source_dd.change(toggle_dir, source_dd, data_dir_tb)

                with gr.Row():
                    max_train = gr.Slider(100, 20000, value=5000,  step=100, label="Max train samples")
                    max_eval  = gr.Slider(50,  2000,  value=500,   step=50,  label="Max eval samples")
                max_dur = gr.Slider(3, 30, value=10, step=1, label="Max audio duration (s)")

                with gr.Row():
                    verify_btn = gr.Button("Verify Dataset", variant="secondary")
                    apply_btn  = gr.Button("Save Config",    variant="primary")
                apply_status = gr.Textbox(label="Status", lines=2, interactive=False)

            with gr.Column():
                verify_log   = gr.Textbox(label="Verification Output", lines=10, interactive=False)
                preview_audio = gr.Audio(label="Sample Audio Preview", interactive=False)

        verify_btn.click(verify_dataset,
                         [source_dd, data_dir_tb, hf_token_tb, max_train],
                         [verify_log, preview_audio])
        apply_btn.click(apply_dataset_config,
                        [source_dd, data_dir_tb, hf_token_tb, max_train, max_eval, max_dur],
                        apply_status)

        gr.Markdown("""
        ---
        | Dataset | Auth needed | Size | Download link |
        |---------|------------|------|---------------|
        | Mozilla Common Voice Arabic | HF login + accept terms | ~14 GB | huggingface.co/datasets/mozilla-foundation/common_voice_17_0 |
        | MASC | None | auto | huggingface.co/datasets/hirundo-io/MASC |
        | Arabic Speech Corpus | Free registration | ~1.5 GB | arabicspeechcorpus.com |
        | EJUST | Instructor provided | — | Google Drive (see instructor) |
        """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TRAIN CNN+LSTM
# ══════════════════════════════════════════════════════════════════════════════

_train_proc   = None
_train_log_q  : queue.Queue = queue.Queue()

def _stream(proc, q):
    for line in iter(proc.stdout.readline, ""):
        q.put(line)
    proc.stdout.close()

def start_training(epochs, batch_size, lr, lstm_hidden, lstm_layers,
                   spec_aug, resume_ckpt):
    global _train_proc
    if _train_proc and _train_proc.poll() is None:
        return "Already running — stop it first.", gr.update()

    _reload_config()
    CONFIG["cnn_lstm"]["num_epochs"]     = int(epochs)
    CONFIG["cnn_lstm"]["batch_size"]     = int(batch_size)
    CONFIG["cnn_lstm"]["learning_rate"]  = float(lr)
    CONFIG["cnn_lstm"]["lstm_hidden_size"]= int(lstm_hidden)
    CONFIG["cnn_lstm"]["lstm_num_layers"] = int(lstm_layers)
    CONFIG["cnn_lstm"]["spec_augment"]   = bool(spec_aug)
    _save_config(CONFIG)

    cmd = [sys.executable, str(ROOT / "training" / "train_cnn_lstm.py"),
           "--config", str(CONFIG_PATH)]
    if resume_ckpt:
        cmd += ["--resume", resume_ckpt]

    _train_proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, cwd=str(ROOT)
    )
    threading.Thread(target=_stream, args=(_train_proc, _train_log_q), daemon=True).start()
    return "Training started...\n", gr.update(interactive=True)

def stop_training():
    global _train_proc
    if _train_proc and _train_proc.poll() is None:
        _train_proc.terminate()
        return "Training stopped."
    return "No training process running."

def poll_train_log(current):
    lines = []
    try:
        while True:
            lines.append(_train_log_q.get_nowait())
    except queue.Empty:
        pass
    return (current or "") + "".join(lines)

def build_train_tab():
    with gr.Tab("2 · Train CNN+LSTM"):
        gr.Markdown("""
        ### Step 2 — Train the custom CNN+BiLSTM+Attention model
        Architecture: **SpecAugment → Residual CNN → BiLSTM × 3 → Multi-Head Attention → CTC**

        Configure hyperparameters then click Start. Logs stream in real time.
        Best checkpoint saved to `outputs/checkpoints/cnn_lstm/best_model.pt`.
        """)
        with gr.Row():
            with gr.Column():
                gr.Markdown("**Hyperparameters**")
                epochs       = gr.Slider(1, 100,  value=50,    step=1,    label="Epochs")
                batch_size   = gr.Slider(4, 64,   value=16,    step=4,    label="Batch size")
                lr           = gr.Number(value=0.0003,                    label="Learning rate", precision=6)
                lstm_hidden  = gr.Slider(128, 1024,value=512,  step=128,  label="LSTM hidden size")
                lstm_layers  = gr.Slider(1, 5,    value=3,     step=1,    label="LSTM layers")
                spec_aug     = gr.Checkbox(value=True,                    label="SpecAugment (recommended)")
                resume_ckpt  = gr.Textbox(label="Resume from checkpoint (optional)",
                                          placeholder="outputs/checkpoints/cnn_lstm/best_model.pt")
                with gr.Row():
                    start_btn = gr.Button("▶  Start Training", variant="primary")
                    stop_btn  = gr.Button("⏹  Stop",           variant="stop")
                train_status = gr.Textbox(label="Status", lines=1, interactive=False)

            with gr.Column():
                gr.Markdown("**Live Training Log**")
                log_box     = gr.Textbox(label="", lines=25, max_lines=25,
                                         interactive=False, autoscroll=True)
                refresh_btn = gr.Button("↻ Refresh Log")

        start_btn.click(start_training,
                        [epochs, batch_size, lr, lstm_hidden, lstm_layers, spec_aug, resume_ckpt],
                        [log_box, start_btn])
        stop_btn.click(stop_training, [], train_status)
        refresh_btn.click(poll_train_log, log_box, log_box)

        gr.Markdown("""
        **Time estimates (50 epochs, 5000 samples):**
        - M2 MacBook Air (MPS) — 2–4 hours (thermals will throttle it)
        - RTX 2060 Super (CUDA) — 25–35 minutes  ← recommended
        - CPU only — 6–10 hours
        """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — EVALUATE ALL MODELS
# ══════════════════════════════════════════════════════════════════════════════

_model_cache = {}

def _prep(audio):
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
    return wav, 16000

def run_evaluation(run_whisper, whisper_size, run_wav2vec, run_cnn_lstm, ckpt_path, n_samples):
    try:
        from data.dataset import load_dataset_by_name
        _reload_config()
        src      = CONFIG["data"].get("source", "common_voice")
        data_dir = CONFIG["data"].get("data_dir")
        hf_token = CONFIG["data"].get("hf_token") or os.environ.get("HF_TOKEN")

        yield "Loading test samples...\n", None, None

        ds = load_dataset_by_name(src, "test", data_dir, int(n_samples), hf_token)
        samples  = list(ds)
        refs     = [s["sentence"] for s in samples]

        yield f"Loaded {len(samples)} test samples.\n", None, None

        from utils.metrics import compute_batch_metrics, format_metrics_report

        all_names, all_wers, all_cers = [], [], []
        all_hyps = {}

        # Whisper
        if run_whisper:
            yield f"Running Whisper-{whisper_size}...\n", None, None
            from models.whisper_asr import WhisperASR
            key = f"whisper_{whisper_size}"
            if key not in _model_cache:
                _model_cache[key] = WhisperASR(whisper_size, "ar", "auto")
            wm = _model_cache[key]
            hyps = []
            for i, s in enumerate(samples):
                wav = s["audio"]["array"].astype(np.float32)
                hyps.append(wm.transcribe_array(wav))
                if (i+1) % 10 == 0:
                    yield f"  Whisper [{i+1}/{len(samples)}]\n", None, None
            m = compute_batch_metrics(refs, hyps)
            all_names.append(f"Whisper-{whisper_size}")
            all_wers.append(m["wer"]); all_cers.append(m["cer"])
            all_hyps["whisper"] = hyps
            yield format_metrics_report(m) + "\n", None, None

        # Wav2Vec2
        if run_wav2vec:
            yield "Running Wav2Vec2 XLSR Arabic...\n", None, None
            from models.wav2vec_asr import Wav2Vec2ASR
            if "wav2vec" not in _model_cache:
                _model_cache["wav2vec"] = Wav2Vec2ASR(device="auto")
            wv = _model_cache["wav2vec"]
            hyps = []
            for i, s in enumerate(samples):
                wav = s["audio"]["array"].astype(np.float32)
                hyps.append(wv.transcribe_array(wav))
                if (i+1) % 10 == 0:
                    yield f"  Wav2Vec2 [{i+1}/{len(samples)}]\n", None, None
            m = compute_batch_metrics(refs, hyps)
            all_names.append("Wav2Vec2-XLSR")
            all_wers.append(m["wer"]); all_cers.append(m["cer"])
            all_hyps["wav2vec"] = hyps
            yield format_metrics_report(m) + "\n", None, None

        # CNN+LSTM
        if run_cnn_lstm:
            ckpt = ckpt_path.strip() or str(ROOT / "outputs/checkpoints/cnn_lstm/best_model.pt")
            if not Path(ckpt).exists():
                yield f"CNN+LSTM checkpoint not found: {ckpt}\nTrain it first.\n", None, None
            else:
                yield "Running CNN+LSTM...\n", None, None
                from models.cnn_lstm_asr import build_model
                from utils.audio_utils import extract_mel_spectrogram, normalize_audio, pad_or_trim
                state   = torch.load(ckpt, map_location="cpu")
                vocab   = state["vocab"]
                idx2ch  = {v: k for k, v in vocab.items()}
                dev     = torch.device(DEVICE)
                cm      = build_model(len(vocab), CONFIG).to(dev)
                cm.load_state_dict(state["model_state"])
                cm.eval()
                audio_cfg = CONFIG["audio"]
                hyps = []
                for s in samples:
                    wav = normalize_audio(s["audio"]["array"].astype(np.float32))
                    wav = pad_or_trim(wav, audio_cfg["max_audio_length"])
                    mel = extract_mel_spectrogram(wav, n_mels=audio_cfg["n_mels"],
                                                  n_fft=audio_cfg["n_fft"],
                                                  hop_length=audio_cfg["hop_length"])
                    mt = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(dev)
                    mt = (mt - mt.mean()) / (mt.std() + 1e-8)
                    with torch.no_grad():
                        lp = cm(mt)
                    ids = cm.greedy_decode(lp)[0]
                    hyps.append("".join(idx2ch.get(i,"") for i in ids if i not in {0,2,3}))
                m = compute_batch_metrics(refs, hyps)
                all_names.append("CNN+LSTM")
                all_wers.append(m["wer"]); all_cers.append(m["cer"])
                all_hyps["cnn_lstm"] = hyps
                yield format_metrics_report(m) + "\n", None, None

        # Save results
        results = {"models": {}}
        for name, wer, cer in zip(all_names, all_wers, all_cers):
            key = name.lower().replace(" ", "_").replace("-", "_")
            results["models"][key] = {"wer": round(wer,4), "cer": round(cer,4)}
            for src_name, hyps in all_hyps.items():
                preds = [{"reference": r, "hypothesis": h} for r, h in zip(refs, hyps)]
                with open(RESULTS_DIR / f"{src_name}_predictions.json", "w", encoding="utf-8") as f:
                    json.dump(preds, f, indent=2, ensure_ascii=False)
        with open(RESULTS_DIR / "all_models_evaluation.json", "w") as f:
            json.dump(results, f, indent=2)

        # Comparison plot
        if len(all_names) >= 1:
            fig = _make_comparison_plot(all_names, all_wers, all_cers)
            fig.savefig(RESULTS_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

        # Side-by-side table (first 10)
        table = "| # | Reference | " + " | ".join(all_names) + " |\n"
        table += "|---|---| " + " | ".join(["---"] * len(all_names)) + " |\n"
        from utils.metrics import compute_wer
        for i, ref in enumerate(refs[:10]):
            row = f"| {i+1} | {ref[:40]} |"
            for model_key in ["whisper","wav2vec","cnn_lstm"]:
                if model_key in all_hyps:
                    h = all_hyps[model_key][i]
                    w = compute_wer([ref],[h])
                    row += f" {h[:35]} (WER:{w:.2f}) |"
            table += row + "\n"

        yield "Done.\n\n" + table, str(RESULTS_DIR/"model_comparison.png") if (RESULTS_DIR/"model_comparison.png").exists() else None, None

    except Exception as e:
        import traceback
        yield f"Error: {e}\n{traceback.format_exc()}", None, None

def _make_comparison_plot(names, wers, cers):
    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")
    b1 = ax.bar(x - w/2, wers, w, label="WER", color="#2980b9", alpha=0.85)
    b2 = ax.bar(x + w/2, cers, w, label="CER", color="#e74c3c", alpha=0.85)
    for b in list(b1)+list(b2):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                f"{b.get_height():.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Error Rate (lower is better)"); ax.set_ylim(0, 1.05)
    ax.set_title("Arabic ASR — Model Comparison", fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    return fig

def build_evaluate_tab():
    with gr.Tab("3 · Evaluate Models"):
        gr.Markdown("""
        ### Step 3 — Evaluate all models on the test set
        Computes **WER** (Word Error Rate) and **CER** (Character Error Rate).
        CNN+LSTM requires a trained checkpoint from Tab 2.
        """)
        with gr.Row():
            with gr.Column(scale=1):
                run_w  = gr.Checkbox(value=True,  label="Whisper")
                w_size = gr.Dropdown(["tiny","base","small","medium","large-v2"],
                                     value="tiny", label="Whisper size")
                run_wv = gr.Checkbox(value=True,  label="Wav2Vec 2.0 XLSR Arabic")
                run_cl = gr.Checkbox(value=True,  label="CNN+LSTM (our model)")
                ckpt   = gr.Textbox(label="CNN+LSTM checkpoint",
                                    value="outputs/checkpoints/cnn_lstm/best_model.pt")
                n_samp = gr.Slider(10, 500, value=100, step=10, label="Test samples")
                eval_btn = gr.Button("▶  Run Evaluation", variant="primary")

            with gr.Column(scale=2):
                eval_log  = gr.Textbox(label="Results & Sample Predictions", lines=22, interactive=False)
                cmp_img   = gr.Image(label="Model Comparison Chart", type="filepath")

        eval_btn.click(run_evaluation,
                       [run_w, w_size, run_wv, run_cl, ckpt, n_samp],
                       [eval_log, cmp_img, gr.State()])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — RESULTS
# ══════════════════════════════════════════════════════════════════════════════

def build_results_tab():
    with gr.Tab("4 · Results"):
        gr.Markdown("### Saved results from training and evaluation runs.")
        refresh_btn = gr.Button("↻ Refresh")
        summary     = gr.Textbox(label="Metrics Summary", lines=20, interactive=False)
        with gr.Row():
            img_train = gr.Image(label="Training Curves",   type="filepath")
            img_cmp   = gr.Image(label="Model Comparison",  type="filepath")
            img_wer   = gr.Image(label="WER Distribution",  type="filepath")
        refresh_btn.click(_load_results_plots, [], [summary, img_train, img_cmp, img_wer])


# ══════════════════════════════════════════════════════════════════════════════
# ASSEMBLE
# ══════════════════════════════════════════════════════════════════════════════

def build_app():
    with gr.Blocks(title="Arabic ASR — Train & Evaluate") as app:
        gr.Markdown("""
        <div style='text-align:center;padding:8px 0'>
        <h1 style='margin:0'>Arabic ASR — Training & Evaluation</h1>
        <p style='color:#666'>CNN+LSTM · Whisper · Wav2Vec 2.0 · Mozilla Common Voice Arabic</p>
        <p style='color:#888;font-size:0.85rem'>Device: <b>{}</b></p>
        </div>
        """.format(DEVICE))
        build_dataset_tab()
        build_train_tab()
        build_evaluate_tab()
        build_results_tab()
    return app

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--port",  type=int, default=7860)
    p.add_argument("--share", action="store_true")
    args = p.parse_args()
    build_app().launch(
        server_port=args.port,
        share=args.share,
        inbrowser=True,
        theme=gr.themes.Soft(primary_hue="blue"),
    )
