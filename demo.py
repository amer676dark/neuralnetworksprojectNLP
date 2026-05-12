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

def _load_seamless():
    if "seamless" not in _models:
        from models.seamless_m4t_asr import SeamlessM4TASR
        _models["seamless"] = SeamlessM4TASR(
            model_name=CONFIG["seamless_m4t"]["model_name"],
            language=CONFIG["seamless_m4t"]["language"],
            device="auto",
        )
    return _models["seamless"]

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

def _run_seamless(wav):
    t0 = time.time()
    m  = _load_seamless()
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
        elif model_choice == "SeamlessM4T-v2":
            text, secs, dev = _run_seamless(wav)
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
                model_radio = gr.Radio(["Whisper","Wav2Vec 2.0","SeamlessM4T-v2","CNN+LSTM (our model)"],
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
        return "No audio", "", "No audio", "", "No audio", "", "No audio", "", None

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

    # SeamlessM4T-v2
    try:
        text, secs, dev = _run_seamless(wav)
        wer = f"WER: {compute_wer([reference],[text]):.3f}" if reference.strip() else ""
        results["seamless"] = (text, f"SeamlessM4T-v2 | {secs}s | {dev} | {wer}")
    except Exception as e:
        results["seamless"] = (f"Error: {e}", "")

    # CNN+LSTM
    try:
        text, secs, dev = _run_cnn_lstm(wav, cnn_ckpt)
        wer = f"WER: {compute_wer([reference],[text]):.3f}" if reference.strip() else ""
        results["cnn_lstm"] = (text, f"CNN+LSTM | {secs}s | {dev} | {wer}")
    except Exception as e:
        results["cnn_lstm"] = (f"Error: {e}", "")

    # Bar chart of WER (if reference provided)
    plot = None
    try:
        names = ["Whisper", "Wav2Vec2", "SeamlessM4T", "CNN+LSTM"]
        wers_val = []
        for key in ["whisper","wav2vec","seamless","cnn_lstm"]:
            txt = results[key][0]
            if reference.strip() and not txt.startswith("Error"):
                wers_val.append(compute_wer([reference],[txt]))
            else:
                wers_val.append(0)
        if reference.strip() and any(w > 0 for w in wers_val):
            fig, ax = plt.subplots(figsize=(8,3))
            colors = ["#2980b9","#e67e22","#16a085","#e74c3c"]
            ax.bar(names, wers_val, color=colors, alpha=0.85)
            for i,(n,v) in enumerate(zip(names,wers_val)):
                ax.text(i, v+0.01, f"{v:.3f}", ha="center", fontsize=10)
            ax.set_ylabel("WER (lower=better)")
            ax.set_title("Live Comparison — WER")
            ax.set_ylim(0,1.1); ax.grid(axis="y",alpha=0.3)
            plt.tight_layout()
            path = str(ROOT / "outputs" / "results" / "live_comparison.png")
            (ROOT / "outputs" / "results").mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
            plot = path
    except Exception:
        pass

    return (results["whisper"][0],  results["whisper"][1],
            results["wav2vec"][0],  results["wav2vec"][1],
            results["seamless"][0], results["seamless"][1],
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
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### SeamlessM4T-v2")
                        sm_out  = gr.Textbox(label="Transcript", lines=4, rtl=True)
                        sm_info = gr.Textbox(label="Info", lines=1, interactive=False)
                    with gr.Column():
                        gr.Markdown("#### CNN+LSTM (our model)")
                        cl_out  = gr.Textbox(label="Transcript", lines=4, rtl=True)
                        cl_info = gr.Textbox(label="Info", lines=1, interactive=False)
                cmp_plot = gr.Image(label="WER Comparison", type="filepath")

        run_btn.click(compare_all,
                      [audio_in, w_size, cnn_ckpt, reference],
                      [w_out, w_info, wv_out, wv_info, sm_out, sm_info, cl_out, cl_info, cmp_plot])


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
                model_radio = gr.Radio(["Whisper","Wav2Vec 2.0","SeamlessM4T-v2","CNN+LSTM (our model)"],
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
# TAB 4 — KEYWORD SPOTTING
# ══════════════════════════════════════════════════════════════════════════════

def run_keyword_spotting(audio, keywords_csv, whisper_size):
    wav, sr = _prep_audio(audio)
    if wav is None:
        return "Please upload or record audio.", ""
    try:
        from advanced.keyword_spotting import spot_keywords, format_hits_table
        text, secs, dev = _run_whisper(wav, whisper_size)
        keywords = [k.strip() for k in keywords_csv.split(",") if k.strip()]
        hits = spot_keywords(text, keywords or None)
        table = format_hits_table(hits)
        info = f"Transcribed in {secs}s on {dev} | {len(hits)} keyword hit(s)"
        return f"**Transcript:** {text}\n\n{table}", info
    except Exception as e:
        return f"Error: {e}", ""

def build_keyword_tab():
    with gr.Tab("4 · Keyword Spotting"):
        gr.Markdown("""
        ### Detect specific keywords in spoken Arabic.
        Provide a comma-separated list of keywords (Arabic or English).
        Audio → Whisper → keyword search with fuzzy matching.
        """)
        with gr.Row():
            with gr.Column(scale=1):
                kw_audio = gr.Audio(label="Audio", sources=["upload","microphone"], type="numpy")
                kw_list  = gr.Textbox(
                    label="Keywords (comma separated)",
                    value="طوارئ, موعد نهائي, امتحان, emergency, deadline, exam",
                    lines=2,
                    rtl=True,
                )
                kw_size  = gr.Dropdown(["tiny","base","small","medium"],
                                       value="small", label="Whisper size")
                kw_btn   = gr.Button("▶  Find Keywords", variant="primary")
            with gr.Column(scale=2):
                kw_out   = gr.Markdown(label="Results")
                kw_info  = gr.Textbox(label="Info", lines=1, interactive=False)

        kw_btn.click(run_keyword_spotting,
                     [kw_audio, kw_list, kw_size],
                     [kw_out, kw_info])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — SPEAKER IDENTIFICATION
# ══════════════════════════════════════════════════════════════════════════════

_speaker = {"id": None}

def _ensure_speaker():
    if _speaker["id"] is None:
        from advanced.speaker_id import SpeakerIdentifier
        _speaker["id"] = SpeakerIdentifier(device="auto")
    return _speaker["id"]

def enroll_speaker(audio, name):
    if not name or not name.strip():
        return "Enter a speaker name.", _list_speakers()
    wav, sr = _prep_audio(audio)
    if wav is None:
        return "Provide an audio clip to enroll.", _list_speakers()
    try:
        s = _ensure_speaker()
        s.enroll(name.strip(), wav, 16000)
        return f"Enrolled '{name.strip()}'.", _list_speakers()
    except Exception as e:
        return f"Error: {e}", _list_speakers()

def _list_speakers():
    if _speaker["id"] is None:
        return "_No speakers enrolled yet._"
    names = _speaker["id"].list_enrolled()
    return "Enrolled: " + ", ".join(names) if names else "_No speakers enrolled yet._"

def identify_speaker(audio, threshold):
    wav, sr = _prep_audio(audio)
    if wav is None:
        return "Provide an audio clip to identify.", "", None
    try:
        s = _ensure_speaker()
        spk, score = s.identify(wav, 16000, float(threshold))
        # Run a simple diarization too
        diar = s.diarize_simple(wav, 16000)
        rows = ["| Start (s) | End (s) | Speaker | Confidence |", "|---|---|---|---|"]
        for seg in diar:
            rows.append(f"| {seg['start']} | {seg['end']} | {seg['speaker']} | {seg['confidence']} |")
        return (f"**Identified speaker:** {spk}  (similarity: {score:.3f})",
                "\n".join(rows),
                None)
    except Exception as e:
        return f"Error: {e}", "", None

def clear_speakers():
    if _speaker["id"] is not None:
        _speaker["id"].clear()
    return "Cleared all enrolled speakers.", _list_speakers()

def build_speaker_tab():
    with gr.Tab("5 · Speaker ID"):
        gr.Markdown("""
        ### Identify speakers by voice (ECAPA-TDNN embeddings).
        1. **Enroll** — provide a clip per known speaker with their name.
        2. **Identify** — provide a new clip; we return the closest enrolled speaker.
        3. **Diarize** — sliding-window scan of the clip showing who speaks when.
        """)
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Enroll a known speaker")
                enroll_audio = gr.Audio(label="Voice sample", sources=["upload","microphone"], type="numpy")
                enroll_name  = gr.Textbox(label="Speaker name", placeholder="e.g. Ahmed")
                enroll_btn   = gr.Button("Enroll Speaker", variant="primary")
                enroll_log   = gr.Textbox(label="Status", interactive=False)
                enroll_list  = gr.Markdown(value=_list_speakers())
                clear_btn    = gr.Button("Clear all enrolled", variant="stop")
            with gr.Column():
                gr.Markdown("### Identify / Diarize")
                id_audio = gr.Audio(label="Audio to identify", sources=["upload","microphone"], type="numpy")
                threshold = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Match threshold")
                id_btn    = gr.Button("Identify", variant="primary")
                id_out    = gr.Markdown()
                diar_out  = gr.Markdown()

        enroll_btn.click(enroll_speaker, [enroll_audio, enroll_name], [enroll_log, enroll_list])
        clear_btn.click(clear_speakers, [], [enroll_log, enroll_list])
        id_btn.click(identify_speaker, [id_audio, threshold], [id_out, diar_out, gr.State()])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — EMOTION DETECTION
# ══════════════════════════════════════════════════════════════════════════════

_emotion_model = {"m": None}

def _ensure_emotion():
    if _emotion_model["m"] is None:
        from advanced.emotion import EmotionRecognizer
        _emotion_model["m"] = EmotionRecognizer(
            model_name=CONFIG["advanced"]["emotion"]["model_name"], device="auto"
        )
    return _emotion_model["m"]

def predict_emotion(audio):
    wav, sr = _prep_audio(audio)
    if wav is None:
        return "Provide an audio clip.", None
    try:
        from advanced.emotion import format_emotion_report
        m = _ensure_emotion()
        result = m.predict(wav, 16000)
        report = format_emotion_report(result)

        # Bar chart of class scores
        fig, ax = plt.subplots(figsize=(7, 3))
        scores = result["all_scores"]
        labels = list(scores.keys())
        vals   = list(scores.values())
        bars = ax.bar(labels, vals, color="#8e44ad", alpha=0.85)
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, v+0.01, f"{v*100:.0f}%", ha="center", fontsize=8)
        ax.set_ylabel("Confidence")
        ax.set_title(f"Emotion: {result['emotion']}")
        ax.set_ylim(0, 1.1); ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        out = str(ROOT / "outputs" / "results" / "emotion_scores.png")
        (ROOT / "outputs" / "results").mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
        return report, out
    except Exception as e:
        return f"Error: {e}", None

def build_emotion_tab():
    with gr.Tab("6 · Emotion"):
        gr.Markdown("""
        ### Detect emotional state in speech.
        Classes: angry, calm, disgust, fear, happy, neutral, sad, surprise.
        Backbone: Wav2Vec2 XLSR.
        """)
        with gr.Row():
            with gr.Column(scale=1):
                em_audio = gr.Audio(label="Audio", sources=["upload","microphone"], type="numpy")
                em_btn   = gr.Button("▶  Analyze Emotion", variant="primary")
            with gr.Column(scale=2):
                em_out   = gr.Markdown()
                em_plot  = gr.Image(label="Class Scores", type="filepath")
        em_btn.click(predict_emotion, [em_audio], [em_out, em_plot])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — SUMMARIZATION + SEARCH (Speech → Text → Summary → Search)
# ══════════════════════════════════════════════════════════════════════════════

_pipeline = {"summarizer": None, "search": None}

def _ensure_summarizer():
    if _pipeline["summarizer"] is None:
        from advanced.summarize_search import ArabicSummarizer
        _pipeline["summarizer"] = ArabicSummarizer(
            model_name=CONFIG["advanced"]["summarization"]["model_name"], device="auto"
        )
    return _pipeline["summarizer"]

def _ensure_search():
    if _pipeline["search"] is None:
        from advanced.summarize_search import TranscriptSearch
        _pipeline["search"] = TranscriptSearch(
            embedding_model=CONFIG["advanced"]["search"]["embedding_model"], device="auto"
        )
    return _pipeline["search"]

def transcribe_and_summarize(audio, whisper_size, do_index, audio_id):
    wav, sr = _prep_audio(audio)
    if wav is None:
        return "Provide an audio clip.", "", "_Index size: 0_"
    try:
        text, secs, dev = _run_whisper(wav, whisper_size)
        summ = _ensure_summarizer()
        summary = summ.summarize(text, max_length=CONFIG["advanced"]["summarization"]["max_length"],
                                  min_length=CONFIG["advanced"]["summarization"]["min_length"])
        idx_info = "_Indexing disabled._"
        if do_index:
            idx = _ensure_search()
            idx.add(audio_id or f"audio_{idx.size()+1}", text, {"summary": summary})
            idx_info = f"_Indexed. Total entries: {idx.size()}_"
        return f"**Transcript:** {text}", f"**Summary:** {summary}", idx_info
    except Exception as e:
        return f"Error: {e}", "", "_Index size: ?_"

def search_corpus(query, top_k):
    try:
        idx = _ensure_search()
        if idx.size() == 0:
            return "_Index is empty — transcribe and index audio first._"
        hits = idx.query(query, int(top_k))
        if not hits:
            return "_No matches._"
        rows = ["| Rank | Score | ID | Summary | Transcript snippet |", "|---|---|---|---|---|"]
        for i, h in enumerate(hits, 1):
            snippet = h["text"][:80].replace("\n", " ")
            summary = h.get("metadata", {}).get("summary", "")
            rows.append(f"| {i} | {h['score']:.3f} | {h['id']} | {summary} | {snippet} |")
        return "\n".join(rows)
    except Exception as e:
        return f"Error: {e}"

def build_pipeline_tab():
    with gr.Tab("7 · Summarize + Search"):
        gr.Markdown("""
        ### Speech → Text → Summary → Search
        1. Upload audio → Whisper transcribes it.
        2. mT5-XLSum summarizes the transcript (Arabic-aware).
        3. Optionally add to a search index.
        4. Query the index with Arabic or English text.
        """)
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 1. Transcribe & Summarize")
                ps_audio = gr.Audio(label="Audio", sources=["upload","microphone"], type="numpy")
                ps_size  = gr.Dropdown(["tiny","base","small","medium"],
                                       value="small", label="Whisper size")
                ps_index = gr.Checkbox(label="Add to search index", value=True)
                ps_id    = gr.Textbox(label="Optional ID for this clip",
                                      placeholder="meeting_2024_07_15")
                ps_btn   = gr.Button("▶  Process", variant="primary")
                ps_out_t = gr.Markdown()
                ps_out_s = gr.Markdown()
                ps_idx   = gr.Markdown(value="_Index size: 0_")
            with gr.Column():
                gr.Markdown("### 2. Search the indexed corpus")
                q     = gr.Textbox(label="Query", placeholder="اكتب استعلامك هنا", rtl=True)
                topk  = gr.Slider(1, 10, value=5, step=1, label="Top K results")
                q_btn = gr.Button("▶  Search", variant="primary")
                q_out = gr.Markdown()

        ps_btn.click(transcribe_and_summarize,
                     [ps_audio, ps_size, ps_index, ps_id],
                     [ps_out_t, ps_out_s, ps_idx])
        q_btn.click(search_corpus, [q, topk], [q_out])


# ══════════════════════════════════════════════════════════════════════════════
# ASSEMBLE
# ══════════════════════════════════════════════════════════════════════════════

def build_app():
    ckpt_exists = Path(DEFAULT_CKPT).exists()
    with gr.Blocks(title="Arabic ASR — Demo") as app:
        gr.Markdown("""
        <div style='text-align:center;padding:8px 0'>
        <h1 style='margin:0'>Arabic Speech Recognition — Demo</h1>
        <p style='color:#666'>Whisper · Wav2Vec 2.0 · SeamlessM4T-v2 · CNN+LSTM (custom)</p>
        <p style='color:#666'>+ Keyword Spotting · Speaker ID · Emotion · Summarize + Search</p>
        <p style='color:#888;font-size:0.85rem'>Device: <b>{}</b> &nbsp;|&nbsp; CNN+LSTM checkpoint: <b>{}</b></p>
        </div>
        """.format(DEVICE, "found ✓" if ckpt_exists else "not found — train first"))
        build_transcribe_tab()
        build_compare_tab()
        build_batch_tab()
        build_keyword_tab()
        build_speaker_tab()
        build_emotion_tab()
        build_pipeline_tab()
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
