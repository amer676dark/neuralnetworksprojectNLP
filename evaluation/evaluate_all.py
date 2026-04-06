"""
Comprehensive evaluation script — runs all three models on the test set
and produces a comparison report.

Usage:
    python evaluation/evaluate_all.py --config configs/config.yaml
    python evaluation/evaluate_all.py --config configs/config.yaml --model whisper
    python evaluation/evaluate_all.py --config configs/config.yaml --model wav2vec
    python evaluation/evaluate_all.py --config configs/config.yaml --model cnn_lstm --checkpoint outputs/checkpoints/cnn_lstm/best_model.pt
"""

import os
import sys
import json
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.metrics import compute_batch_metrics, format_metrics_report
from utils.visualization import plot_model_comparison, plot_confusion_matrix_wer


def evaluate_whisper(config: dict, test_samples: list) -> dict:
    from models.whisper_asr import WhisperASR

    model = WhisperASR(
        model_size=config["whisper"]["model_size"],
        language=config["whisper"]["language"],
        device=config["whisper"]["device"],
    )

    references = [s["sentence"] for s in test_samples]
    hypotheses = []

    print(f"Evaluating Whisper ({config['whisper']['model_size']}) on {len(test_samples)} samples...")
    for sample in tqdm(test_samples):
        audio = sample["audio"]["array"].astype(np.float32)
        hyp = model.transcribe_array(audio)
        hypotheses.append(hyp)

    metrics = compute_batch_metrics(references, hypotheses)
    metrics["model"] = f"Whisper-{config['whisper']['model_size']}"
    return metrics, references, hypotheses


def evaluate_wav2vec(config: dict, test_samples: list) -> dict:
    from models.wav2vec_asr import Wav2Vec2ASR

    model = Wav2Vec2ASR(
        model_name=config["wav2vec"]["model_name"],
        device=config["wav2vec"]["device"],
    )

    references = [s["sentence"] for s in test_samples]
    hypotheses = []

    print(f"Evaluating Wav2Vec2 on {len(test_samples)} samples...")
    for sample in tqdm(test_samples):
        audio = sample["audio"]["array"].astype(np.float32)
        hyp = model.transcribe_array(audio)
        hypotheses.append(hyp)

    metrics = compute_batch_metrics(references, hypotheses)
    metrics["model"] = "Wav2Vec2-XLSR-Arabic"
    return metrics, references, hypotheses


def evaluate_cnn_lstm(config: dict, checkpoint_path: str, test_samples: list) -> dict:
    import torch
    from models.cnn_lstm_asr import build_model
    from utils.audio_utils import extract_mel_spectrogram, normalize_audio, pad_or_trim

    state = torch.load(checkpoint_path, map_location="cpu")
    vocab = state["vocab"]
    idx2char = {v: k for k, v in vocab.items()}

    device_str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(device_str)

    model = build_model(vocab_size=len(vocab), config=config).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()

    audio_cfg = config["audio"]
    references = [s["sentence"] for s in test_samples]
    hypotheses = []

    print(f"Evaluating CNN+LSTM on {len(test_samples)} samples...")
    for sample in tqdm(test_samples):
        audio = sample["audio"]["array"].astype(np.float32)
        audio = normalize_audio(audio)
        audio = pad_or_trim(audio, audio_cfg["max_audio_length"])

        mel = extract_mel_spectrogram(
            audio,
            n_mels=audio_cfg["n_mels"],
            n_fft=audio_cfg["n_fft"],
            hop_length=audio_cfg["hop_length"],
            win_length=audio_cfg["win_length"],
        )
        mel_t = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        mel_t = (mel_t - mel_t.mean()) / (mel_t.std() + 1e-8)

        with torch.no_grad():
            log_probs = model(mel_t)

        ids = model.greedy_decode(log_probs)[0]
        hyp = "".join(idx2char.get(i, "") for i in ids if i not in {0, 2, 3})
        hypotheses.append(hyp)

    metrics = compute_batch_metrics(references, hypotheses)
    metrics["model"] = "CNN+LSTM"
    return metrics, references, hypotheses


def main():
    parser = argparse.ArgumentParser(description="Evaluate Arabic ASR models")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model", choices=["whisper", "wav2vec", "cnn_lstm", "all"], default="all")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/cnn_lstm/best_model.pt")
    parser.add_argument("--num_samples", type=int, default=200)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load test data
    print("Loading test dataset...")
    from data.dataset import load_common_voice_arabic
    test_ds = load_common_voice_arabic("test", max_samples=args.num_samples)
    test_samples = list(test_ds)

    results = {}
    model_names, wer_scores, cer_scores = [], [], []

    if args.model in ("whisper", "all"):
        m, refs, hyps = evaluate_whisper(config, test_samples)
        results["whisper"] = m
        model_names.append(m["model"])
        wer_scores.append(m["wer"])
        cer_scores.append(m["cer"])
        print(format_metrics_report(m))

        # Save predictions
        _save_predictions(refs, hyps, "outputs/results/whisper_predictions.json")

    if args.model in ("wav2vec", "all"):
        m, refs, hyps = evaluate_wav2vec(config, test_samples)
        results["wav2vec"] = m
        model_names.append(m["model"])
        wer_scores.append(m["wer"])
        cer_scores.append(m["cer"])
        print(format_metrics_report(m))

        _save_predictions(refs, hyps, "outputs/results/wav2vec_predictions.json")

    if args.model in ("cnn_lstm", "all") and Path(args.checkpoint).exists():
        m, refs, hyps = evaluate_cnn_lstm(config, args.checkpoint, test_samples)
        results["cnn_lstm"] = m
        model_names.append(m["model"])
        wer_scores.append(m["wer"])
        cer_scores.append(m["cer"])
        print(format_metrics_report(m))

        _save_predictions(refs, hyps, "outputs/results/cnn_lstm_predictions.json")
        plot_confusion_matrix_wer(refs, hyps, save_path="outputs/results/cnn_lstm_wer_dist.png")

    # Save full results
    results_path = "outputs/results/all_models_evaluation.json"
    Path("outputs/results").mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nAll results saved to {results_path}")

    # Comparison plot
    if len(model_names) > 1:
        plot_model_comparison(
            model_names, wer_scores, cer_scores,
            save_path="outputs/results/model_comparison.png",
        )


def _save_predictions(references, hypotheses, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = [{"reference": r, "hypothesis": h} for r, h in zip(references, hypotheses)]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
