"""
Local smoke test — no dataset, no GPU required.

Verifies:
  - All project modules import without error
  - The CNN+LSTM model builds and runs forward + greedy decode on a dummy tensor
  - CTC loss computes without NaN
  - Audio utilities run on a synthetic waveform
  - Metrics (WER/CER) compute on Arabic strings
  - Whisper / Wav2Vec2 wrappers can be imported (does NOT download weights)

Run:
    python smoke_test.py

If this passes locally, you can confidently rent a GPU and run training.
"""

import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def section(name):
    print(f"\n=== {name} ===")


def main():
    ok = True

    section("1. Import project modules")
    try:
        from utils.audio_utils import (
            extract_mel_spectrogram, normalize_audio, pad_or_trim, apply_augmentation,
        )
        from utils.metrics import compute_wer, compute_cer, compute_batch_metrics
        from utils.visualization import plot_training_curves, plot_model_comparison
        from models.cnn_lstm_asr import build_model
        from data.dataset import load_dataset_by_name  # noqa: F401
        print("  All project modules imported.")
    except Exception as e:
        print(f"  IMPORT FAILED: {e}")
        traceback.print_exc()
        return 1

    section("2. Load config")
    cfg = yaml.safe_load(open(ROOT / "configs" / "config.yaml"))
    print(f"  audio.n_mels={cfg['audio']['n_mels']}  "
          f"lstm_hidden={cfg['cnn_lstm']['lstm_hidden_size']}")

    section("3. Audio utilities on synthetic waveform")
    sr = 16000
    wav = (np.random.randn(sr * 3) * 0.1).astype(np.float32)  # 3-second clip
    wav = normalize_audio(wav)
    wav = pad_or_trim(wav, cfg["audio"]["max_audio_length"])
    mel = extract_mel_spectrogram(
        wav,
        n_mels=cfg["audio"]["n_mels"],
        n_fft=cfg["audio"]["n_fft"],
        hop_length=cfg["audio"]["hop_length"],
        win_length=cfg["audio"]["win_length"],
    )
    print(f"  waveform: {wav.shape}  mel: {mel.shape}")
    assert mel.shape[0] == cfg["audio"]["n_mels"]

    section("4. Build CNN+LSTM model and run forward (CPU)")
    # Use a tiny override locally so the test doesn't allocate hundreds of MB
    light_cfg = json.loads(json.dumps(cfg))  # deep copy via JSON
    light_cfg["cnn_lstm"]["cnn_channels"] = [16, 32, 64]
    light_cfg["cnn_lstm"]["lstm_hidden_size"] = 64
    light_cfg["cnn_lstm"]["lstm_num_layers"] = 1
    light_cfg["cnn_lstm"]["attention_heads"] = 4
    light_cfg["cnn_lstm"]["spec_augment"] = False
    vocab_size = 60
    model = build_model(vocab_size, light_cfg).eval()
    n_params = model.count_parameters()
    print(f"  model built — {n_params:,} parameters (lightweight test variant)")

    # Forward with a short raw waveform — exercises GPU-mel code path on CPU
    wav_t = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)  # (1, samples)
    with torch.no_grad():
        log_probs = model(wav_t)
    print(f"  forward OK — output shape {tuple(log_probs.shape)}")

    section("5. CTC loss on dummy targets")
    # Compute valid input_lengths via the model (post-subsampling time axis)
    targets = torch.tensor([[5, 10, 15, 20, 25, 30]], dtype=torch.long)
    input_lengths = torch.tensor([log_probs.shape[1]], dtype=torch.long)
    target_lengths = torch.tensor([targets.shape[1]], dtype=torch.long)
    loss = model.ctc_loss(log_probs, targets, input_lengths, target_lengths)
    print(f"  CTC loss = {loss.item():.4f}  (finite: {torch.isfinite(loss).item()})")
    assert torch.isfinite(loss).item(), "CTC loss is not finite"

    section("6. Greedy decode")
    decoded = model.greedy_decode(log_probs)
    print(f"  greedy decoded length: {len(decoded[0])}")

    section("7. WER / CER on Arabic strings")
    refs = ["مرحبا بالعالم", "هذا اختبار للنظام"]
    hyps = ["مرحبا عالم",    "هذا اختبار النظام"]
    wer = compute_wer(refs, hyps)
    cer = compute_cer(refs, hyps)
    print(f"  WER={wer:.3f}  CER={cer:.3f}")

    section("8. Import ASR model wrappers (no model download)")
    try:
        from models.whisper_asr import WhisperASR  # noqa: F401
        from models.wav2vec_asr import Wav2Vec2ASR  # noqa: F401
        from models.seamless_m4t_asr import SeamlessM4TASR  # noqa: F401
        print("  Whisper + Wav2Vec2 + SeamlessM4T wrappers importable.")
    except Exception as e:
        print(f"  WRAPPER IMPORT FAILED: {e}")
        ok = False

    section("8b. Import advanced task modules")
    try:
        from advanced.keyword_spotting import spot_keywords, format_hits_table
        from advanced import speaker_id as _sid_mod  # noqa: F401 — speechbrain may be missing locally
        from advanced import emotion as _em_mod      # noqa: F401
        from advanced import summarize_search as _ss # noqa: F401
        print("  advanced.* modules importable (keyword_spotting / speaker_id / emotion / summarize_search).")
    except Exception as e:
        print(f"  ADVANCED MODULE IMPORT FAILED: {e}")
        ok = False

    section("8c. Keyword spotting on a synthetic Arabic transcript")
    try:
        transcript = "اجتماع طوارئ غدا قبل الامتحان النهائي وموعد نهائي للتقرير"
        hits = spot_keywords(transcript, keywords=["طوارئ", "امتحان", "موعد نهائي", "deadline"])
        print(f"  keyword hits: {[(h.keyword, h.confidence) for h in hits]}")
        assert len(hits) >= 2, "expected at least 2 hits"
    except Exception as e:
        print(f"  KEYWORD SPOTTING FAILED: {e}")
        ok = False

    section("8d. Kaggle arabic_tts dataset loader (path-only check, no audio)")
    try:
        from data.dataset import load_dataset_by_name  # noqa: F401
        print("  load_dataset_by_name imported (Kaggle loader available via name='kaggle_arabic_tts').")
    except Exception as e:
        print(f"  DATASET LOADER IMPORT FAILED: {e}")
        ok = False

    section("9. Visualization (headless save to /tmp)")
    try:
        import matplotlib
        matplotlib.use("Agg")
        out = Path("/tmp/smoke_training_curves.png")
        plot_training_curves(
            train_losses=[3.2, 2.5, 1.9, 1.5],
            val_losses=[3.4, 2.8, 2.2, 1.9],
            val_wers=[0.8, 0.6, 0.5, 0.45],
            save_path=str(out),
        )
        print(f"  plot saved to {out} (exists: {out.exists()})")
    except Exception as e:
        print(f"  PLOT FAILED: {e}")
        traceback.print_exc()
        ok = False

    print("\n" + ("=" * 50))
    print("SMOKE TEST PASSED — safe to rent a GPU." if ok else "SMOKE TEST FAILED — fix above before paying for GPU.")
    print("=" * 50)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
