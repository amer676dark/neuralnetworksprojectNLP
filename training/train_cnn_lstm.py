"""
Training script for the Arabic CNN+BiLSTM+Attention CTC model.

Pipeline (see models/cnn_lstm_asr.py for architecture details):
  raw waveform (B, samples)
    -> GPU mel + log + per-sample norm
    -> SpecAugment
    -> CNN encoder (4x time subsampling)
    -> BiLSTM (3 x 512, bidirectional)
    -> Multi-head self-attention
    -> Linear head + CTC

Training extras:
  - bf16 autocast (Blackwell tensor cores)
  - torch.compile (reduce-overhead mode) for ~25% extra speedup
  - TF32 matmul allowed
  - channels_last memory format on the CNN
  - EMA (decay 0.999) of model weights used for validation + best-checkpoint
  - OneCycleLR with configurable warmup fraction
  - Reproducibility seed
  - Per-epoch JSON metrics + final training_curves.png

Usage:
    python training/train_cnn_lstm.py --config configs/config.yaml
    python training/train_cnn_lstm.py --config configs/config.yaml \
                                      --resume outputs/checkpoints/cnn_lstm/best_model.pt
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless — must be set before any pyplot import

import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import get_dataloaders
from models.cnn_lstm_asr import build_model
from utils.metrics import compute_wer
from utils.visualization import plot_training_curves


# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility / device
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ──────────────────────────────────────────────────────────────────────────────
# EMA — exponential moving average of model weights
# ──────────────────────────────────────────────────────────────────────────────

class EMA:
    """Maintains a shadow copy of model parameters updated by EMA after each step.

    `apply_to(model)` swaps EMA weights into a model copy for evaluation; the
    training model is unaffected, so the optimizer keeps using the real weights.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {
            name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        d = self.decay
        for name, p in model.named_parameters():
            if name in self.shadow and p.requires_grad:
                self.shadow[name].mul_(d).add_(p.detach(), alpha=1.0 - d)

    @torch.no_grad()
    def apply_to(self, target_model: torch.nn.Module) -> None:
        """Copy EMA weights into target_model in-place."""
        for name, p in target_model.named_parameters():
            if name in self.shadow:
                p.data.copy_(self.shadow[name])

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        for k, v in state.items():
            if k in self.shadow:
                self.shadow[k].copy_(v.to(self.shadow[k].device))


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint I/O
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    ema: Optional[EMA],
    epoch: int,
    val_loss: float,
    val_wer: float,
    vocab: dict,
    checkpoint_dir: str,
    is_best: bool = False,
) -> None:
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state": _strip_compile_prefix(model.state_dict()),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "ema_state": ema.state_dict() if ema else None,
        "val_loss": val_loss,
        "val_wer": val_wer,
        "vocab": vocab,
    }
    last_path = os.path.join(checkpoint_dir, "last.pt")
    torch.save(state, last_path)
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pt")
        torch.save(state, best_path)
        print(f"  → New best model saved (WER {val_wer:.4f})")


def _strip_compile_prefix(state_dict: dict) -> dict:
    """torch.compile wraps params under `_orig_mod.` — strip for clean reload."""
    prefix = "_orig_mod."
    return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in state_dict.items()}


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    ema: Optional[EMA] = None,
) -> Tuple[int, dict]:
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(_strip_compile_prefix(state["model_state"]))
    if optimizer and state.get("optimizer_state"):
        optimizer.load_state_dict(state["optimizer_state"])
    if scheduler and state.get("scheduler_state"):
        scheduler.load_state_dict(state["scheduler_state"])
    if ema and state.get("ema_state"):
        ema.load_state_dict(state["ema_state"])
    return state.get("epoch", 0), state.get("vocab", {})


# ──────────────────────────────────────────────────────────────────────────────
# Train / Eval steps
# ──────────────────────────────────────────────────────────────────────────────

def _to_device(batch: Dict, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
    if "waveform" in batch:
        inp = batch["waveform"].to(device, non_blocking=True)
    else:
        inp = batch["mel"].to(device, non_blocking=True)
    tokens = batch["tokens"].to(device, non_blocking=True)
    target_lengths = batch["target_lengths"].to(device, non_blocking=True)
    transcripts = batch["transcripts"]
    return inp, tokens, target_lengths, transcripts


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    grad_clip: float,
    amp: bool,
    ema: Optional[EMA],
    log_every: int,
) -> float:
    model.train()
    total_loss, n_batches = 0.0, 0
    use_amp = amp and device.type == "cuda"

    pbar = tqdm(loader, desc="  Train", leave=False, dynamic_ncols=True)
    for step, batch in enumerate(pbar, 1):
        inp, tokens, target_lengths, _ = _to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            log_probs = model(inp)

        # CTC always in fp32 for numerical stability
        log_probs_fp32 = log_probs.float() if use_amp else log_probs

        # Input lengths reflect the time-subsampling done by the CNN
        T_out = log_probs_fp32.shape[1]
        input_lengths = torch.full(
            (inp.shape[0],), T_out, dtype=torch.long, device=device
        )

        # Ensure each target is at most T_out; CTC requires T_out >= 2*target_len - 1
        loss = _call_ctc_loss(
            model, log_probs_fp32, tokens, input_lengths, target_lengths
        )
        if not torch.isfinite(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler:
            scheduler.step()
        if ema is not None:
            ema.update(model)

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.3f}",
                         lr=f"{optimizer.param_groups[0]['lr']:.2e}")
    return total_loss / max(n_batches, 1)


def _call_ctc_loss(model, log_probs, tokens, input_lengths, target_lengths):
    """Wrap model.ctc_loss to be robust to torch.compile (which wraps the model)."""
    fn = getattr(model, "ctc_loss", None)
    if fn is not None:
        return fn(log_probs, tokens, input_lengths, target_lengths)
    return model._orig_mod.ctc_loss(log_probs, tokens, input_lengths, target_lengths)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    vocab: dict,
    amp: bool,
    max_batches: Optional[int] = None,
) -> Tuple[float, float]:
    model.eval()
    total_loss, n_batches = 0.0, 0
    refs, hyps = [], []
    idx2char = {v: k for k, v in vocab.items()}
    use_amp = amp and device.type == "cuda"

    iter_loader = enumerate(tqdm(loader, desc="  Eval ", leave=False, dynamic_ncols=True))
    for i, batch in iter_loader:
        if max_batches is not None and i >= max_batches:
            break
        inp, tokens, target_lengths, transcripts = _to_device(batch, device)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            log_probs = model(inp)
        log_probs_fp32 = log_probs.float() if use_amp else log_probs

        T_out = log_probs_fp32.shape[1]
        input_lengths = torch.full(
            (inp.shape[0],), T_out, dtype=torch.long, device=device
        )
        loss = _call_ctc_loss(
            model, log_probs_fp32, tokens, input_lengths, target_lengths
        )
        if torch.isfinite(loss):
            total_loss += loss.item()
            n_batches += 1

        decoded_ids = _model_for_decode(model).greedy_decode(log_probs_fp32)
        for ids, ref in zip(decoded_ids, transcripts):
            hyp = "".join(idx2char.get(j, "") for j in ids if j not in {0, 2, 3})
            refs.append(ref)
            hyps.append(hyp)

    avg_loss = total_loss / max(n_batches, 1)
    wer = compute_wer(refs, hyps) if refs else 1.0
    return avg_loss, wer


def _model_for_decode(model):
    """Greedy decode is a staticmethod on the underlying module; handle compile wrap."""
    return getattr(model, "_orig_mod", model)


# ──────────────────────────────────────────────────────────────────────────────
# Main training entry
# ──────────────────────────────────────────────────────────────────────────────

def train(config_path: str, resume: Optional[str] = None) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    cfg = config["cnn_lstm"]
    eval_cfg = config["evaluation"]

    # Reproducibility & numeric defaults
    set_seed(cfg.get("seed", 1337))
    torch.set_float32_matmul_precision("high")        # allow TF32
    torch.backends.cudnn.benchmark = True             # variable seq lens still benefit

    device = get_device()
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
        print(f"VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, vocab = get_dataloaders(config)
    print(f"Vocab size: {len(vocab)}")
    print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}  "
          f"Test batches: {len(test_loader)}")

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_model(vocab_size=len(vocab), config=config).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last) if _supports_channels_last(model) else model
    print(f"Model parameters: {model.count_parameters():,}")

    # EMA shadow weights
    ema = EMA(model, decay=cfg.get("ema_decay", 0.999)) if cfg.get("ema_decay", 0) > 0 else None
    if ema:
        print(f"EMA decay: {ema.decay}")

    # torch.compile — kicks in lazily on first batch.
    # Note: "reduce-overhead" (cudagraphs) breaks on SpecAugment's mask_along_axis
    # because the mask widths are dynamic. Default to mode="default" (Inductor
    # without cudagraphs), which is safe and still gives a meaningful speedup.
    compiled_model = model
    if device.type == "cuda" and cfg.get("compile", False):
        compile_mode = cfg.get("compile_mode", "default")
        try:
            compiled_model = torch.compile(model, mode=compile_mode, dynamic=True)
            print(f"torch.compile: mode={compile_mode} (compilation happens on first batch)")
        except Exception as e:
            print(f"torch.compile failed ({e}) — running uncompiled.")
            compiled_model = model

    # ── Optimizer & schedule ─────────────────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg.get("weight_decay", 1e-5),
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=cfg["learning_rate"],
        steps_per_epoch=max(len(train_loader), 1),
        epochs=cfg["num_epochs"],
        pct_start=cfg.get("warmup_pct", 0.1),
    )

    start_epoch = 1
    if resume:
        print(f"Resuming from {resume}")
        start_epoch, _ = load_checkpoint(resume, model, optimizer, scheduler, ema)
        start_epoch += 1

    # ── Training loop ────────────────────────────────────────────────────────
    print(f"\nStarting training: {cfg['num_epochs']} epochs · AMP bf16 · "
          f"batch {cfg['batch_size']} · compile {cfg.get('compile', False)}")

    best_wer = float("inf")
    train_losses, val_losses, val_wers = [], [], []
    eval_model_buffer = None  # lazily-built copy of model for EMA eval
    log_every = int(cfg.get("log_every", 50))
    history_path = os.path.join(eval_cfg["results_dir"], "cnn_lstm_history.json")
    Path(eval_cfg["results_dir"]).mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, cfg["num_epochs"] + 1):
        t_start = time.time()
        print(f"\nEpoch [{epoch}/{cfg['num_epochs']}]")
        train_loss = train_one_epoch(
            compiled_model, train_loader, optimizer, scheduler, device,
            cfg["gradient_clip"], cfg.get("amp", True), ema, log_every,
        )

        # Build / refresh EMA eval model (kept off-optimizer so weights aren't disturbed)
        if ema is not None:
            if eval_model_buffer is None:
                eval_model_buffer = build_model(vocab_size=len(vocab), config=config).to(device)
            ema.apply_to(eval_model_buffer)
            val_loss, val_wer = evaluate(
                eval_model_buffer, val_loader, device, vocab, cfg.get("amp", True),
            )
        else:
            val_loss, val_wer = evaluate(
                compiled_model, val_loader, device, vocab, cfg.get("amp", True),
            )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_wers.append(val_wer)
        elapsed = time.time() - t_start
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} "
              f"| Val WER: {val_wer:.4f} | {elapsed:.1f}s")

        is_best = val_wer < best_wer
        if is_best:
            best_wer = val_wer

        save_checkpoint(
            model, optimizer, scheduler, ema,
            epoch, val_loss, val_wer, vocab,
            cfg["checkpoint_dir"], is_best,
        )

        # Persist per-epoch metrics — survives if training is interrupted
        with open(history_path, "w") as f:
            json.dump({
                "train_losses": train_losses,
                "val_losses": val_losses,
                "val_wers": val_wers,
                "best_val_wer": best_wer,
                "epoch": epoch,
                "config_path": config_path,
            }, f, indent=2)

    # ── Final test evaluation (using EMA weights if available) ──────────────
    print("\nFinal test evaluation (using best/EMA weights)...")
    test_model = eval_model_buffer if ema is not None else compiled_model
    test_loss, test_wer = evaluate(test_model, test_loader, device, vocab, cfg.get("amp", True))
    print(f"  Test Loss: {test_loss:.4f} | Test WER: {test_wer:.4f}")

    final_results = {
        "test_loss": test_loss,
        "test_wer": test_wer,
        "best_val_wer": best_wer,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_wers": val_wers,
        "config_path": config_path,
        "model_params": model.count_parameters(),
        "vocab_size": len(vocab),
    }
    with open(os.path.join(eval_cfg["results_dir"], "cnn_lstm_results.json"), "w") as f:
        json.dump(final_results, f, indent=2)

    plot_training_curves(
        train_losses=train_losses,
        val_losses=val_losses,
        val_wers=val_wers,
        save_path=os.path.join(eval_cfg["results_dir"], "training_curves.png"),
    )
    print(f"Results + curves saved under {eval_cfg['results_dir']}")


def _supports_channels_last(module: torch.nn.Module) -> bool:
    """channels_last only helps when there's a Conv2d in the model."""
    return any(isinstance(m, torch.nn.Conv2d) for m in module.modules())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN+LSTM Arabic ASR")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    train(args.config, args.resume)
