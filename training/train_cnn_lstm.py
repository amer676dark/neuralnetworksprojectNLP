"""
Training script for CNN+LSTM Arabic ASR model with CTC loss.

Usage:
    python training/train_cnn_lstm.py --config configs/config.yaml
    python training/train_cnn_lstm.py --config configs/config.yaml --resume outputs/checkpoints/cnn_lstm/checkpoint_epoch10.pt
"""

import os
import sys
import json
import argparse
import matplotlib
matplotlib.use("Agg")  # headless — must be set before any pyplot import
import yaml
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import get_dataloaders
from models.cnn_lstm_asr import build_model
from utils.metrics import compute_wer
from utils.visualization import plot_training_curves


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(
    model,
    optimizer,
    scheduler,
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
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "val_loss": val_loss,
        "val_wer": val_wer,
        "vocab": vocab,
    }
    path = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch:03d}.pt")
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pt")
        torch.save(state, best_path)
        print(f"  -> New best model saved (WER: {val_wer:.4f})")


def load_checkpoint(path: str, model, optimizer=None, scheduler=None):
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model_state"])
    if optimizer and state.get("optimizer_state"):
        optimizer.load_state_dict(state["optimizer_state"])
    if scheduler and state.get("scheduler_state"):
        scheduler.load_state_dict(state["scheduler_state"])
    return state["epoch"], state.get("vocab", {})


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    device,
    grad_clip: float = 5.0,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc="  Train", leave=False)
    for batch in pbar:
        mel = batch["mel"].to(device)
        tokens = batch["tokens"].to(device)
        input_lengths = batch["input_lengths"].to(device)
        target_lengths = batch["target_lengths"].to(device)

        optimizer.zero_grad()

        log_probs = model(mel)  # (B, T, V)

        # CTC requires input_lengths in time-steps (T after CNN)
        # T_out = T_in (no time reduction in our CNN)
        ctc_input_lengths = torch.full(
            (mel.shape[0],), log_probs.shape[1], dtype=torch.long, device=device
        )

        loss = model.ctc_loss(log_probs, tokens, ctc_input_lengths, target_lengths)

        if not torch.isfinite(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device, vocab, max_batches: int = 50) -> tuple:
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_refs, all_hyps = [], []
    idx2char = {v: k for k, v in vocab.items()}

    for i, batch in enumerate(tqdm(loader, desc="  Eval ", leave=False)):
        if i >= max_batches:
            break

        mel = batch["mel"].to(device)
        tokens = batch["tokens"].to(device)
        target_lengths = batch["target_lengths"].to(device)
        transcripts = batch["transcripts"]

        log_probs = model(mel)
        ctc_input_lengths = torch.full(
            (mel.shape[0],), log_probs.shape[1], dtype=torch.long, device=device
        )
        loss = model.ctc_loss(log_probs, tokens, ctc_input_lengths, target_lengths)

        if torch.isfinite(loss):
            total_loss += loss.item()
            num_batches += 1

        # Greedy decode
        decoded_ids = model.greedy_decode(log_probs)
        for ids, ref in zip(decoded_ids, transcripts):
            hyp = "".join(idx2char.get(i, "") for i in ids if i not in {0, 2, 3})
            all_refs.append(ref)
            all_hyps.append(hyp)

    avg_loss = total_loss / max(num_batches, 1)
    wer = compute_wer(all_refs, all_hyps) if all_refs else 1.0
    return avg_loss, wer


def train(config_path: str, resume: str = None) -> None:
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = get_device()
    print(f"Training on device: {device}")

    # Data
    train_loader, val_loader, test_loader, vocab = get_dataloaders(config)
    print(f"Vocab size: {len(vocab)}")

    # Model
    model = build_model(vocab_size=len(vocab), config=config).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    # Optimizer & scheduler
    cfg = config["cnn_lstm"]
    optimizer = optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=1e-5)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=cfg["learning_rate"],
        steps_per_epoch=len(train_loader),
        epochs=cfg["num_epochs"],
        pct_start=0.1,
    )

    start_epoch = 1
    if resume:
        print(f"Resuming from {resume}")
        start_epoch, _ = load_checkpoint(resume, model, optimizer, scheduler)
        start_epoch += 1

    checkpoint_dir = cfg["checkpoint_dir"]
    best_wer = float("inf")
    train_losses, val_losses = [], []
    train_wers, val_wers = [], []

    print(f"\nStarting training for {cfg['num_epochs']} epochs...")
    for epoch in range(start_epoch, cfg["num_epochs"] + 1):
        print(f"\nEpoch [{epoch}/{cfg['num_epochs']}]")

        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, cfg["gradient_clip"])
        val_loss, val_wer = evaluate(model, val_loader, device, vocab)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_wers.append(val_wer)

        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val WER: {val_wer:.4f}")

        is_best = val_wer < best_wer
        if is_best:
            best_wer = val_wer

        save_checkpoint(
            model, optimizer, scheduler,
            epoch, val_loss, val_wer, vocab,
            checkpoint_dir, is_best,
        )

    # Final test evaluation
    print("\nRunning final test evaluation...")
    test_loss, test_wer = evaluate(model, test_loader, device, vocab, max_batches=200)
    print(f"  Test Loss: {test_loss:.4f} | Test WER: {test_wer:.4f}")

    # Save results
    results = {
        "test_loss": test_loss,
        "test_wer": test_wer,
        "best_val_wer": best_wer,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_wers": val_wers,
    }
    results_path = os.path.join(config["evaluation"]["results_dir"], "cnn_lstm_results.json")
    Path(config["evaluation"]["results_dir"]).mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    # Training curve
    plot_training_curves(
        train_losses, val_losses, val_wers=val_wers,
        save_path=os.path.join(config["evaluation"]["results_dir"], "training_curves.png"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN+LSTM Arabic ASR")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    train(args.config, args.resume)
