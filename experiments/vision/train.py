"""
Train all 4 ViT combinations on CIFAR-10:
  standard-small, standard-medium, gated-small, gated-medium

Run from repo root:
    python -m experiments.vision.train [--config experiments/vision/config.yaml]
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.vit import CONFIGS, GatedViT, ViT

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def build_loaders(batch_size: int, num_workers: int = 4):
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    val_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    data_root = ROOT / "data"
    train_ds = torchvision.datasets.CIFAR10(data_root, train=True, download=True, transform=train_tf)
    val_ds = torchvision.datasets.CIFAR10(data_root, train=False, download=True, transform=val_tf)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# LR schedule: linear warmup + cosine decay
# ---------------------------------------------------------------------------

def cosine_schedule_with_warmup(optimizer, warmup_epochs: int, total_epochs: int):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Attention snapshot
# ---------------------------------------------------------------------------

def save_attn_snapshot(model, tag: str, run_id: str):
    weights = model.get_attn_weights()
    snapshot = {
        f"layer_{i}": w.cpu().tolist() if w is not None else None
        for i, w in enumerate(weights)
    }
    path = RESULTS_DIR / f"{run_id}_attn_{tag}.json"
    with open(path, "w") as f:
        json.dump(snapshot, f)


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss, n = 0.0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler is not None):
            logits = model(imgs)
            loss = criterion(logits, labels)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        n += imgs.size(0)
    return total_loss / n, correct / n


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_experiment(model_type: str, model_size: str, hparams: dict, device: torch.device):
    print(f"\n{'='*60}")
    print(f"  {model_type.upper()} | {model_size.upper()}")
    print(f"{'='*60}")

    cfg = CONFIGS[model_size].copy()
    model_cls = ViT if model_type == "standard" else GatedViT
    model = model_cls(cfg).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    hp = hparams[model_size]
    epochs = hp["epochs"]
    warmup = hp["warmup_epochs"]

    train_loader, val_loader = build_loaders(hp["batch_size"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"]
    )
    scheduler = cosine_schedule_with_warmup(optimizer, warmup, epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    run_id = f"{model_type}_{model_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    records = []
    mid_epoch = epochs // 2

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        record = {
            "experiment": run_id,
            "model_type": model_type,
            "model_size": model_size,
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_accuracy": round(val_acc, 6),
        }
        records.append(record)

        print(
            f"  Epoch {epoch:3d}/{epochs} | "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_acc={val_acc*100:.2f}%  lr={scheduler.get_last_lr()[0]:.2e}"
        )

        # Attention snapshots at epoch 1, mid, and final
        if epoch == 1:
            save_attn_snapshot(model, "epoch1", run_id)
        if epoch == mid_epoch:
            save_attn_snapshot(model, "mid", run_id)
        if epoch == epochs:
            save_attn_snapshot(model, "final", run_id)

    # Save per-epoch results
    results_path = RESULTS_DIR / f"{run_id}.json"
    with open(results_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"  Results saved to {results_path}")

    return records


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    parser.add_argument("--model_type", choices=["standard", "gated", "all"], default="all")
    parser.add_argument("--model_size", choices=["small", "medium", "all"], default="all")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    hparams = cfg["hparams"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_types = ["standard", "gated"] if args.model_type == "all" else [args.model_type]
    model_sizes = ["small", "medium"] if args.model_size == "all" else [args.model_size]

    all_results = []
    for mtype in model_types:
        for msize in model_sizes:
            results = run_experiment(mtype, msize, hparams, device)
            all_results.extend(results)

    summary_path = RESULTS_DIR / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull summary saved to {summary_path}")


if __name__ == "__main__":
    main()
