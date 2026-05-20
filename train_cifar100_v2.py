"""CIFAR-100 training harness for MPKx v2 (The Hidden Supervisor).

GPU-side upsampling from 32x32 to 224x224 -- no disk cache needed.
Optimized for RTX 3090 (24GB VRAM).

Baseline: MPKx v1 -> 46.5% CIFAR-100 val accuracy, 874K params
"""

from __future__ import annotations

import math
import os
import pickle
import sys
import tarfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from MPKx_v2 import MPKx_v2

IMAGE_SIZE = 224
NUM_CLASSES = 100
EVAL_BATCH_SIZE = 256
CACHE_DIR = Path(os.path.expanduser("~")) / ".cache" / "autoresearch_vision" / "cifar100"
ARCHIVE = CACHE_DIR / "cifar-100-python.tar.gz"
DATA_DIR = CACHE_DIR / "cifar-100-python"
URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
TENSOR_CACHE = CACHE_DIR / "tensor_cache"
TRAIN_TENSORS = TENSOR_CACHE / "train.pt"
VAL_TENSORS = TENSOR_CACHE / "val.pt"


@dataclass
class TrainConfig:
    batch_size: int = 1024  # maximize VRAM usage
    channels: int = 48  # ~831K params
    context_ch: int = 32
    lr: float = 0.008  # scaled for batch_size=1024
    weight_decay: float = 0.01
    epochs: int = 100
    warmup_ratio: float = 0.05
    final_lr_frac: float = 0.01
    num_workers: int = 4  # parallel CPU workers
    dropout: float = 0.2
    label_smoothing: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    eval_every_epochs: int = 1


CFG = TrainConfig()


def download_cifar100() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if ARCHIVE.exists():
        return
    print("Downloading CIFAR-100...")
    urllib.request.urlretrieve(URL, str(ARCHIVE))


def _load_pickle(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f, encoding="latin1")


def _to_tensor(raw: dict) -> tuple[torch.Tensor, torch.Tensor]:
    data = raw["data"].reshape(-1, 3, 32, 32).astype(np.uint8)
    labels = np.asarray(raw["fine_labels"], dtype=np.int64)
    return torch.from_numpy(data).contiguous(), torch.from_numpy(labels).long()


def load_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    download_cifar100()
    TENSOR_CACHE.mkdir(parents=True, exist_ok=True)

    if TRAIN_TENSORS.exists() and VAL_TENSORS.exists():
        x_train, y_train = torch.load(TRAIN_TENSORS, weights_only=True)
        x_val, y_val = torch.load(VAL_TENSORS, weights_only=True)
        return x_train, y_train, x_val, y_val

    if not DATA_DIR.exists():
        print("Extracting CIFAR-100...")
        with tarfile.open(ARCHIVE, "r:gz") as tf:
            tf.extractall(CACHE_DIR)

    train_raw = _load_pickle(DATA_DIR / "train")
    test_raw = _load_pickle(DATA_DIR / "test")

    x_train, y_train = _to_tensor(train_raw)
    x_val, y_val = _to_tensor(test_raw)

    torch.save((x_train, y_train), TRAIN_TENSORS)
    torch.save((x_val, y_val), VAL_TENSORS)

    return x_train, y_train, x_val, y_val


class CIFAR100_32(Dataset):
    """Returns 32x32 uint8 tensors -- upsampling happens on GPU."""
    def __init__(self, x_uint8: torch.Tensor, y: torch.Tensor):
        self.x = x_uint8
        self.y = y

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        # Return float32 normalized, still 32x32
        img = self.x[idx].float() / 255.0
        return img, int(self.y[idx])


def get_dataloaders():
    x_train, y_train, x_val, y_val = load_tensors()

    train_ds = CIFAR100_32(x_train, y_train)
    val_ds = CIFAR100_32(x_val, y_val)

    train_loader = DataLoader(
        train_ds, batch_size=CFG.batch_size, shuffle=True,
        num_workers=CFG.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False,
        num_workers=CFG.num_workers, pin_memory=True,
    )
    return train_loader, val_loader


@torch.no_grad()
def upsample_on_gpu(x: torch.Tensor) -> torch.Tensor:
    """Upsample [B,3,32,32] to [B,3,224,224] on GPU."""
    return F.interpolate(x, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.inference_mode()
def evaluate_acc(model: nn.Module, val_loader, device: str) -> dict:
    model.eval()
    total = correct = correct_top5 = 0
    for x, y in val_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        x = upsample_on_gpu(x)
        logits = model(x)
        _, pred = logits.max(1)
        total += y.size(0)
        correct += pred.eq(y).sum().item()
        _, top5 = logits.topk(5, dim=1)
        correct_top5 += top5.eq(y.unsqueeze(1)).any(dim=1).sum().item()
    return {"val_acc": correct / total, "val_top5": correct_top5 / total}


def cosine_schedule(progress: float) -> float:
    if progress < CFG.warmup_ratio:
        return progress / CFG.warmup_ratio
    t = (progress - CFG.warmup_ratio) / (1.0 - CFG.warmup_ratio)
    return CFG.final_lr_frac + 0.5 * (1.0 - CFG.final_lr_frac) * (1.0 + math.cos(math.pi * min(t, 1.0)))


def train_augment(x: torch.Tensor) -> torch.Tensor:
    """Horizontal flip (GPU-side)."""
    mask = torch.rand(x.size(0), device=x.device) < 0.5
    x[mask] = torch.flip(x[mask], [-1])
    return x


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    print(f"Device: {CFG.device}")
    train_loader, val_loader = get_dataloaders()

    print("Building MPKx v2...")
    model = MPKx_v2(
        num_classes=NUM_CLASSES,
        ch=CFG.channels,
        use_stereo=True,
        context_ch=CFG.context_ch,
        dropout=CFG.dropout,
    ).to(CFG.device)

    n_params = count_trainable_params(model)
    print(f"Parameters: {n_params:,}")

    # Sanity check
    dummy = torch.randn(2, 3, 32, 32, device=CFG.device)
    dummy = upsample_on_gpu(dummy)
    out = model(dummy)
    print(f"Forward pass OK: 32x32 -> 224x224 -> {out.shape}")

    criterion = nn.CrossEntropyLoss(label_smoothing=CFG.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scaler = torch.cuda.amp.GradScaler()
    best_val = 0.0
    best_epoch = 0
    total_steps = len(train_loader) * CFG.epochs
    step = 0
    results_log = []

    for epoch in range(1, CFG.epochs + 1):
        model.train()
        running_loss = running_correct = running_total = 0
        t0 = time.time()

        for x, y in train_loader:
            x, y = x.to(CFG.device, non_blocking=True), y.to(CFG.device, non_blocking=True)
            x = upsample_on_gpu(x)
            x = train_augment(x)

            lr_scale = cosine_schedule(step / total_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = CFG.lr * lr_scale

            optimizer.zero_grad(set_to_none=True)
            # Mixed precision forward pass
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * x.size(0)
            _, pred = logits.max(1)
            running_correct += pred.eq(y).sum().item()
            running_total += y.size(0)
            step += 1

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total
        epoch_time = time.time() - t0

        if epoch % CFG.eval_every_epochs == 0 or epoch == 1:
            metrics = evaluate_acc(model, val_loader, CFG.device)
            val_acc = metrics["val_acc"]
            val_top5 = metrics["val_top5"]

            if val_acc > best_val:
                best_val = val_acc
                best_epoch = epoch
                torch.save(model.state_dict(), "mpkx_v2_best.pt")

            current_lr = CFG.lr * cosine_schedule(step / total_steps)
            print(
                f"Ep {epoch:3d}/{CFG.epochs} | "
                f"loss={train_loss:.3f} train={train_acc:.4f} | "
                f"val={val_acc:.4f} top5={val_top5:.4f} | "
                f"best={best_val:.4f}@{best_epoch} | "
                f"lr={current_lr:.6f} | {epoch_time:.1f}s"
            )
            results_log.append({
                "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                "val_acc": val_acc, "val_top5": val_top5,
            })
        else:
            print(f"Ep {epoch:3d}/{CFG.epochs} | loss={train_loss:.3f} train={train_acc:.4f} | {epoch_time:.1f}s")

    # Summary
    print("\n" + "=" * 60)
    print("MPKx v2 CIFAR-100 Results")
    print("=" * 60)
    print(f"Parameters:          {n_params:,}")
    print(f"Best val accuracy:   {best_val:.4f} (epoch {best_epoch})")
    print(f"Baseline (MPKx v1):  0.4649 (46.49%)")
    print(f"Improvement:         {(best_val - 0.4649)*100:+.2f}%")
    print("=" * 60)

    with open("mpkx_v2_results.tsv", "w") as f:
        f.write("epoch\ttrain_loss\ttrain_acc\tval_acc\tval_top5\n")
        for r in results_log:
            f.write(f"{r['epoch']}\t{r['train_loss']:.4f}\t{r['train_acc']:.4f}\t{r['val_acc']:.4f}\t{r['val_top5']:.4f}\n")
    print("Results saved to mpkx_v2_results.tsv")


if __name__ == "__main__":
    main()
