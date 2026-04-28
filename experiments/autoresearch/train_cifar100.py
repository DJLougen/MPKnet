"""CIFAR-100 training harness for MPKx.

Runs the same architecture-first setup as the TinyImageNet MPK experiments:
no dropout, no mixup, no label smoothing, simple horizontal flip only, and
per-epoch train/validation curve logging.
"""

from __future__ import annotations

import math
import os
import pickle
import tarfile
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from model import MPKx

IMAGE_SIZE = 224
NUM_CLASSES = 100
PARAM_BUDGET = 1_500_000
EVAL_BATCH_SIZE = 128
CACHE_DIR = Path(os.path.expanduser("~")) / ".cache" / "autoresearch_vision" / "cifar100"
ARCHIVE = CACHE_DIR / "cifar-100-python.tar.gz"
DATA_DIR = CACHE_DIR / "cifar-100-python"
URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
TENSOR_CACHE = CACHE_DIR / "tensor_cache"
TRAIN_TENSORS = TENSOR_CACHE / "train.pt"
VAL_TENSORS = TENSOR_CACHE / "val.pt"


@dataclass
class TrainConfig:
    batch_size: int = 80
    channels: int = 56
    lr: float = 4e-3
    weight_decay: float = 1.0e-2
    warmup_ratio: float = 0.08
    final_lr_frac: float = 0.0
    label_smoothing: float = 0.0
    grad_clip: float = 1.0
    use_stereo: bool = True
    disparity_range: int = 2
    kernel_size: int = 5
    num_workers: int = 4
    mixup_alpha: float = 0.0
    epochs: int = 100
    eval_every_epochs: int = 1


CFG = TrainConfig()


def download_cifar100() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if DATA_DIR.exists():
        return
    if not ARCHIVE.exists():
        print(f"[cifar100] downloading {URL}", flush=True)
        urllib.request.urlretrieve(URL, ARCHIVE)
    print("[cifar100] extracting", flush=True)
    with tarfile.open(ARCHIVE, "r:gz") as tf:
        tf.extractall(CACHE_DIR)


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
        tr = torch.load(TRAIN_TENSORS, map_location="cpu", weights_only=True)
        va = torch.load(VAL_TENSORS, map_location="cpu", weights_only=True)
        return tr["x"], tr["y"], va["x"], va["y"]
    x_train, y_train = _to_tensor(_load_pickle(DATA_DIR / "train"))
    x_val, y_val = _to_tensor(_load_pickle(DATA_DIR / "test"))
    torch.save({"x": x_train, "y": y_train}, TRAIN_TENSORS)
    torch.save({"x": x_val, "y": y_val}, VAL_TENSORS)
    print(f"[cifar100] train {tuple(x_train.shape)} val {tuple(x_val.shape)}", flush=True)
    return x_train, y_train, x_val, y_val


class CIFAR100Upsampled(Dataset):
    def __init__(self, x_uint8: torch.Tensor, y: torch.Tensor):
        self.x = x_uint8
        self.y = y

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        img = self.x[idx].float() / 255.0
        img = F.interpolate(img.unsqueeze(0), size=IMAGE_SIZE, mode="bilinear", align_corners=False).squeeze(0)
        return img, int(self.y[idx])


def get_dataloaders(batch_size: int, num_workers: int):
    x_train, y_train, x_val, y_val = load_tensors()
    train_loader = DataLoader(
        CIFAR100Upsampled(x_train, y_train), batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        CIFAR100Upsampled(x_val, y_val), batch_size=EVAL_BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader


def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.inference_mode()
def evaluate_acc(model: torch.nn.Module, val_loader, device: str = "cuda") -> dict:
    model.eval()
    correct = 0
    correct5 = 0
    total = 0
    t0 = time.time()
    for x, y in val_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred1 = logits.argmax(dim=-1)
        correct += (pred1 == y).sum().item()
        top5 = logits.topk(5, dim=-1).indices
        correct5 += (top5 == y.unsqueeze(-1)).any(dim=-1).sum().item()
        total += y.numel()
    return {
        "val_acc": correct / max(total, 1),
        "val_top5": correct5 / max(total, 1),
        "n": total,
        "elapsed_s": time.time() - t0,
    }


def cosine_schedule(progress: float) -> float:
    if progress < CFG.warmup_ratio:
        return max(progress / max(CFG.warmup_ratio, 1e-8), 1e-3)
    t = (progress - CFG.warmup_ratio) / max(1.0 - CFG.warmup_ratio, 1e-8)
    return CFG.final_lr_frac + 0.5 * (1.0 - CFG.final_lr_frac) * (1.0 + math.cos(math.pi * min(t, 1.0)))


def train_augment(x: torch.Tensor) -> torch.Tensor:
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, [-1])
    return x


def main() -> None:
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    print(f"Config: {asdict(CFG)}", flush=True)
    print(f"Task: CIFAR-100 @ {IMAGE_SIZE}px", flush=True)

    train_loader, val_loader = get_dataloaders(CFG.batch_size, CFG.num_workers)
    model = MPKx(
        num_classes=NUM_CLASSES,
        ch=CFG.channels,
        use_stereo=CFG.use_stereo,
        disparity_range=CFG.disparity_range,
        kernel_size=CFG.kernel_size,
    ).to(device)

    n_params = count_trainable_params(model)
    if n_params > PARAM_BUDGET:
        raise RuntimeError(f"parameter budget exceeded: {n_params} > {PARAM_BUDGET}")
    print(f"Trainable params: {n_params:,} / budget {PARAM_BUDGET:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    step = 0
    total_seen = 0
    smooth_loss = 0.0
    total_steps = len(train_loader) * CFG.epochs
    gate_mode = getattr(model, "k_gate_mode", "facilitatory_sigmoid")
    curve_dir = Path("training_curves")
    curve_dir.mkdir(exist_ok=True)
    curve_path = curve_dir / f"cifar100_ch{CFG.channels}_batch{CFG.batch_size}_epochs{CFG.epochs}_{gate_mode}.tsv"
    curve_path.write_text("epoch\tstep\tseen\ttrain_loss\tsmooth_loss\tval_acc\tval_top5\tlr\n")
    last_metrics = None

    for epoch in range(CFG.epochs):
        model.train()
        epoch_loss_sum = 0.0
        epoch_loss_n = 0
        last_lr = 0.0
        for x, y in train_loader:
            progress = step / max(total_steps - 1, 1)
            lr_mult = cosine_schedule(progress)
            for group in optimizer.param_groups:
                group["lr"] = CFG.lr * lr_mult
            x = train_augment(x.to(device, non_blocking=True))
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx:
                logits = model(x)
                loss = F.cross_entropy(logits, y, label_smoothing=CFG.label_smoothing)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if CFG.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            batch_n = y.numel()
            total_seen += batch_n
            loss_f = float(loss.detach())
            if not math.isfinite(loss_f):
                raise RuntimeError("non-finite loss")
            smooth_loss = loss_f if step == 0 else 0.95 * smooth_loss + 0.05 * loss_f
            epoch_loss_sum += loss_f * batch_n
            epoch_loss_n += batch_n
            last_lr = CFG.lr * lr_mult
            if step % 20 == 0:
                print(
                    f"epoch {epoch + 1:03d}/{CFG.epochs:03d} | step {step:06d}/{total_steps:06d} "
                    f"({100*progress:5.1f}%) | loss {smooth_loss:.4f} | lr {last_lr:.2e} | seen {total_seen:,}",
                    flush=True,
                )
            step += 1

        train_loss = epoch_loss_sum / max(epoch_loss_n, 1)
        if CFG.eval_every_epochs and ((epoch + 1) % CFG.eval_every_epochs == 0 or (epoch + 1) == CFG.epochs):
            last_metrics = evaluate_acc(model, val_loader, device="cuda")
            with curve_path.open("a") as f:
                f.write(
                    f"{epoch + 1}\t{step}\t{total_seen}\t{train_loss:.6f}\t{smooth_loss:.6f}\t"
                    f"{last_metrics['val_acc']:.6f}\t{last_metrics['val_top5']:.6f}\t{last_lr:.8e}\n"
                )
            print(
                f"epoch_summary {epoch + 1:03d}/{CFG.epochs:03d} | train_loss {train_loss:.4f} | "
                f"val_acc {last_metrics['val_acc']:.4f} | val_top5 {last_metrics['val_top5']:.4f} | lr {last_lr:.2e}",
                flush=True,
            )

    metrics = last_metrics or evaluate_acc(model, val_loader, device="cuda")
    torch.cuda.synchronize()
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    print("---")
    print(f"val_acc:          {metrics['val_acc']:.6f}")
    print(f"val_top5:         {metrics['val_top5']:.6f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {n_params / 1e6:.3f}")
    print(f"channels:         {CFG.channels}")


if __name__ == "__main__":
    main()
