"""Matched-parameter residual CNN control for MPKx vision experiments."""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_cifar100 import CIFAR100Upsampled, load_tensors as load_cifar100_tensors
from train_mpk_smallvision import UpsampledDataset, load_dataset

IMAGE_SIZE = 224
PARAM_BUDGET = 1_500_000
EVAL_BATCH_SIZE = 128


@dataclass
class TrainConfig:
    dataset: str
    num_classes: int
    batch_size: int = 80
    width: int = 36
    lr: float = 4e-3
    weight_decay: float = 1.0e-2
    warmup_ratio: float = 0.08
    final_lr_frac: float = 0.0
    label_smoothing: float = 0.0
    grad_clip: float = 1.0
    num_workers: int = 4
    epochs: int = 100
    eval_every_epochs: int = 1


class BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        return F.relu(y + self.skip(x), inplace=True)


class SmallResNetControl(nn.Module):
    def __init__(self, num_classes: int, width: int = 36):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(BasicBlock(width, width), BasicBlock(width, width))
        self.stage2 = nn.Sequential(BasicBlock(width, width * 2, stride=2), BasicBlock(width * 2, width * 2))
        self.stage3 = nn.Sequential(BasicBlock(width * 2, width * 4, stride=2), BasicBlock(width * 4, width * 4))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(width * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def get_dataloaders(dataset: str, batch_size: int, num_workers: int):
    if dataset == "cifar100":
        x_train, y_train, x_val, y_val = load_cifar100_tensors()
        num_classes = 100
        train_ds = CIFAR100Upsampled(x_train, y_train)
        val_ds = CIFAR100Upsampled(x_val, y_val)
    else:
        x_train, y_train, x_val, y_val, num_classes = load_dataset(dataset)
        train_ds = UpsampledDataset(x_train, y_train)
        val_ds = UpsampledDataset(x_val, y_val)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, num_classes


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.inference_mode()
def evaluate_acc(model: nn.Module, val_loader, device: str = "cuda") -> dict:
    model.eval()
    correct = correct5 = total = 0
    t0 = time.time()
    for x, y in val_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        correct += (logits.argmax(dim=-1) == y).sum().item()
        topk = min(5, logits.shape[-1])
        correct5 += (logits.topk(topk, dim=-1).indices == y.unsqueeze(-1)).any(dim=-1).sum().item()
        total += y.numel()
    return {"val_acc": correct / max(total, 1), "val_top5": correct5 / max(total, 1), "elapsed_s": time.time() - t0}


def cosine_schedule(progress: float, cfg: TrainConfig) -> float:
    if progress < cfg.warmup_ratio:
        return max(progress / max(cfg.warmup_ratio, 1e-8), 1e-3)
    t = (progress - cfg.warmup_ratio) / max(1.0 - cfg.warmup_ratio, 1e-8)
    return cfg.final_lr_frac + 0.5 * (1.0 - cfg.final_lr_frac) * (1.0 + math.cos(math.pi * min(t, 1.0)))


def train_augment(x: torch.Tensor) -> torch.Tensor:
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, [-1])
    return x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cifar100", "stl10", "caltech101"], required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--width", type=int, default=36)
    args = parser.parse_args()

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    train_loader, val_loader, num_classes = get_dataloaders(args.dataset, 80, 4)
    cfg = TrainConfig(dataset=args.dataset, num_classes=num_classes, epochs=args.epochs, width=args.width)
    model = SmallResNetControl(num_classes=num_classes, width=cfg.width).to(device)
    n_params = count_trainable_params(model)
    if n_params > PARAM_BUDGET:
        raise RuntimeError(f"parameter budget exceeded: {n_params} > {PARAM_BUDGET}")
    print(f"Config: {asdict(cfg)}", flush=True)
    print(f"Task: {args.dataset} control @ {IMAGE_SIZE}px", flush=True)
    print(f"Trainable params: {n_params:,} / budget {PARAM_BUDGET:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    total_steps = len(train_loader) * cfg.epochs
    step = total_seen = 0
    smooth_loss = 0.0
    last_metrics = None
    curve_dir = Path("training_curves")
    curve_dir.mkdir(exist_ok=True)
    curve_path = curve_dir / f"control_resnet_w{cfg.width}_{args.dataset}_batch{cfg.batch_size}_epochs{cfg.epochs}.tsv"
    curve_path.write_text("epoch\tstep\tseen\ttrain_loss\tsmooth_loss\tval_acc\tval_top5\tlr\n")

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss_sum = 0.0
        epoch_loss_n = 0
        last_lr = 0.0
        for x, y in train_loader:
            progress = step / max(total_steps - 1, 1)
            lr_mult = cosine_schedule(progress, cfg)
            for group in optimizer.param_groups:
                group["lr"] = cfg.lr * lr_mult
            x = train_augment(x.to(device, non_blocking=True))
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx:
                loss = F.cross_entropy(model(x), y, label_smoothing=cfg.label_smoothing)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
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
            last_lr = cfg.lr * lr_mult
            if step % 20 == 0:
                print(
                    f"epoch {epoch + 1:03d}/{cfg.epochs:03d} | step {step:06d}/{total_steps:06d} "
                    f"({100*progress:5.1f}%) | loss {smooth_loss:.4f} | lr {last_lr:.2e} | seen {total_seen:,}",
                    flush=True,
                )
            step += 1

        train_loss = epoch_loss_sum / max(epoch_loss_n, 1)
        if cfg.eval_every_epochs and ((epoch + 1) % cfg.eval_every_epochs == 0 or (epoch + 1) == cfg.epochs):
            last_metrics = evaluate_acc(model, val_loader, device="cuda")
            with curve_path.open("a") as f:
                f.write(
                    f"{epoch + 1}\t{step}\t{total_seen}\t{train_loss:.6f}\t{smooth_loss:.6f}\t"
                    f"{last_metrics['val_acc']:.6f}\t{last_metrics['val_top5']:.6f}\t{last_lr:.8e}\n"
                )
            print(
                f"epoch_summary {epoch + 1:03d}/{cfg.epochs:03d} | train_loss {train_loss:.4f} | "
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
    print(f"width:            {cfg.width}")


if __name__ == "__main__":
    main()
