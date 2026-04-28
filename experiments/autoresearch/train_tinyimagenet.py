"""
Vision training harness for MPKnet/MPKx.

Training is controlled by data passes. Dataset
preparation, evaluation, and parameter limits live in prepare_vision.py.
"""

import math
from pathlib import Path
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from model import MPKx
from prepare_vision import (
    IMAGE_SIZE,
    NUM_CLASSES,
    PARAM_BUDGET,
    check_param_budget,
    evaluate_acc,
    get_dataloaders,
)


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
    compile_model: bool = False
    mixup_alpha: float = 0.0
    epochs: int = 100
    eval_every_epochs: int = 1


CFG = TrainConfig()


def cosine_schedule(progress: float) -> float:
    if progress < CFG.warmup_ratio:
        return max(progress / max(CFG.warmup_ratio, 1e-8), 1e-3)
    t = (progress - CFG.warmup_ratio) / max(1.0 - CFG.warmup_ratio, 1e-8)
    return CFG.final_lr_frac + 0.5 * (1.0 - CFG.final_lr_frac) * (1.0 + math.cos(math.pi * min(t, 1.0)))


def normalize(x: torch.Tensor) -> torch.Tensor:
    mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x - mean) / std


def train_augment(x: torch.Tensor) -> torch.Tensor:
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, [-1])
    return x


def main():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    print(f"Config: {asdict(CFG)}", flush=True)
    print(f"Task: TinyImageNet-{NUM_CLASSES} @ {IMAGE_SIZE}px", flush=True)

    train_loader, val_loader = get_dataloaders(CFG.batch_size, CFG.num_workers)
    model = MPKx(
        num_classes=NUM_CLASSES,
        ch=CFG.channels,
        use_stereo=CFG.use_stereo,
        disparity_range=CFG.disparity_range,
        kernel_size=CFG.kernel_size,
    ).to(device)

    n_params, within_budget = check_param_budget(model)
    if not within_budget:
        raise RuntimeError(f"parameter budget exceeded: {n_params} > {PARAM_BUDGET}")
    print(f"Trainable params: {n_params:,} / budget {PARAM_BUDGET:,}", flush=True)

    if CFG.compile_model:
        model = torch.compile(model, dynamic=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    step = 0
    total_seen = 0
    smooth_loss = 0.0
    total_steps = len(train_loader) * CFG.epochs
    curve_dir = Path("training_curves")
    curve_dir.mkdir(exist_ok=True)
    gate_mode = getattr(model, "k_gate_mode", "facilitatory_sigmoid")
    curve_path = curve_dir / f"ch{CFG.channels}_batch{CFG.batch_size}_epochs{CFG.epochs}_{gate_mode}.tsv"
    curve_path.write_text("epoch\tstep\tseen\ttrain_loss\tsmooth_loss\tval_acc\tval_top5\tlr\n")
    last_metrics = None

    for epoch in range(CFG.epochs):
        model.train()
        epoch_loss_sum = 0.0
        epoch_loss_n = 0
        last_lr = 0.0
        for batch_idx, (x, y) in enumerate(train_loader):
            progress = step / max(total_steps - 1, 1)
            lr_mult = cosine_schedule(progress)
            for group in optimizer.param_groups:
                group["lr"] = CFG.lr * lr_mult

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            x = train_augment(x)

            # Mixup: blend image pairs for regularization
            if CFG.mixup_alpha > 0 and step % 2 == 0:
                lam = torch.distributions.Beta(CFG.mixup_alpha, CFG.mixup_alpha).sample().item()
                perm = torch.randperm(x.size(0), device=x.device)
                x = lam * x + (1 - lam) * x[perm]
                y_mix = y[perm]
            else:
                lam = 1.0
                y_mix = None

            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx:
                logits = model(x)
                if lam < 1.0:
                    loss = lam * F.cross_entropy(logits, y, label_smoothing=CFG.label_smoothing) + \
                           (1 - lam) * F.cross_entropy(logits, y_mix, label_smoothing=CFG.label_smoothing)
                else:
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
                print("FAIL: non-finite loss", flush=True)
                raise SystemExit(1)
            smooth_loss = loss_f if step == 0 else 0.95 * smooth_loss + 0.05 * loss_f
            epoch_loss_sum += loss_f * batch_n
            epoch_loss_n += batch_n
            last_lr = CFG.lr * lr_mult

            if step % 20 == 0:
                print(
                    f"epoch {epoch + 1:03d}/{CFG.epochs:03d} | step {step:06d}/{total_steps:06d} "
                    f"({100*progress:5.1f}%) | loss {smooth_loss:.4f} | "
                    f"lr {CFG.lr * lr_mult:.2e} | seen {total_seen:,}",
                    flush=True,
                )
            step += 1

        train_loss = epoch_loss_sum / max(epoch_loss_n, 1)
        if CFG.eval_every_epochs and ((epoch + 1) % CFG.eval_every_epochs == 0 or (epoch + 1) == CFG.epochs):
            with torch.inference_mode():
                last_metrics = evaluate_acc(model, val_loader=val_loader, device="cuda")
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

    model.eval()
    metrics = last_metrics or evaluate_acc(model, val_loader=val_loader, device="cuda")
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
