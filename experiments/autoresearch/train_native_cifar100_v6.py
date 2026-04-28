"""Native 32x32 CIFAR-100 reproduction for original MPKNet V6."""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, "/workspace/MPKnet/public")

from mpknet_v6 import BinocularMPKNetV6  # noqa: E402
from train_cifar100 import load_tensors  # noqa: E402


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


class NativeCIFAR100(Dataset):
    def __init__(self, x_uint8: torch.Tensor, y: torch.Tensor):
        self.x = x_uint8
        self.y = y
        self.mean = torch.tensor(CIFAR100_MEAN).view(3, 1, 1)
        self.std = torch.tensor(CIFAR100_STD).view(3, 1, 1)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        img = self.x[idx].float() / 255.0
        img = (img - self.mean) / self.std
        return img, int(self.y[idx])


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach())
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.numel()
    return total_loss / max(len(loader), 1), correct / max(total, 1)


@torch.inference_mode()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += float(loss.detach())
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.numel()
    return total_loss / max(len(loader), 1), correct / max(total, 1)


def main() -> None:
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    epochs = 100
    batch_size = 128
    x_train, y_train, x_val, y_val = load_tensors()
    train_loader = DataLoader(
        NativeCIFAR100(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        NativeCIFAR100(x_val, y_val),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = BinocularMPKNetV6(num_classes=100, ch=48, use_stereo=True, kernel_size=5).to(device)
    n_params = count_params(model)
    print("Task: CIFAR-100 native 32x32", flush=True)
    print("Model: original BinocularMPKNetV6 ch=48 kernel=5 sigmoid K gates", flush=True)
    print(f"Trainable params: {n_params:,} ({n_params / 1e6:.3f}M)", flush=True)
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}", flush=True)
    print("Optimizer: SGD lr=0.1 momentum=0.9 wd=5e-4 cosine T_max=100", flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    curve_dir = Path("training_curves")
    curve_dir.mkdir(exist_ok=True)
    curve_path = curve_dir / "native_cifar100_original_v6_sgd.tsv"
    curve_path.write_text("epoch\ttrain_loss\ttrain_acc\tval_loss\tval_acc\tlr\tseconds\n")

    best_acc = 0.0
    best_epoch = 0
    start_all = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0
        lr = scheduler.get_last_lr()[0]
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), "native_cifar100_original_v6_best.pth")
        with curve_path.open("a") as f:
            f.write(f"{epoch}\t{train_loss:.6f}\t{train_acc:.6f}\t{val_loss:.6f}\t{val_acc:.6f}\t{lr:.8e}\t{elapsed:.3f}\n")
        print(
            f"epoch {epoch:03d}/{epochs:03d} | train {100 * train_acc:.2f}% | "
            f"val {100 * val_acc:.2f}% | best {100 * best_acc:.2f}% @ {best_epoch} | "
            f"lr {lr:.6f} | {elapsed:.1f}s",
            flush=True,
        )

    print("---", flush=True)
    print(f"best_val_acc:     {best_acc:.6f}", flush=True)
    print(f"best_epoch:       {best_epoch}", flush=True)
    print(f"final_seconds:    {time.time() - start_all:.1f}", flush=True)
    print(f"num_params_M:     {n_params / 1e6:.3f}", flush=True)


if __name__ == "__main__":
    main()
