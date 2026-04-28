"""Train MPKx on small image datasets upsampled to 224px."""

from __future__ import annotations

import argparse
import gzip
import math
import os
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
PARAM_BUDGET = 1_500_000
EVAL_BATCH_SIZE = 128


@dataclass
class TrainConfig:
    dataset: str
    num_classes: int
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
    epochs: int = 100
    eval_every_epochs: int = 1


class UpsampledDataset(Dataset):
    def __init__(self, x_uint8: torch.Tensor, y: torch.Tensor):
        self.x = x_uint8
        self.y = y

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        img = self.x[idx].float() / 255.0
        img = F.interpolate(img.unsqueeze(0), size=IMAGE_SIZE, mode="bilinear", align_corners=False).squeeze(0)
        return img, int(self.y[idx])


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"[download] {url}", flush=True)
        urllib.request.urlretrieve(url, path)


def load_stl10(cache_root: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    cache_dir = cache_root / "stl10"
    archive = cache_dir / "stl10_binary.tar.gz"
    data_dir = cache_dir / "stl10_binary"
    tensor_dir = cache_dir / "tensor_cache"
    train_cache = tensor_dir / "train.pt"
    val_cache = tensor_dir / "test.pt"
    if train_cache.exists() and val_cache.exists():
        tr = torch.load(train_cache, map_location="cpu", weights_only=True)
        va = torch.load(val_cache, map_location="cpu", weights_only=True)
        return tr["x"], tr["y"], va["x"], va["y"], 10
    download("http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz", archive)
    if not data_dir.exists():
        print("[stl10] extracting", flush=True)
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(cache_dir)
    raw_train = np.fromfile(data_dir / "train_X.bin", dtype=np.uint8).reshape(-1, 3, 96, 96)
    raw_test = np.fromfile(data_dir / "test_X.bin", dtype=np.uint8).reshape(-1, 3, 96, 96)
    x_train = torch.from_numpy(np.transpose(raw_train, (0, 1, 3, 2)).copy()).contiguous()
    x_val = torch.from_numpy(np.transpose(raw_test, (0, 1, 3, 2)).copy()).contiguous()
    y_train = torch.from_numpy(np.fromfile(data_dir / "train_y.bin", dtype=np.uint8).astype(np.int64) - 1).long()
    y_val = torch.from_numpy(np.fromfile(data_dir / "test_y.bin", dtype=np.uint8).astype(np.int64) - 1).long()
    tensor_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"x": x_train, "y": y_train}, train_cache)
    torch.save({"x": x_val, "y": y_val}, val_cache)
    return x_train, y_train, x_val, y_val, 10


def _read_idx_images(path: Path) -> torch.Tensor:
    with gzip.open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    n = int.from_bytes(data[4:8].tobytes(), "big")
    rows = int.from_bytes(data[8:12].tobytes(), "big")
    cols = int.from_bytes(data[12:16].tobytes(), "big")
    images = data[16:].reshape(n, 1, rows, cols)
    return torch.from_numpy(np.repeat(images, 3, axis=1).copy()).contiguous()


def _read_idx_labels(path: Path) -> torch.Tensor:
    with gzip.open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return torch.from_numpy(data[8:].astype(np.int64).copy()).long()


def load_fashion_mnist(cache_root: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    cache_dir = cache_root / "fashion_mnist"
    tensor_dir = cache_dir / "tensor_cache"
    train_cache = tensor_dir / "train.pt"
    val_cache = tensor_dir / "test.pt"
    if train_cache.exists() and val_cache.exists():
        tr = torch.load(train_cache, map_location="cpu", weights_only=True)
        va = torch.load(val_cache, map_location="cpu", weights_only=True)
        return tr["x"], tr["y"], va["x"], va["y"], 10
    base = "https://fashion-mnist.s3-website.eu-central-1.amazonaws.com"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }
    for filename in files.values():
        download(f"{base}/{filename}", cache_dir / filename)
    x_train = _read_idx_images(cache_dir / files["train_images"])
    y_train = _read_idx_labels(cache_dir / files["train_labels"])
    x_val = _read_idx_images(cache_dir / files["test_images"])
    y_val = _read_idx_labels(cache_dir / files["test_labels"])
    tensor_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"x": x_train, "y": y_train}, train_cache)
    torch.save({"x": x_val, "y": y_val}, val_cache)
    return x_train, y_train, x_val, y_val, 10


def load_caltech101(cache_root: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    cache_dir = cache_root / "caltech101"
    archive = cache_dir / "101_ObjectCategories.tar.gz"
    data_dir = cache_dir / "101_ObjectCategories"
    tensor_dir = cache_dir / "tensor_cache"
    train_cache = tensor_dir / "train.pt"
    val_cache = tensor_dir / "test.pt"
    if train_cache.exists() and val_cache.exists():
        tr = torch.load(train_cache, map_location="cpu", weights_only=True)
        va = torch.load(val_cache, map_location="cpu", weights_only=True)
        return tr["x"], tr["y"], va["x"], va["y"], int(tr["num_classes"])
    download("https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip", cache_dir / "caltech-101.zip")
    if not data_dir.exists():
        import zipfile
        print("[caltech101] extracting", flush=True)
        with zipfile.ZipFile(cache_dir / "caltech-101.zip") as zf:
            zf.extractall(cache_dir)
        nested = cache_dir / "caltech-101" / "101_ObjectCategories"
        if nested.exists():
            nested.rename(data_dir)
    classes = sorted(p for p in data_dir.iterdir() if p.is_dir() and p.name != "BACKGROUND_Google")
    class_to_idx = {p.name: i for i, p in enumerate(classes)}
    train_items: list[tuple[Path, int]] = []
    val_items: list[tuple[Path, int]] = []
    rng = np.random.default_rng(42)
    for cls_dir in classes:
        images = sorted([p for p in cls_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        order = rng.permutation(len(images))
        split = min(30, max(1, len(images) // 2))
        idx = class_to_idx[cls_dir.name]
        for j in order[:split]:
            train_items.append((images[int(j)], idx))
        for j in order[split:]:
            val_items.append((images[int(j)], idx))

    def load_items(items: list[tuple[Path, int]]) -> tuple[torch.Tensor, torch.Tensor]:
        xs = []
        ys = []
        for path, label in items:
            img = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.uint8).transpose(2, 0, 1).copy()
            xs.append(torch.from_numpy(arr))
            ys.append(label)
        return torch.stack(xs, dim=0).contiguous(), torch.tensor(ys, dtype=torch.long)

    x_train, y_train = load_items(train_items)
    x_val, y_val = load_items(val_items)
    tensor_dir.mkdir(parents=True, exist_ok=True)
    payload_train = {"x": x_train, "y": y_train, "num_classes": len(classes)}
    payload_val = {"x": x_val, "y": y_val, "num_classes": len(classes)}
    torch.save(payload_train, train_cache)
    torch.save(payload_val, val_cache)
    print(f"[caltech101] train {tuple(x_train.shape)} val {tuple(x_val.shape)} classes {len(classes)}", flush=True)
    return x_train, y_train, x_val, y_val, len(classes)


def load_dataset(name: str):
    cache_root = Path(os.path.expanduser("~")) / ".cache" / "autoresearch_vision"
    if name == "stl10":
        return load_stl10(cache_root)
    if name == "fashion_mnist":
        return load_fashion_mnist(cache_root)
    if name == "caltech101":
        return load_caltech101(cache_root)
    raise ValueError(f"unknown dataset: {name}")


def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.inference_mode()
def evaluate_acc(model: torch.nn.Module, val_loader, device: str = "cuda") -> dict:
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


def train_augment(x: torch.Tensor) -> torch.Tensor:
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, [-1])
    return x


def cosine_schedule(progress: float, cfg: TrainConfig) -> float:
    if progress < cfg.warmup_ratio:
        return max(progress / max(cfg.warmup_ratio, 1e-8), 1e-3)
    t = (progress - cfg.warmup_ratio) / max(1.0 - cfg.warmup_ratio, 1e-8)
    return cfg.final_lr_frac + 0.5 * (1.0 - cfg.final_lr_frac) * (1.0 + math.cos(math.pi * min(t, 1.0)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["stl10", "fashion_mnist", "caltech101"], required=True)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    x_train, y_train, x_val, y_val, num_classes = load_dataset(args.dataset)
    cfg = TrainConfig(dataset=args.dataset, num_classes=num_classes, epochs=args.epochs)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    train_loader = DataLoader(UpsampledDataset(x_train, y_train), batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(UpsampledDataset(x_val, y_val), batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    model = MPKx(num_classes=cfg.num_classes, ch=cfg.channels, use_stereo=cfg.use_stereo, disparity_range=cfg.disparity_range, kernel_size=cfg.kernel_size).to(device)
    n_params = count_trainable_params(model)
    if n_params > PARAM_BUDGET:
        raise RuntimeError(f"parameter budget exceeded: {n_params} > {PARAM_BUDGET}")
    print(f"Config: {asdict(cfg)}", flush=True)
    print(f"Task: {args.dataset} @ {IMAGE_SIZE}px", flush=True)
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
    gate_mode = getattr(model, "k_gate_mode", "facilitatory_sigmoid")
    curve_path = curve_dir / f"{args.dataset}_ch{cfg.channels}_batch{cfg.batch_size}_epochs{cfg.epochs}_{gate_mode}.tsv"
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
                print(f"epoch {epoch + 1:03d}/{cfg.epochs:03d} | step {step:06d}/{total_steps:06d} ({100*progress:5.1f}%) | loss {smooth_loss:.4f} | lr {last_lr:.2e} | seen {total_seen:,}", flush=True)
            step += 1
        train_loss = epoch_loss_sum / max(epoch_loss_n, 1)
        if cfg.eval_every_epochs and ((epoch + 1) % cfg.eval_every_epochs == 0 or (epoch + 1) == cfg.epochs):
            last_metrics = evaluate_acc(model, val_loader, device="cuda")
            with curve_path.open("a") as f:
                f.write(f"{epoch + 1}\t{step}\t{total_seen}\t{train_loss:.6f}\t{smooth_loss:.6f}\t{last_metrics['val_acc']:.6f}\t{last_metrics['val_top5']:.6f}\t{last_lr:.8e}\n")
            print(f"epoch_summary {epoch + 1:03d}/{cfg.epochs:03d} | train_loss {train_loss:.4f} | val_acc {last_metrics['val_acc']:.4f} | val_top5 {last_metrics['val_top5']:.4f} | lr {last_lr:.2e}", flush=True)

    metrics = last_metrics or evaluate_acc(model, val_loader, device="cuda")
    torch.cuda.synchronize()
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    print("---")
    print(f"val_acc:          {metrics['val_acc']:.6f}")
    print(f"val_top5:         {metrics['val_top5']:.6f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {n_params / 1e6:.3f}")
    print(f"channels:         {cfg.channels}")


if __name__ == "__main__":
    main()
