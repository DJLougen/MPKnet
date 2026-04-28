"""
One-time data preparation for autoresearch — VISION adapter (TinyImageNet @ 224).

Replaces karpathy/autoresearch's text/BPE prepare.py. Mirrors the same contract:
- This file provides the fixed dataset, evaluation metric, and parameter limit.
- Training length is controlled by train.py data-pass settings.

Usage:
    uv run prepare.py            # download + extract + cache
    uv run prepare.py --check    # verify cache only
"""

import argparse
import os
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np


# ---------------------------------------------------------------------------
# Constants (fixed — do not modify)
# ---------------------------------------------------------------------------

IMAGE_SIZE = 224                    # full-resolution MPKNet target
NUM_CLASSES = 200                   # TinyImageNet classes
EVAL_BATCH_SIZE = 128               # eval-time batch (kept fixed for consistent timing)
PARAM_BUDGET = 1_500_000            # max trainable params allowed (1.5M)


# ---------------------------------------------------------------------------
# Cache layout
# ---------------------------------------------------------------------------

CACHE_DIR = Path(os.path.expanduser("~")) / ".cache" / "autoresearch_vision"
DATA_DIR = CACHE_DIR / "tiny-imagenet-200"
TENSOR_CACHE = CACHE_DIR / "tensor_cache"  # preprocessed tensors live here
TRAIN_TENSORS = TENSOR_CACHE / "train.pt"
VAL_TENSORS = TENSOR_CACHE / "val.pt"

ZIP_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


# ---------------------------------------------------------------------------
# Download + extract
# ---------------------------------------------------------------------------

def download_tinyimagenet(force: bool = False):
    """Download + extract TinyImageNet-200 if not already present."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = CACHE_DIR / "tiny-imagenet-200.zip"

    if DATA_DIR.exists() and not force:
        print(f"[prepare] dataset already extracted at {DATA_DIR}", flush=True)
        return

    if not zip_path.exists() or force:
        print(f"[prepare] downloading {ZIP_URL} ...", flush=True)
        t0 = time.time()
        urllib.request.urlretrieve(ZIP_URL, zip_path)
        print(f"[prepare] downloaded in {time.time()-t0:.1f}s "
              f"({zip_path.stat().st_size/1e6:.0f} MB)", flush=True)

    print(f"[prepare] extracting...", flush=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(CACHE_DIR)
    print(f"[prepare] extracted to {DATA_DIR}", flush=True)


# ---------------------------------------------------------------------------
# Image loader (PIL -> tensor) without torchvision.transforms heavy stack
# ---------------------------------------------------------------------------

def _load_image(path: str) -> torch.Tensor:
    """Load a JPEG/PNG, return [3, 64, 64] uint8 tensor (TinyImageNet native)."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)  # [H, W, 3]
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # [3, H, W]


def _build_class_mapping():
    """Read wnids.txt for ordered class index -> wnid mapping."""
    wnids_path = DATA_DIR / "wnids.txt"
    with open(wnids_path) as f:
        wnids = [line.strip() for line in f if line.strip()]
    return {w: i for i, w in enumerate(wnids)}


def preprocess_train_tensors(force: bool = False) -> tuple:
    """Walk train/<wnid>/images/*.JPEG, stack into one big uint8 tensor."""
    if TRAIN_TENSORS.exists() and not force:
        d = torch.load(TRAIN_TENSORS, map_location="cpu", weights_only=True)
        return d["x"], d["y"]
    TENSOR_CACHE.mkdir(parents=True, exist_ok=True)

    cls_map = _build_class_mapping()
    train_dir = DATA_DIR / "train"
    xs, ys = [], []
    n = 0
    t0 = time.time()
    for wnid in sorted(cls_map.keys()):
        cls_dir = train_dir / wnid / "images"
        if not cls_dir.exists():
            continue
        label = cls_map[wnid]
        for fp in sorted(cls_dir.glob("*.JPEG")):
            xs.append(_load_image(str(fp)))
            ys.append(label)
            n += 1
            if n % 5000 == 0:
                print(f"[prepare] train: {n} images in {time.time()-t0:.1f}s",
                      flush=True)
    x = torch.stack(xs)  # [N, 3, 64, 64] uint8
    y = torch.tensor(ys, dtype=torch.long)
    torch.save({"x": x, "y": y}, TRAIN_TENSORS)
    print(f"[prepare] train cached: {x.shape}, labels {y.shape}", flush=True)
    return x, y


def preprocess_val_tensors(force: bool = False) -> tuple:
    """Read val/val_annotations.txt to get labels, build val tensor."""
    if VAL_TENSORS.exists() and not force:
        d = torch.load(VAL_TENSORS, map_location="cpu", weights_only=True)
        return d["x"], d["y"]
    TENSOR_CACHE.mkdir(parents=True, exist_ok=True)

    cls_map = _build_class_mapping()
    val_dir = DATA_DIR / "val"
    ann_path = val_dir / "val_annotations.txt"
    name_to_label = {}
    with open(ann_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                name_to_label[parts[0]] = cls_map[parts[1]]
    images_dir = val_dir / "images"
    xs, ys = [], []
    for name in sorted(name_to_label.keys()):
        xs.append(_load_image(str(images_dir / name)))
        ys.append(name_to_label[name])
    x = torch.stack(xs)
    y = torch.tensor(ys, dtype=torch.long)
    torch.save({"x": x, "y": y}, VAL_TENSORS)
    print(f"[prepare] val cached: {x.shape}, labels {y.shape}", flush=True)
    return x, y


# ---------------------------------------------------------------------------
# Dataset / DataLoader (upsampled to 224×224, no augmentation)
# ---------------------------------------------------------------------------

class TinyImageNet224(Dataset):
    """Pre-loaded uint8 tensor dataset, upsampled to 224 on retrieval."""

    def __init__(self, x_uint8: torch.Tensor, y: torch.Tensor, train: bool):
        self.x = x_uint8  # [N, 3, 64, 64] uint8
        self.y = y
        self.train = train

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        # uint8 [3,64,64] -> float32 [3,64,64]/255 -> upsample to 224
        img = self.x[idx].float() / 255.0
        img = F.interpolate(img.unsqueeze(0), size=IMAGE_SIZE,
                            mode="bilinear", align_corners=False).squeeze(0)
        return img, int(self.y[idx])


def get_dataloaders(batch_size: int, num_workers: int = 4):
    """Return (train_loader, val_loader). batch_size is the only knob.
    The agent passes batch_size from train.py; everything else fixed."""
    x_train, y_train = preprocess_train_tensors()
    x_val, y_val = preprocess_val_tensors()
    train_ds = TinyImageNet224(x_train, y_train, train=True)
    val_ds = TinyImageNet224(x_val, y_val, train=False)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Evaluation — the GROUND TRUTH metric. Do not modify.
# ---------------------------------------------------------------------------

@torch.inference_mode()
def evaluate_acc(model: torch.nn.Module, val_loader=None,
                 device: str = "cuda") -> dict:
    """
    Compute top-1 accuracy on the full TinyImageNet val set.

    Calls model(x_left, x_right) if the model accepts two arguments
    (binocular), else model(x). Auto-detected via signature inspection.

    Returns: {"val_acc": float, "val_top5": float, "n": int, "elapsed_s": float}
    """
    import inspect
    if val_loader is None:
        _, val_loader = get_dataloaders(batch_size=128, num_workers=2)

    sig = inspect.signature(model.forward)
    n_params = len(sig.parameters)
    binocular = n_params >= 2  # treat 2+ args as (left, right)

    model.eval()
    correct = 0
    correct5 = 0
    total = 0
    t0 = time.time()
    for x, y in val_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if binocular:
            logits = model(x, x)
        else:
            logits = model(x)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        pred1 = logits.argmax(dim=-1)
        correct += (pred1 == y).sum().item()
        # top-5
        top5 = logits.topk(5, dim=-1).indices
        correct5 += (top5 == y.unsqueeze(-1)).any(dim=-1).sum().item()
        total += y.numel()
    return {
        "val_acc": correct / max(total, 1),
        "val_top5": correct5 / max(total, 1),
        "n": total,
        "elapsed_s": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# Param counter (helper for the agent + the constraint validator)
# ---------------------------------------------------------------------------

def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_param_budget(model: torch.nn.Module) -> tuple:
    n = count_trainable_params(model)
    return n, n <= PARAM_BUDGET


# ---------------------------------------------------------------------------
# Main: full prep
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if args.check:
        ok = (TRAIN_TENSORS.exists() and VAL_TENSORS.exists() and
              DATA_DIR.exists())
        print(f"cache_ok={ok}  cache_dir={CACHE_DIR}")
        sys.exit(0 if ok else 1)

    download_tinyimagenet(force=args.force)
    x_train, y_train = preprocess_train_tensors(force=args.force)
    x_val, y_val = preprocess_val_tensors(force=args.force)
    print(f"\n[prepare] DONE")
    print(f"  train: {x_train.shape}  labels: {y_train.shape}  "
          f"unique={y_train.unique().numel()}")
    print(f"  val:   {x_val.shape}  labels: {y_val.shape}")
    print(f"  cache: {CACHE_DIR}")
    print(f"  IMAGE_SIZE={IMAGE_SIZE}  NUM_CLASSES={NUM_CLASSES}  PARAM_BUDGET={PARAM_BUDGET:,}")


if __name__ == "__main__":
    main()
