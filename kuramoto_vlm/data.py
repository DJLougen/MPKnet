"""Image-classification datasets exposed as (image, label, class-name) triples.

CIFAR-10/100 auto-download via torchvision; ImageNet-style trees load via
ImageFolder (point ``--data-root`` at a ``train/<class>/*.jpg`` layout — full
ImageNet, Tiny-ImageNet-200, Imagenette, etc.). Images are returned as float
CHW tensors in [0, 1] at ``image_size`` (the head's retinal front-end expects
that range).
"""

from __future__ import annotations

from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

_TORCHVISION = {"cifar10": datasets.CIFAR10, "cifar100": datasets.CIFAR100}


def _tf(image_size: int, train: bool) -> transforms.Compose:
    steps = [transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC)]
    if train:
        steps.append(transforms.RandomHorizontalFlip())
    steps.append(transforms.ToTensor())  # -> CHW float in [0, 1]
    return transforms.Compose(steps)


def _pretty(names: list[str]) -> list[str]:
    """Human-readable class names (underscores -> spaces, ImageNet takes 1st synonym)."""
    out = []
    for n in names:
        n = n.split(",")[0].strip().replace("_", " ")
        out.append(n)
    return out


IMAGENETTE_NAMES = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}


def get_dataset(name: str, *, root: str, image_size: int, train: bool):
    """Return ``(dataset, class_names)``; dataset yields ``(image_tensor, label_int)``."""
    key = name.lower()
    transform = _tf(image_size, train)
    if key in _TORCHVISION:
        ds = _TORCHVISION[key](root=root, train=train, download=True, transform=transform)
        return ds, _pretty(list(ds.classes))
    if key == "imagenette":
        from torchvision.datasets import Imagenette

        split = "train" if train else "val"
        try:
            ds = Imagenette(root, split=split, size="160px", download=True, transform=transform)
        except RuntimeError:
            ds = Imagenette(root, split=split, size="160px", download=False, transform=transform)
        names = []
        for c in ds.classes:
            wnid = c if isinstance(c, str) else c[0]
            names.append(IMAGENETTE_NAMES.get(wnid, str(wnid).replace("_", " ")))
        return ds, names
    # ImageNet / Tiny-ImageNet / Imagenette style folder tree.
    split = "train" if train else "val"
    import os

    split_root = os.path.join(root, split)
    folder = split_root if os.path.isdir(split_root) else root
    ds = datasets.ImageFolder(folder, transform=transform)
    return ds, _pretty(list(ds.classes))
