# data_loader.py

import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple

DATASET_STATS = {
    "CIFAR10":   {"mean": [0.4914, 0.4822, 0.4465], "std": [0.2023, 0.1994, 0.2010]},
    "CIFAR100":  {"mean": [0.5071, 0.4867, 0.4408], "std": [0.2675, 0.2565, 0.2761]},
    "MNIST":     {"mean": [0.1307],                 "std": [0.3081]},
    "SVHN":      {"mean": [0.4377, 0.4438, 0.4728], "std": [0.1980, 0.2010, 0.1970]},
    "STL10":     {"mean": [0.4467, 0.4398, 0.4066], "std": [0.2603, 0.2566, 0.2713]},
    "IMAGENET200": {"mean": [0.485, 0.456, 0.406],  "std": [0.229, 0.224, 0.225]},
    "IMAGENET1K":  {"mean": [0.485, 0.456, 0.406],  "std": [0.229, 0.224, 0.225]},
    "STANFORDCARS": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "INATURALIST": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "UCF101": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
}

IMAGENET_LIKE = {"IMAGENET200", "IMAGENET1K", "STANFORDCARS", "STL10", "INATURALIST", "UCF101"}
def get_transforms(dataset: str, img_size: int = None):
    """
    Return appropriate transforms for the given dataset.
    If img_size is None, chooses sensible defaults:
      - 224 for ImageNet-like datasets (ImageNet, Stanford Cars, STL10, iNaturalist)
      - 32 for CIFAR/SVHN
      - 28->img_size for MNIST (default 32)
    """
    ds = dataset.upper()
    stats = DATASET_STATS.get(ds, DATASET_STATS["CIFAR10"])

    # choose default size if not specified
    if img_size is None:
        img_size = 224 if ds in IMAGENET_LIKE else 32

    if ds == "MNIST":
        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(stats["mean"], stats["std"]),
        ])
        test_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(stats["mean"], stats["std"]),
        ])
        return train_tf, test_tf

    if ds in IMAGENET_LIKE:
        # Standard ImageNet-style transforms
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(stats["mean"], stats["std"]),
        ])
        test_tf = transforms.Compose([
            transforms.Resize(int(img_size * 256 / 224)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(stats["mean"], stats["std"]),
        ])
        return train_tf, test_tf

    # default: CIFAR/SVHN style
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(stats["mean"], stats["std"]),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(stats["mean"], stats["std"]),
    ])
    return train_tf, test_tf


def load_dataset(
    name: str,
    batch_size: int = 128,
    root: str = "./data",
    num_workers: int = 2,
    img_size: int | None = None,
    use_torchvision: bool = False,  # for StanfordCars optional path
    inat_version: str = "2021_train_mini",  # iNaturalist version
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Load a dataset by name.
    Returns: train_loader, test_loader, num_classes

    Datasets supported:
      - CIFAR10, CIFAR100, MNIST, SVHN, STL10
      - IMAGENET200  (expects: root/tiny-imagenet-200/{train,val})
      - IMAGENET1K   (expects: root/imagenet/{train,val})
      - STANFORDCARS
         * If use_torchvision=False (default):
             expects: root/stanford_cars/{train,test}/<class>/*.jpg
         * If use_torchvision=True:
             uses torchvision.datasets.StanfordCars with split='train'/'test'
      - INATURALIST (iNaturalist species classification)
         * Auto-downloads using torchvision.datasets.INaturalist
         * inat_version options: "2021_train", "2021_train_mini", "2021_valid"
         * 2021_train_mini: ~50k images, 10k species (recommended for testing)
         * 2021_train: ~2.7M images, 10k species (full dataset)
    """
    ds = name.upper()
    train_tf, test_tf = get_transforms(ds, img_size)

    if ds == "CIFAR10":
        train_ds = datasets.CIFAR10(root, train=True, transform=train_tf, download=True)
        test_ds  = datasets.CIFAR10(root, train=False, transform=test_tf, download=True)
        num_classes = 10

    elif ds == "CIFAR100":
        train_ds = datasets.CIFAR100(root, train=True, transform=train_tf, download=True)
        test_ds  = datasets.CIFAR100(root, train=False, transform=test_tf, download=True)
        num_classes = 100

    elif ds == "MNIST":
        train_ds = datasets.MNIST(root, train=True, transform=train_tf, download=True)
        test_ds  = datasets.MNIST(root, train=False, transform=test_tf, download=True)
        num_classes = 10

    elif ds == "SVHN":
        train_ds = datasets.SVHN(root, split="train", transform=train_tf, download=True)
        test_ds  = datasets.SVHN(root, split="test",  transform=test_tf, download=True)
        num_classes = 10

    elif ds == "STL10":
        train_ds = datasets.STL10(root, split="train", transform=train_tf, download=True)
        test_ds  = datasets.STL10(root, split="test",  transform=test_tf, download=True)
        num_classes = 10

    elif ds == "IMAGENET200":
        train_path = os.path.join(root, "tiny-imagenet-200/train")
        test_path  = os.path.join(root, "tiny-imagenet-200/val")
        train_ds = datasets.ImageFolder(train_path, transform=train_tf)
        test_ds  = datasets.ImageFolder(test_path,  transform=test_tf)
        num_classes = len(train_ds.classes)

    elif ds == "IMAGENET1K":
        # Expect standard layout: root/imagenet/{train,val}/wnid/*.JPEG
        train_path = os.path.join(root, "imagenet/train")
        val_path   = os.path.join(root, "imagenet/val")
        train_ds = datasets.ImageFolder(train_path, transform=train_tf)
        test_ds  = datasets.ImageFolder(val_path,   transform=test_tf)
        num_classes = len(train_ds.classes)  # should be 1000

    elif ds == "STANFORDCARS":
        if use_torchvision:
            # Uses torchvision dataset (requires files present; download is unreliable)
            train_ds = datasets.StanfordCars(root=root, split="train", transform=train_tf, download=True)
            test_ds  = datasets.StanfordCars(root=root, split="test",  transform=test_tf,  download=True)
        else:
            # Expect Kaggle "by-classes" folders: root/stanford_cars/{train,test}/<class>/*.jpg
            train_path = os.path.join(root, "stanford_cars/train")
            test_path  = os.path.join(root, "stanford_cars/test")
            train_ds = datasets.ImageFolder(train_path, transform=train_tf)
            test_ds  = datasets.ImageFolder(test_path,  transform=test_tf)
        num_classes = 196
    elif ds == "UCF101":
        # Expects pre-extracted frames
        train_path = os.path.join(root, "UCF-101_16f/train")
        test_path = os.path.join(root, "UCF-101_16f/val")
        train_ds = datasets.ImageFolder(train_path, transform=train_tf)
        test_ds = datasets.ImageFolder(test_path, transform=test_tf)
        num_classes = 101     

    elif ds == "INATURALIST":
        # iNaturalist dataset - auto-downloads
        # Version options:
        #   "2021_train" - full training set (~2.7M images, 10k species)
        #   "2021_train_mini" - mini training set (~50k images, 10k species) 
        #   "2021_valid" - validation set (~100k images, 10k species)
        
        print(f"Loading iNaturalist dataset (version: {inat_version})...")
        print("This may take a while on first download...")
        
        # For training, use train or train_mini
        if "train" in inat_version:
            train_ds = datasets.INaturalist(
                root=root, 
                version=inat_version,
                transform=train_tf,
                download=True
            )
            # Use validation set for testing
            test_ds = datasets.INaturalist(
                root=root,
                version="2021_valid",
                transform=test_tf,
                download=True
            )
        else:
            # If only validation is specified, split it
            print("Warning: Using validation set for both train and test")
            full_ds = datasets.INaturalist(
                root=root,
                version=inat_version,
                transform=train_tf,
                download=True
            )
            # Split validation set 80/20
            train_size = int(0.8 * len(full_ds))
            test_size = len(full_ds) - train_size
            train_ds, test_ds = torch.utils.data.random_split(
                full_ds, [train_size, test_size]
            )
        
        num_classes = 10000  # iNaturalist 2021 has 10k species

    else:
        raise ValueError(f"Unsupported dataset: {name}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False, persistent_workers=(num_workers > 0)
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False, persistent_workers=(num_workers > 0)
    )

    return train_loader, test_loader, num_classes