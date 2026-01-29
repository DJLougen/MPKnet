"""
Train MPKNet V6 (full) on Kvasir-v2.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os

from mpknet_v6 import BinocularMPKNetV6


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Kvasir-v2 transforms
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load Kvasir-v2
    data_dir = os.path.expanduser('~/mpknet/data/kvasir-dataset-v2')
    if not os.path.exists(data_dir):
        data_dir = '/Users/djl/mpknet/data/kvasir-dataset-v2'

    full_dataset = datasets.ImageFolder(data_dir, transform=transform_train)
    test_dataset = datasets.ImageFolder(data_dir, transform=transform_test)

    # 80/20 split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, _ = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    _, val_dataset = random_split(
        test_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Classes: {full_dataset.classes}")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Full V6 model
    model = BinocularMPKNetV6(num_classes=8, ch=48, use_stereo=True).to(device)
    print(f"Parameters: {count_params(model)/1e3:.1f}K")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    best_acc = 0
    epochs = 100

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (out.argmax(1) == y).sum().item()
            train_total += y.size(0)

        # Eval
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                val_correct += (out.argmax(1) == y).sum().item()
                val_total += y.size(0)

        train_acc = train_correct / train_total * 100
        val_acc = val_correct / val_total * 100

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'v6_kvasir_best.pth')

        scheduler.step()

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d}: Train {train_acc:.1f}% | Val {val_acc:.1f}% | Best {best_acc:.1f}%")

    print(f"\nTraining complete! Best accuracy: {best_acc:.1f}%")


if __name__ == '__main__':
    main()
