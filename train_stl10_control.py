#!/usr/bin/env python3
"""
Control experiment: Vanilla CNN with matched capacity (~0.14M params).

Tests whether MPKNet's performance comes from:
A) The parallel M/P/K pathway structure (biological hypothesis)
B) Simply having ~0.14M parameters in a deep network (capacity hypothesis)

If (B), this vanilla CNN should perform similarly to full BinocularMPKNet.
If (A), BinocularMPKNet should outperform despite equal capacity.

Usage:
    python train_stl10_control.py
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class VanillaCNN(nn.Module):
    """
    Vanilla CNN with ~0.14M parameters to match BinocularMPKNet.

    7 conv layers (matching total depth of M[2] + P[3] + K[2] pathways)
    No parallel streams, no gating, no biological structure.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()

        # Target: ~142K params
        # Strategy: 7 conv layers with channels tuned to match param count
        # Using kernel size 3 throughout (no biological receptive field priors)

        self.features = nn.Sequential(
            # Layer 1: 3 -> 32
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Layer 2: 32 -> 32
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Layer 3: 32 -> 48
            nn.Conv2d(32, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),

            # Layer 4: 48 -> 48
            nn.Conv2d(48, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),

            # Layer 5: 48 -> 64
            nn.Conv2d(48, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Layer 6: 64 -> 64
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Layer 7: 64 -> 64
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.fc(x)


def get_args():
    parser = argparse.ArgumentParser(description='Control CNN on STL-10')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--img_size', type=int, default=96)
    return parser.parse_args()


def main():
    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Transforms (no augmentation - same as ablation experiments)
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Load STL-10
    print("Loading STL-10 dataset...")
    train_dataset = datasets.STL10(root=args.data_dir, split='train',
                                    download=True, transform=transform)
    test_dataset = datasets.STL10(root=args.data_dir, split='test',
                                   download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False,
                             num_workers=4, pin_memory=True)

    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    # Model
    model = VanillaCNN(num_classes=10).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"CONTROL: VanillaCNN (no M/P/K structure)")
    print(f"Parameters: {num_params:,} ({num_params/1e6:.3f}M)")
    print(f"Layers: 7 conv + GAP + FC")
    print(f"{'='*60}\n")

    # Training setup (same as ablation)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # TensorBoard
    writer = SummaryWriter('runs/stl10_control_vanilla')
    print("TensorBoard: runs/stl10_control_vanilla")

    best_acc = 0

    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'acc': f'{100.*train_correct/train_total:.1f}%'
            })

        scheduler.step()

        # Validation
        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        train_acc = 100. * train_correct / train_total
        test_acc = 100. * test_correct / test_total

        print(f'Epoch {epoch+1}: Train: {train_acc:.2f}%, Test: {test_acc:.2f}%, LR: {scheduler.get_last_lr()[0]:.6f}')

        # TensorBoard
        writer.add_scalar('Loss/train', train_loss / len(train_loader), epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/test', test_acc, epoch)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_acc': best_acc,
            }, 'stl10_control_vanilla_best.pth')
            print(f'  -> New best: {best_acc:.2f}%')

    writer.close()

    # Final summary
    print("\n" + "="*60)
    print("STL-10 CONTROL RESULTS: VanillaCNN")
    print("="*60)
    print(f"Parameters: {num_params:,} ({num_params/1e6:.3f}M)")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print("="*60)


if __name__ == '__main__':
    main()
