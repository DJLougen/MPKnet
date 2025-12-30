#!/usr/bin/env python3
"""
Train BinocularMPKNet on CIFAR-10 with augmentation options.

Usage:
    python train_cifar10_augment.py --augment standard --epochs 100
    python train_cifar10_augment.py --augment heavy --epochs 100
    python train_cifar10_augment.py --augment none --epochs 100
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from mpknet_binocular import BinocularMPKNet


def get_args():
    parser = argparse.ArgumentParser(description='Train BinocularMPKNet on CIFAR-10')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--ch', type=int, default=48)
    parser.add_argument('--use_stereo', action='store_true', default=True)
    parser.add_argument('--disparity', type=int, default=2)
    parser.add_argument('--monocular_ratio', type=float, default=0.5)
    parser.add_argument('--augment', type=str, default='none',
                        choices=['none', 'standard', 'heavy'],
                        help='Augmentation level')
    return parser.parse_args()


def get_transforms(augment: str, img_size: int):
    """Get train/test transforms based on augmentation level."""

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    if augment == 'none':
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize
        ])
    elif augment == 'standard':
        # Standard augmentation: flip + crop
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(img_size, padding=4),
            transforms.ToTensor(),
            normalize
        ])
    elif augment == 'heavy':
        # Heavy augmentation: flip + crop + rotation + color jitter + cutout
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2))  # Cutout-like
        ])

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])

    return train_transform, test_transform


def main():
    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Transforms
    train_transform, test_transform = get_transforms(args.augment, args.img_size)

    print(f"\nAugmentation: {args.augment}")
    print(f"Train transform: {train_transform}")

    # Load CIFAR-10
    print("\nLoading CIFAR-10 dataset...")
    train_dataset = datasets.CIFAR10(root=args.data_dir, train=True,
                                      download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=args.data_dir, train=False,
                                     download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False,
                             num_workers=4, pin_memory=True)

    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    # Model
    model = BinocularMPKNet(
        num_classes=10,
        ch=args.ch,
        use_stereo=args.use_stereo,
        disparity_range=args.disparity,
        monocular_ratio=args.monocular_ratio
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nBinocularMPKNet: {num_params/1e6:.3f}M params")
    print(f"Stereo: {args.use_stereo}, Disparity: {args.disparity}")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # TensorBoard
    run_name = f"cifar10_binocular_aug_{args.augment}_{args.epochs}ep"
    writer = SummaryWriter(f'runs/{run_name}')
    print(f"TensorBoard: runs/{run_name}")

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
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, f'cifar10_binocular_aug_{args.augment}_best.pth')
            print(f'  -> New best: {best_acc:.2f}%')

    writer.close()

    # Final summary
    print("\n" + "="*60)
    print(f"CIFAR-10 RESULTS: BinocularMPKNet + {args.augment} augmentation")
    print("="*60)
    print(f"Parameters: {num_params/1e6:.3f}M")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print("="*60)


if __name__ == '__main__':
    main()
