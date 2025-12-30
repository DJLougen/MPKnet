#!/usr/bin/env python3
"""
Train BinocularMPKNet on STL-10 dataset.

STL-10: 96x96 images, 10 classes
- 5000 labeled training images (500 per class)
- 8000 test images
- 100000 unlabeled images (not used here)

Usage:
    python train_stl10.py --epochs 100 --batch 32 --lr 0.01
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os


def get_args():
    parser = argparse.ArgumentParser(description='Train BinocularMPKNet on STL-10')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Data directory')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--img_size', type=int, default=96,
                        help='Image size (STL-10 native is 96)')
    parser.add_argument('--ch', type=int, default=48,
                        help='Base channel count')
    parser.add_argument('--use_stereo', action='store_true',
                        help='Use stereo disparity simulation')
    parser.add_argument('--disparity', type=int, default=2,
                        help='Disparity range for stereo')
    parser.add_argument('--monocular_ratio', type=float, default=0.5,
                        help='Ratio of monocular channels')
    parser.add_argument('--augment', action='store_true',
                        help='Use data augmentation')
    parser.add_argument('--download', action='store_true',
                        help='Download dataset if not present')
    return parser.parse_args()


def main():
    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Transforms
    if args.augment:
        train_transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    test_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Load STL-10 dataset
    print("Loading STL-10 dataset...")
    train_dataset = datasets.STL10(
        root=args.data_dir,
        split='train',
        download=args.download,
        transform=train_transform
    )
    test_dataset = datasets.STL10(
        root=args.data_dir,
        split='test',
        download=args.download,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch, shuffle=True,
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch, shuffle=False,
        num_workers=4, pin_memory=True
    )

    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    # Model
    from mpknet_binocular import BinocularMPKNet

    model = BinocularMPKNet(
        num_classes=10,
        ch=args.ch,
        use_stereo=args.use_stereo,
        disparity_range=args.disparity,
        monocular_ratio=args.monocular_ratio
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nBinocularMPKNet: {num_params/1e6:.3f}M params")
    print(f"Image size: {args.img_size}x{args.img_size}")
    print(f"Stereo: {args.use_stereo}, Disparity: {args.disparity}")
    print(f"Augmentation: {args.augment}")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # TensorBoard
    run_name = f"stl10_ch{args.ch}_stereo{args.use_stereo}_aug{args.augment}"
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

        print(f'Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%, LR: {scheduler.get_last_lr()[0]:.6f}')

        # TensorBoard logging
        writer.add_scalar('Loss/train', train_loss / len(train_loader), epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/test', test_acc, epoch)
        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, 'stl10_best.pth')
            print(f'  -> New best: {best_acc:.2f}%')

    writer.close()
    print(f'\nTraining complete! Best accuracy: {best_acc:.2f}%')

    # Final summary
    print("\n" + "="*60)
    print("STL-10 RESULTS SUMMARY")
    print("="*60)
    print(f"Model: BinocularMPKNet ({num_params/1e6:.3f}M params)")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print(f"Augmentation: {args.augment}")
    print(f"Stereo: {args.use_stereo}")
    print("="*60)


if __name__ == '__main__':
    main()
