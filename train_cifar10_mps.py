#!/usr/bin/env python3
"""
Train BinocularMPKNet on CIFAR-10 using MPS (Apple Silicon).

Usage:
    python train_cifar10_mps.py --epochs 300 --ch 48
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
    parser = argparse.ArgumentParser(description='Train BinocularMPKNet on CIFAR-10 (MPS)')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--ch', type=int, default=48)
    parser.add_argument('--use_stereo', action='store_true', default=True)
    parser.add_argument('--disparity', type=int, default=2)
    parser.add_argument('--monocular_ratio', type=float, default=0.5)
    return parser.parse_args()


def main():
    args = get_args()

    # Device selection: prefer MPS on Apple Silicon
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using device: MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device('cpu')
        print("Using device: CPU")

    # Transforms (no augmentation)
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Load CIFAR-10
    print("\nLoading CIFAR-10 dataset...")
    train_dataset = datasets.CIFAR10(root=args.data_dir, train=True,
                                      download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=args.data_dir, train=False,
                                     download=True, transform=transform)

    # Fewer workers for MPS to avoid issues
    num_workers = 2 if device.type == 'mps' else 4

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True,
                              num_workers=num_workers, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False,
                             num_workers=num_workers, pin_memory=False)

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
    print(f"\nBinocularMPKNet: {num_params/1e6:.3f}M params (ch={args.ch})")
    print(f"Stereo: {args.use_stereo}, Disparity: {args.disparity}")
    print(f"Training for {args.epochs} epochs")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # TensorBoard
    run_name = f"cifar10_binocular_ch{args.ch}_{args.epochs}ep_mps"
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

        # Print every epoch
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
            }, f'cifar10_binocular_ch{args.ch}_{args.epochs}ep_best.pth')
            print(f'  -> New best: {best_acc:.2f}%')

    writer.close()

    # Final summary
    print("\n" + "="*60)
    print(f"CIFAR-10 RESULTS: BinocularMPKNet ch={args.ch}")
    print("="*60)
    print(f"Parameters: {num_params/1e6:.3f}M")
    print(f"Epochs: {args.epochs}")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print("="*60)


if __name__ == '__main__':
    main()
