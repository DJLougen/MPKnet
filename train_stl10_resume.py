#!/usr/bin/env python3
"""
Resume training BinocularMPKNet on STL-10 from checkpoint.

Usage:
    python train_stl10_resume.py --checkpoint stl10_best.pth --epochs 200
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
    parser = argparse.ArgumentParser(description='Resume BinocularMPKNet training on STL-10')
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint to resume from')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--epochs', type=int, default=200, help='Additional epochs to train')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--img_size', type=int, default=96)
    parser.add_argument('--ch', type=int, default=48)
    parser.add_argument('--disparity', type=int, default=2)
    parser.add_argument('--monocular_ratio', type=float, default=0.5)
    return parser.parse_args()


def main():
    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    start_epoch = checkpoint['epoch'] + 1
    best_acc = checkpoint['best_acc']
    print(f"Resuming from epoch {start_epoch}, best acc: {best_acc:.2f}%")

    # Transforms (no augmentation)
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
    model = BinocularMPKNet(
        num_classes=10,
        ch=args.ch,
        use_stereo=True,
        disparity_range=args.disparity,
        monocular_ratio=args.monocular_ratio
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])

    num_params = sum(p.numel() for p in model.parameters())
    total_epochs = start_epoch + args.epochs
    print(f"\nBinocularMPKNet: {num_params/1e6:.3f}M params")
    print(f"Training epochs {start_epoch} -> {total_epochs}")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    # Load optimizer state if available
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # New cosine schedule for the additional epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # TensorBoard
    writer = SummaryWriter(f'runs/stl10_binocular_{total_epochs}ep')
    print(f"TensorBoard: runs/stl10_binocular_{total_epochs}ep")

    for epoch in range(start_epoch, total_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{total_epochs}')
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
            }, f'stl10_binocular_{total_epochs}ep_best.pth')
            print(f'  -> New best: {best_acc:.2f}%')

    writer.close()

    # Final summary
    print("\n" + "="*60)
    print(f"STL-10 RESULTS: BinocularMPKNet {total_epochs} epochs")
    print("="*60)
    print(f"Parameters: {num_params/1e6:.3f}M")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print("="*60)


if __name__ == '__main__':
    main()
