#!/usr/bin/env python3
"""
Train BinocularMPKNet on SVHN (Street View House Numbers).
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--ch', type=int, default=48)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # SVHN dataset (32x32 RGB, 10 classes: digits 0-9)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
    ])

    print("Loading SVHN dataset...")
    train_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
    test_dataset = datasets.SVHN(root='./data', split='test', download=True, transform=transform)
    
    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    num_workers = 4 if device.type == 'cuda' else 2
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    # Model
    model = BinocularMPKNet(num_classes=10, ch=args.ch, use_stereo=True, disparity_range=2).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"\nBinocularMPKNet: {params/1e6:.3f}M params (ch={args.ch})")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # TensorBoard
    run_name = f"runs/svhn_binocular_ch{args.ch}_{args.epochs}ep"
    writer = SummaryWriter(run_name)
    print(f"TensorBoard: {run_name}")

    best_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
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
            
            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{100.*train_correct/train_total:.1f}%")
        
        train_acc = 100. * train_correct / train_total
        
        # Eval
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        test_acc = 100. * test_correct / test_total
        lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch}: Train: {train_acc:.2f}%, Test: {test_acc:.2f}%, LR: {lr:.6f}")
        
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/test', test_acc, epoch)
        writer.add_scalar('Loss/train', train_loss / len(train_loader), epoch)
        writer.add_scalar('LR', lr, epoch)
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f"svhn_binocular_ch{args.ch}_best.pth")
            print(f"  -> New best: {best_acc:.2f}%")
        
        scheduler.step()

    print(f"\nTraining complete. Best test accuracy: {best_acc:.2f}%")
    writer.close()

if __name__ == '__main__':
    main()
