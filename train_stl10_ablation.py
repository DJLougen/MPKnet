#!/usr/bin/env python3
"""
Ablation study for BinocularMPKNet on STL-10.

Tests which components are load-bearing:
1. Full model (M+P+K gating) - baseline
2. No K-gating (M+P only, gates fixed at 1.0)
3. No M pathway (P+K only)
4. No P pathway (M+K only)

Usage:
    python train_stl10_ablation.py --ablation none     # Full model
    python train_stl10_ablation.py --ablation no_kgate # Disable K-gating
    python train_stl10_ablation.py --ablation no_M     # Disable M pathway
    python train_stl10_ablation.py --ablation no_P     # Disable P pathway
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
import os
from typing import Tuple


# ============================================================================
# BinocularMPKNet with ablation support
# ============================================================================

class BinocularPreMPK(nn.Module):
    def __init__(self, sigma: float = 1.0):
        super().__init__()
        self.sigma = sigma
        ks = int(4 * sigma + 1) | 1
        ax = torch.arange(ks, dtype=torch.float32) - ks // 2
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        self.register_buffer('gauss', kernel.unsqueeze(0).unsqueeze(0))
        self.ks = ks

    def _blur(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        kernel = self.gauss.expand(C, 1, self.ks, self.ks)
        return F.conv2d(x, kernel, padding=self.ks // 2, groups=C)

    def forward(self, x_left: torch.Tensor, x_right: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        blur_L = self._blur(x_left)
        P_left = x_left - blur_L
        lum_L = x_left.mean(dim=1, keepdim=True)
        M_left = self._blur(lum_L).expand(-1, 3, -1, -1)

        blur_R = self._blur(x_right)
        P_right = x_right - blur_R
        lum_R = x_right.mean(dim=1, keepdim=True)
        M_right = self._blur(lum_R).expand(-1, 3, -1, -1)

        return P_left, M_left, P_right, M_right


class OcularDominanceConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 monocular_ratio: float = 0.5):
        super().__init__()
        self.out_ch = out_ch
        n_mono = int(out_ch * monocular_ratio)
        n_mono_per_eye = n_mono // 2
        n_bino = out_ch - 2 * n_mono_per_eye

        self.conv_left = nn.Conv2d(in_ch, n_mono_per_eye, kernel_size, padding=kernel_size//2)
        self.conv_right = nn.Conv2d(in_ch, n_mono_per_eye, kernel_size, padding=kernel_size//2)
        self.conv_bino_L = nn.Conv2d(in_ch, n_bino, kernel_size, padding=kernel_size//2)
        self.conv_bino_R = nn.Conv2d(in_ch, n_bino, kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x_left: torch.Tensor, x_right: torch.Tensor) -> torch.Tensor:
        left_only = self.conv_left(x_left)
        right_only = self.conv_right(x_right)
        bino = self.conv_bino_L(x_left) + self.conv_bino_R(x_right)
        out = torch.cat([left_only, right_only, bino], dim=1)
        return F.relu(self.bn(out))


class BinocularMPKPathway(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_sizes: list,
                 monocular_ratio: float = 0.5):
        super().__init__()

        layers = []
        ch = in_ch
        for i, ks in enumerate(kernel_sizes):
            if i == 0:
                layers.append(OcularDominanceConv(ch, out_ch, ks, monocular_ratio))
            else:
                layers.append(nn.Sequential(
                    nn.Conv2d(out_ch if i > 0 else ch, out_ch, ks, padding=ks//2),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                ))
            ch = out_ch

        self.first_layer = layers[0]
        self.rest = nn.Sequential(*layers[1:]) if len(layers) > 1 else nn.Identity()

    def forward(self, x_left: torch.Tensor, x_right: torch.Tensor) -> torch.Tensor:
        x = self.first_layer(x_left, x_right)
        return self.rest(x)


class StereoDisparity(nn.Module):
    def __init__(self, disparity_range: int = 2):
        super().__init__()
        self.disparity_range = disparity_range

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.training:
            d = torch.randint(-self.disparity_range, self.disparity_range + 1, (1,)).item()
        else:
            d = 1

        if d == 0:
            return x, x

        if d > 0:
            x_left = F.pad(x[:, :, :, d:], (0, d, 0, 0), mode='replicate')
            x_right = F.pad(x[:, :, :, :-d], (d, 0, 0, 0), mode='replicate')
        else:
            d = -d
            x_left = F.pad(x[:, :, :, :-d], (d, 0, 0, 0), mode='replicate')
            x_right = F.pad(x[:, :, :, d:], (0, d, 0, 0), mode='replicate')

        return x_left, x_right


class BinocularMPKNetAblation(nn.Module):
    """
    BinocularMPKNet with ablation flags.

    ablation options:
    - 'none': Full model (baseline)
    - 'no_kgate': K pathway exists but gates fixed at 1.0
    - 'no_M': M pathway zeroed out
    - 'no_P': P pathway zeroed out
    """
    def __init__(self, num_classes: int = 10, ch: int = 48,
                 use_stereo: bool = True, disparity_range: int = 2,
                 monocular_ratio: float = 0.5, ablation: str = 'none'):
        super().__init__()

        self.use_stereo = use_stereo
        self.ablation = ablation
        self.ch = ch

        if use_stereo:
            self.stereo = StereoDisparity(disparity_range)

        self.pre_mpk = BinocularPreMPK(sigma=1.0)

        # M pathway
        self.M_pathway = BinocularMPKPathway(
            in_ch=3, out_ch=ch,
            kernel_sizes=[7, 5],
            monocular_ratio=monocular_ratio
        )

        # P pathway
        self.P_pathway = BinocularMPKPathway(
            in_ch=3, out_ch=ch,
            kernel_sizes=[3, 3, 3],
            monocular_ratio=monocular_ratio
        )

        # K pathway (still needed for no_kgate to compare fair param count)
        self.K_pathway = BinocularMPKPathway(
            in_ch=3, out_ch=ch // 2,
            kernel_sizes=[5, 5],
            monocular_ratio=monocular_ratio
        )

        # K-gating mechanism
        self.k_gate_M = nn.Sequential(
            nn.Linear(ch // 2, ch),
            nn.Sigmoid()
        )
        self.k_gate_P = nn.Sequential(
            nn.Linear(ch // 2, ch),
            nn.Sigmoid()
        )

        # Fusion - adjust for ablations
        if ablation in ['no_M', 'no_P']:
            # Only one pathway active
            self.fuse = nn.Sequential(
                nn.Conv2d(ch, ch, 1),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
            )
            self.fc = nn.Linear(ch, num_classes)
        else:
            # Both pathways
            self.fuse = nn.Sequential(
                nn.Conv2d(ch * 2, ch * 2, 1),
                nn.BatchNorm2d(ch * 2),
                nn.ReLU(inplace=True),
            )
            self.fc = nn.Linear(ch * 2, num_classes)

        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Create stereo views
        if self.use_stereo:
            x_left, x_right = self.stereo(x)
        else:
            x_left, x_right = x, x

        # Preprocessing
        P_left, M_left, P_right, M_right = self.pre_mpk(x_left, x_right)

        # Process pathways
        M = self.M_pathway(M_left, M_right)
        P = self.P_pathway(P_left, P_right)
        K = self.K_pathway((P_left + P_right) / 2, (P_left + P_right) / 2)

        # Match sizes
        target_size = min(M.shape[-1], P.shape[-1], K.shape[-1])
        if M.shape[-1] != target_size:
            M = F.adaptive_avg_pool2d(M, target_size)
        if P.shape[-1] != target_size:
            P = F.adaptive_avg_pool2d(P, target_size)
        if K.shape[-1] != target_size:
            K = F.adaptive_avg_pool2d(K, target_size)

        # K-gating (or not)
        k_ctx = self.gap(K).flatten(1)

        if self.ablation == 'no_kgate':
            # Gates fixed at 1.0 - no modulation
            gate_M = torch.ones(M.shape[0], self.ch, 1, 1, device=M.device)
            gate_P = torch.ones(P.shape[0], self.ch, 1, 1, device=P.device)
        else:
            gate_M = self.k_gate_M(k_ctx).unsqueeze(-1).unsqueeze(-1)
            gate_P = self.k_gate_P(k_ctx).unsqueeze(-1).unsqueeze(-1)

        M = M * gate_M
        P = P * gate_P

        # Apply ablation
        if self.ablation == 'no_M':
            z = self.fuse(P)
        elif self.ablation == 'no_P':
            z = self.fuse(M)
        else:
            z = self.fuse(torch.cat([M, P], dim=1))

        z = self.gap(z).flatten(1)
        return self.fc(z)


# ============================================================================
# Training
# ============================================================================

def get_args():
    parser = argparse.ArgumentParser(description='Ablation study on STL-10')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--img_size', type=int, default=96)
    parser.add_argument('--ch', type=int, default=48)
    parser.add_argument('--disparity', type=int, default=2)
    parser.add_argument('--monocular_ratio', type=float, default=0.5)
    parser.add_argument('--ablation', type=str, default='none',
                        choices=['none', 'no_kgate', 'no_M', 'no_P'],
                        help='Ablation type')
    return parser.parse_args()


def main():
    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Transforms (no augmentation for fair comparison)
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

    # Model with ablation
    model = BinocularMPKNetAblation(
        num_classes=10,
        ch=args.ch,
        use_stereo=True,
        disparity_range=args.disparity,
        monocular_ratio=args.monocular_ratio,
        ablation=args.ablation
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"ABLATION: {args.ablation}")
    print(f"Parameters: {num_params/1e6:.3f}M")
    print(f"{'='*60}\n")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # TensorBoard
    run_name = f"stl10_ablation_{args.ablation}"
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
                'best_acc': best_acc,
                'ablation': args.ablation
            }, f'stl10_ablation_{args.ablation}_best.pth')
            print(f'  -> New best: {best_acc:.2f}%')

    writer.close()

    # Final summary
    print("\n" + "="*60)
    print(f"STL-10 ABLATION RESULTS: {args.ablation}")
    print("="*60)
    print(f"Parameters: {num_params/1e6:.3f}M")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print("="*60)


if __name__ == '__main__':
    main()
