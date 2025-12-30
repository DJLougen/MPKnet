#!/usr/bin/env python3
"""
Scaling Study for BinocularMPKNet.

Tests how scaling individual pathway parameters affects performance.
Supports CIFAR-10, CIFAR-100, and STL-10 datasets.

Scaling dimensions:
1. ch (base channels): 24, 32, 48, 64, 96
2. M depth: 1, 2, 3 layers
3. P depth: 2, 3, 4 layers
4. K depth: 1, 2, 3 layers
5. K channel ratio: 0.25, 0.5, 0.75 (relative to ch)

Usage:
    python scaling_study.py --dataset cifar10 --scale ch --value 64
    python scaling_study.py --dataset stl10 --scale M_depth --value 3
    python scaling_study.py --dataset cifar10 --scale all  # Run full grid
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
from typing import Tuple, Dict, List
import json
import os
from datetime import datetime


# ============================================================================
# Configurable BinocularMPKNet
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


class ScalableBinocularMPKNet(nn.Module):
    """
    BinocularMPKNet with configurable scaling parameters.

    Config dict should contain:
    - ch: base channel count (default 48)
    - M_kernels: list of kernel sizes for M pathway (default [7, 5])
    - P_kernels: list of kernel sizes for P pathway (default [3, 3, 3])
    - K_kernels: list of kernel sizes for K pathway (default [5, 5])
    - K_ratio: K channel multiplier relative to ch (default 0.5)
    - monocular_ratio: ratio of monocular channels (default 0.5)
    """
    def __init__(self, num_classes: int = 10, config: Dict = None):
        super().__init__()

        # Default config (baseline)
        default_config = {
            'ch': 48,
            'M_kernels': [7, 5],
            'P_kernels': [3, 3, 3],
            'K_kernels': [5, 5],
            'K_ratio': 0.5,
            'monocular_ratio': 0.5,
            'disparity_range': 2,
        }

        if config is None:
            config = {}
        self.config = {**default_config, **config}

        ch = self.config['ch']
        k_ch = int(ch * self.config['K_ratio'])

        self.stereo = StereoDisparity(self.config['disparity_range'])
        self.pre_mpk = BinocularPreMPK(sigma=1.0)

        # M pathway
        self.M_pathway = BinocularMPKPathway(
            in_ch=3, out_ch=ch,
            kernel_sizes=self.config['M_kernels'],
            monocular_ratio=self.config['monocular_ratio']
        )

        # P pathway
        self.P_pathway = BinocularMPKPathway(
            in_ch=3, out_ch=ch,
            kernel_sizes=self.config['P_kernels'],
            monocular_ratio=self.config['monocular_ratio']
        )

        # K pathway
        self.K_pathway = BinocularMPKPathway(
            in_ch=3, out_ch=k_ch,
            kernel_sizes=self.config['K_kernels'],
            monocular_ratio=self.config['monocular_ratio']
        )

        # K-gating
        self.k_gate_M = nn.Sequential(
            nn.Linear(k_ch, ch),
            nn.Sigmoid()
        )
        self.k_gate_P = nn.Sequential(
            nn.Linear(k_ch, ch),
            nn.Sigmoid()
        )

        # Fusion
        self.fuse = nn.Sequential(
            nn.Conv2d(ch * 2, ch * 2, 1),
            nn.BatchNorm2d(ch * 2),
            nn.ReLU(inplace=True),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(ch * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_left, x_right = self.stereo(x)
        P_left, M_left, P_right, M_right = self.pre_mpk(x_left, x_right)

        M = self.M_pathway(M_left, M_right)
        P = self.P_pathway(P_left, P_right)
        K = self.K_pathway((P_left + P_right) / 2, (P_left + P_right) / 2)

        target_size = min(M.shape[-1], P.shape[-1], K.shape[-1])
        if M.shape[-1] != target_size:
            M = F.adaptive_avg_pool2d(M, target_size)
        if P.shape[-1] != target_size:
            P = F.adaptive_avg_pool2d(P, target_size)
        if K.shape[-1] != target_size:
            K = F.adaptive_avg_pool2d(K, target_size)

        k_ctx = self.gap(K).flatten(1)
        gate_M = self.k_gate_M(k_ctx).unsqueeze(-1).unsqueeze(-1)
        gate_P = self.k_gate_P(k_ctx).unsqueeze(-1).unsqueeze(-1)

        M = M * gate_M
        P = P * gate_P

        z = self.fuse(torch.cat([M, P], dim=1))
        z = self.gap(z).flatten(1)
        return self.fc(z)


# ============================================================================
# Scaling configurations
# ============================================================================

def get_scaling_configs() -> Dict[str, List[Dict]]:
    """Returns all scaling configurations to test."""

    configs = {
        # Base channel scaling
        'ch': [
            {'ch': 24, 'name': 'ch_24'},
            {'ch': 32, 'name': 'ch_32'},
            {'ch': 48, 'name': 'ch_48_baseline'},
            {'ch': 64, 'name': 'ch_64'},
            {'ch': 96, 'name': 'ch_96'},
        ],

        # M pathway depth scaling
        'M_depth': [
            {'M_kernels': [7], 'name': 'M_1layer'},
            {'M_kernels': [7, 5], 'name': 'M_2layer_baseline'},
            {'M_kernels': [7, 5, 5], 'name': 'M_3layer'},
        ],

        # P pathway depth scaling
        'P_depth': [
            {'P_kernels': [3, 3], 'name': 'P_2layer'},
            {'P_kernels': [3, 3, 3], 'name': 'P_3layer_baseline'},
            {'P_kernels': [3, 3, 3, 3], 'name': 'P_4layer'},
        ],

        # K pathway depth scaling
        'K_depth': [
            {'K_kernels': [5], 'name': 'K_1layer'},
            {'K_kernels': [5, 5], 'name': 'K_2layer_baseline'},
            {'K_kernels': [5, 5, 5], 'name': 'K_3layer'},
        ],

        # K channel ratio scaling
        'K_ratio': [
            {'K_ratio': 0.25, 'name': 'Kratio_0.25'},
            {'K_ratio': 0.5, 'name': 'Kratio_0.5_baseline'},
            {'K_ratio': 0.75, 'name': 'Kratio_0.75'},
            {'K_ratio': 1.0, 'name': 'Kratio_1.0'},
        ],
    }

    return configs


def get_single_config(scale_type: str, value) -> Dict:
    """Get a single config for a specific scaling parameter."""
    base = {}

    if scale_type == 'ch':
        base['ch'] = int(value)
        base['name'] = f'ch_{value}'
    elif scale_type == 'M_depth':
        depth = int(value)
        kernels = [7] + [5] * (depth - 1) if depth > 1 else [7]
        base['M_kernels'] = kernels
        base['name'] = f'M_{depth}layer'
    elif scale_type == 'P_depth':
        depth = int(value)
        base['P_kernels'] = [3] * depth
        base['name'] = f'P_{depth}layer'
    elif scale_type == 'K_depth':
        depth = int(value)
        base['K_kernels'] = [5] * depth
        base['name'] = f'K_{depth}layer'
    elif scale_type == 'K_ratio':
        base['K_ratio'] = float(value)
        base['name'] = f'Kratio_{value}'
    else:
        raise ValueError(f"Unknown scale type: {scale_type}")

    return base


# ============================================================================
# Training
# ============================================================================

def get_dataset(args):
    """Load dataset based on args.dataset."""

    # Determine image size based on dataset if not specified
    if args.img_size is None:
        if args.dataset == 'stl10':
            img_size = 96
        else:
            img_size = 32
    else:
        img_size = args.img_size

    # Transforms (no augmentation)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=args.data_dir, train=True,
                                          download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root=args.data_dir, train=False,
                                         download=True, transform=transform)
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=args.data_dir, train=True,
                                           download=True, transform=transform)
        test_dataset = datasets.CIFAR100(root=args.data_dir, train=False,
                                          download=True, transform=transform)
        num_classes = 100
    elif args.dataset == 'stl10':
        train_dataset = datasets.STL10(root=args.data_dir, split='train',
                                        download=True, transform=transform)
        test_dataset = datasets.STL10(root=args.data_dir, split='test',
                                       download=True, transform=transform)
        num_classes = 10
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    return train_dataset, test_dataset, num_classes, img_size


def train_single_config(config: Dict, args) -> Dict:
    """Train a single configuration and return results."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    train_dataset, test_dataset, num_classes, img_size = get_dataset(args)

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False,
                             num_workers=4, pin_memory=True)

    # Model
    name = config.pop('name', 'unnamed')
    model = ScalableBinocularMPKNet(num_classes=num_classes, config=config).to(device)
    num_params = sum(p.numel() for p in model.parameters())

    print(f"\n{'='*60}")
    print(f"CONFIG: {name}")
    print(f"Parameters: {num_params:,} ({num_params/1e6:.3f}M)")
    print(f"Config: {config}")
    print(f"{'='*60}")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # TensorBoard
    run_name = f"scaling_{args.dataset}_{name}_{args.epochs}ep"
    writer = SummaryWriter(f'runs/{run_name}')

    best_acc = 0
    best_epoch = 0

    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f'{name} E{epoch+1}/{args.epochs}', leave=False)
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

        # Log every 10 epochs or at end
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            print(f'{name} E{epoch+1}: Train {train_acc:.1f}%, Test {test_acc:.1f}%')

        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/test', test_acc, epoch)
        writer.add_scalar('Loss/train', train_loss / len(train_loader), epoch)

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1

    writer.close()

    result = {
        'name': name,
        'config': config,
        'params': num_params,
        'best_test_acc': best_acc,
        'best_epoch': best_epoch,
        'final_train_acc': train_acc,
    }

    return result


def get_args():
    parser = argparse.ArgumentParser(description='BinocularMPKNet Scaling Study')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'stl10'],
                        help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--img_size', type=int, default=None,
                        help='Image size (default: 32 for CIFAR, 96 for STL-10)')

    # Scaling options
    parser.add_argument('--scale', type=str, default='ch',
                        choices=['ch', 'M_depth', 'P_depth', 'K_depth', 'K_ratio', 'all'],
                        help='Which parameter to scale')
    parser.add_argument('--value', type=str, default=None,
                        help='Specific value to test (for single runs)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for results (default: scaling_results_{dataset}.json)')

    return parser.parse_args()


def main():
    args = get_args()

    # Default output filename based on dataset
    if args.output is None:
        args.output = f'scaling_results_{args.dataset}.json'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    all_results = []

    if args.value is not None:
        # Single configuration run
        config = get_single_config(args.scale, args.value)
        result = train_single_config(config.copy(), args)
        all_results.append(result)
    elif args.scale == 'all':
        # Run all scaling configurations
        all_configs = get_scaling_configs()
        for scale_type, configs in all_configs.items():
            print(f"\n{'#'*60}")
            print(f"# SCALING: {scale_type}")
            print(f"{'#'*60}")
            for config in configs:
                result = train_single_config(config.copy(), args)
                all_results.append(result)
    else:
        # Run all values for a specific scaling type
        all_configs = get_scaling_configs()
        if args.scale in all_configs:
            for config in all_configs[args.scale]:
                result = train_single_config(config.copy(), args)
                all_results.append(result)
        else:
            print(f"Unknown scale type: {args.scale}")
            return

    # Save results
    output_file = args.output
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print("\n" + "="*80)
    print("SCALING STUDY RESULTS")
    print("="*80)
    print(f"{'Name':<25} {'Params':>10} {'Best Test':>10} {'Best Epoch':>10}")
    print("-"*80)
    for r in sorted(all_results, key=lambda x: x['best_test_acc'], reverse=True):
        print(f"{r['name']:<25} {r['params']/1e6:>9.3f}M {r['best_test_acc']:>9.2f}% {r['best_epoch']:>10}")
    print("="*80)
    print(f"Results saved to: {output_file}")


if __name__ == '__main__':
    main()
