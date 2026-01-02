"""
MPKx: An Interpretable, LGN-Inspired Architecture for Efficient Visual Processing

Daniel J. Lougen
University of Toronto
d.lougen@mail.utoronto.ca

Patent pending (US 63/950,391)

Key insight: M/P/K pathways differ in spatial sampling density, not kernel shape.
All pathways use the same 3x3 kernel but different strides:
- P: stride=1 (dense sampling, fine detail)
- K: stride=2 (intermediate, modulatory)
- M: stride=3 (sparse sampling, global gist)

Biologically accurate: M cells tile space more sparsely than P cells.
Eye segregation persists through LGN blocks, fusion only at V1.
K pathway modulates M and P streams (cross-stream attention).

Results:
- TinyImageNet-200: 40.6% accuracy, 0.23M params (matches ResNet18 with 48x fewer params)
- Kvasir-v2: 89.2% accuracy, no pretraining, no augmentation
- STL-10: 72.2% accuracy, 0.21M params
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# =============================================================================
# COMPONENTS
# =============================================================================

class BinocularPreMPK(nn.Module):
    """
    Simulates retinal + LGN preprocessing for both eyes.
    Each eye gets its own center-surround filtering.

    Biological motivation:
    - Retinal ganglion cells have center-surround receptive fields
    - M cells respond to luminance changes (motion/gist)
    - P cells respond to color/detail (high-pass filtered)
    """
    def __init__(self, sigma: float = 1.0):
        super().__init__()
        self.sigma = sigma
        ks = int(4 * sigma + 1) | 1  # ensure odd
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
        """
        Returns (P_left, M_left, P_right, M_right)
        P = high-pass (center - surround) for detail
        M = low-pass luminance for motion/gist
        """
        # Left eye
        blur_L = self._blur(x_left)
        P_left = x_left - blur_L  # high-pass (Parvo-like)
        lum_L = x_left.mean(dim=1, keepdim=True)
        M_left = self._blur(lum_L).expand(-1, 3, -1, -1)  # low-pass luminance (Magno-like)

        # Right eye
        blur_R = self._blur(x_right)
        P_right = x_right - blur_R
        lum_R = x_right.mean(dim=1, keepdim=True)
        M_right = self._blur(lum_R).expand(-1, 3, -1, -1)

        return P_left, M_left, P_right, M_right


class StereoDisparity(nn.Module):
    """
    Creates stereo disparity by horizontally shifting left/right views.
    Simulates the slight positional difference between two eyes.

    disparity_range: maximum pixel shift (positive = crossed disparity)
    """
    def __init__(self, disparity_range: int = 2):
        super().__init__()
        self.disparity_range = disparity_range

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Takes single image, returns (left_view, right_view) with disparity.
        For training, uses random disparity. For inference, uses fixed small disparity.
        """
        B, C, H, W = x.shape

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


class StridedMonocularBlock(nn.Module):
    """
    Monocular pathway block with configurable stride.
    Keeps left/right eyes separate, uses stride to control spatial sampling.

    Used for stride-based pathway differentiation:
    - P pathway: stride=1 (dense, fine detail)
    - K pathway: stride=2 (intermediate, modulatory)
    - M pathway: stride=3 (sparse, global gist)
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.conv_left = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.conv_right = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_left: torch.Tensor, x_right: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.conv_left(x_left), self.conv_right(x_right)


# =============================================================================
# MAIN MODEL
# =============================================================================

class MPKx(nn.Module):
    """
    MPKx: LGN-inspired architecture with stride-based pathway differentiation.

    Architecture:
    - Retinal preprocessing: center-surround filtering (parameter-free)
    - P pathway: stride=1, dense sampling for fine detail
    - K pathway: stride=2, generates cross-stream attention gates
    - M pathway: stride=3, sparse sampling for global gist
    - V1 fusion: late integration of all streams

    Key features:
    - Cross-stream K-gating (distinct from SE self-attention)
    - Binocular processing with late fusion
    - Every component maps to known neuroscience
    """
    def __init__(self, num_classes: int = 10, ch: int = 48,
                 use_stereo: bool = True, disparity_range: int = 2,
                 kernel_size: int = 3):
        super().__init__()

        self.use_stereo = use_stereo
        self.kernel_size = kernel_size

        if use_stereo:
            self.stereo = StereoDisparity(disparity_range)

        self.pre_mpk = BinocularPreMPK(sigma=1.0)

        # ========== BLOCK 1 ==========
        # P pathway: stride=1 (dense), 2 layers
        self.P_block1_layer1 = StridedMonocularBlock(3, ch, kernel_size, stride=1)
        self.P_block1_layer2 = StridedMonocularBlock(ch, ch, kernel_size, stride=1)

        # K pathway: stride=2 (intermediate), 1 layer
        self.K_block1 = StridedMonocularBlock(3, ch // 2, kernel_size, stride=2)

        # M pathway: stride=3 (sparse), 1 layer
        self.M_block1 = StridedMonocularBlock(3, ch, kernel_size, stride=3)

        # K gates for block 1 (cross-stream modulation)
        self.k_gate1_M_left = nn.Sequential(nn.Linear(ch // 2, ch), nn.Sigmoid())
        self.k_gate1_M_right = nn.Sequential(nn.Linear(ch // 2, ch), nn.Sigmoid())
        self.k_gate1_P_left = nn.Sequential(nn.Linear(ch // 2, ch), nn.Sigmoid())
        self.k_gate1_P_right = nn.Sequential(nn.Linear(ch // 2, ch), nn.Sigmoid())

        # ========== BLOCK 2 ==========
        # All stride=1 now (already at different resolutions)
        self.P_block2_layer1 = StridedMonocularBlock(ch, ch, kernel_size, stride=1)
        self.P_block2_layer2 = StridedMonocularBlock(ch, ch, kernel_size, stride=1)

        self.K_block2 = StridedMonocularBlock(ch // 2, ch // 2, kernel_size, stride=1)

        self.M_block2 = StridedMonocularBlock(ch, ch, kernel_size, stride=1)

        # K gates for block 2 (cross-stream modulation)
        self.k_gate2_M_left = nn.Sequential(nn.Linear(ch // 2, ch), nn.Sigmoid())
        self.k_gate2_M_right = nn.Sequential(nn.Linear(ch // 2, ch), nn.Sigmoid())
        self.k_gate2_P_left = nn.Sequential(nn.Linear(ch // 2, ch), nn.Sigmoid())
        self.k_gate2_P_right = nn.Sequential(nn.Linear(ch // 2, ch), nn.Sigmoid())

        # ========== V1 FUSION ==========
        self.v1_fusion = nn.Sequential(
            nn.Conv2d(ch * 4, ch * 2, 1),
            nn.BatchNorm2d(ch * 2),
            nn.ReLU(inplace=True),
        )

        # Classification head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(ch * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Create stereo views
        if self.use_stereo:
            x_left, x_right = self.stereo(x)
        else:
            x_left, x_right = x, x

        # Retinal preprocessing (center-surround filtering)
        P_left, M_left, P_right, M_right = self.pre_mpk(x_left, x_right)

        # ========== BLOCK 1 ==========
        # K first (from original input, stride=2)
        K_left, K_right = self.K_block1(P_left, P_right)

        # P pathway: stride=1, 2 layers
        P_left, P_right = self.P_block1_layer1(P_left, P_right)
        P_left, P_right = self.P_block1_layer2(P_left, P_right)

        # M pathway: stride=3, 1 layer
        M_left, M_right = self.M_block1(M_left, M_right)

        # K-gating 1: cross-stream modulation (GAP makes it resolution-independent)
        k_ctx1_left = self.gap(K_left).flatten(1)
        k_ctx1_right = self.gap(K_right).flatten(1)

        gate1_M_left = self.k_gate1_M_left(k_ctx1_left).unsqueeze(-1).unsqueeze(-1)
        gate1_M_right = self.k_gate1_M_right(k_ctx1_right).unsqueeze(-1).unsqueeze(-1)
        gate1_P_left = self.k_gate1_P_left(k_ctx1_left).unsqueeze(-1).unsqueeze(-1)
        gate1_P_right = self.k_gate1_P_right(k_ctx1_right).unsqueeze(-1).unsqueeze(-1)

        M_left = M_left * gate1_M_left
        M_right = M_right * gate1_M_right
        P_left = P_left * gate1_P_left
        P_right = P_right * gate1_P_right

        # ========== BLOCK 2 ==========
        P_left, P_right = self.P_block2_layer1(P_left, P_right)
        P_left, P_right = self.P_block2_layer2(P_left, P_right)

        K_left, K_right = self.K_block2(K_left, K_right)

        M_left, M_right = self.M_block2(M_left, M_right)

        # K-gating 2: cross-stream modulation
        k_ctx2_left = self.gap(K_left).flatten(1)
        k_ctx2_right = self.gap(K_right).flatten(1)

        gate2_M_left = self.k_gate2_M_left(k_ctx2_left).unsqueeze(-1).unsqueeze(-1)
        gate2_M_right = self.k_gate2_M_right(k_ctx2_right).unsqueeze(-1).unsqueeze(-1)
        gate2_P_left = self.k_gate2_P_left(k_ctx2_left).unsqueeze(-1).unsqueeze(-1)
        gate2_P_right = self.k_gate2_P_right(k_ctx2_right).unsqueeze(-1).unsqueeze(-1)

        M_left = M_left * gate2_M_left
        M_right = M_right * gate2_M_right
        P_left = P_left * gate2_P_left
        P_right = P_right * gate2_P_right

        # ========== V1 FUSION ==========
        # Match spatial sizes only at fusion (pool to smallest)
        target_size = M_left.shape[-1]  # M is smallest (stride=3)
        if P_left.shape[-1] != target_size:
            P_left = F.adaptive_avg_pool2d(P_left, target_size)
            P_right = F.adaptive_avg_pool2d(P_right, target_size)

        # Combine all four streams (binocular M and P)
        z = torch.cat([M_left, M_right, P_left, P_right], dim=1)
        z = self.v1_fusion(z)

        # Classification
        z = self.gap(z).flatten(1)
        return self.fc(z)


def count_params(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters())


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    # Example: STL-10 (96x96, 10 classes)
    model = MPKx(num_classes=10, ch=48, use_stereo=True)
    print(f"MPKx parameters: {count_params(model)/1e6:.3f}M")

    x = torch.randn(2, 3, 96, 96)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")

    # Example: TinyImageNet (64x64, 200 classes)
    model_tiny = MPKx(num_classes=200, ch=48, use_stereo=True)
    print(f"\nMPKx for TinyImageNet: {count_params(model_tiny)/1e6:.3f}M parameters")

    x_tiny = torch.randn(2, 3, 64, 64)
    y_tiny = model_tiny(x_tiny)
    print(f"Input: {x_tiny.shape}, Output: {y_tiny.shape}")
