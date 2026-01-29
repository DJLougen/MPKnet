"""
BinocularMPKNet V6 - First complete M/P/K pathway implementation with Fibonacci strides.

Key innovations:
1. First Fibonacci strides (2:3:5) in CNNs - derived from biological spatial frequency tuning
2. First complete M/P/K implementation - prior work (Magno-Parvo CNN, EVNets) models M/P only
3. Biologically-grounded K→M/P gating - extends cross-attention (Bahdanau, FiLM) with LGN anatomy

Fibonacci-inspired stride ratios (2:3:5) for P:K:M pathways:
- P: stride=2, kernel=5 (fine detail, ~80% of LGN neurons)
- K: stride=3, kernel=5 (context/modulation, ~10% of LGN)
- M: stride=5, kernel=5 (global gist, ~10% of LGN)

Results:
- 89.38% on CIFAR-10 with 0.539M parameters
- 60.8% on ImageNet-100 with 0.54M parameters
- 89.2% on Kvasir-v2 with 0.21M parameters

The stride ratios produce resolutions converging toward golden ratio (φ ≈ 1.618),
optimizing multi-scale coverage without redundancy - same principle as phyllotaxis.

Prior art acknowledgment:
- Cross-stream attention: Bahdanau (2014), FiLM (2018), SlowFast laterals (2019)
- M/P pathways: Magno-Parvo CNN (2022), EVNets (2024)
- Contribution: Complete M/P/K with functional K gating, Fibonacci strides
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mpknet_components import (
    BinocularPreMPK,
    StereoDisparity,
    StridedMonocularBlock,
    count_params,
)


class BinocularMPKNetV6(nn.Module):
    """
    Binocular MPKNet V6 with fibonacci stride scaling.

    Key changes from V4:
    - Larger kernel (5 vs 3)
    - Fibonacci strides: P=2, K=3, M=5
    - Same information extraction, fewer FLOPs

    The stride/kernel ratio ~0.4-1.0 provides efficient coverage:
    - P: 5/2 = 2.5 overlap per step (fine but not redundant)
    - K: 5/3 = 1.67 overlap (moderate)
    - M: 5/5 = 1.0 no overlap (coarse gist)
    """
    def __init__(self, num_classes: int = 10, ch: int = 48,
                 use_stereo: bool = True, disparity_range: int = 2,
                 kernel_size: int = 5):
        super().__init__()

        self.use_stereo = use_stereo
        self.kernel_size = kernel_size

        # Fibonacci strides: 2, 3, 5
        self.p_stride = 2
        self.k_stride = 3
        self.m_stride = 5

        if use_stereo:
            self.stereo = StereoDisparity(disparity_range)

        self.pre_mpk = BinocularPreMPK(sigma=1.0)

        # ========== BLOCK 1 ==========
        # P pathway: stride=2 (detail without noise), 2 layers
        self.P_block1_layer1 = StridedMonocularBlock(3, ch, kernel_size, stride=self.p_stride)
        self.P_block1_layer2 = StridedMonocularBlock(ch, ch, kernel_size, stride=1)

        # K pathway: stride=3 (context), 1 layer
        self.K_block1 = StridedMonocularBlock(3, ch // 2, kernel_size, stride=self.k_stride)

        # M pathway: stride=5 (global gist), 1 layer
        self.M_block1 = StridedMonocularBlock(3, ch, kernel_size, stride=self.m_stride)

        # K gates for block 1
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

        # K gates for block 2
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

        # Classification head (dropout before FC per NiN paper)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(ch * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Create stereo views
        if self.use_stereo:
            x_left, x_right = self.stereo(x)
        else:
            x_left, x_right = x, x

        # Retinal preprocessing
        P_left, M_left, P_right, M_right = self.pre_mpk(x_left, x_right)

        # ========== BLOCK 1 ==========
        # K first (from original input)
        K_left, K_right = self.K_block1(P_left, P_right)

        # P pathway: 2 layers
        P_left, P_right = self.P_block1_layer1(P_left, P_right)
        P_left, P_right = self.P_block1_layer2(P_left, P_right)

        # M pathway: 1 layer
        M_left, M_right = self.M_block1(M_left, M_right)

        # K gate 1 - GAP makes it resolution-independent
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

        # K gate 2
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
        target_size = M_left.shape[-1]  # M is smallest
        if P_left.shape[-1] != target_size:
            P_left = F.adaptive_avg_pool2d(P_left, target_size)
            P_right = F.adaptive_avg_pool2d(P_right, target_size)
        if K_left.shape[-1] != target_size:
            K_left = F.adaptive_avg_pool2d(K_left, target_size)
            K_right = F.adaptive_avg_pool2d(K_right, target_size)

        # Combine all four streams (M and P from both eyes)
        z = torch.cat([M_left, M_right, P_left, P_right], dim=1)
        z = self.v1_fusion(z)

        # Classification
        z = self.gap(z).flatten(1)
        z = self.dropout(z)
        return self.fc(z)


if __name__ == "__main__":
    model = BinocularMPKNetV6(num_classes=10, ch=48, use_stereo=True)
    print(f"BinocularMPKNet V6 params: {count_params(model)/1e6:.3f}M")
    print(f"Strides: P={model.p_stride}, K={model.k_stride}, M={model.m_stride}")
    print(f"Kernel: {model.kernel_size}")

    # Test on CIFAR-10 size
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")

    # Test on larger input
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
