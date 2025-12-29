# mpknet_binocular.py
# Binocular MPKNet - dual-eye processing inspired by LGN ocular organization
#
# Biological motivation:
# - LGN has eye-specific layers (layers 1,4,6 = contralateral; 2,3,5 = ipsilateral)
# - Each M/P/K pathway receives segregated eye input
# - Binocular integration primarily in V1, but LGN shows modulatory effects
# - Stereo disparity processing begins with slightly different eye views

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class BinocularPreMPK(nn.Module):
    """
    Simulates retinal + LGN preprocessing for both eyes.
    Each eye gets its own center-surround filtering.
    """
    def __init__(self, sigma: float = 1.0):
        super().__init__()
        self.sigma = sigma
        # Gaussian kernel for surround
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


class OcularDominanceConv(nn.Module):
    """
    Convolution with ocular dominance - channels are assigned to left/right eye
    with graded mixing (some purely monocular, some binocular).

    Inspired by V1 ocular dominance columns but applied at LGN stage
    for computational efficiency.
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 monocular_ratio: float = 0.5):
        super().__init__()
        self.out_ch = out_ch
        self.monocular_ratio = monocular_ratio

        # Number of purely monocular channels (half left, half right)
        n_mono = int(out_ch * monocular_ratio)
        n_mono_per_eye = n_mono // 2
        n_bino = out_ch - 2 * n_mono_per_eye

        self.n_left = n_mono_per_eye
        self.n_right = n_mono_per_eye
        self.n_bino = n_bino

        # Separate convs for left-eye-only, right-eye-only, and binocular
        self.conv_left = nn.Conv2d(in_ch, n_mono_per_eye, kernel_size, padding=kernel_size//2)
        self.conv_right = nn.Conv2d(in_ch, n_mono_per_eye, kernel_size, padding=kernel_size//2)
        self.conv_bino_L = nn.Conv2d(in_ch, n_bino, kernel_size, padding=kernel_size//2)
        self.conv_bino_R = nn.Conv2d(in_ch, n_bino, kernel_size, padding=kernel_size//2)

        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x_left: torch.Tensor, x_right: torch.Tensor) -> torch.Tensor:
        # Monocular channels
        left_only = self.conv_left(x_left)
        right_only = self.conv_right(x_right)

        # Binocular channels (sum of both eyes)
        bino = self.conv_bino_L(x_left) + self.conv_bino_R(x_right)

        # Concatenate: [left_mono | right_mono | binocular]
        out = torch.cat([left_only, right_only, bino], dim=1)
        return F.relu(self.bn(out))


class BinocularMPKPathway(nn.Module):
    """
    Single pathway (M, P, or K) with binocular processing.
    Receives left and right eye inputs, produces fused output.
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_sizes: list,
                 monocular_ratio: float = 0.5):
        super().__init__()

        layers = []
        ch = in_ch
        for i, ks in enumerate(kernel_sizes):
            is_first = (i == 0)
            if is_first:
                # First layer: ocular dominance processing
                layers.append(OcularDominanceConv(ch, out_ch, ks, monocular_ratio))
            else:
                # Subsequent layers: standard conv (already fused)
                layers.append(nn.Sequential(
                    nn.Conv2d(out_ch if i > 0 else ch, out_ch, ks, padding=ks//2),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                ))
            ch = out_ch

        self.first_layer = layers[0]
        self.rest = nn.Sequential(*layers[1:]) if len(layers) > 1 else nn.Identity()

    def forward(self, x_left: torch.Tensor, x_right: torch.Tensor) -> torch.Tensor:
        # Binocular fusion in first layer
        x = self.first_layer(x_left, x_right)
        # Rest is monocular (already fused)
        return self.rest(x)


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
            # Random disparity during training
            d = torch.randint(-self.disparity_range, self.disparity_range + 1, (1,)).item()
        else:
            # Fixed small disparity during eval
            d = 1

        if d == 0:
            return x, x

        # Shift images horizontally
        if d > 0:
            x_left = F.pad(x[:, :, :, d:], (0, d, 0, 0), mode='replicate')
            x_right = F.pad(x[:, :, :, :-d], (d, 0, 0, 0), mode='replicate')
        else:
            d = -d
            x_left = F.pad(x[:, :, :, :-d], (d, 0, 0, 0), mode='replicate')
            x_right = F.pad(x[:, :, :, d:], (0, d, 0, 0), mode='replicate')

        return x_left, x_right


class BinocularMPKNet(nn.Module):
    """
    Binocular MPKNet with dual-eye processing.

    Architecture:
    1. Stereo disparity simulation (optional)
    2. BinocularPreMPK: separate retinal processing per eye
    3. Three parallel pathways (M, P, K) with ocular dominance
    4. Konio gating of M and P streams
    5. Fusion and classification

    This adds ~50% more parameters but provides:
    - Biologically plausible eye-specific processing
    - Ocular dominance organization
    - Stereo disparity sensitivity
    """
    def __init__(self, num_classes: int = 10, ch: int = 48,
                 use_stereo: bool = True, disparity_range: int = 2,
                 monocular_ratio: float = 0.5):
        super().__init__()

        self.use_stereo = use_stereo

        # Stereo disparity simulation
        if use_stereo:
            self.stereo = StereoDisparity(disparity_range)

        # Binocular preprocessing (retina + early LGN)
        self.pre_mpk = BinocularPreMPK(sigma=1.0)

        # M pathway: large kernels, motion/gist (from luminance)
        # Input: 3 channels (expanded luminance), binocular
        self.M_pathway = BinocularMPKPathway(
            in_ch=3, out_ch=ch,
            kernel_sizes=[7, 5],
            monocular_ratio=monocular_ratio
        )

        # P pathway: small kernels, fine detail (from high-pass)
        self.P_pathway = BinocularMPKPathway(
            in_ch=3, out_ch=ch,
            kernel_sizes=[3, 3, 3],
            monocular_ratio=monocular_ratio
        )

        # K pathway: intermediate, generates attention gates
        self.K_pathway = BinocularMPKPathway(
            in_ch=3, out_ch=ch // 2,
            kernel_sizes=[5, 5],
            monocular_ratio=monocular_ratio
        )

        # Konio gating mechanism
        self.k_gate_M = nn.Sequential(
            nn.Linear(ch // 2, ch),
            nn.Sigmoid()
        )
        self.k_gate_P = nn.Sequential(
            nn.Linear(ch // 2, ch),
            nn.Sigmoid()
        )

        # Fusion
        self.fuse = nn.Sequential(
            nn.Conv2d(ch * 2, ch * 2, 1),
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

        # Binocular preprocessing
        P_left, M_left, P_right, M_right = self.pre_mpk(x_left, x_right)

        # Process through M/P/K pathways (binocular fusion happens in first layer)
        M = self.M_pathway(M_left, M_right)
        P = self.P_pathway(P_left, P_right)
        K = self.K_pathway((P_left + P_right) / 2, (P_left + P_right) / 2)  # K uses averaged input

        # Downsample to match sizes
        target_size = min(M.shape[-1], P.shape[-1], K.shape[-1])
        if M.shape[-1] != target_size:
            M = F.adaptive_avg_pool2d(M, target_size)
        if P.shape[-1] != target_size:
            P = F.adaptive_avg_pool2d(P, target_size)
        if K.shape[-1] != target_size:
            K = F.adaptive_avg_pool2d(K, target_size)

        # Konio gating
        k_ctx = self.gap(K).flatten(1)  # [B, ch//2]
        gate_M = self.k_gate_M(k_ctx).unsqueeze(-1).unsqueeze(-1)  # [B, ch, 1, 1]
        gate_P = self.k_gate_P(k_ctx).unsqueeze(-1).unsqueeze(-1)

        M = M * gate_M
        P = P * gate_P

        # Fuse and classify
        z = self.fuse(torch.cat([M, P], dim=1))
        z = self.gap(z).flatten(1)
        return self.fc(z)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    # Test
    model = BinocularMPKNet(num_classes=10, ch=48, use_stereo=True)
    print(f"BinocularMPKNet params: {count_params(model)/1e6:.3f}M")

    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")

    # Compare to original
    model_no_stereo = BinocularMPKNet(num_classes=10, ch=48, use_stereo=False)
    print(f"Without stereo: {count_params(model_no_stereo)/1e6:.3f}M")
