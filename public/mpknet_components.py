"""
Shared components for all MPKNet model variants.

Contains building blocks used across V1, V2, V3, V4 and detection models:
- RGCLayer: Biologically accurate retinal ganglion cell preprocessing
- BinocularPreMPK: Legacy retinal preprocessing (deprecated, use RGCLayer)
- StereoDisparity: Stereo disparity simulation
- OcularDominanceConv: Convolution with ocular dominance channels
- BinocularMPKPathway: Pathway with binocular processing
- MonocularPathwayBlock: Pathway keeping eyes separate
- StridedMonocularBlock: Strided pathway for V4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class RGCLayer(nn.Module):
    """
    Biologically accurate Retinal Ganglion Cell layer.

    Based on Kim et al. 2021 "Retinal Ganglion Cells—Diversity of Cell Types
    and Clinical Relevance" (Front. Neurol. 12:661938).

    Models three main RGC types that feed the P/K/M pathways:

    1. MIDGET RGCs (~70% of RGCs):
       - Small receptive field (5-10 μm dendritic field)
       - Center-surround via Difference of Gaussians (DoG)
       - Red-Green color opponency (L-M or M-L)
       - Feeds PARVOCELLULAR (P) pathway
       - High spatial acuity, low temporal resolution

    2. PARASOL RGCs (~10% of RGCs):
       - Large receptive field (30-300 μm dendritic field)
       - Center-surround DoG on luminance
       - Achromatic (no color, L+M pooled)
       - Feeds MAGNOCELLULAR (M) pathway
       - Motion detection, high temporal resolution

    3. SMALL BISTRATIFIED RGCs (~5-8% of RGCs):
       - Medium receptive field
       - S-cone ON center, (L+M) OFF surround
       - Blue-Yellow opponency
       - Feeds KONIOCELLULAR (K) pathway
       - Color context, particularly blue

    Key biological details implemented:
    - DoG (Difference of Gaussians) for center-surround RF
    - RF size ratios: Midget < Bistratified < Parasol
    - Surround ~3-6x larger than center (we use 3x)
    - ON-center and OFF-center populations (we use ON-center)
    """

    def __init__(
        self,
        midget_sigma: float = 0.8,    # Small RF for fine detail
        parasol_sigma: float = 2.5,    # Large RF for motion/gist
        bistrat_sigma: float = 1.2,    # Medium RF for color context
        surround_ratio: float = 3.0,   # Surround is 3x center
    ):
        super().__init__()

        self.midget_sigma = midget_sigma
        self.parasol_sigma = parasol_sigma
        self.bistrat_sigma = bistrat_sigma
        self.surround_ratio = surround_ratio

        # Create DoG kernels for each cell type
        self.register_buffer('midget_center', self._make_gaussian(midget_sigma))
        self.register_buffer('midget_surround', self._make_gaussian(midget_sigma * surround_ratio))

        self.register_buffer('parasol_center', self._make_gaussian(parasol_sigma))
        self.register_buffer('parasol_surround', self._make_gaussian(parasol_sigma * surround_ratio))

        self.register_buffer('bistrat_center', self._make_gaussian(bistrat_sigma))
        self.register_buffer('bistrat_surround', self._make_gaussian(bistrat_sigma * surround_ratio))

        # Store kernel sizes for padding calculation
        self.midget_ks = self.midget_surround.shape[-1]
        self.parasol_ks = self.parasol_surround.shape[-1]
        self.bistrat_ks = self.bistrat_surround.shape[-1]

    def _make_gaussian(self, sigma: float) -> torch.Tensor:
        """Create a normalized 2D Gaussian kernel."""
        ks = int(6 * sigma + 1) | 1  # Ensure odd, cover 3 sigma each side
        ax = torch.arange(ks, dtype=torch.float32) - ks // 2
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()  # Normalize
        return kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    def _apply_dog(
        self,
        x: torch.Tensor,
        center_kernel: torch.Tensor,
        surround_kernel: torch.Tensor,
        kernel_size: int
    ) -> torch.Tensor:
        """Apply Difference of Gaussians (center - surround)."""
        B, C, H, W = x.shape
        padding = kernel_size // 2

        # Expand kernels for all channels
        center_k = center_kernel.expand(C, 1, -1, -1)
        surround_k = surround_kernel.expand(C, 1, -1, -1)

        # Pad surround kernel to match size if needed
        c_size = center_k.shape[-1]
        s_size = surround_k.shape[-1]
        if c_size < s_size:
            pad_amt = (s_size - c_size) // 2
            center_k = F.pad(center_k, (pad_amt, pad_amt, pad_amt, pad_amt))

        # Apply center and surround
        center_response = F.conv2d(x, center_k, padding=padding, groups=C)
        surround_response = F.conv2d(x, surround_k, padding=padding, groups=C)

        # DoG: ON-center response (center - surround)
        return center_response - surround_response

    def forward(
        self,
        x_left: torch.Tensor,
        x_right: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process left and right eye inputs through RGC populations.

        Returns:
            P_left, P_right: Midget RGC output (R-G opponency) -> P pathway
            M_left, M_right: Parasol RGC output (luminance DoG) -> M pathway
            K_left, K_right: Bistratified RGC output (S vs L+M) -> K pathway
        """
        # ========== MIDGET RGCs -> P pathway ==========
        # Red-Green opponency: L-cone vs M-cone
        # Approximate: R channel vs G channel
        # DoG on the opponent signal

        # Extract R and G channels (approximating L and M cones)
        R_left, G_left = x_left[:, 0:1], x_left[:, 1:2]
        R_right, G_right = x_right[:, 0:1], x_right[:, 1:2]

        # L-M opponency (R-G) with small receptive field DoG
        rg_left = R_left - G_left
        rg_right = R_right - G_right

        P_left = self._apply_dog(rg_left, self.midget_center, self.midget_surround, self.midget_ks)
        P_right = self._apply_dog(rg_right, self.midget_center, self.midget_surround, self.midget_ks)

        # Expand back to 3 channels for compatibility
        P_left = P_left.expand(-1, 3, -1, -1)
        P_right = P_right.expand(-1, 3, -1, -1)

        # ========== PARASOL RGCs -> M pathway ==========
        # Achromatic: pool L+M (approximate as luminance)
        # Large RF DoG for motion sensitivity

        lum_left = 0.299 * x_left[:, 0:1] + 0.587 * x_left[:, 1:2] + 0.114 * x_left[:, 2:3]
        lum_right = 0.299 * x_right[:, 0:1] + 0.587 * x_right[:, 1:2] + 0.114 * x_right[:, 2:3]

        M_left = self._apply_dog(lum_left, self.parasol_center, self.parasol_surround, self.parasol_ks)
        M_right = self._apply_dog(lum_right, self.parasol_center, self.parasol_surround, self.parasol_ks)

        # Expand to 3 channels
        M_left = M_left.expand(-1, 3, -1, -1)
        M_right = M_right.expand(-1, 3, -1, -1)

        # ========== SMALL BISTRATIFIED RGCs -> K pathway ==========
        # S-cone ON center, (L+M) OFF surround
        # Blue-Yellow opponency: S vs (L+M)

        # S-cone approximated by B channel
        # (L+M) approximated by (R+G)/2
        S_left = x_left[:, 2:3]  # Blue
        S_right = x_right[:, 2:3]
        LM_left = (x_left[:, 0:1] + x_left[:, 1:2]) / 2
        LM_right = (x_right[:, 0:1] + x_right[:, 1:2]) / 2

        # S - (L+M) opponency with medium RF
        by_left = S_left - LM_left
        by_right = S_right - LM_right

        K_left = self._apply_dog(by_left, self.bistrat_center, self.bistrat_surround, self.bistrat_ks)
        K_right = self._apply_dog(by_right, self.bistrat_center, self.bistrat_surround, self.bistrat_ks)

        # Expand to 3 channels
        K_left = K_left.expand(-1, 3, -1, -1)
        K_right = K_right.expand(-1, 3, -1, -1)

        return P_left, M_left, K_left, P_right, M_right, K_right


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

        n_mono = int(out_ch * monocular_ratio)
        n_mono_per_eye = n_mono // 2
        n_bino = out_ch - 2 * n_mono_per_eye

        self.n_left = n_mono_per_eye
        self.n_right = n_mono_per_eye
        self.n_bino = n_bino

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


class MonocularPathwayBlock(nn.Module):
    """
    Single pathway block that keeps left/right eyes separate.
    Used for LGN processing where eye segregation persists.
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int):
        super().__init__()
        self.conv_left = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.conv_right = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_left: torch.Tensor, x_right: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.conv_left(x_left), self.conv_right(x_right)


class StridedMonocularBlock(nn.Module):
    """
    Monocular pathway block with configurable stride.
    Keeps left/right eyes separate, uses stride to control spatial sampling.

    Used in V4 for stride-based pathway differentiation.
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


def count_params(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters())
