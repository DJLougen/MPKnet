"""Image-driven MPK-Kuramoto vision head producing Qwen-compatible tokens.

Pipeline: image -> parameter-free retinal decomposition (the same 5 signals as
``MPKx.RetinaPreprocess`` / the Koniocellular manuscript) -> per-patch pathway
drives -> :class:`MPKKuramotoField` (M/P/K oscillator fields, run per patch in a
big ``B*P`` batch with shared learned coupling) -> sin/cos readout of M, P and K
(K encodes too) with a K-state diffuse gain injected on the integrated vector ->
linear projector to the LLM hidden size. Output is a sequence of vision tokens
``(B, num_tokens, hidden)`` that splice into an LLM's input-embedding stream at
image-placeholder positions, exactly like a ViT connector.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .kuramoto import MPKKuramotoField


def _gaussian_kernel(sigma: float, ksize: int) -> Tensor:
    """Normalized 2-D Gaussian kernel ``(ksize, ksize)``."""
    ax = torch.arange(ksize, dtype=torch.float32) - (ksize - 1) / 2.0
    g1 = torch.exp(-(ax * ax) / (2.0 * sigma * sigma))
    g1 = g1 / g1.sum()
    return torch.outer(g1, g1)


class RetinaPatchFront(nn.Module):
    """Parameter-free retinal decomposition, pooled into per-patch drives.

    Signals (RGB proxies, matching ``MPKx.RetinaPreprocess``):
      * ``low_pass``  = blur(luminance)          -> M drive (encoding)
      * ``high_pass`` = x - blur(x)  (3-channel)  -> P drive (encoding)
      * ``s_cone``    = B - (R+G)/2, normalized   -> K drive (K34 chromatic encoding)
      * ``orient_saliency`` = |grad(luminance)|   -> K drive (K12 orienting)
      * ``irradiance`` = [global mean lum, local var] -> K drive (K56 state)

    The K drive bundles chromatic + orienting + state, because K is a hybrid
    circuit-family (it encodes and carries higher-area context). Everything here
    is fixed (no learnable weights); the learnable maps live in the fields.
    """

    def __init__(self, *, patch_size: int = 8, pool: int = 2, blur_sigma: float = 1.0) -> None:
        super().__init__()
        if patch_size < pool:
            raise ValueError(f"patch_size ({patch_size}) must be >= pool ({pool}).")
        self.patch_size = int(patch_size)
        self.pool = int(pool)

        ksize = 5
        self.register_buffer("blur", _gaussian_kernel(blur_sigma, ksize).view(1, 1, ksize, ksize))
        self.register_buffer("lum_weight", torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_x.t().reshape(1, 1, 3, 3).contiguous())

        # Per-patch drive widths (see forward).
        c = pool * pool
        self.drive_dim_m = c  # low_pass (1ch) pooled
        self.drive_dim_p = 3 * c  # high_pass (3ch) pooled
        # K drive = [s_cone mean (K34), saliency (K12), global lum, patch var (K56)]
        self.drive_dim_k = 4

    def _blur1(self, x1: Tensor) -> Tensor:
        return F.conv2d(x1, self.blur, padding=self.blur.shape[-1] // 2)

    def _blur3(self, x3: Tensor) -> Tensor:
        weight = self.blur.expand(3, 1, -1, -1)
        return F.conv2d(x3, weight, padding=self.blur.shape[-1] // 2, groups=3)

    def _patch_pool(self, sig: Tensor) -> Tensor:
        """``(B, C, H, W)`` -> ``(B, P, C*pool*pool)`` by per-patch adaptive pool."""
        b, c, h, w = sig.shape
        p = self.patch_size
        nh, nw = h // p, w // p
        x = sig.unfold(2, p, p).unfold(3, p, p)  # (B, C, nh, nw, p, p)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(b * nh * nw, c, p, p)
        x = F.adaptive_avg_pool2d(x, self.pool)  # (B*P, C, pool, pool)
        return x.reshape(b, nh * nw, c * self.pool * self.pool)

    def _patch_stats1(self, sig1: Tensor) -> tuple[Tensor, Tensor]:
        """Per-patch mean and variance of a 1-channel signal -> two ``(B, P)``."""
        b, _, h, w = sig1.shape
        p = self.patch_size
        nh, nw = h // p, w // p
        x = sig1.unfold(2, p, p).unfold(3, p, p).reshape(b, nh * nw, p * p)
        return x.mean(dim=-1), x.var(dim=-1, unbiased=False)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Decompose ``x`` ``(B, 3, H, W)`` into per-patch pathway drives."""
        if x.dim() != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected (B, 3, H, W), got {tuple(x.shape)}.")
        h, w = x.shape[-2:]
        if h % self.patch_size or w % self.patch_size:
            raise ValueError(f"H,W ({h},{w}) must be divisible by patch {self.patch_size}.")

        lum = (x * self.lum_weight).sum(dim=1, keepdim=True)
        low_pass = self._blur1(lum)
        high_pass = x - self._blur3(x)
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        s_cone = b - 0.5 * (r + g)
        s_cone = s_cone / s_cone.abs().amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        grad = torch.sqrt(
            F.conv2d(lum, self.sobel_x, padding=1) ** 2
            + F.conv2d(lum, self.sobel_y, padding=1) ** 2
            + 1e-12
        )

        drive_m = self._patch_pool(low_pass)  # (B, P, pool^2)
        drive_p = self._patch_pool(high_pass)  # (B, P, 3*pool^2)
        s_mean, _ = self._patch_stats1(s_cone)  # (B, P)  K34 chromatic
        sal_mean, _ = self._patch_stats1(grad)  # (B, P)  K12 orienting
        _, lum_var = self._patch_stats1(lum)  # (B, P)   K56 local state
        global_lum = lum.mean(dim=(1, 2, 3)).unsqueeze(-1)  # (B, 1)  K56 global
        global_lum = global_lum.expand(-1, s_mean.shape[1])  # (B, P)

        drive_k = torch.stack([s_mean, sal_mean, global_lum, lum_var], dim=-1)  # (B, P, 4)
        return {"m": drive_m, "p": drive_p, "k": drive_k}


class KuramotoVisionHead(nn.Module):
    """MPK-Kuramoto image encoder emitting ``(B, num_tokens, hidden)`` tokens.

    Args:
        hidden_size: LLM hidden size (Qwen3.5-2B = 2048); the projector target.
        image_size: Square input side. Must be divisible by ``patch_size``.
        patch_size: One vision token per ``patch_size`` x ``patch_size`` patch.
        n_m, n_p, n_k: Oscillator counts per pathway.
        num_steps: Euler integration steps of the fields.
        pool: Retinal per-patch pooling grid.
    """

    def __init__(
        self,
        *,
        hidden_size: int = 2048,
        image_size: int = 32,
        patch_size: int = 8,
        n_m: int = 64,
        n_p: int = 64,
        n_k: int = 48,
        num_steps: int = 6,
        pool: int = 2,
        k_encode: bool = True,
        k_modulate: bool = True,
        freeze_coupling: bool = False,
    ) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError(f"image_size {image_size} not divisible by patch {patch_size}.")
        self.hidden_size = int(hidden_size)
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.num_tokens = (image_size // patch_size) ** 2

        self.retina = RetinaPatchFront(patch_size=patch_size, pool=pool)
        self.field = MPKKuramotoField(
            n_m=n_m,
            n_p=n_p,
            n_k=n_k,
            drive_dim_m=self.retina.drive_dim_m,
            drive_dim_p=self.retina.drive_dim_p,
            drive_dim_k=self.retina.drive_dim_k,
            num_steps=num_steps,
            k_encode=k_encode,
            k_modulate=k_modulate,
            freeze_coupling=freeze_coupling,
        )
        self.projector = nn.Sequential(
            nn.LayerNorm(self.field.readout_dim),
            nn.Linear(self.field.readout_dim, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

    def forward(self, pixel_values: Tensor) -> Tensor:
        """Encode ``(B, 3, H, W)`` -> vision tokens ``(B, num_tokens, hidden)``."""
        drives = self.retina(pixel_values)
        b, p = drives["m"].shape[:2]

        def flat(name: str) -> Tensor:
            return drives[name].reshape(b * p, -1)

        out = self.field(flat("m"), flat("p"), flat("k"))
        tokens = self.projector(out.readout)  # (B*P, hidden)
        return tokens.reshape(b, p, self.hidden_size)
