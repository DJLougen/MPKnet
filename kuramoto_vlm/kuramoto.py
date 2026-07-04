"""Kuramoto oscillator fields for the MPK vision head.

The coupled-oscillator primitive is Un-0's (unconv-ai/Un-0): the Kuramoto
velocity is expanded via the angle-difference identity so the pairwise term
`sum_j K_ij sin(theta_j - theta_i)` becomes two matmuls per block instead of a
per-sample outer product. Un-0 uses this as a *generative* substrate driven by a
class index and random seed. We invert that: the oscillators are *driven by the
image* (initial phases come from retinal patch features), and the population is
split into magno/parvo/konio pathways per the Koniocellular Circuit-Family
manuscript.

Pathway roles (faithful to the manuscript, corrected):

  * M (magnocellular): fast oscillators, luminance gist  — encoding relay.
  * P (parvocellular): mid-band oscillators, fine detail — encoding relay.
  * K (koniocellular): a HYBRID circuit-family — it both *encodes* (K3/K4 S-cone
    chromatic + coarse spatial) and *injects higher-area modulation* (K12
    orienting / K56 state gain from pulvinar/SC/arousal).

Key correction: K does **not** directly modulate the M and P relays. M/P/K each
evolve under their own coupling, independently. K contributes its own phases to
the readout (encoding), and a K-state-derived diffuse gain is injected onto the
**integrated decision vector** (what M and P feed into) — i.e. K "gates cortical
circuits fed by M and P" (Cheong et al. 2011), never the relays themselves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


def kuramoto_velocity(theta: Tensor, omega: Tensor, coupling: Tensor) -> Tensor:
    """Kuramoto phase velocity via the angle-difference identity.

    ``dtheta_i/dt = omega_i + sum_j K_ij sin(theta_j - theta_i)``
    ``            = omega_i + cos(theta_i)(sin(theta) @ K^T) - sin(theta_i)(cos(theta) @ K^T)``

    Args:
        theta: Phases ``(B, n)``.
        omega: Natural frequencies ``(1, n)`` (broadcast over the batch).
        coupling: Coupling matrix ``(n, n)`` with a zeroed diagonal.

    Returns:
        Phase velocity ``(B, n)``.
    """
    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)
    weighted_sin = sin_t @ coupling.transpose(-1, -2)
    weighted_cos = cos_t @ coupling.transpose(-1, -2)
    return omega + cos_t * weighted_sin - sin_t * weighted_cos


def order_parameter(theta: Tensor) -> Tensor:
    """Kuramoto order parameter (phase coherence) ``r in [0, 1]`` per sample.

    ``r = |(1/n) sum_j exp(i theta_j)|``. High ``r`` = the population is
    synchronized; low ``r`` = incoherent.

    Args:
        theta: Phases ``(B, n)``.

    Returns:
        Coherence ``(B, 1)``.
    """
    mean_cos = torch.cos(theta).mean(dim=-1, keepdim=True)
    mean_sin = torch.sin(theta).mean(dim=-1, keepdim=True)
    return torch.sqrt((mean_cos * mean_cos + mean_sin * mean_sin).clamp_min(1e-12))


class PathwayField(nn.Module):
    """One pathway's population of image-driven Kuramoto oscillators.

    Each pathway is *tuned to its own preference* two ways: natural frequencies
    ``omega`` are initialized in a pathway-specific band (``freq_scale``), and
    the image drive is projected into initial phases through a pathway-specific
    linear map. The intra-pathway coupling ``K`` is learned and shared across
    all patches/images (only the initial phases vary with the image).

    Args:
        n_oscillators: Population size ``n``.
        drive_dim: Dimensionality of the per-patch retinal drive feeding this
            pathway.
        freq_scale: Std of the natural-frequency init. Larger = faster pathway
            (M > P > K).
        init_k_scale: Coupling init scale (before the ``1/sqrt(n)`` factor).
    """

    def __init__(
        self,
        *,
        n_oscillators: int,
        drive_dim: int,
        freq_scale: float,
        init_k_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if n_oscillators < 2:
            raise ValueError(f"n_oscillators must be >= 2, got {n_oscillators}.")
        if drive_dim < 1:
            raise ValueError(f"drive_dim must be >= 1, got {drive_dim}.")
        self.n = int(n_oscillators)
        self.drive_dim = int(drive_dim)

        self.omega = nn.Parameter(freq_scale * torch.randn(1, self.n))
        coupling = init_k_scale * (self.n**-0.5) * torch.randn(self.n, self.n)
        coupling.fill_diagonal_(0.0)
        self.coupling = nn.Parameter(coupling)

        # Image -> initial phase. Bounded to (-pi, pi) via tanh so the drive sets
        # a genuine phase (matching Un-0's uniform [-pi, pi) seed range).
        self.to_phase = nn.Linear(self.drive_dim, self.n)

    def coupling_nodiag(self) -> Tensor:
        """Coupling with the diagonal removed (no self-coupling), grad-safe."""
        return self.coupling - torch.diag_embed(self.coupling.diagonal())

    def initial_phase(self, drive: Tensor) -> Tensor:
        """Map a per-patch drive ``(B, drive_dim)`` to phases ``(B, n)``."""
        return math.pi * torch.tanh(self.to_phase(drive))

    def integrate(self, theta: Tensor, *, num_steps: int, dt: float) -> Tensor:
        """Evolve phases under this pathway's own coupling (explicit Euler)."""
        coupling = self.coupling_nodiag()
        for _ in range(num_steps):
            theta = theta + dt * kuramoto_velocity(theta, self.omega, coupling)
        return theta


@dataclass
class MPKFieldOutput:
    """Bundle of an :class:`MPKKuramotoField` forward pass."""

    readout: Tensor  # (B, 2*(n_m + n_p + n_k)) — sin/cos of M, P AND K phases
    mod_gain: Tensor  # (B, 1) — K-derived diffuse gain injected on the DV
    coherence_k: Tensor  # (B, 1) — K-pathway order parameter at readout


class MPKKuramotoField(nn.Module):
    """M/P/K oscillator fields with K as an encoding + modulatory circuit-family.

    Dynamics (explicit Euler, ``num_steps`` steps over ``integration_time``):

      * M, P and K each evolve **independently** under their own coupling. K does
        NOT gate the M/P relays (the manuscript's coordination is not
        relay-to-relay).
      * Readout concatenates sin/cos of M, P **and K** — K is a genuine encoder
        (S-cone chromatic + coarse spatial), not merely a controller.
      * A diffuse gain read from the K population state (coherence + mean phase)
        multiplies the **integrated decision vector** — higher-area modulation
        injected where M and P converge (K "gates cortical circuits fed by M and
        P", Cheong et al. 2011), never the relays.
    """

    def __init__(
        self,
        *,
        n_m: int,
        n_p: int,
        n_k: int,
        drive_dim_m: int,
        drive_dim_p: int,
        drive_dim_k: int,
        num_steps: int = 6,
        integration_time: float = 1.0,
        freq_scale_m: float = 1.0,
        freq_scale_p: float = 0.5,
        freq_scale_k: float = 0.1,
        max_gain: float = 2.0,
        k_encode: bool = True,
        k_modulate: bool = True,
        freeze_coupling: bool = False,
    ) -> None:
        super().__init__()
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}.")
        self.n_m, self.n_p, self.n_k = int(n_m), int(n_p), int(n_k)
        self.num_steps = int(num_steps)
        self.dt = float(integration_time) / float(num_steps)
        self.max_gain = float(max_gain)
        self.k_encode = bool(k_encode)
        self.k_modulate = bool(k_modulate)

        # M/P faster than K; K is the slow sub-beta stream (manuscript 3.3).
        self.magno = PathwayField(n_oscillators=n_m, drive_dim=drive_dim_m, freq_scale=freq_scale_m)
        self.parvo = PathwayField(n_oscillators=n_p, drive_dim=drive_dim_p, freq_scale=freq_scale_p)
        self.konio = PathwayField(n_oscillators=n_k, drive_dim=drive_dim_k, freq_scale=freq_scale_k)
        if freeze_coupling:  # reservoir ablation: dynamics fixed at init, readout trains
            for field in (self.magno, self.parvo, self.konio):
                for param in field.parameters():
                    param.requires_grad_(False)

        # K population state -> a single diffuse gain injected onto the integrated
        # decision vector. Context = [mean cos, mean sin, coherence] of K phases.
        # Bias 0 keeps the initial gain near max_gain/2 (~1.0, gentle); the weight
        # is small but NONZERO so K's coupling/frequencies receive gradient via the
        # modulation path in addition to the readout path.
        self.mod = nn.Linear(3, 1)
        nn.init.normal_(self.mod.weight, std=0.1)
        nn.init.zeros_(self.mod.bias)

    @property
    def readout_dim(self) -> int:
        """Width of the sin/cos readout (M, P, and K when K encoding is on)."""
        base = 2 * (self.n_m + self.n_p)
        return base + (2 * self.n_k if self.k_encode else 0)

    def _mod_gain(self, theta_k: Tensor) -> Tensor:
        """Diffuse modulatory gain in (0, max_gain) from K population state."""
        ctx = torch.cat(
            [
                torch.cos(theta_k).mean(dim=-1, keepdim=True),
                torch.sin(theta_k).mean(dim=-1, keepdim=True),
                order_parameter(theta_k),
            ],
            dim=-1,
        )
        return self.max_gain * torch.sigmoid(self.mod(ctx))

    def forward(self, drive_m: Tensor, drive_p: Tensor, drive_k: Tensor) -> MPKFieldOutput:
        """Integrate the three pathways independently and integrate their readout.

        Args:
            drive_m: Magno drive ``(B, drive_dim_m)`` (luminance gist).
            drive_p: Parvo drive ``(B, drive_dim_p)`` (fine detail).
            drive_k: Konio drive ``(B, drive_dim_k)`` (S-cone chromatic + orienting
                saliency + irradiance state) — K encodes and carries context.
        """
        theta_m = self.magno.integrate(
            self.magno.initial_phase(drive_m), num_steps=self.num_steps, dt=self.dt
        )
        theta_p = self.parvo.integrate(
            self.parvo.initial_phase(drive_p), num_steps=self.num_steps, dt=self.dt
        )
        parts = [torch.sin(theta_m), torch.cos(theta_m), torch.sin(theta_p), torch.cos(theta_p)]

        theta_k = None
        if self.k_encode or self.k_modulate:
            theta_k = self.konio.integrate(
                self.konio.initial_phase(drive_k), num_steps=self.num_steps, dt=self.dt
            )
        if self.k_encode:
            parts += [torch.sin(theta_k), torch.cos(theta_k)]
        encode = torch.cat(parts, dim=-1)

        if self.k_modulate:
            gain = self._mod_gain(theta_k)
            readout = encode * gain
        else:
            gain = encode.new_ones(encode.shape[0], 1)
            readout = encode
        coherence = (
            order_parameter(theta_k) if theta_k is not None else encode.new_zeros(encode.shape[0], 1)
        )
        return MPKFieldOutput(readout=readout, mod_gain=gain, coherence_k=coherence)
