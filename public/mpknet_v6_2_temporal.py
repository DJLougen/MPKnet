"""
BinocularMPKNet V6.2 - Sequential Temporal M-Pathway

Key insight: M processes 8 frames sequentially, computing deltas between
consecutive frames. P sees only the current frame for detail.

M stream: f0→f1→f2→f3→f4→f5→f6→f7 (8 frames, 7 deltas)
P stream: f7 only (current frame, full detail)

This is Conv2D-based temporal processing:
- Same M weights applied to each frame
- Deltas capture motion between consecutive frames
- Fusion learns motion patterns (acceleration, direction change, etc.)

For static images: use augmented "pseudo-frames" (scales, shifts)
For video: use actual consecutive frames
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


class SequentialTemporalMPathway(nn.Module):
    """
    M-pathway that processes 8 frames sequentially.

    Computes features for each frame, then deltas between consecutive frames.
    Fuses all 7 deltas into a single motion representation.
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int, num_frames: int = 8):
        super().__init__()

        self.num_frames = num_frames
        self.num_deltas = num_frames - 1

        # Shared M block (same weights for all frames)
        self.m_block = StridedMonocularBlock(in_ch, out_ch, kernel_size, stride)

        # Delta processing: learn how to combine consecutive features
        # Instead of raw subtraction, learn the comparison
        self.delta_conv = nn.Conv2d(out_ch * 2, out_ch, kernel_size=1, bias=False)

        # Temporal fusion: combine all deltas into motion features
        self.temporal_fuse = nn.Sequential(
            nn.Conv2d(out_ch * self.num_deltas, out_ch * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch * 2, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, frames_left: torch.Tensor, frames_right: torch.Tensor):
        """
        Args:
            frames_left: [B, num_frames, C, H, W] - left eye frame sequence
            frames_right: [B, num_frames, C, H, W] - right eye frame sequence

        Returns:
            motion_left, motion_right: [B, out_ch, H', W'] - motion features
        """
        B = frames_left.shape[0]

        # Process all frames through M (shared weights)
        feats_left = []
        feats_right = []

        for t in range(self.num_frames):
            fl, fr = self.m_block(frames_left[:, t], frames_right[:, t])
            feats_left.append(fl)
            feats_right.append(fr)

        # Compute deltas between consecutive frames
        deltas_left = []
        deltas_right = []

        for t in range(self.num_deltas):
            # Concatenate consecutive features and learn the delta
            pair_left = torch.cat([feats_left[t], feats_left[t+1]], dim=1)
            pair_right = torch.cat([feats_right[t], feats_right[t+1]], dim=1)

            delta_l = self.delta_conv(pair_left)
            delta_r = self.delta_conv(pair_right)

            deltas_left.append(delta_l)
            deltas_right.append(delta_r)

        # Fuse all deltas into motion representation
        all_deltas_left = torch.cat(deltas_left, dim=1)   # [B, out_ch*7, H, W]
        all_deltas_right = torch.cat(deltas_right, dim=1)

        motion_left = self.temporal_fuse(all_deltas_left)
        motion_right = self.temporal_fuse(all_deltas_right)

        return motion_left, motion_right


class BinocularMPKNetV6_2(nn.Module):
    """
    Binocular MPKNet V6.2 with sequential temporal M-pathway.

    For video:
        - M sees 8 consecutive frames, extracts motion
        - P sees current frame only, extracts detail
        - Fusion combines motion + detail

    For static images (training):
        - Generate pseudo-frames via augmentation (scale, shift, blur)
        - M learns to detect "change" even without real motion
        - Transfers to real video at test time
    """
    def __init__(self, num_classes: int = 10, ch: int = 48,
                 use_stereo: bool = True, disparity_range: int = 2,
                 kernel_size: int = 5, num_frames: int = 8):
        super().__init__()

        self.use_stereo = use_stereo
        self.kernel_size = kernel_size
        self.num_frames = num_frames

        # Fibonacci strides
        self.p_stride = 2
        self.k_stride = 3
        self.m_stride = 5

        if use_stereo:
            self.stereo = StereoDisparity(disparity_range)

        self.pre_mpk = BinocularPreMPK(sigma=1.0)

        # ========== BLOCK 1 ==========
        # P pathway: single frame, high detail
        self.P_block1_layer1 = StridedMonocularBlock(3, ch, kernel_size, stride=self.p_stride)
        self.P_block1_layer2 = StridedMonocularBlock(ch, ch, kernel_size, stride=1)

        # K pathway: single frame, context
        self.K_block1 = StridedMonocularBlock(3, ch // 2, kernel_size, stride=self.k_stride)

        # M pathway: TEMPORAL - processes num_frames frames
        self.M_block1 = SequentialTemporalMPathway(3, ch, kernel_size, stride=self.m_stride, num_frames=num_frames)

        # K gates for block 1
        self.k_gate1_M_left = nn.Sequential(nn.Linear(ch // 2, ch), nn.Sigmoid())
        self.k_gate1_M_right = nn.Sequential(nn.Linear(ch // 2, ch), nn.Sigmoid())
        self.k_gate1_P_left = nn.Sequential(nn.Linear(ch // 2, ch), nn.Sigmoid())
        self.k_gate1_P_right = nn.Sequential(nn.Linear(ch // 2, ch), nn.Sigmoid())

        # ========== BLOCK 2 ==========
        self.P_block2_layer1 = StridedMonocularBlock(ch, ch, kernel_size, stride=1)
        self.P_block2_layer2 = StridedMonocularBlock(ch, ch, kernel_size, stride=1)

        self.K_block2 = StridedMonocularBlock(ch // 2, ch // 2, kernel_size, stride=1)

        # M block 2: single frame processing on motion features
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

        # Classification head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(ch * 2, num_classes)

    def forward(self, x: torch.Tensor, frames: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] - current frame (for P pathway)
            frames: [B, num_frames, C, H, W] - frame sequence (for M pathway)
                    If None, generates pseudo-frames from x
        """
        # Generate pseudo-frames if not provided (for static image training)
        if frames is None:
            frames = self._generate_pseudo_frames(x)

        # Create stereo views for current frame
        if self.use_stereo:
            x_left, x_right = self.stereo(x)
            # Also create stereo for all frames
            frames_left = torch.stack([self.stereo(frames[:, t])[0] for t in range(self.num_frames)], dim=1)
            frames_right = torch.stack([self.stereo(frames[:, t])[1] for t in range(self.num_frames)], dim=1)
        else:
            x_left, x_right = x, x
            frames_left = frames
            frames_right = frames

        # Retinal preprocessing for current frame (P pathway)
        P_left, _, P_right, _ = self.pre_mpk(x_left, x_right)

        # Retinal preprocessing for all frames (M pathway)
        M_frames_left = []
        M_frames_right = []
        for t in range(self.num_frames):
            _, ml, _, mr = self.pre_mpk(frames_left[:, t], frames_right[:, t])
            M_frames_left.append(ml)
            M_frames_right.append(mr)
        M_frames_left = torch.stack(M_frames_left, dim=1)  # [B, num_frames, C, H, W]
        M_frames_right = torch.stack(M_frames_right, dim=1)

        # ========== BLOCK 1 ==========
        # K pathway (from current frame)
        K_left, K_right = self.K_block1(P_left, P_right)

        # P pathway (current frame only)
        P_left, P_right = self.P_block1_layer1(P_left, P_right)
        P_left, P_right = self.P_block1_layer2(P_left, P_right)

        # M pathway (all frames - TEMPORAL)
        M_left, M_right = self.M_block1(M_frames_left, M_frames_right)

        # K gating
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

        # K gating
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
        target_size = M_left.shape[-1]
        if P_left.shape[-1] != target_size:
            P_left = F.adaptive_avg_pool2d(P_left, target_size)
            P_right = F.adaptive_avg_pool2d(P_right, target_size)
        if K_left.shape[-1] != target_size:
            K_left = F.adaptive_avg_pool2d(K_left, target_size)
            K_right = F.adaptive_avg_pool2d(K_right, target_size)

        z = torch.cat([M_left, M_right, P_left, P_right], dim=1)
        z = self.v1_fusion(z)

        # Classification
        z = self.gap(z).flatten(1)
        z = self.dropout(z)
        return self.fc(z)

    def _generate_pseudo_frames(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate pseudo-frames from a single image for static image training.

        Creates a sequence by applying progressive transformations:
        - Scales (zoom in/out slightly)
        - Small translations
        - Blur levels

        This teaches M to detect "change" even without real motion.
        """
        B, C, H, W = x.shape
        frames = []

        # Frame 0: most different (smaller scale, slight blur)
        frames.append(self._augment_frame(x, scale=0.85, blur=1.5))

        # Frames 1-6: gradual progression toward original
        for i in range(1, self.num_frames - 1):
            t = i / (self.num_frames - 1)  # 0 to 1
            scale = 0.85 + 0.15 * t  # 0.85 to 1.0
            blur = 1.5 * (1 - t)  # 1.5 to 0
            frames.append(self._augment_frame(x, scale=scale, blur=blur))

        # Frame 7: original (current frame)
        frames.append(x)

        return torch.stack(frames, dim=1)  # [B, num_frames, C, H, W]

    def _augment_frame(self, x: torch.Tensor, scale: float = 1.0, blur: float = 0.0) -> torch.Tensor:
        """Apply scale and blur augmentation to create pseudo-frame."""
        B, C, H, W = x.shape

        # Scale
        if scale != 1.0:
            new_H, new_W = int(H * scale), int(W * scale)
            x = F.interpolate(x, size=(new_H, new_W), mode='bilinear', align_corners=False)
            # Pad or crop back to original size
            if scale < 1.0:
                pad_h = (H - new_H) // 2
                pad_w = (W - new_W) // 2
                x = F.pad(x, (pad_w, W - new_W - pad_w, pad_h, H - new_H - pad_h), mode='reflect')
            else:
                start_h = (new_H - H) // 2
                start_w = (new_W - W) // 2
                x = x[:, :, start_h:start_h+H, start_w:start_w+W]

        # Blur (simple box blur approximation)
        if blur > 0:
            kernel_size = max(3, int(blur) * 2 + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            x = F.avg_pool2d(x, kernel_size, stride=1, padding=kernel_size//2)

        return x


if __name__ == "__main__":
    from mpknet_v6 import BinocularMPKNetV6
    from mpknet_v6_1 import BinocularMPKNetV6_1

    print("=" * 60)
    print("V6 vs V6.1 vs V6.2 (Temporal) Comparison")
    print("=" * 60)

    v6 = BinocularMPKNetV6(num_classes=10, ch=48)
    v6_1 = BinocularMPKNetV6_1(num_classes=10, ch=48)
    v6_2 = BinocularMPKNetV6_2(num_classes=10, ch=48, num_frames=8)

    print(f"V6   params: {count_params(v6)/1e6:.3f}M")
    print(f"V6.1 params: {count_params(v6_1)/1e6:.3f}M (dual-pass M)")
    print(f"V6.2 params: {count_params(v6_2)/1e6:.3f}M (8-frame temporal M)")
    print()

    # Test forward pass with static image (pseudo-frames)
    x = torch.randn(2, 3, 32, 32)

    print("Testing V6.2 with static image (generates pseudo-frames):")
    y = v6_2(x)
    print(f"  Input: {x.shape} → Output: {y.shape}")
    print()

    # Test forward pass with actual video frames
    frames = torch.randn(2, 8, 3, 32, 32)  # 8 frames
    print("Testing V6.2 with video frames:")
    y = v6_2(x, frames=frames)
    print(f"  Current frame: {x.shape}")
    print(f"  Frame sequence: {frames.shape}")
    print(f"  Output: {y.shape}")
    print()

    print("Architecture summary:")
    print("  P pathway: sees current frame only (detail)")
    print("  K pathway: sees current frame only (context/gating)")
    print("  M pathway: sees 8 frames, computes 7 deltas (motion)")
    print()
    print("For static images: pseudo-frames via scale/blur progression")
    print("For video: actual consecutive frames")
