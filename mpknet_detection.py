# mpknet_detection.py
# Object detection with BinocularMPKNet backbone
# Simple anchor-free detection head inspired by FCOS/CenterNet
#
# Design philosophy:
# - Keep BinocularMPKNet backbone unchanged (task agnostic)
# - Add minimal detection head
# - Predict: (class_scores, bbox_offsets, centerness) per spatial location

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict
import math


class BinocularPreMPK(nn.Module):
    """Retinal + LGN preprocessing for both eyes."""
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
    """Convolution with ocular dominance channels."""
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
    """Single pathway with binocular processing."""
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
    """Creates stereo disparity by horizontal shifts."""
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


class MPKNetBackbone(nn.Module):
    """
    BinocularMPKNet as a feature extraction backbone.
    Returns feature maps at multiple scales for detection.
    """
    def __init__(self, ch: int = 48, use_stereo: bool = True,
                 disparity_range: int = 2, monocular_ratio: float = 0.5):
        super().__init__()
        self.use_stereo = use_stereo
        self.out_channels = ch * 2  # After M+P fusion

        if use_stereo:
            self.stereo = StereoDisparity(disparity_range)

        self.pre_mpk = BinocularPreMPK(sigma=1.0)

        self.M_pathway = BinocularMPKPathway(
            in_ch=3, out_ch=ch, kernel_sizes=[7, 5],
            monocular_ratio=monocular_ratio
        )
        self.P_pathway = BinocularMPKPathway(
            in_ch=3, out_ch=ch, kernel_sizes=[3, 3, 3],
            monocular_ratio=monocular_ratio
        )
        self.K_pathway = BinocularMPKPathway(
            in_ch=3, out_ch=ch // 2, kernel_sizes=[5, 5],
            monocular_ratio=monocular_ratio
        )

        self.k_gate_M = nn.Sequential(nn.Linear(ch // 2, ch), nn.Sigmoid())
        self.k_gate_P = nn.Sequential(nn.Linear(ch // 2, ch), nn.Sigmoid())

        self.fuse = nn.Sequential(
            nn.Conv2d(ch * 2, ch * 2, 1),
            nn.BatchNorm2d(ch * 2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns fused feature map [B, ch*2, H', W']"""
        if self.use_stereo:
            x_left, x_right = self.stereo(x)
        else:
            x_left, x_right = x, x

        P_left, M_left, P_right, M_right = self.pre_mpk(x_left, x_right)

        M = self.M_pathway(M_left, M_right)
        P = self.P_pathway(P_left, P_right)
        K = self.K_pathway((P_left + P_right) / 2, (P_left + P_right) / 2)

        # Match spatial sizes
        target_size = min(M.shape[-1], P.shape[-1], K.shape[-1])
        if M.shape[-1] != target_size:
            M = F.adaptive_avg_pool2d(M, target_size)
        if P.shape[-1] != target_size:
            P = F.adaptive_avg_pool2d(P, target_size)
        if K.shape[-1] != target_size:
            K = F.adaptive_avg_pool2d(K, target_size)

        # K-gating
        k_ctx = F.adaptive_avg_pool2d(K, 1).flatten(1)
        gate_M = self.k_gate_M(k_ctx).unsqueeze(-1).unsqueeze(-1)
        gate_P = self.k_gate_P(k_ctx).unsqueeze(-1).unsqueeze(-1)

        M = M * gate_M
        P = P * gate_P

        # Fuse M and P
        features = self.fuse(torch.cat([M, P], dim=1))
        return features


class DetectionHead(nn.Module):
    """
    Simple anchor-free detection head (FCOS-style).
    Predicts at each spatial location:
    - class scores (num_classes)
    - bbox offsets (4: left, top, right, bottom distances to edges)
    - centerness (1: how close to object center)
    """
    def __init__(self, in_channels: int, num_classes: int, num_convs: int = 2):
        super().__init__()
        self.num_classes = num_classes

        # Shared convs
        shared_layers = []
        for _ in range(num_convs):
            shared_layers.extend([
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.GroupNorm(8, in_channels),
                nn.ReLU(inplace=True)
            ])
        self.shared = nn.Sequential(*shared_layers)

        # Classification branch
        self.cls_head = nn.Conv2d(in_channels, num_classes, 3, padding=1)

        # Regression branch (4 bbox offsets + 1 centerness)
        self.reg_head = nn.Conv2d(in_channels, 4, 3, padding=1)
        self.centerness_head = nn.Conv2d(in_channels, 1, 3, padding=1)

        # Initialize
        self._init_weights()

    def _init_weights(self):
        for m in [self.cls_head, self.reg_head, self.centerness_head]:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0)
        # Bias init for classification (focal loss style)
        prior_prob = 0.01
        nn.init.constant_(self.cls_head.bias, -math.log((1 - prior_prob) / prior_prob))

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [B, C, H, W] from backbone

        Returns:
            dict with:
                - cls_logits: [B, num_classes, H, W]
                - bbox_reg: [B, 4, H, W] (l, t, r, b distances)
                - centerness: [B, 1, H, W]
        """
        x = self.shared(features)
        return {
            'cls_logits': self.cls_head(x),
            'bbox_reg': F.relu(self.reg_head(x)),  # distances must be positive
            'centerness': self.centerness_head(x)
        }


class MPKNetDetector(nn.Module):
    """
    Complete object detector using BinocularMPKNet backbone.

    Architecture:
    1. MPKNetBackbone: M/P/K pathway processing with binocular fusion
    2. DetectionHead: FCOS-style anchor-free predictions

    For each spatial location in the output feature map, predicts:
    - Class probabilities
    - Bounding box as (left, top, right, bottom) distances from that location
    - Centerness score (confidence that location is near object center)
    """
    def __init__(self, num_classes: int = 20, ch: int = 48,
                 use_stereo: bool = True, disparity_range: int = 2,
                 monocular_ratio: float = 0.5):
        super().__init__()

        self.backbone = MPKNetBackbone(
            ch=ch, use_stereo=use_stereo,
            disparity_range=disparity_range,
            monocular_ratio=monocular_ratio
        )

        self.head = DetectionHead(
            in_channels=self.backbone.out_channels,
            num_classes=num_classes,
            num_convs=2
        )

        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W] input image

        Returns:
            dict with cls_logits, bbox_reg, centerness
        """
        features = self.backbone(x)
        return self.head(features)

    def compute_loss(self, predictions: Dict[str, torch.Tensor],
                     targets: List[Dict[str, torch.Tensor]],
                     input_size: Tuple[int, int]) -> Dict[str, torch.Tensor]:
        """
        Compute detection loss.

        Args:
            predictions: output from forward()
            targets: list of dicts with 'boxes' [N, 4] and 'labels' [N]
            input_size: (H, W) of input image

        Returns:
            dict with cls_loss, reg_loss, centerness_loss, total_loss
        """
        cls_logits = predictions['cls_logits']  # [B, C, H', W']
        bbox_reg = predictions['bbox_reg']       # [B, 4, H', W']
        centerness = predictions['centerness']   # [B, 1, H', W']

        B, C, H, W = cls_logits.shape
        device = cls_logits.device

        # Generate grid of locations
        stride_h = input_size[0] / H
        stride_w = input_size[1] / W

        # Create coordinate grid
        y_coords = (torch.arange(H, device=device) + 0.5) * stride_h
        x_coords = (torch.arange(W, device=device) + 0.5) * stride_w
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        locations = torch.stack([xx, yy], dim=-1)  # [H, W, 2]

        total_cls_loss = 0
        total_reg_loss = 0
        total_ctr_loss = 0
        num_pos = 0

        for batch_idx in range(B):
            boxes = targets[batch_idx]['boxes']   # [N, 4] in (x1, y1, x2, y2)
            labels = targets[batch_idx]['labels'] # [N]

            if len(boxes) == 0:
                # No objects - all background
                cls_target = torch.zeros(H, W, dtype=torch.long, device=device)
                total_cls_loss += F.cross_entropy(
                    cls_logits[batch_idx].view(C, -1).t(),
                    cls_target.view(-1),
                    reduction='mean'
                )
                continue

            # Assign each location to a GT box (if inside any box)
            # [H, W, N] - check if location is inside each box
            in_box = (
                (locations[:, :, 0:1] >= boxes[:, 0]) &
                (locations[:, :, 0:1] <= boxes[:, 2]) &
                (locations[:, :, 1:2] >= boxes[:, 1]) &
                (locations[:, :, 1:2] <= boxes[:, 3])
            )  # [H, W, N]

            # Compute regression targets (l, t, r, b distances)
            l = locations[:, :, 0:1] - boxes[:, 0]  # [H, W, N]
            t = locations[:, :, 1:2] - boxes[:, 1]
            r = boxes[:, 2] - locations[:, :, 0:1]
            b = boxes[:, 3] - locations[:, :, 1:2]
            reg_targets = torch.stack([l, t, r, b], dim=-1)  # [H, W, N, 4]

            # For each location, pick the smallest box it's inside
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])  # [N]
            areas_expanded = areas.view(1, 1, -1).expand(H, W, -1)
            areas_expanded = torch.where(in_box, areas_expanded, torch.tensor(float('inf'), device=device))
            min_area_idx = areas_expanded.argmin(dim=-1)  # [H, W]

            # Gather targets
            is_pos = in_box.any(dim=-1)  # [H, W]
            cls_target = torch.zeros(H, W, dtype=torch.long, device=device)
            cls_target[is_pos] = labels[min_area_idx[is_pos].long()]

            # Regression targets for positive locations
            h_idx = torch.arange(H, device=device).view(-1, 1).expand(H, W)
            w_idx = torch.arange(W, device=device).view(1, -1).expand(H, W)
            reg_target = reg_targets[h_idx, w_idx, min_area_idx.long()]  # [H, W, 4]

            # Centerness target
            lr = reg_target[:, :, 0:1], reg_target[:, :, 2:3]
            tb = reg_target[:, :, 1:2], reg_target[:, :, 3:4]
            ctr_target = torch.sqrt(
                (torch.min(lr[0], lr[1]) / (torch.max(lr[0], lr[1]) + 1e-6)) *
                (torch.min(tb[0], tb[1]) / (torch.max(tb[0], tb[1]) + 1e-6))
            ).squeeze(-1)  # [H, W]

            # Classification loss (focal loss would be better, using CE for simplicity)
            total_cls_loss += F.cross_entropy(
                cls_logits[batch_idx].view(C, -1).t(),
                cls_target.view(-1),
                reduction='mean'
            )

            # Regression loss (only for positive locations)
            if is_pos.sum() > 0:
                pos_reg_pred = bbox_reg[batch_idx].permute(1, 2, 0)[is_pos]  # [num_pos, 4]
                pos_reg_target = reg_target[is_pos]  # [num_pos, 4]
                total_reg_loss += F.smooth_l1_loss(pos_reg_pred, pos_reg_target)

                pos_ctr_pred = centerness[batch_idx, 0][is_pos]  # [num_pos]
                pos_ctr_target = ctr_target[is_pos]
                total_ctr_loss += F.binary_cross_entropy_with_logits(
                    pos_ctr_pred, pos_ctr_target
                )
                num_pos += is_pos.sum().item()

        # Average over batch
        cls_loss = total_cls_loss / B
        reg_loss = total_reg_loss / max(1, B) if num_pos > 0 else torch.tensor(0., device=device)
        ctr_loss = total_ctr_loss / max(1, B) if num_pos > 0 else torch.tensor(0., device=device)

        return {
            'cls_loss': cls_loss,
            'reg_loss': reg_loss,
            'centerness_loss': ctr_loss,
            'total_loss': cls_loss + reg_loss + ctr_loss
        }

    @torch.no_grad()
    def predict(self, x: torch.Tensor, score_thresh: float = 0.3,
                nms_thresh: float = 0.5) -> List[Dict[str, torch.Tensor]]:
        """
        Run inference and return detected boxes.

        Args:
            x: [B, 3, H, W] input image
            score_thresh: minimum score to keep
            nms_thresh: NMS IoU threshold

        Returns:
            list of dicts with 'boxes', 'labels', 'scores' for each image
        """
        from torchvision.ops import nms

        self.eval()
        preds = self(x)

        cls_logits = preds['cls_logits']  # [B, C, H', W']
        bbox_reg = preds['bbox_reg']
        centerness = preds['centerness']

        B, C, H, W = cls_logits.shape
        input_h, input_w = x.shape[2:]
        stride_h = input_h / H
        stride_w = input_w / W

        # Create location grid
        y_coords = (torch.arange(H, device=x.device) + 0.5) * stride_h
        x_coords = (torch.arange(W, device=x.device) + 0.5) * stride_w
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

        results = []
        for b in range(B):
            cls_scores = cls_logits[b].sigmoid()  # [C, H, W]
            ctr_scores = centerness[b, 0].sigmoid()  # [H, W]

            # Combine class and centerness scores
            scores = cls_scores * ctr_scores.unsqueeze(0)  # [C, H, W]

            # Get predictions above threshold
            max_scores, max_labels = scores.max(dim=0)  # [H, W]
            mask = max_scores > score_thresh

            if mask.sum() == 0:
                results.append({
                    'boxes': torch.zeros(0, 4, device=x.device),
                    'labels': torch.zeros(0, dtype=torch.long, device=x.device),
                    'scores': torch.zeros(0, device=x.device)
                })
                continue

            # Get box predictions
            l, t, r, b_reg = bbox_reg[b]  # each [H, W]
            x1 = xx - l
            y1 = yy - t
            x2 = xx + r
            y2 = yy + b_reg

            boxes = torch.stack([x1, y1, x2, y2], dim=-1)  # [H, W, 4]

            # Filter by mask
            boxes_filtered = boxes[mask]
            scores_filtered = max_scores[mask]
            labels_filtered = max_labels[mask]

            # NMS
            keep = nms(boxes_filtered, scores_filtered, nms_thresh)

            results.append({
                'boxes': boxes_filtered[keep],
                'labels': labels_filtered[keep],
                'scores': scores_filtered[keep]
            })

        return results


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    # Test
    model = MPKNetDetector(num_classes=20, ch=48, use_stereo=True)
    print(f"MPKNetDetector params: {count_params(model)/1e6:.3f}M")

    # Test forward
    x = torch.randn(2, 3, 416, 416)
    preds = model(x)
    print(f"Input: {x.shape}")
    print(f"cls_logits: {preds['cls_logits'].shape}")
    print(f"bbox_reg: {preds['bbox_reg'].shape}")
    print(f"centerness: {preds['centerness'].shape}")

    # Test loss
    targets = [
        {'boxes': torch.tensor([[50, 50, 150, 150], [200, 200, 300, 300]]),
         'labels': torch.tensor([1, 5])},
        {'boxes': torch.tensor([[100, 100, 200, 200]]),
         'labels': torch.tensor([3])}
    ]
    losses = model.compute_loss(preds, targets, (416, 416))
    print(f"\nLosses:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")

    # Test inference
    results = model.predict(x)
    print(f"\nPredictions:")
    for i, r in enumerate(results):
        print(f"  Image {i}: {len(r['boxes'])} boxes")
