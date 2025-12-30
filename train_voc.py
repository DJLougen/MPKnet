# train_voc.py
# Training script for MPKNetDetector on PASCAL VOC
# Anchor-free object detection with BinocularMPKNet backbone

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection
from tqdm import tqdm
import argparse
import os
from mpknet_detection import MPKNetDetector, count_params

# Device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using MPS (Apple Silicon)")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")

# VOC class names (background is 0)
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
CLASS_TO_IDX = {c: i + 1 for i, c in enumerate(VOC_CLASSES)}  # 1-indexed, 0 is background


def parse_voc_annotation(annotation):
    """Parse VOC XML annotation to boxes and labels."""
    objects = annotation['annotation'].get('object', [])
    if not isinstance(objects, list):
        objects = [objects]

    boxes = []
    labels = []

    for obj in objects:
        if obj.get('difficult', '0') == '1':
            continue

        name = obj['name']
        if name not in CLASS_TO_IDX:
            continue

        bbox = obj['bndbox']
        x1 = float(bbox['xmin'])
        y1 = float(bbox['ymin'])
        x2 = float(bbox['xmax'])
        y2 = float(bbox['ymax'])

        boxes.append([x1, y1, x2, y2])
        labels.append(CLASS_TO_IDX[name])

    return boxes, labels


class VOCDataset(torch.utils.data.Dataset):
    """PASCAL VOC dataset wrapper for detection."""

    def __init__(self, root: str, year: str = '2007', image_set: str = 'train',
                 img_size: int = 416, augment: bool = False):
        self.voc = VOCDetection(root=root, year=year, image_set=image_set, download=True)
        self.img_size = img_size
        self.augment = augment

        # Transforms
        if augment:
            self.transform = T.Compose([
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.voc)

    def __getitem__(self, idx):
        img, annotation = self.voc[idx]

        # Original size
        orig_w, orig_h = img.size

        # Resize image
        img = img.resize((self.img_size, self.img_size))
        img = self.transform(img)

        # Parse and scale boxes
        boxes, labels = parse_voc_annotation(annotation)

        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            # Scale boxes to new size
            boxes[:, [0, 2]] *= self.img_size / orig_w
            boxes[:, [1, 3]] *= self.img_size / orig_h
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            boxes = torch.zeros(0, 4, dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.long)

        # Random horizontal flip during training
        if self.augment and torch.rand(1) > 0.5:
            img = torch.flip(img, dims=[2])
            if len(boxes) > 0:
                boxes[:, [0, 2]] = self.img_size - boxes[:, [2, 0]]

        return img, {'boxes': boxes, 'labels': labels}


def collate_fn(batch):
    """Custom collate for variable number of boxes."""
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets


def compute_iou(boxes1, boxes2):
    """Compute IoU between two sets of boxes."""
    x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-6)


@torch.no_grad()
def evaluate(model, loader, iou_thresh=0.5):
    """Compute mAP on validation set."""
    model.eval()

    all_predictions = []  # (image_idx, class, score, box)
    all_targets = []      # (image_idx, class, box)

    for batch_idx, (images, targets) in enumerate(tqdm(loader, desc="eval", leave=False)):
        images = images.to(DEVICE)

        # Get predictions
        results = model.predict(images, score_thresh=0.01, nms_thresh=0.5)

        for i, (result, target) in enumerate(zip(results, targets)):
            img_idx = batch_idx * loader.batch_size + i

            # Store predictions
            for box, label, score in zip(result['boxes'], result['labels'], result['scores']):
                all_predictions.append({
                    'image_idx': img_idx,
                    'class': label.item(),
                    'score': score.item(),
                    'box': box.cpu()
                })

            # Store targets
            for box, label in zip(target['boxes'], target['labels']):
                all_targets.append({
                    'image_idx': img_idx,
                    'class': label.item(),
                    'box': box
                })

    if len(all_predictions) == 0 or len(all_targets) == 0:
        return 0.0

    # Compute AP per class
    aps = []
    for cls in range(1, len(VOC_CLASSES) + 1):
        # Get predictions and targets for this class
        cls_preds = [p for p in all_predictions if p['class'] == cls]
        cls_targets = [t for t in all_targets if t['class'] == cls]

        if len(cls_targets) == 0:
            continue

        if len(cls_preds) == 0:
            aps.append(0.0)
            continue

        # Sort predictions by score
        cls_preds.sort(key=lambda x: x['score'], reverse=True)

        # Track which targets have been matched
        target_matched = [False] * len(cls_targets)

        tp = []
        fp = []

        for pred in cls_preds:
            # Find matching target
            best_iou = 0
            best_idx = -1

            for t_idx, target in enumerate(cls_targets):
                if target['image_idx'] != pred['image_idx']:
                    continue
                if target_matched[t_idx]:
                    continue

                iou = compute_iou(
                    pred['box'].unsqueeze(0),
                    target['box'].unsqueeze(0)
                )[0, 0].item()

                if iou > best_iou:
                    best_iou = iou
                    best_idx = t_idx

            if best_iou >= iou_thresh and best_idx >= 0:
                tp.append(1)
                fp.append(0)
                target_matched[best_idx] = True
            else:
                tp.append(0)
                fp.append(1)

        # Compute precision-recall
        tp = torch.tensor(tp).cumsum(0)
        fp = torch.tensor(fp).cumsum(0)

        recall = tp / len(cls_targets)
        precision = tp / (tp + fp)

        # AP using 11-point interpolation
        ap = 0
        for r_thresh in torch.linspace(0, 1, 11):
            mask = recall >= r_thresh
            if mask.sum() > 0:
                ap += precision[mask].max().item() / 11

        aps.append(ap)

    return sum(aps) / len(aps) if aps else 0.0


def train_epoch(model, loader, optimizer, img_size):
    """Train one epoch."""
    model.train()
    total_loss = 0
    total_cls = 0
    total_reg = 0
    total_ctr = 0

    pbar = tqdm(loader, desc="train", leave=False)
    for images, targets in pbar:
        images = images.to(DEVICE)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        preds = model(images)
        losses = model.compute_loss(preds, targets, (img_size, img_size))

        losses['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss += losses['total_loss'].item()
        total_cls += losses['cls_loss'].item()
        total_reg += losses['reg_loss'].item()
        total_ctr += losses['centerness_loss'].item()

        pbar.set_postfix(loss=f"{losses['total_loss'].item():.2f}")

    n = len(loader)
    return {
        'total': total_loss / n,
        'cls': total_cls / n,
        'reg': total_reg / n,
        'ctr': total_ctr / n
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Path to VOC data")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--img_size", type=int, default=416)
    parser.add_argument("--ch", type=int, default=48)
    parser.add_argument("--use_stereo", action="store_true", default=True)
    parser.add_argument("--no_stereo", action="store_true")
    parser.add_argument("--disparity", type=int, default=2)
    parser.add_argument("--monocular_ratio", type=float, default=0.5)
    parser.add_argument("--augment", action="store_true")
    args = parser.parse_args()

    if args.no_stereo:
        args.use_stereo = False

    # Datasets
    print("Loading PASCAL VOC 2007...")
    train_ds = VOCDataset(args.data_dir, year='2007', image_set='train',
                          img_size=args.img_size, augment=args.augment)
    val_ds = VOCDataset(args.data_dir, year='2007', image_set='val',
                        img_size=args.img_size, augment=False)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=4, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=4, pin_memory=True, collate_fn=collate_fn)

    # Model
    model = MPKNetDetector(
        num_classes=len(VOC_CLASSES) + 1,  # +1 for background
        ch=args.ch,
        use_stereo=args.use_stereo,
        disparity_range=args.disparity,
        monocular_ratio=args.monocular_ratio
    ).to(DEVICE)

    print(f"\nMPKNetDetector: {count_params(model)/1e6:.3f}M params")
    print(f"Image size: {args.img_size}x{args.img_size}")
    print(f"Stereo: {args.use_stereo}, Disparity: {args.disparity}")
    print(f"Augmentation: {args.augment}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_map = 0
    save_name = "mpknet_voc_best.pth"

    for epoch in range(1, args.epochs + 1):
        train_losses = train_epoch(model, train_loader, optimizer, args.img_size)

        # Evaluate every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            mAP = evaluate(model, val_loader)
        else:
            mAP = -1

        scheduler.step()

        if mAP > best_map:
            best_map = mAP
            torch.save(model.state_dict(), save_name)

        lr = optimizer.param_groups[0]['lr']
        if mAP >= 0:
            print(f"[Epoch {epoch:03d}] lr {lr:.4e} | Loss: {train_losses['total']:.3f} "
                  f"(cls={train_losses['cls']:.3f}, reg={train_losses['reg']:.3f}) | "
                  f"mAP: {mAP*100:.2f}% | Best: {best_map*100:.2f}%")
        else:
            print(f"[Epoch {epoch:03d}] lr {lr:.4e} | Loss: {train_losses['total']:.3f} "
                  f"(cls={train_losses['cls']:.3f}, reg={train_losses['reg']:.3f})")

    print(f"\n{'='*60}")
    print(f"FINAL: MPKNetDetector on PASCAL VOC 2007")
    print(f"Params: {count_params(model)/1e6:.3f}M")
    print(f"Best mAP@0.5: {best_map*100:.2f}%")
    print(f"Model saved to: {save_name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
