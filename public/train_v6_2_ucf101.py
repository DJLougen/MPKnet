#!/usr/bin/env python3
"""
Train V6.2 (temporal M-pathway) on UCF-101 video dataset.

Usage:
  python train_v6_2_ucf101.py --ucf101 --data_dir ~/data/ucf101_frames

Expects extracted frames in:
  ucf101_frames/
    ApplyEyeMakeup/
      v_ApplyEyeMakeup_g01_c01/
        frame_0001.jpg
        frame_0002.jpg
        ...
"""

import argparse
import os
import time
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from mpknet_v6 import BinocularMPKNetV6
from mpknet_components import count_params


# UCF-10 classes (subset of UCF-101)
UCF10_CLASSES = [
    'ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling', 'BalanceBeam',
    'BandMarching', 'BaseballPitch', 'Basketball', 'BasketballDunk', 'BenchPress'
]

# Full UCF-101 classes
UCF101_CLASSES = [
    'ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling', 'BalanceBeam',
    'BandMarching', 'BaseballPitch', 'Basketball', 'BasketballDunk', 'BenchPress',
    'Biking', 'Billiards', 'BlowDryHair', 'BlowingCandles', 'BodyWeightSquats',
    'Bowling', 'BoxingPunchingBag', 'BoxingSpeedBag', 'BreastStroke', 'BrushingTeeth',
    'CleanAndJerk', 'CliffDiving', 'CricketBowling', 'CricketShot', 'CuttingInKitchen',
    'Diving', 'Drumming', 'Fencing', 'FieldHockeyPenalty', 'FloorGymnastics',
    'FrisbeeCatch', 'FrontCrawl', 'GolfSwing', 'Haircut', 'Hammering',
    'HammerThrow', 'HandstandPushups', 'HandstandWalking', 'HeadMassage', 'HighJump',
    'HorseRace', 'HorseRiding', 'HulaHoop', 'IceDancing', 'JavelinThrow',
    'JugglingBalls', 'JumpingJack', 'JumpRope', 'Kayaking', 'Knitting',
    'LongJump', 'Lunges', 'MilitaryParade', 'Mixing', 'MoppingFloor',
    'Nunchucks', 'ParallelBars', 'PizzaTossing', 'PlayingCello', 'PlayingDaf',
    'PlayingDhol', 'PlayingFlute', 'PlayingGuitar', 'PlayingPiano', 'PlayingSitar',
    'PlayingTabla', 'PlayingViolin', 'PoleVault', 'PommelHorse', 'PullUps',
    'Punch', 'PushUps', 'Rafting', 'RockClimbingIndoor', 'RopeClimbing',
    'Rowing', 'SalsaSpin', 'ShavingBeard', 'Shotput', 'SkateBoarding',
    'Skiing', 'Skijet', 'SkyDiving', 'SoccerJuggling', 'SoccerPenalty',
    'StillRings', 'SumoWrestling', 'Surfing', 'Swing', 'TableTennisShot',
    'TaiChi', 'TennisSwing', 'ThrowDiscus', 'TrampolineJumping', 'Typing',
    'UnevenBars', 'VolleyballSpiking', 'WalkingWithDog', 'WallPushups', 'WritingOnBoard',
    'YoYo'
]


class UCFDataset(Dataset):
    """
    UCF dataset loader for extracted frames.

    Expects directory structure:
    ucf101_frames/
      ApplyEyeMakeup/
        v_ApplyEyeMakeup_g01_c01/
          frame_0001.jpg
          frame_0002.jpg
          ...
      ApplyLipstick/
        ...
    """
    def __init__(self, root_dir, split='train', num_frames=8, transform=None,
                 frame_sampling='uniform', train_ratio=0.8, seed=42, classes=None):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.transform = transform
        self.frame_sampling = frame_sampling

        # Use provided classes or default to UCF10
        self.classes = classes if classes is not None else UCF10_CLASSES
        self.samples = []  # List of (frame_paths, class_idx)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # Collect all videos for each class
        all_videos = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"Warning: class {class_name} not found at {class_dir}")
                continue

            class_idx = self.class_to_idx[class_name]

            # Each subfolder is a video
            for video_name in sorted(os.listdir(class_dir)):
                video_path = os.path.join(class_dir, video_name)
                if os.path.isdir(video_path):
                    frames = sorted([
                        os.path.join(video_path, f)
                        for f in os.listdir(video_path)
                        if f.endswith(('.jpg', '.png'))
                    ])
                    if len(frames) >= num_frames:
                        all_videos.append((frames, class_idx, class_name))

        # Split into train/test
        random.seed(seed)
        random.shuffle(all_videos)

        split_idx = int(len(all_videos) * train_ratio)
        if split == 'train':
            selected = all_videos[:split_idx]
        else:
            selected = all_videos[split_idx:]

        self.samples = [(frames, class_idx) for frames, class_idx, _ in selected]

        # Count per class
        class_counts = {}
        for _, class_idx, class_name in (all_videos[:split_idx] if split == 'train' else all_videos[split_idx:]):
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        print(f"Found {len(self.samples)} video sequences in {split} ({len(self.classes)} classes)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, label = self.samples[idx]

        # Sample frames
        if self.frame_sampling == 'uniform':
            # Uniformly sample num_frames across the video
            indices = torch.linspace(0, len(frame_paths) - 1, self.num_frames).long()
        elif self.frame_sampling == 'random':
            # Random consecutive segment
            max_start = max(0, len(frame_paths) - self.num_frames)
            start = random.randint(0, max_start)
            indices = list(range(start, min(start + self.num_frames, len(frame_paths))))
            # Pad if needed
            while len(indices) < self.num_frames:
                indices.append(indices[-1])
        else:
            # First num_frames
            indices = list(range(min(self.num_frames, len(frame_paths))))
            while len(indices) < self.num_frames:
                indices.append(indices[-1])

        # Load frames
        frames = []
        for i in indices:
            img = Image.open(frame_paths[i]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        frames = torch.stack(frames)  # [num_frames, C, H, W]
        current_frame = frames[-1]  # Last frame for P pathway

        return current_frame, frames, label


def train_epoch(model, loader, criterion, optimizer, device, use_temporal=True):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for current_frame, frames, labels in loader:
        current_frame = current_frame.to(device)
        frames = frames.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if use_temporal:
            outputs = model(current_frame, frames=frames)
        else:
            outputs = model(current_frame)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), 100.0 * correct / total


@torch.no_grad()
def test_model(model, loader, criterion, device, use_temporal=True):
    model.train(False)
    total_loss = 0
    correct = 0
    total = 0

    for current_frame, frames, labels in loader:
        current_frame = current_frame.to(device)
        frames = frames.to(device)
        labels = labels.to(device)

        if use_temporal:
            outputs = model(current_frame, frames=frames)
        else:
            outputs = model(current_frame)

        loss = criterion(outputs, labels)

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="v6.2", choices=["v6", "v6.2"])
    parser.add_argument("--data_dir", type=str, default="~/data/ucf101_frames")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--ch", type=int, default=48)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ucf101", action="store_true", help="Use all 101 classes instead of 10")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--start_epoch", type=int, default=0, help="Starting epoch (for resumed training)")
    parser.add_argument("--best_acc", type=float, default=0.0, help="Best accuracy so far (for resumed training)")
    args = parser.parse_args()

    # Expand path
    args.data_dir = os.path.expanduser(args.data_dir)

    # Select class list
    classes = UCF101_CLASSES if args.ucf101 else UCF10_CLASSES
    num_classes = len(classes)
    dataset_name = "UCF-101" if args.ucf101 else "UCF-10"

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((112, 112)),  # Standard for action recognition
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Datasets
    print(f"\nLoading {dataset_name} from {args.data_dir}...")
    train_dataset = UCFDataset(args.data_dir, 'train', args.num_frames, transform, 'random', classes=classes)
    test_dataset = UCFDataset(args.data_dir, 'test', args.num_frames, transform, 'uniform', classes=classes)

    if len(train_dataset) == 0:
        print("\nNo training data found!")
        print("Expected directory structure:")
        print("  ucf101_frames/")
        print("    ApplyEyeMakeup/")
        print("      v_ApplyEyeMakeup_g01_c01/")
        print("        frame_0001.jpg")
        print("        ...")
        return

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    use_temporal = (args.model == "v6.2")

    if args.model == "v6":
        model = BinocularMPKNetV6(num_classes=num_classes, ch=args.ch, use_stereo=True)
        name = "V6"
    else:
        from mpknet_v6_2_temporal import BinocularMPKNetV6_2
        model = BinocularMPKNetV6_2(num_classes=num_classes, ch=args.ch, use_stereo=True, num_frames=args.num_frames)
        name = "V6.2-Temporal"

    model = model.to(device)
    print(f"\n{name} params: {count_params(model)/1e6:.3f}M")

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nLoading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"Resumed from epoch {args.start_epoch} with best_acc {args.best_acc:.2f}%")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Fast-forward scheduler if resuming
    if args.start_epoch > 0:
        for _ in range(args.start_epoch):
            scheduler.step()

    print(f"\nTraining {name} on {dataset_name}")
    print(f"Frames per clip: {args.num_frames}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    print("=" * 60)

    best_acc = args.best_acc
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, use_temporal)
        test_loss, test_acc = test_model(model, test_loader, criterion, device, use_temporal)
        scheduler.step()

        is_best = test_acc > best_acc
        if is_best:
            best_acc = test_acc
            save_name = f"{args.model}_ucf101_best.pth" if args.ucf101 else f"{args.model}_ucf10_best.pth"
            torch.save(model.state_dict(), save_name)

        epoch_time = time.time() - t0
        lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train: {train_acc:.2f}% | Test: {test_acc:.2f}% | "
              f"LR: {lr:.4f} | Time: {epoch_time:.1f}s"
              f"{' *' if is_best else ''}")

    total_time = time.time() - start_time
    print("=" * 60)
    print(f"Done! Best test accuracy: {best_acc:.2f}%")
    print(f"Total time: {total_time/60:.1f} min")
    save_name = f"{args.model}_ucf101_best.pth" if args.ucf101 else f"{args.model}_ucf10_best.pth"
    print(f"Saved: {save_name}")


if __name__ == "__main__":
    main()
