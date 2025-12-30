# train_mpknet_sgd_swa.py

#-------------------------------
# Library imports
#-------------------------------
import os
import copy
import math
import random
import argparse
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="nolds.datasets")

from nolds import dfa
from pyts.image import RecurrencePlot
from tbLogger import TensorboardLogger # Custom Tensorboard logger with recurrence plot & fractal metrics
from modelData import load_dataset #Script containing dataet loading functions

# -------------------------------
# Lightweight params / FLOPs profiler (Conv2d + Linear)
# -------------------------------
def count_params(model: nn.Module) -> int: # total number of parameters
    return sum(p.numel() for p in model.parameters())

@torch.no_grad()
def profile_gflops(model: nn.Module, input_shape: Tuple[int, int, int, int], device: torch.device) -> float:
    """ 
    Returns FLOPs in billions (GFLOPs) for a single forward pass.
    Counts Conv2d and Linear layers. Ignores BN, activations, pooling, interpolate.
    """
    flops = 0.0
    handles = []

    def conv_hook(m: nn.Conv2d, inp, out):
        nonlocal flops
        x = inp[0]
        N = x.shape[0]
        Cout = m.out_channels
        Hout, Wout = out.shape[2], out.shape[3]
        kernel_ops = (m.kernel_size[0] * m.kernel_size[1]) * (m.in_channels // m.groups)
        output_elements = Cout * Hout * Wout
        macs = N * output_elements * kernel_ops
        flops += 2.0 * macs
        if m.bias is not None:
            flops += N * output_elements

    def linear_hook(m: nn.Linear, inp, out):
        nonlocal flops
        x = inp[0]
        N = x.shape[0]
        macs = N * m.in_features * m.out_features
        flops += 2.0 * macs
        if m.bias is not None:
            flops += N * m.out_features

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))

    model_was_training = model.training
    model.eval()
    dummy = torch.zeros(input_shape, device=device)
    _ = model(dummy)
    if model_was_training:
        model.train()

    for h in handles:
        h.remove()

    return float(flops) / 1e9

# -------------------------------
# Device setup (CUDA > MPS > CPU)
# -------------------------------
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using MPS (Apple Silicon)")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")

torch.set_float32_matmul_precision("high")

# -------------------------------
# Utils (unchanged, minor additions)
# -------------------------------
def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def gaussian_kernel2d(ks: int = 7, sigma: float = 1.2, device=None, dtype=torch.float32):
    ax = torch.arange(ks, device=device, dtype=dtype) - (ks - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    k = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return k / k.sum()

class FixedDepthwise(nn.Module):
    def __init__(self, kernel_2d: torch.Tensor, channels: int, padding: str = "same"):
        super().__init__()
        ks = kernel_2d.shape[0]
        weight = torch.zeros(channels, 1, ks, ks)
        weight[:, 0, :, :] = kernel_2d
        self.register_buffer("weight", weight)
        self.groups = channels
        self.ks = ks
        self.pad = padding

    def forward(self, x):
        pad = self.ks // 2 if self.pad == "same" else 0
        return F.conv2d(x, self.weight, bias=None, stride=1, padding=pad, groups=self.groups)

class PreMPK(nn.Module):
    def __init__(self, channels=3, lpf_ks=7, lpf_sigma=1.2, hpf_ks=7, hpf_sigma=1.0, hpf_alpha=1.0, magno_luminance=False):
        super().__init__()
        g_l = gaussian_kernel2d(lpf_ks, lpf_sigma)
        self.magno_lpf = FixedDepthwise(g_l, channels)

        g_p = gaussian_kernel2d(hpf_ks, hpf_sigma)
        delta = torch.zeros_like(g_p); delta[hpf_ks//2, hpf_ks//2] = 1.0
        hpf = delta - hpf_alpha * g_p
        hpf = (hpf - hpf.mean()) / (hpf.abs().sum() + 1e-8)
        self.parvo_hpf = FixedDepthwise(hpf, channels)

        self.magno_luminance = magno_luminance
        if magno_luminance:
            self.register_buffer("lum_w", torch.tensor([0.299, 0.587, 0.114]).view(1,3,1,1))

    def forward(self, x):
        x_mag = (x * self.lum_w).sum(1, keepdim=True).repeat(1, x.size(1), 1, 1) if self.magno_luminance else x
        return self.parvo_hpf(x), self.magno_lpf(x_mag)

# -------------------------------
# CellPop downsample block (3x3 structured)
# -------------------------------
class CellPopDownsample3x3(nn.Module):
    def __init__(self, cin: int, kout_per_channel: int = 4, cout: int | None = None,
                 bias: bool = False, use_bn: bool = True, act: str = "relu", downscale: int = 3):
        super().__init__()
        assert downscale in (2,3), "downscale must be 2 or 3"
        self.k = kout_per_channel
        self.cout = cout
        self.s = downscale

        self.pop = nn.Conv2d(cin * (self.s**2), cin * self.k, 1, groups=cin, bias=bias)
        self.bn1 = nn.BatchNorm2d(cin * self.k) if use_bn else nn.Identity()
        self.act1 = nn.ReLU6(inplace=True) if act.lower()=="relu6" else (nn.PReLU() if act.lower()=="prelu" else nn.ReLU(inplace=True))

        if cout is not None:
            self.mix = nn.Conv2d(cin * self.k, cout, 1, bias=bias)
            self.bn2 = nn.BatchNorm2d(cout) if use_bn else nn.Identity()
            self.act2 = nn.ReLU(inplace=True)
        else:
            self.mix = None
            self.bn2 = nn.Identity()
            self.act2 = nn.Identity()

    def forward(self, x):
        # pad to multiple of s so pixel_unshuffle doesn't crash
        B, C, H, W = x.shape
        s = self.s
        pad_h = (s - (H % s)) % s
        pad_w = (s - (W % s)) % s
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")  # [left,right,top,bottom]

        x = F.pixel_unshuffle(x, downscale_factor=s)  # [B, s^2*C, H/s, W/s]
        x = self.act1(self.bn1(self.pop(x)))
        if self.mix is not None:
            x = self.act2(self.bn2(self.mix(x)))

        # crop back to floor(H/s), floor(W/s) if we padded
        if pad_h or pad_w:
            Hout = H // s
            Wout = W // s
            x = x[..., :Hout, :Wout]
        return x


# -------------------------------
# Model (streams now accept stem stride)
# -------------------------------
class Magno(nn.Module):
    def __init__(self, in_ch=3, ch=64, first_stride=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_ch, ch, 7, first_stride, 3, bias=False), nn.BatchNorm2d(ch), nn.PReLU(),
            nn.Conv2d(ch, ch, 9, 2, 4, bias=False), nn.BatchNorm2d(ch), nn.PReLU(),
        )
    def forward(self, x): return self.layers(x)

class Parvo(nn.Module):
    def __init__(self, in_ch=3, ch=64, first_stride=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_ch, ch, 4, first_stride, 1, bias=False), nn.BatchNorm2d(ch), nn.PReLU(),
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False), nn.BatchNorm2d(ch), nn.PReLU(),
            nn.Conv2d(ch, ch, 2, 1, 1, bias=False), nn.BatchNorm2d(ch), nn.PReLU(),
            nn.Conv2d(ch, ch, 2, 1, 1, bias=False), nn.BatchNorm2d(ch), nn.PReLU()
        )
    def forward(self, x): return self.layers(x)

class Konio(nn.Module):
    def __init__(self, in_ch=3, ch=16, first_stride=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_ch, ch, 5, first_stride, 2, bias=True), nn.BatchNorm2d(ch), nn.PReLU(),
            nn.Conv2d(ch, ch, 5, 2, 2, bias=True), nn.BatchNorm2d(ch), nn.PReLU(),
            nn.Conv2d(ch, ch, 5, 1, 2, bias=True), nn.BatchNorm2d(ch), nn.PReLU(),
        )
    def forward(self, x): return self.layers(x)

class MPKNet(nn.Module):
    """
    MPKNet with Konio Gating mechanism.

    Konio pathway acts as a context relay that modulates the Parvo and Magno
    streams via learned channel attention gates. This implements the biological
    hypothesis that K cells carry modulatory/contextual signals that weight
    the relative contributions of P (detail) and M (gist) pathways.
    """
    def __init__(self, num_classes=100, ch=64, hid=128, use_prefilters=True,
                 use_cellpop=False, cellpop_k=4, parvo_stem=24, magno_stem=24, konio_stem=12,
                 cellpop_stride=3):
        super().__init__()
        self.pref = PreMPK() if use_prefilters else None
        self.use_cellpop = use_cellpop

        if use_cellpop:
            self.parvo_pop = CellPopDownsample3x3(cin=3, kout_per_channel=cellpop_k,
                                                  cout=parvo_stem, downscale=cellpop_stride, bias=False)
            self.magno_pop = CellPopDownsample3x3(cin=3, kout_per_channel=cellpop_k,
                                                  cout=magno_stem, downscale=cellpop_stride, bias=False)
            self.konio_pop = CellPopDownsample3x3(cin=3, kout_per_channel=cellpop_k,
                                                  cout=konio_stem, downscale=cellpop_stride, bias=False)

            # keep first convs at stride=1 since CellPop already downsamples
            self.parvo = Parvo(in_ch=parvo_stem, ch=ch, first_stride=1)
            self.magno = Magno(in_ch=magno_stem, ch=ch, first_stride=1)
            self.konio = Konio(in_ch=konio_stem, ch=ch//4, first_stride=1)
        else:
            self.parvo = Parvo(3, ch, first_stride=2)
            self.magno = Magno(3, ch, first_stride=2)
            self.konio = Konio(3, ch//4, first_stride=2)

        # Konio gating mechanism: K generates channel attention for P and M
        # K output is ch//4 (16), we project to ch (64) for each gate
        k_ch = ch // 4  # Konio output channels
        self.k_gate_P = nn.Sequential(
            nn.Linear(k_ch, ch),
            nn.Sigmoid()
        )
        self.k_gate_M = nn.Sequential(
            nn.Linear(k_ch, ch),
            nn.Sigmoid()
        )

        # Fuse: now only P + M since K is used for gating (not concatenated)
        self.fuse = nn.Sequential(
            nn.Conv2d(ch * 2, hid, 1, bias=False), nn.BatchNorm2d(hid), nn.ReLU(inplace=True)
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(hid, num_classes)
        )

    def forward(self, x):
        x_par, x_mag = self.pref(x) if self.pref else (x, x)

        if self.use_cellpop:
            # 3x3 structured downsample per pathway; spatial dims reduced by /s
            xP = self.parvo_pop(x_par)
            xM = self.magno_pop(x_mag)
            xK = self.konio_pop(x)
        else:
            xP, xM, xK = x_par, x_mag, x

        P, M, K = self.parvo(xP), self.magno(xM), self.konio(xK)

        # Align spatial sizes before gating
        target_hw = P.shape[2:]
        if M.shape[2:] != target_hw:
            M = F.interpolate(M, size=target_hw, mode="nearest")
        if K.shape[2:] != target_hw:
            K = F.interpolate(K, size=target_hw, mode="nearest")

        # Konio generates gates for P and M
        K_attn = F.adaptive_avg_pool2d(K, 1).flatten(1)  # [B, k_ch] - global context
        gate_P = self.k_gate_P(K_attn).unsqueeze(-1).unsqueeze(-1)  # [B, ch, 1, 1]
        gate_M = self.k_gate_M(K_attn).unsqueeze(-1).unsqueeze(-1)  # [B, ch, 1, 1]

        # Modulate P and M with Konio-derived gates
        P = P * gate_P
        M = M * gate_M

        # Fuse gated P and M (K is consumed in gating, not concatenated)
        z = self.fuse(torch.cat([P, M], dim=1))
        return self.head(z)

# -------------------------------
# EMA helper
# -------------------------------
class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)
    def update(self, model):
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if k in msd:
                    v.copy_(v * self.decay + msd[k].detach() * (1.0 - self.decay))

# -------------------------------
# Mixup
# -------------------------------
def mixup_data(x, y, alpha=0.2, device='cpu'):
    if alpha <= 0:
        return x, y, None, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b,)

# -------------------------------
# Train / Eval (modified to accept mixup)
# -------------------------------
def train_one_epoch(model, loader, optimizer, device, mixup_alpha=0.0, criterion=None, ema: ModelEMA=None):
    model.train()
    total, correct, total_loss = 0, 0, 0.0
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        atype = "cpu" if device.type == "mps" else device.type
        with torch.autocast(atype):
            if mixup_alpha > 0:
                x, y_a, y_b, lam = mixup_data(x, y, alpha=mixup_alpha, device=device)
                logits = model(x)
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            else:
                logits = model(x)
                loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(1)
        if mixup_alpha > 0:
            # rough proxy to keep curve shape; do not trust absolute value
            correct += (preds == y).sum().item()
        else:
            correct += (preds == y).sum().item()

        total += y.size(0)
        total_loss += loss.item() * y.size(0)

        if ema is not None:
            ema.update(model)

    return correct / total, total_loss / total

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, correct = 0, 0
    pred_trace = []
    for x, y in tqdm(loader, desc="eval", leave=False):
        x, y = x.to(device), y.to(device)
        atype = "cpu" if device.type == "mps" else device.type
        with torch.autocast(atype):
            logits = model(x)
        preds = logits.argmax(1)
        pred_trace.extend(preds.cpu().numpy().tolist())
        correct += (preds == y).sum().item()
        total += y.size(0)
    acc = correct / total
    return acc, pred_trace

def plot_recurrence(pred_trace, writer, epoch):
    rp = RecurrencePlot(threshold='point', percentage=20)
    img = rp.fit_transform([pred_trace])[0]
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='binary', origin='lower')
    ax.set_title("Recurrence Plot")
    writer.add_figure("Recurrence", fig, epoch)
    plt.close(fig)

# -------------------------------
# Main (SGD + warmup + SWA)
# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--no_aug", action="store_true")
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--dataset", type=str, default="CIFAR100")
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--mixup", type=float, default=0.2)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--swa", action="store_true")
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--swa_lr", type=float, default=0.05)

    # CellPop args
    parser.add_argument("--use_cellpop", action="store_true",
                        help="Enable 3x3 structured CellPop downsampling stems for P/M/K")
    parser.add_argument("--cellpop_k", type=int, default=4,
                        help="Outputs per input channel after 9->k compression")
    parser.add_argument("--parvo_stem", type=int, default=24,
                        help="Parvo CellPop output channels")
    parser.add_argument("--magno_stem", type=int, default=24,
                        help="Magno CellPop output channels")
    parser.add_argument("--konio_stem", type=int, default=12,
                        help="Konio CellPop output channels")
    # in main() after other CellPop args
    parser.add_argument("--cellpop_stride", type=int, default=3, choices=[2,3],
                        help="Downscale factor used by CellPop (pixel_unshuffle). Use 2 for 32/64 images.")


    args = parser.parse_args()

    set_seed(args.seed)

    # Transforms
    if args.no_aug:
        train_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
    else:
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.RandAugment(num_ops=2, magnitude=10),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

    val_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    # Load data (your load_dataset should accept transforms or handle them internally)
    train_loader, val_loader, num_classes = load_dataset(
        name=args.dataset,
        batch_size=args.batch,
        root=args.data_root,
        img_size=args.img_size
    )

    model = MPKNet(
        num_classes=num_classes,
        ch=64,
        hid=128,
        use_prefilters=True,
        use_cellpop=args.use_cellpop,
        cellpop_k=args.cellpop_k,
        parvo_stem=args.parvo_stem,
        magno_stem=args.magno_stem,
        konio_stem=args.konio_stem,
        cellpop_stride=args.cellpop_stride
    ).to(DEVICE)

    # --- Report params and FLOPs (per 1 image of size img_size x img_size) ---
    total_params = count_params(model)
    gflops = profile_gflops(model, (1, 3, args.img_size, args.img_size), DEVICE)
    print(f"Params: {total_params/1e6:.3f}M")
    print(f"Model FLOPs: {gflops:.3f} GFLOPs (per {args.img_size}x{args.img_size} image, batch=1)")

    # optimizer: SGD
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.weight_decay)

    # learning-rate schedule handled manually per epoch (warmup + cosine)
    epochs = args.epochs
    warmup_epochs = args.warmup_epochs
    def set_lr(optimizer, epoch):
        if epoch <= warmup_epochs:
            lr = args.lr * (0.01 + 0.99 * (epoch / float(max(1, warmup_epochs))))
        else:
            t = (epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
            lr = 0.5 * args.lr * (1.0 + math.cos(math.pi * t))
        for g in optimizer.param_groups:
            g['lr'] = lr
        return lr

    # SWA setup
    swa_start = int(0.525 * epochs)
    if args.swa:
        from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=args.swa_lr)
    else:
        swa_model = None

    # EMA
    ema = ModelEMA(model) if args.ema else None

    # loss -- label smoothing included
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    run_name = f"{model.__class__.__name__}_{args.dataset.upper()}_seed{args.seed}_run{args.run}_SGD_SWA{args.swa}_CP{args.use_cellpop}"
    logger = TensorboardLogger(model_name=run_name)

    best_acc = 0.0
    best_ckpt = None
    best_ema_acc = 0.0
    best_swa_acc = 0.0

    for epoch in range(1, epochs + 1):
        lr = set_lr(optimizer, epoch)

        tr_acc, tr_loss = train_one_epoch(model, train_loader, optimizer, DEVICE, mixup_alpha=args.mixup, criterion=criterion, ema=ema)

        # Evaluate model (FP32)
        val_acc, pred_trace = evaluate(model, val_loader, DEVICE)
        dfa_val = dfa(np.array(pred_trace, dtype=np.float32))

        # EMA snapshot
        if ema is not None:
            ema_acc, _ = evaluate(ema.ema, val_loader, DEVICE)
        else:
            ema_acc = 0.0

        # SWA update
        if args.swa and epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            swa_acc, _ = evaluate(swa_model.module if hasattr(swa_model, "module") else swa_model, val_loader, DEVICE)
        else:
            swa_acc = 0.0

        # Tensorboard & logging
        try:
            hurst_val, divider_dim = TensorboardLogger.compute_fractal_metrics(np.asarray(pred_trace, dtype=np.float64))
        except Exception:
            hurst_val, divider_dim = 0.0, 0.0

        logger.log_scalar("Train/Accuracy", float(tr_acc), epoch)
        logger.log_scalar("Train/Loss",     float(tr_loss), epoch)
        logger.log_scalar("Eval/Accuracy",  float(val_acc), epoch)
        logger.log_scalar("Eval/DFA",       float(dfa_val), epoch)
        logger.log_scalar("Eval/Hurst",     float(hurst_val), epoch)
        logger.log_scalar("Eval/DividerDim",float(divider_dim), epoch)
        logger.log_scalar("LR", float(lr), epoch)
        if ema is not None:
            logger.log_scalar("Eval/EMA_Accuracy", float(ema_acc), epoch)
        if args.swa:
            logger.log_scalar("Eval/SWA_Accuracy", float(swa_acc), epoch)

        if epoch % 5 == 0:
            logger.log_recurrence_plot(pred_trace, epoch)

        print(f"[Epoch {epoch:03d}] lr {lr:.4e} | Train Acc: {tr_acc:.4f} | Train Loss: {tr_loss:.4f} | Val Acc: {val_acc:.4f} | DFA: {dfa_val:.4f} | EMA: {ema_acc:.4f} | SWA: {swa_acc:.4f}")

        # Save best FP32 checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_acc": val_acc
            }, f"checkpoints/best_{run_name}.pt")

        # Save best EMA checkpoint
        if ema is not None and ema_acc > best_ema_acc:
            best_ema_acc = ema_acc
            torch.save({
                "epoch": epoch,
                "model": ema.ema.state_dict(),
                "val_acc": ema_acc
            }, f"checkpoints/best_ema_{run_name}.pt")

        # Save SWA model interim (optional)
        if args.swa and swa_acc > best_swa_acc:
            best_swa_acc = swa_acc
            torch.save({
                "epoch": epoch,
                "model": swa_model.state_dict(),
                "val_acc": swa_acc
            }, f"checkpoints/best_swa_{run_name}.pt")

    # AFTER training: finalize SWA (update BN then evaluate & save)
    if args.swa:
        print("Finalizing SWA: updating BatchNorm statistics...")
        from torch.optim.swa_utils import update_bn
        update_bn(train_loader, swa_model, device=DEVICE)
        swa_final_acc, _ = evaluate(swa_model.module if hasattr(swa_model, "module") else swa_model, val_loader, DEVICE)
        torch.save({
            "epoch": epochs,
            "model": swa_model.state_dict(),
            "val_acc": swa_final_acc
        }, f"checkpoints/final_swa_{run_name}.pt")
        print(f"SWA final val acc: {swa_final_acc:.4f}")

    logger.close()
    print(f"Best Val Acc: {best_acc:.4f} | Best EMA: {best_ema_acc:.4f} | Best SWA (interim): {best_swa_acc:.4f}")

if __name__ == "__main__":
    main()

    # Optional: quick latency test on a random batch
    import time, torch
    from mpkSGD import MPKNet  # or however you import your model

    DEVICE = torch.device("mps")
    model = MPKNet(num_classes=200, use_cellpop=True, cellpop_stride=3).to(DEVICE)
    model.eval()
    x = torch.randn(64, 3, 224, 224, device=DEVICE)

    # warmup
    for _ in range(10):
        _ = model(x)
    torch.mps.synchronize()

    # measure
    t0 = time.time()
    for _ in range(50):
        _ = model(x)
    torch.mps.synchronize()
    print("avg ms:", (time.time() - t0) * 1000 / 50)
