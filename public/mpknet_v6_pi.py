"""
MPKNet V6-Pi - Raspberry Pi 5 optimized version with INT8 quantization.

This is the extreme edge variant: 15.5K parameters, 76KB model, 58 FPS on RPi5.

Results:
- 76% on Kvasir-v2 (medical imaging)
- 82% on CIFAR-10
- 58 FPS forward pass / 33 FPS end-to-end on Raspberry Pi 5

Designed for:
1. Real-time inference on $35 Raspberry Pi
2. Deployment where larger models cannot fit
3. Privacy-preserving local inference (no cloud needed)

Optimizations:
- Reduced channels (ch=24 instead of 48)
- Kernel size 3 (vs 5) for speed
- QuantStub/DeQuantStub for INT8 quantization
- Fibonacci strides preserved: P=2, K=3, M=5

Trade-off: V6-Pi has 161× fewer parameters than MobileNetV3-S but lower accuracy.
The value is enabling inference on hardware where larger models cannot run.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub


class PiMonocularBlock(nn.Module):
    """Lightweight conv block for Pi."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, groups=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride,
                              padding=padding, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class BinocularMPKNetV6Pi(nn.Module):
    """
    V6-Pi: Raspberry Pi optimized MPKNet with INT8 quantization support.

    Key changes from V6:
    - Smaller channel count (24 vs 48)
    - Kernel size 3 (vs 5) for speed
    - QuantStub/DeQuantStub for INT8 quantization
    - Fibonacci strides preserved: P=2, K=3, M=5

    ~15K params FP32, ~4K params INT8
    """
    def __init__(self, num_classes: int = 10, ch: int = 24,
                 kernel_size: int = 3, use_depthwise: bool = False):
        super().__init__()

        self.ch = ch
        self.kernel_size = kernel_size
        self.use_depthwise = use_depthwise

        # Quantization stubs
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        # Fibonacci strides
        self.p_stride = 2
        self.k_stride = 3
        self.m_stride = 5

        # Simple center-surround preprocessing (no learnable params)
        self.blur = nn.AvgPool2d(3, stride=1, padding=1)

        # ========== P PATHWAY ==========
        # stride=2 for detail without noise
        self.P_block1 = PiMonocularBlock(3, ch, kernel_size, stride=self.p_stride)
        self.P_block2 = PiMonocularBlock(ch, ch, kernel_size, stride=1)

        # ========== K PATHWAY ==========
        # stride=3 for context
        self.K_block1 = PiMonocularBlock(3, ch // 2, kernel_size, stride=self.k_stride)
        self.K_block2 = PiMonocularBlock(ch // 2, ch // 2, kernel_size, stride=1)

        # ========== M PATHWAY ==========
        # stride=5 for global gist
        if use_depthwise:
            # Depthwise separable for efficiency
            self.M_block1 = nn.Sequential(
                nn.Conv2d(3, 3, kernel_size, stride=self.m_stride,
                          padding=kernel_size//2, groups=3, bias=False),
                nn.Conv2d(3, ch, 1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True)
            )
        else:
            self.M_block1 = PiMonocularBlock(3, ch, kernel_size, stride=self.m_stride)
        self.M_block2 = PiMonocularBlock(ch, ch, kernel_size, stride=1)

        # ========== K GATES ==========
        # Lightweight gates
        self.k_gate_M = nn.Sequential(
            nn.Linear(ch // 2, ch),
            nn.Sigmoid()
        )
        self.k_gate_P = nn.Sequential(
            nn.Linear(ch // 2, ch),
            nn.Sigmoid()
        )

        # ========== FUSION ==========
        # 1x1 conv to combine M + P
        self.fusion = nn.Sequential(
            nn.Conv2d(ch * 2, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True)
        )

        # Classification
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simple preprocessing BEFORE quantization (not quantizable ops)
        blurred = self.blur(x)
        P_input = x - blurred  # High-freq detail
        M_input = blurred  # Low-freq gist

        # Quantize after preprocessing
        P_input = self.quant(P_input)
        M_input = self.quant(M_input)

        # P pathway
        P = self.P_block1(P_input)
        P = self.P_block2(P)

        # K pathway (for gating)
        K = self.K_block1(P_input)
        K = self.K_block2(K)

        # M pathway
        M = self.M_block1(M_input)
        M = self.M_block2(M)

        # K gating
        k_ctx = self.gap(K).flatten(1)
        gate_M = self.k_gate_M(k_ctx).unsqueeze(-1).unsqueeze(-1)
        gate_P = self.k_gate_P(k_ctx).unsqueeze(-1).unsqueeze(-1)

        M = M * gate_M
        P = P * gate_P

        # Match sizes for fusion (use interpolate for MPS compatibility)
        target_size = M.shape[-1]
        if P.shape[-1] != target_size:
            P = F.interpolate(P, size=(target_size, target_size), mode='bilinear', align_corners=False)

        # Fusion
        z = torch.cat([M, P], dim=1)
        z = self.fusion(z)

        # Classification
        z = self.gap(z).flatten(1)
        z = self.fc(z)

        # Dequantize output
        z = self.dequant(z)
        return z

    def fuse_model(self):
        """Fuse Conv-BN-ReLU for quantization."""
        for module in [self.P_block1, self.P_block2,
                       self.K_block1, self.K_block2,
                       self.M_block2]:
            torch.quantization.fuse_modules(
                module, ['conv', 'bn', 'relu'], inplace=True
            )
        # M_block1 may be depthwise separable, handle separately
        if not self.use_depthwise:
            torch.quantization.fuse_modules(
                self.M_block1, ['conv', 'bn', 'relu'], inplace=True
            )


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def quantize_model(model, calibration_loader=None):
    """
    Quantize model to INT8 for deployment.

    Args:
        model: trained FP32 model
        calibration_loader: DataLoader for calibration (optional)

    Returns:
        INT8 quantized model
    """
    model.eval()

    # Fuse Conv-BN-ReLU
    model.fuse_model()

    # Set quantization config for ARM CPUs (QNNPACK backend)
    model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    torch.backends.quantized.engine = 'qnnpack'

    # Prepare for quantization
    torch.quantization.prepare(model, inplace=True)

    # Calibrate if loader provided
    if calibration_loader is not None:
        with torch.no_grad():
            for x, _ in calibration_loader:
                model(x)
                break  # Single batch usually sufficient

    # Convert to INT8
    torch.quantization.convert(model, inplace=True)

    return model


def get_model_size(model):
    """Get model size in bytes."""
    import io
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell()


if __name__ == "__main__":
    # Test model
    model = BinocularMPKNetV6Pi(num_classes=10, ch=24, kernel_size=3)
    print(f"V6-Pi params: {count_params(model)/1e3:.2f}K")
    print(f"V6-Pi FP32 size: {get_model_size(model)/1024:.1f}KB")

    # Test on CIFAR size
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")

    # Test on Kvasir size
    x_kvasir = torch.randn(1, 3, 224, 224)
    y_kvasir = model(x_kvasir)
    print(f"Input: {x_kvasir.shape}, Output: {y_kvasir.shape}")

    # FLOPs estimate for 32x32
    def conv_flops(in_ch, out_ch, k, h, w, s=1, groups=1):
        out_h, out_w = h // s, w // s
        return 2 * (in_ch // groups) * out_ch * k * k * out_h * out_w

    H, W = 32, 32
    ch = 24
    k = 3

    p_flops = conv_flops(3, ch, k, H, W, 2) + conv_flops(ch, ch, k, H//2, W//2, 1)
    k_flops = conv_flops(3, ch//2, k, H, W, 3) + conv_flops(ch//2, ch//2, k, H//3, W//3, 1)
    m_flops = conv_flops(3, ch, k, H, W, 5) + conv_flops(ch, ch, k, H//5, W//5, 1)
    fusion_flops = 2 * (ch*2) * ch * 1 * 1 * (H//5) * (H//5)

    total_flops = p_flops + k_flops + m_flops + fusion_flops
    print(f"Estimated FLOPs on 32x32: {total_flops/1e6:.2f}M")

    # INT8 quantization info
    print("\n--- INT8 Quantization for Pi ---")
    print(f"FP32 size: {get_model_size(model)/1024:.1f}KB")
    print(f"Expected INT8 size: ~{get_model_size(model)/1024/4:.1f}KB")
    print("\nTo deploy INT8 on Raspberry Pi:")
    print("1. Export to ONNX: torch.onnx.export(model, x, 'v6_pi.onnx')")
    print("2. Quantize with onnxruntime: python -m onnxruntime.quantization.preprocess ...")
    print("3. Run with: ort_session = onnxruntime.InferenceSession('v6_pi_int8.onnx')")
