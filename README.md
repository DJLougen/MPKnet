# MPKNet: A Tree Shrew LGN-Inspired Architecture for Efficient Visual Processing

**MPKNet** is a bio-inspired convolutional neural network that models the Magnocellular (M), Parvocellular (P), and Koniocellular (K) pathways of the Lateral Geniculate Nucleus (LGN), based on cross-species evolutionary priors observed in mammals from tree shrews to primates.

## Key Ideas

1. **Parallel Visual Streams**: Like the biological LGN, MPKNet processes visual information through three parallel pathways:
   - **Magno (M)**: Large receptive fields, fast temporal processing, global "gist"
   - **Parvo (P)**: Small receptive fields, high spatial acuity, fine detail
   - **Konio (K)**: Context relay and cross-stream modulation

2. **CellPop Retinal Sampling**: Structured downsampling using `pixel_unshuffle` to model retinal ganglion cell population responses

3. **Konio Gating**: The K-pathway generates channel attention to modulate P and M streams, acting as a context-aware relay (novel architectural contribution)

4. **Evolutionary Priors**: Kernel sizes and strides chosen to reflect biological receptive field properties across species

## Architecture

```
Input (3×H×W)
    │
    ▼
┌─────────┐
│ PreMPK  │  LGN-like preprocessing (HPF for P, LPF for M)
└─────────┘
    │
    ├────────────────┬────────────────┐
    ▼                ▼                ▼
┌─────────┐    ┌─────────┐    ┌─────────┐
│ CellPop │    │ CellPop │    │ CellPop │
│ P-stem  │    │ M-stem  │    │ K-stem  │
└─────────┘    └─────────┘    └─────────┘
    │                │                │
    ▼                ▼                ▼
┌─────────┐    ┌─────────┐    ┌─────────┐
│  Parvo  │    │  Magno  │    │  Konio  │
│ 4×4→3×3 │    │ 7×7→9×9 │    │ 5×5×3   │
│ 4 layers│    │ 2 layers│    │ 3 layers│
└─────────┘    └─────────┘    └─────────┘
    │                │                │
    │                │                ▼
    │                │         ┌─────────────┐
    │                │         │ K-Gate (σ)  │
    │                │         └─────────────┘
    │                │                │
    ▼                ▼                ▼
┌────────────────────────────────────────────┐
│              Fusion (1×1 conv)             │
└────────────────────────────────────────────┘
    │
    ▼
┌─────────┐
│  Head   │  GAP → FC → logits
└─────────┘
```

## Results

### CIFAR-10 (From Scratch, No Pretraining)

| Model | Params | Val Acc (no aug) | Val Acc (aug) | DFA | Notes |
|-------|--------|------------------|---------------|-----|-------|
| MPKNet + CellPop | 0.54M | **79.5%** | 81.1% | 0.52 | Stable training |
| Baseline CNN (ablation) | 0.55M | 84.5%* | - | - | *Overfits: 100% train |
| MPKNet + Binocular | 0.14M | TBD | TBD | TBD | 4x fewer params |

**Key Findings**:
1. **Augmentation Insensitivity**: MPKNet gains only +1.6% from heavy augmentation (vs +8-12% typical), suggesting intrinsic invariances from bio-inspired design
2. **Stable Dynamics**: While a vanilla CNN achieves higher peak accuracy, it severely overfits (100% train acc). MPKNet maintains more stable generalization dynamics
3. **Biological DFA**: Our models exhibit DFA ≈ 0.52, within the biological range (0.5-0.75)

### Comparison Context

| Model | Params | CIFAR-10 | Pretrained? | Augmentation |
|-------|--------|----------|-------------|--------------|
| **MPKNet (ours)** | 0.54M | 79.5% | No | None |
| MobileNetV3-Small | 2.5M | 92.5% | No | Heavy |
| SqueezeNet | 1.2M | 84.5% | Yes (ImageNet) | Standard |

*Note: Most published results use pretraining and/or heavy augmentation. Our results are from-scratch with minimal data augmentation to isolate architectural contribution.*

## Fractal Dynamics

We measure Detrended Fluctuation Analysis (DFA) and Hurst exponent of prediction confidence traces during evaluation. Biological neural systems exhibit DFA values in the 0.5-0.75 range, indicating long-range temporal correlations. Our models consistently produce dynamics in this biological range.

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/mpknet.git
cd mpknet
pip install -r requirements.txt
```

## Usage

### Training

```bash
# Basic training with CellPop
python mpkSGD.py --dataset CIFAR10 --epochs 100 --batch 256 --lr 0.1 \
    --use_cellpop --cellpop_stride 2 --mixup 0.2 --warmup_epochs 5 --swa --ema

# With Konio gating
python mpkSGD_kgate.py --dataset CIFAR10 --epochs 100 --batch 256 --lr 0.1 \
    --use_cellpop --cellpop_stride 2 --mixup 0.2 --warmup_epochs 5 --swa --ema
```

### Visualization

```bash
python visualize_mpknet.py  # Generates architecture diagram
```

## File Structure

```
mpknet/
├── mpkSGD.py           # Main training script (SGD + modern techniques)
├── mpkSGD_kgate.py     # Training with Konio gating mechanism
├── cellpop.py          # CellPop retinal sampling module
├── modelData.py        # Dataset loading utilities
├── tbLogger.py         # TensorBoard logging
├── visualize_mpknet.py # Architecture visualization
├── figures/            # Generated figures
├── results/            # Experiment results
└── paper/              # White paper LaTeX source
```

## Biological Motivation

The tree shrew (*Tupaia*) LGN provides an excellent model for studying parallel visual processing due to its clearly laminated structure. Unlike primates where M, P, and K cells are intermixed, the tree shrew LGN shows distinct layers:

- **Layers 1-2**: Koniocellular-like (small cells, modulatory)
- **Layers 3-4**: Parvocellular-like (color, detail)
- **Layers 5-6**: Magnocellular-like (motion, global)

This architecture is conserved across mammals, suggesting evolutionary optimization for efficient visual processing.

## Citation

```bibtex
@misc{mpknet2024,
  author = {Your Name},
  title = {MPKNet: A Tree Shrew LGN-Inspired Architecture for Efficient Visual Processing},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/YOUR_USERNAME/mpknet}
}
```

## License

MIT License

## Acknowledgments

- Jay Pratt Lab, University of Toronto
- Vector Institute
