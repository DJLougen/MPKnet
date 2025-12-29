# MPKNet: A LGN-Inspired Architecture for Efficient Visual Processing

**MPKNet** is a bio-inspired convolutional neural network that models the Magnocellular (M), Parvocellular (P), and Koniocellular (K) pathways of the Lateral Geniculate Nucleus (LGN), based on cross-species evolutionary priors observed in mammals from tree shrews to primates.

## Motivation

This project was largely inspired by [Yamins et al. (2014)](https://www.pnas.org/doi/10.1073/pnas.1403112111) on performance-optimized hierarchical models.

This project is a side project to my PhD research on the LGN at the University of Toronto. I had the idea a couple years ago and built it over the summer of 2025. After reviewing prior attempts at bio-inspired neural networks, I felt existing approaches did not satisfactorily capture the actual organizational principles of biological visual systems. They often borrowed surface-level inspiration (like Gabor filters) without modeling the fundamental parallel-stream architecture that evolution has conserved across mammals. Another approach I encountered was EEG-guided training, which can reveal correlations between neural activity and image processing, but seemed to me more about pattern matching brain waves than capturing the underlying structure of how biological vision is organized.

While it might seem naive, I was curious whether directly modeling the anatomical structure I study would produce something interesting. MPKNet is both a learning exercise in deep learning and an attempt to put forth a new approach: rather than cherry-picking biological features, I directly model the laminar organization of the LGN as observed in humans, [tree shrews](https://pubmed.ncbi.nlm.nih.gov/40550685/), and macaques, where M, P, and K pathways are clearly separated into distinct layers. This explores whether taking biological structure seriously leads to networks with different computational properties.

This work also represents an alternative to current scaling methodologies in AI. Rather than emphasizing parameter count and data volume, MPKNet explores a **structural approach to scaling**: the idea that architectural organization inspired by biological systems may provide computational benefits that brute-force scaling cannot.

I am open to suggestions and collaboration. The parallel pathway architecture might be useful for robotics applications where efficient, real-time visual processing matters more than benchmark accuracy.

## Key Ideas

For a thorough explanation of the LGN and its pathways, see [Solomon (2021)](https://pubmed.ncbi.nlm.nih.gov/33832683/).

1. **Parallel Visual Streams**: Like the biological LGN, MPKNet processes visual information through three parallel pathways:
   - **Magno (M)**: Large receptive fields, fast temporal processing, global "gist"
   - **Parvo (P)**: Small receptive fields, high spatial acuity, fine detail
   - **Konio (K)**: Context relay and cross-stream modulation

2. **CellPop Retinal Sampling**: Structured downsampling using `pixel_unshuffle` to model retinal ganglion cell population responses

3. **Konio Gating**: The K-pathway generates channel attention to modulate P and M streams, acting as a context-aware relay (novel architectural contribution)

4. **Evolutionary Priors**: Kernel sizes and strides chosen to reflect biological receptive field properties across species

5. **Late Pooling**: Pooling is deferred until the final GAP layer. This preserves spatial noise throughout the network—the hypothesis being that "what is not" (negative space, noise patterns) may carry information that aids discrimination, similar to how biological systems may use absence of signal as informative

## Architecture

![MPKNet Architecture](figures/mpknet_architecture.png)

## Results

### CIFAR-10 (From Scratch, No Pretraining)

| Model | Params | Val Acc (no aug) | Val Acc (aug) | DFA | Train Acc |
|-------|--------|------------------|---------------|-----|-----------|
| MPKNet + CellPop | 0.54M | 79.5% | 81.1% | 0.52 | ~95% |
| Baseline CNN (ablation) | 0.55M | 84.6% | - | - | 100% (overfits) |
| MPKNet + Binocular | 0.14M | 83.0% | - | - | ~93% |

**Key Findings**:

1. **Augmentation Insensitivity**: MPKNet gains only **+1.6%** from heavy augmentation (vs +8-12% typical for CNNs). This suggests the parallel M/P/K pathway structure provides intrinsic invariances that standard architectures must learn through data augmentation.

2. **Implicit Regularization**: While the baseline CNN achieves higher peak accuracy (84.6%), it memorizes the training set (100% train acc). MPKNet's biological structure acts as implicit regularization, preventing perfect memorization.

3. **Biological Dynamics**: The models exhibit DFA ≈ 0.52, within the biological range (0.5-0.75), indicating long-range temporal correlations characteristic of neural systems at criticality.

**Note on Evaluation**: This project explores bio-inspired design principles rather than pursuing SOTA accuracy. The value lies in understanding how biological organizational principles (parallel visual streams, cross-stream modulation) translate to computational properties in artificial systems.

### Comparison Context

| Model | Params | CIFAR-10 | CIFAR-100 | Pretrained? | Augmentation |
|-------|--------|----------|-----------|-------------|--------------|
| **BinocularMPKNet (ours)** | 0.14M | 83.0% | - | No | None |
| MobileNetV3-Small | 2.5M | 92.5% | 75.4% | No | Heavy |
| SqueezeNet | 1.2M | 84.5% | 58.5% | Yes (ImageNet) | Standard |

*Comparison numbers from [Benchmark Analysis of Deep Learning Models on CIFAR-10/100](https://arxiv.org/abs/2505.03303). Most published results use pretraining and/or heavy augmentation. BinocularMPKNet results are from-scratch without augmentation to isolate architectural contribution.*

## Fractal Dynamics

I measure Detrended Fluctuation Analysis (DFA) and Hurst exponent of prediction confidence traces during evaluation. Biological neural systems exhibit DFA values in the 0.5-0.75 range, indicating long-range temporal correlations. The models consistently produce dynamics in this biological range. Whether this is meaningful or simply reflects how the data is organized is an open question.

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
└── results/            # Experiment results
```

## Biological Motivation

The tree shrew (*Tupaia*) LGN provides an excellent model for studying parallel visual processing due to its clearly laminated structure. Unlike primates where M, P, and K cells are intermixed, the tree shrew LGN shows distinct layers:

- **Layers 1-2**: Koniocellular-like (small cells, modulatory)
- **Layers 3-4**: Parvocellular-like (color, detail)
- **Layers 5-6**: Magnocellular-like (motion, global)

This architecture is conserved across mammals, suggesting evolutionary optimization for efficient visual processing.

## Current Focus: Binocular Processing

The binocular extension (`mpknet_binocular.py`) is the current active development focus. This adds:

- **Ocular dominance organization**: Channels are assigned to left/right eye with graded mixing; some purely monocular, some binocular
- **Stereo disparity simulation**: Horizontal shifts between eye views during training
- **Eye-specific LGN layers**: Modeling the contralateral/ipsilateral layer organization (layers 1,4,6 vs 2,3,5)

The binocular model is significantly smaller (0.14M params) while adding biologically plausible dual-eye processing, achieving 83.0% validation accuracy on CIFAR-10.

## What This Project Deliberately Ignores

Several biological features are intentionally omitted. The reasoning:

**Temporal dynamics / spiking**: The LGN exhibits rich temporal processing; M cells respond transiently, P cells have sustained responses. I chose to focus on the spatial/structural organization first. Adding temporal dynamics (e.g., spiking networks, LSTM-like recurrence) would complicate the architecture before validating that the parallel stream structure itself provides value.

**Recurrent connections**: Real visual processing involves massive feedback from V1 to LGN and lateral connections within LGN. These are omitted because feedforward CNNs are better understood and easier to train. Recurrence is a natural next step but adds training complexity. See [arXiv:2506.21734](https://arxiv.org/abs/2506.21734) for a potentially relevant approach.

**Foveation / eccentricity**: Biological retinas have varying resolution across the visual field. This could be added via attention mechanisms or non-uniform sampling, but would require larger images than CIFAR-10's 32x32 to be meaningful.

**Color opponent channels**: The P pathway in particular carries color opponent signals (red-green, blue-yellow). The current implementation uses standard RGB. True color opponency might improve the biological fidelity of the P stream.

**Cortical processing (V1+)**: This model stops at LGN-level processing. Real vision involves extensive cortical computation. The fusion layer is a crude stand-in for V1 integration. A proper V1 model with orientation columns and complex cells would be a substantial extension. I am thinking about how the [Kakeya conjecture](https://en.wikipedia.org/wiki/Kakeya_set) might be used with Fourier-transformed data to efficiently encode and predict orientations—inspired by a recent [video](https://www.youtube.com/watch?v=5J3tYU_-IZI) on the 3D Kakeya conjecture being solved ([arXiv:2502.17655](https://arxiv.org/abs/2502.17655)).

**Attention / top-down modulation**: Beyond K-gating, biological vision involves attentional modulation from higher areas. This is ignored to keep the model simple and feedforward.

The philosophy is to start with the most fundamental structural feature (parallel M/P/K streams) and validate that before adding complexity. Each ignored feature represents a potential future direction.

## White Paper

*Coming soon* — once I figure out how to write it!

## Citation

```bibtex
@misc{MPKNet,
  author = {Lougen, D.J.
  title = {MPKNet: A Tree Shrew LGN-Inspired Architecture for Efficient Visual Processing},
  year = {2024},
  publisher = {GitHub},
  url = https://github.com/DJLougen/MPKnet
}
```

## License

MIT License


