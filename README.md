<h1 align="center">MPKNet V6</h1>

<p align="center">
  <strong>Bio-inspired vision that runs on a Raspberry Pi</strong>
</p>

<p align="center">
  <a href="#benchmarks">Benchmarks</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#license--commercial-use">License</a>
</p>

<p align="center">
  <code>76KB</code> model • <code>33 FPS</code> on RPi5 • <code>89%</code> Kvasir-v2 • <code>No pretraining</code> • <code>No augmentation</code> • <code>161× fewer params than MobileNetV3</code>
</p>

---

MPKNet V6 implements the **parallel visual pathways** found in mammalian brains. Instead of stacking more layers, it asks: what if *how* information flows matters more than parameter count?

| | MobileNetV3-S | **MPKNet V6** | **MPKNet V6-Pi** |
|---|---------------|---------------|------------------|
| Parameters | 2.5M | **0.21M** | **15.5K** |
| Model size | 10MB | **0.89MB** | **76KB** |
| FPS (RPi5) | 5-8 | — | **33** |
| Accuracy (Kvasir) | ~92%* | **89%** | **82%** |

*MobileNetV3-S accuracy from published benchmarks (not my evaluation). V6/V6-Pi measured on Kvasir-v2 val set (1600 samples). Direct comparison requires same-dataset evaluation with identical training protocol.*

**161× fewer parameters** than MobileNetV3-S. Train in an hour, not a week. Deploy on a $35 Raspberry Pi, not a cloud GPU.

**V6 is not about beating SOTA.** It's about *competitive accuracy at a fraction of the cost*.

---

## Benchmarks

### Image Classification

| Dataset | Classes | Resolution | Accuracy | Params | Notes |
|---------|---------|------------|----------|--------|-------|
| **Kvasir-v2** | 8 | 224×224 | **89%** | 0.21M | Medical endoscopy (research only) |
| **TinyImageNet** | 200 | 64×64 | **40.6%** | 0.21M | ResNet18 gets ~41.5% with 52× more params |
| **CIFAR-100** | 100 | 32×32 | **58.8%** | 0.22M | |
| **STL-10** | 10 | 96×96 | **71.7%** | 0.21M | Only 5K training samples |
| **ImageNet-100** | 100 | 224×224 | **60.8%** | 0.54M | |

### Video Classification (V6.2 Temporal)

| Dataset | Classes | Resolution | Accuracy | Params | Notes |
|---------|---------|------------|----------|--------|-------|
| **UCF-101** | 101 | 112×112 | **77%** | 0.58M | 8-frame temporal M-pathway, from scratch |

V6.2 adds a **sequential temporal M-pathway** — the M stream processes 8 consecutive frames and computes inter-frame deltas for motion detection, while P sees only the current frame for spatial detail. K gates both. This mirrors the biological role of magnocellular neurons in motion processing.

### Edge Deployment

| Device | Model | Size | FPS | Accuracy |
|--------|-------|------|-----|----------|
| Raspberry Pi 5 (no heatsink) | V6-Pi | 76KB | 33 | 82% |
| MacBook M3 | V6 | 0.89MB | 200+ | 89% |

### Finding: Augmentation Hurts at Small Scales

| Dataset | No Augmentation | With Augmentation | Change |
|---------|-----------------|-------------------|--------|
| CIFAR-100 (32×32) | **52.8%** | 46.0% | -6.8% |
| TinyImageNet (64×64) | **40.6%** | 24.1% | -16.5% |
| ImageNet-100 (224×224) | 60.8% | ~62% | +1-2% |

This is consistent with NetAug (Cai et al., 2022), which showed regularization hurts tiny models that underfit rather than overfit. At small resolutions, the Fibonacci stride architecture provides sufficient multi-scale coverage that augmentation becomes redundant noise. At 224×224, mild augmentation helps marginally.

---

## Quick Start

```python
import torch
from MPKx import MPKNetV6

# Create model
model = MPKNetV6(num_classes=8)  # e.g., Kvasir-v2
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
# Output: Parameters: 210,000

# Inference
x = torch.randn(1, 3, 224, 224)
out = model(x)  # [1, 8]
```

### Training

```bash
# Clone and install
git clone https://github.com/DJLougen/MPKnet.git
cd MPKnet
pip install -r requirements.txt

# Train on your dataset
python train.py --dataset kvasir --epochs 100
```

---

## Architecture

MPKNet models the **Lateral Geniculate Nucleus (LGN)** - the relay station between retina and visual cortex.

<p align="center">
  <img src="figures/mpknet_architecture.png" alt="MPKNet V6 Architecture" width="600"/>
</p>

### The Three Pathways

| Pathway | Biological Role | Implementation | What It Captures |
|---------|-----------------|----------------|------------------|
| **M** (Magnocellular) | ~10% of LGN, motion, global gist | Stride 5 (coarse) | Shape, motion, layout |
| **P** (Parvocellular) | ~80% of LGN, fine detail, color | Stride 2 (fine) | Texture, edges, color |
| **K** (Koniocellular) | ~10% of LGN, projects to M and P | Stride 3 (intermediate) | Context-dependent gating |

### Core Principles

1. **Same kernel, different stride** - All pathways use 5×5 kernels. Fibonacci strides (2:3:5) differentiate them, producing resolutions that converge toward the golden ratio.
2. **Parallel processing** - M/P/K run independently until fusion. No cross-talk within pathways.
3. **Late fusion only** - No pooling within pathways. Global pool only at the end.
4. **K modulates, doesn't process** - K-pathway generates cross-stream attention gates for M and P (extends Bahdanau attention, FiLM with biological grounding).

### What's Novel

- **First Fibonacci strides in CNNs** - Derived from biological spatial frequency tuning, not empirical search
- **First complete M/P/K implementation** - Prior work (Magno-Parvo CNN, EVNets, SlowFast) models M/P only
- **Biologically-grounded cross-stream gating** - K→M/P gating mirrors koniocellular projections in LGN

### Why This Works

Biology processes vision with **20 watts**. One hypothesis: efficiency comes from the *wiring diagram*, not raw neuron count.

MPKNet borrows this principle: I restrict **where** multiplication happens. M and P process in parallel streams before fusion. K modulates both. The math is standard convolutions. The connectivity pattern is inspired by biology.

> *"It's what you multiply and where you multiply."*

---

## Ablation Study

Pathway ablations currently running across a variety of datasets. Results forthcoming.

---

## Interpretable Failures

**Method:** I evaluated V6-Pi on Kvasir-v2 validation set (1600 samples), tracking all misclassifications with confidence scores.

**Key finding:** 63% of errors (183/292) cluster in just two bidirectional pairs.

### Per-Class Accuracy

| Class | Accuracy | Confusion Pattern |
|-------|----------|-------------------|
| esophagitis | 67.9% | → normal-z-line (58 errors) |
| dyed-lifted-polyps | 70.4% | → dyed-resection-margins (51 errors) |
| polyps | 76.9% | Scattered across multiple classes |
| normal-pylorus | 99.0% | Nearly perfect |

### Top Confusion Pairs (with Confidence)

| True Class | → Predicted | Count | Mean Conf | Range |
|------------|-------------|-------|-----------|-------|
| esophagitis | → normal-z-line | 58 | 68% | 50-94% |
| dyed-lifted-polyps | → dyed-resection-margins | 51 | 69% | 34-97% |
| dyed-resection-margins | → dyed-lifted-polyps | 40 | 60% | 30-96% |
| normal-z-line | → esophagitis | 34 | 61% | 48-81% |

**What it means:** The discriminative signal between these pairs is weak enough that errors concentrate here. External context (patient history, procedure timeline, endoscope position) would help, but is unavailable to any vision-only system.

### Failure Categories

| Type | Count | % of Failures | Meaning |
|------|-------|---------------|---------|
| **Confident failures** (≥80% conf) | 44 | 15% | Model is wrong but sure — miscalibrated |
| **Ambiguous failures** (<50% conf) | 22 | 8% | Model knows it doesn't know — honest |
| **Close calls** (<15% margin) | 69 | 24% | True class almost won — fixable |

### Semantic Group Confusion

| Direction | Errors | Clinical Impact |
|-----------|--------|-----------------|
| pathology to normal | 66 | **Missed disease** |
| normal to pathology | 39 | False alarm |
| polyp to procedure | 52 | Dye similarity |
| procedure to polyp | 42 | Dye similarity |

### Clinical Limitations

**This model is a research prototype, not a clinical tool.**

| Metric | V6-Pi Result | Clinical Requirement |
|--------|--------------|----------------------|
| Polyp sensitivity | ~75% | >=95% for screening |
| Pathology to Normal errors | 66 cases | Near zero |
| Confident false negatives | 44 @ 88% conf | Unacceptable |

**Why it's interpretable:** Failures cluster in predictable, explainable pairs rather than scattering randomly across 8 classes. You know *which* cases need human review and *why* the model failed.

---

## Roadmap

MPKNet V6 implements the **LGN stage** of mammalian vision. What I'm working on next:

**Biological extensions:**
- [ ] **Surround suppression** - V1-like center-surround for better edge discrimination
- [ ] **Temporal M pathway** - 3D convolutions in M pathway for video (matches M-cell motion sensitivity)

### Biological Extensions
- [ ] **RGC layer** - Midget/Parasol/Bistratified cells feeding M/P/K pathways
- [ ] **Retinotectal pathway** - Superior colliculus for saccades
- [ ] **V1 orientation columns** - Edge detection specialization
- [ ] **Thalamo-cortical loops** - Exploring whether attention-like behavior emerges from architecture alone

### Applications
- [ ] **Detection head** - YOLO-style head using M/P as multi-scale FPN
- [ ] **Medical uncertainty** - MC Dropout for epistemic uncertainty quantification
- [ ] **VLM encoder** - Lightweight vision encoder for vision-language models
- [ ] **Webcam eye tracking** - Real-time gaze estimation from eye crops
- [ ] **Thermal glider fire detection** - 3D-printed gliders for wildfire monitoring




## V1-to-LGN Feedback Variant

MPKNet now includes an experimental recurrent cortical feedback variant. The feedforward pass builds LGN-like M/P streams and a V1 representation, then V1 sends corticogeniculate gain back to the LGN-like M/P streams before recomputing the refined V1 state. This keeps the model compact while making the visual loop explicit:

```text
retina -> LGN(M/P) -> V1 -> LGN gain feedback -> refined V1 -> classifier
```

The main `MPKx.py` contains the feedback-enabled MPKx implementation. A public standalone variant is also available at `public/mpknet_v6_feedback.py`.

## Citation

```bibtex
@misc{MPKNet,
  author = {Lougen, D.J.},
  title = {MPKNet: A LGN-Inspired Architecture for Efficient Visual Processing},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/DJLougen/MPKnet}
}
```

Patent pending: US 63/950,391

---

## License & Commercial Use

**PolyForm Small Business License** with Humanitarian Exception.

| Use Case | Cost |
|----------|------|
| Academic research | Free |
| Personal projects | Free |
| Startups (<$100K revenue) | Free |
| Non-profits & NGOs | Free |
| Educational institutions | Free |
| Low-income region deployment | Free |
| Commercial (>$100K revenue) | [Contact me](mailto:d.lougen@mail.utoronto.ca) |

### Why This License?

A 76KB model on a $35 Raspberry Pi can enable:
- **Research prototypes** for medical image analysis (not clinical deployment)
- Agricultural monitoring on small farms
- Educational tools in underfunded schools
- Disaster response with limited infrastructure

**These use cases should never be paywalled.**

**Note:** For medical applications, see [Clinical Limitations](#clinical-limitations). This model is a research tool, not a diagnostic device.

For commercial licensing: [d.lougen@mail.utoronto.ca](mailto:d.lougen@mail.utoronto.ca)

---

## Acknowledgements

Thanks to Paul Dassonville (UO) for introducing me to these cells, and Jay Pratt (U of T) for ongoing collaboration on koniocellular research.


---

<p align="center">
  <a href="https://djlougen.github.io/PersonalWebsite/">Daniel J. Lougen</a> · University of Toronto
</p>
