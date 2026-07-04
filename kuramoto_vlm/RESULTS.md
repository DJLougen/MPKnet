# Results — MPK-Kuramoto vision head on frozen Qwen3.5-2B

Setup: the image-driven MPK-Kuramoto head (6.98M trainable params) produces 16
vision tokens spliced into a **frozen** `Qwen/Qwen3.5-2B` at its image-placeholder
slots; classification is framed as vision-language (predict the class name),
trained by next-token cross-entropy on the answer span only. Metric is
**teacher-forced exact-match** of the class name (chance = 1/num_classes).
Optimizer AdamW, cosine LR with warmup. Main runs on an NVIDIA GB10 (DGX Spark);
ablation on an RTX 3090.

## Main runs (corrected architecture)

| dataset | classes | steps / batch | exact-match curve | final token-acc | final val-loss | chance |
|---|---|---|---|---|---|---|
| CIFAR-100 | 100 | 600 / 128 | 6.5 → 12.9 → 19.1 → **19.6%** | 65.4% | 1.51 | 1% |
| Imagenette | 10 | 400 / 64 (64px) | 38.9 → 43.5 → 50.7 → **50.9%** | 85.2% | 10% |

Both curves are monotonic; the frozen LLM never sees pixels, only the oscillator
tokens, so the accuracy is attributable to the head.

## Ablation (Imagenette, equal budget: 250 steps, batch 32, same seed)

| variant | K role | final exact | token-acc | val-loss |
|---|---|---|---|---|
| **full** | encode + modulate DV | 45.3% | 83.9% | 0.529 |
| **encode_only** | encode only | **46.9%** | 84.4% | 0.528 |
| **modulate_only** | modulate DV only (not read out) | 31.9% | 79.9% | 0.638 |
| **no_k** | M/P only | 31.1% | 79.7% | 0.637 |
| **reservoir** | full, dynamics frozen at random init | 44.5% | 83.7% | 0.536 |

### Reading

1. **K's encoding is decisive (+~15 pts).** full / encode_only (~45–47%) vs
   modulate_only / no_k (~31%); the sole difference is whether K's phases are
   read out. K as a genuine encoder (chromatic + coarse spatial) is doing real
   work — not a controller bolted on.
2. **The diffuse DV modulation does not change final accuracy here.** full ≈
   encode_only, and modulate_only ≈ no_k. It sped early learning (44% vs 34% at
   step 125) but washed out by convergence. An honest negative on clean
   Imagenette; the manuscript predicts K state-gain should help under
   illumination/exposure shift, which is the fairer test (not run here).
3. **Learned ≈ random dynamics at this budget.** reservoir (frozen random
   coupling, 44.5%) ≈ full (45.3%). Most of the value is the oscillator *readout
   structure*, not yet the *learned* coupling — consistent with Un-0's own
   finding that a random Kuramoto reservoir gets most of the way and trained
   dynamics only separate with long training. Showing the couplings learn
   something a reservoir cannot needs a much longer run.

## Open, motivated next steps

- Long-budget CIFAR/Imagenette to separate trained dynamics from the reservoir.
- Robustness test (brightness/exposure shift) where K's state-modulation is
  predicted to matter, rather than clean-set accuracy.
- Free-generation accuracy (vs teacher-forced) and larger oscillator counts.

## Reproduce

```bash
python -m kuramoto_vlm.train_classify --dataset cifar100  --batch-size 128 --max-steps 600 --eval-every 150 --out runs/cifar100
python -m kuramoto_vlm.train_classify --dataset imagenette --image-size 64 --patch 16 --batch-size 64 --max-steps 400 --eval-every 100 --out runs/imagenette
bash kuramoto_vlm/ablation.sh
```
