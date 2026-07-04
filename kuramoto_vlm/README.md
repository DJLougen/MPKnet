# kuramoto_vlm — an image-driven MPK-Kuramoto vision head for a Qwen VLM

A coupled-oscillator vision encoder that feeds a frozen large language model.
It takes the Kuramoto primitive from [Un-0](https://github.com/unconv-ai/Un-0)
(Unconventional AI) — where a population of coupled oscillators is a *generative*
substrate driven by a class label — and **inverts it into an image-driven
encoder**, organized into the magno / parvo / konio (M/P/K) pathways of the
Koniocellular Circuit-Family account of early primate vision.

The oscillator phases become a sequence of vision tokens that splice into a
Qwen LLM's input-embedding stream at its image-placeholder slots, exactly like a
ViT connector. Only the head is trained; the LLM stays frozen.

## Idea

- **M (magnocellular)** and **P (parvocellular)** are the fast/mid **encoding**
  streams (luminance gist, fine detail).
- **K (koniocellular)** is a **hybrid circuit-family**: it *encodes* (K3/K4
  S-cone chromatic + coarse spatial, read out alongside M/P) **and** injects a
  diffuse, higher-area **modulation** onto the integrated decision vector — it
  does **not** modulate the M/P relays directly (it "gates cortical circuits fed
  by M and P", Cheong et al. 2011).
- M/P/K each evolve under their **own** learned coupling (Un-0's angle-identity
  velocity: two matmuls per step, no per-sample outer product). Per-pathway
  natural-frequency bands make each oscillator "tuned to its pathway's
  preference" (M fast > P mid > K slow sub-beta).

The discriminative-oscillator regime this builds on is
[AKOrN](https://arxiv.org/abs/2410.13821) (Miyato et al., 2025); Un-0 is the
evidence the primitive scales to image-grade generation.

## Layout

```
kuramoto.py      kuramoto_velocity, PathwayField, MPKKuramotoField (+ ablation flags)
vision_head.py   RetinaPatchFront (parameter-free 5-signal front end) + KuramotoVisionHead
backbone.py      load a frozen Qwen (or any causal LM) + resolve image-token id / hidden size
qwen_glue.py     VLMWithKuramotoHead — splice vision tokens into the embedding stream
data.py          CIFAR-10/100 (torchvision) and ImageNet-style / Imagenette loaders
train_classify.py  classification-as-VLM trainer (frozen LLM, next-token CE on the class name)
smoke_test.py    CPU plumbing test (field invariants, injection, grads reach K/M/P, overfit)
ablation.sh      K-role + reservoir ablation on equal budget
```

## Setup

Python ≥ 3.11, PyTorch ≥ 2.11, `transformers >= 5.12` (needed for the `qwen3_5`
architecture), `torchvision`, `datasets`, `accelerate`. A CUDA GPU is required
for real training; the smoke test runs on CPU.

## Use

```bash
# CPU plumbing test — no GPU, no downloads
KVLM_FORCE_CPU=1 python -m kuramoto_vlm.smoke_test

# validate the full stack on GPU (loads the LLM, one synthetic step)
python -m kuramoto_vlm.train_classify --dry-run

# train the head on CIFAR-100 (LLM frozen)
python -m kuramoto_vlm.train_classify --dataset cifar100 --batch-size 128 \
    --max-steps 600 --eval-every 150 --out runs/cifar100

# train on Imagenette (an ImageNet subset)
python -m kuramoto_vlm.train_classify --dataset imagenette --image-size 64 \
    --patch 16 --batch-size 64 --max-steps 400 --eval-every 100 --out runs/imagenette

# K-role + reservoir ablation (equal budget)
bash kuramoto_vlm/ablation.sh
```

`--llm` selects the backbone (default `Qwen/Qwen3.5-2B`). Ablation flags:
`--no-k-encode`, `--no-k-modulate`, `--freeze-coupling`.

## Contract

`KuramotoVisionHead(pixel_values)` → `(B, num_tokens, hidden)` vision tokens,
`num_tokens = (image_size / patch_size)**2`, `hidden` = the LLM's hidden size.
`VLMWithKuramotoHead` replaces the embeddings at `image_token_id` positions with
those tokens and runs the frozen LLM with `inputs_embeds` + `labels`.

See [RESULTS.md](RESULTS.md) for CIFAR-100 / Imagenette curves and the ablation.

## Credit

Kuramoto velocity, sin/cos readout, and resize-conv decoder pattern adapted from
Un-0 (unconv-ai/Un-0, MIT). Biological framing from the Koniocellular
Circuit-Family manuscript. This is exploratory research, shared to collaborate.
