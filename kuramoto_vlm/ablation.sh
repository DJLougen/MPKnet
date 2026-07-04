#!/bin/bash
# K-role ablation + reservoir control for the MPK-Kuramoto vision head.
# Equal budget, same seed. Requires `python` on PATH with the project env
# (torch, transformers>=5.12, torchvision, datasets) and a CUDA GPU.
#
#   full          : K encodes (read out) AND injects a diffuse gain on the DV
#   encode_only   : K encodes, no DV modulation
#   modulate_only : K modulates the DV, not read out
#   no_k          : M/P only
#   reservoir     : full model, but M/P/K dynamics frozen at random init
set -u
DATA=${DATA:-./data}
OUT=${OUT:-runs/abl}
COMMON="--dataset imagenette --data-root $DATA --image-size 64 --patch 16 \
        --batch-size 32 --max-steps 250 --eval-every 125 --log-every 50 --lr 5e-4 --warmup 30"

run () { name="$1"; shift; python -m kuramoto_vlm.train_classify $COMMON --out "$OUT/$name" "$@"; }

run full
run encode_only   --no-k-modulate
run modulate_only --no-k-encode
run no_k          --no-k-encode --no-k-modulate
run reservoir     --freeze-coupling
