#!/usr/bin/env bash
set -euo pipefail
cd /workspace/research/autoresearch
export CUDA_VISIBLE_DEVICES=GPU-27a6a0cb-3c38-8941-5bfb-e2669ec3e6a7
export PYTHONUNBUFFERED=1

echo "[$(date -Is)] starting CIFAR-100 on RTX 6000"
/workspace/venv-mamba3/bin/python train_cifar100.py 2>&1 | tee run_cifar100_ch56_bidir_100ep_rtx6000.log

echo "[$(date -Is)] starting STL-10 on RTX 6000"
/workspace/venv-mamba3/bin/python train_mpk_smallvision.py --dataset stl10 --epochs 100 2>&1 | tee run_stl10_ch56_bidir_100ep_rtx6000.log

echo "[$(date -Is)] starting Caltech-101 on RTX 6000"
/workspace/venv-mamba3/bin/python train_mpk_smallvision.py --dataset caltech101 --epochs 100 2>&1 | tee run_caltech101_ch56_bidir_100ep_rtx6000.log

echo "[$(date -Is)] all dataset runs complete"