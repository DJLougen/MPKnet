#!/usr/bin/env bash
set -euo pipefail
cd /workspace/research/autoresearch
export CUDA_VISIBLE_DEVICES=GPU-27a6a0cb-3c38-8941-5bfb-e2669ec3e6a7
export PYTHONUNBUFFERED=1

echo "[$(date -Is)] waiting for ResNet control queue"
while pgrep -f "train_conv_control.py|run_control_resnet_three_datasets_rtx6000.sh" >/dev/null; do
  sleep 60
done

echo "[$(date -Is)] starting native CIFAR-100 original V6 baseline"
/workspace/venv-mamba3/bin/python train_native_cifar100_v6.py 2>&1 | tee run_native_cifar100_original_v6_sgd_rtx6000.log

echo "[$(date -Is)] native CIFAR-100 original V6 baseline complete"
