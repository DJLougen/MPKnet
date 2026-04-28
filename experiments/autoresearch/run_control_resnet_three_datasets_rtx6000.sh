#!/usr/bin/env bash
set -euo pipefail
cd /workspace/research/autoresearch
export CUDA_VISIBLE_DEVICES=GPU-27a6a0cb-3c38-8941-5bfb-e2669ec3e6a7
export PYTHONUNBUFFERED=1

echo "[$(date -Is)] starting ResNet control CIFAR-100 on RTX 6000"
/workspace/venv-mamba3/bin/python train_conv_control.py --dataset cifar100 --epochs 100 --width 36 2>&1 | tee run_control_resnet_w36_cifar100_rtx6000.log

echo "[$(date -Is)] starting ResNet control STL-10 on RTX 6000"
/workspace/venv-mamba3/bin/python train_conv_control.py --dataset stl10 --epochs 100 --width 36 2>&1 | tee run_control_resnet_w36_stl10_rtx6000.log

echo "[$(date -Is)] starting ResNet control Caltech-101 on RTX 6000"
/workspace/venv-mamba3/bin/python train_conv_control.py --dataset caltech101 --epochs 100 --width 36 2>&1 | tee run_control_resnet_w36_caltech101_rtx6000.log

echo "[$(date -Is)] all ResNet control runs complete"
