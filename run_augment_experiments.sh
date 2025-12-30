#!/bin/bash
# Run augmentation experiments after 300-epoch training completes

cd /home/djl/mpknet

echo "Waiting for 300-epoch training to complete..."
while pgrep -f "train_cifar10_mps.py" > /dev/null; do
    sleep 60
done

echo "300-epoch training complete. Starting augmentation experiments..."

# Run standard augmentation
echo "Starting standard augmentation..."
python train_cifar10_augment.py --augment standard --epochs 100 > augment_standard.log 2>&1
echo "Standard augmentation complete."

# Run heavy augmentation
echo "Starting heavy augmentation..."
python train_cifar10_augment.py --augment heavy --epochs 100 > augment_heavy.log 2>&1
echo "Heavy augmentation complete."

echo "All experiments done!"
