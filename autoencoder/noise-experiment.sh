#!/bin/bash
set -x

LATENT=16
DIM=32
LR=0.001
BS=8
EPOCHS=10
NOISE="0.00 0.01 0.02 0.03 0.04 0.05"
NUM=$1

for noise in $NOISE; do
    echo fspool $noise
    python train.py --mnist --dim $DIM --latent $LATENT --batch-size $BS --train-only --loss direct --epochs $EPOCHS --name mnist-direct-$noise-$NUM --noise $noise --lr $LR
    echo baseline $noise
    python train.py --mnist --dim $DIM --latent $LATENT --batch-size $BS --train-only --loss chamfer --epochs $EPOCHS --name mnist-chamfer-$noise-$NUM --noise $noise --lr $LR --encoder SumEncoder --decoder MLPDecoder
done
