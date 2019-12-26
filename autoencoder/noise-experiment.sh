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
    python train.py --mnist --dim $DIM --latent $LATENT --batch-size $BS --train-only --loss direct --epochs $EPOCHS --name mnist-fs-$noise-$NUM --noise $noise --lr $LR

    echo baseline sum $noise
    python train.py --mnist --dim $DIM --latent $LATENT --batch-size $BS --train-only --loss chamfer --epochs $EPOCHS --name mnist-sum-$noise-$NUM --noise $noise --lr $LR --encoder SumEncoder --decoder MLPDecoder
    echo baseline max $noise
    python train.py --mnist --dim $DIM --latent $LATENT --batch-size $BS --train-only --loss chamfer --epochs $EPOCHS --name mnist-max-$noise-$NUM --noise $noise --lr $LR --encoder MaxEncoder --decoder MLPDecoder
    echo baseline max $noise
    python train.py --mnist --dim $DIM --latent $LATENT --batch-size $BS --train-only --loss chamfer --epochs $EPOCHS --name mnist-mean-$noise-$NUM --noise $noise --lr $LR --encoder MaxEncoder --decoder MLPDecoder

    echo baseline sum + rnn decoder $noise
    python train.py --mnist --dim $DIM --latent $LATENT --batch-size $BS --train-only --loss chamfer --epochs $EPOCHS --name mnist-sum-rnn-$noise-$NUM --noise $noise --lr $LR --encoder SumEncoder --decoder RNNDecoder
done
