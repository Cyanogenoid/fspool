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
    python train.py --mnist --dim $DIM --latent $LATENT --batch-size $BS --loss direct --epochs $EPOCHS --name mnistmask-fs-$noise-$NUM --noise $noise --lr $LR --masked

    echo baseline sum $noise
    python train.py --mnist --dim $DIM --latent $LATENT --batch-size $BS --loss chamfer --epochs $EPOCHS --name mnistmask-sum-$noise-$NUM --noise $noise --lr $LR --encoder SumEncoder --decoder MLPDecoder --masked
    echo baseline max $noise
    python train.py --mnist --dim $DIM --latent $LATENT --batch-size $BS --loss chamfer --epochs $EPOCHS --name mnistmask-max-$noise-$NUM --noise $noise --lr $LR --encoder MaxEncoder --decoder MLPDecoder --masked
    echo baseline max $noise
    python train.py --mnist --dim $DIM --latent $LATENT --batch-size $BS --loss chamfer --epochs $EPOCHS --name mnistmask-mean-$noise-$NUM --noise $noise --lr $LR --encoder MaxEncoder --decoder MLPDecoder --masked

    echo baseline sum + rnn decoder $noise
    python train.py --mnist --dim $DIM --latent $LATENT --batch-size $BS --loss chamfer --epochs $EPOCHS --name mnistmask-sum-rnn-$noise-$NUM --noise $noise --lr $LR --encoder SumEncoder --decoder RNNDecoder --masked
done
