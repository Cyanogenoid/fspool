#!/bin/bash
set -x

LATENT=16
DIM=32
LR=0.001
BS=8
EPOCHS=10
NOISE=0.05
NUM=$1

# fspool + frozen encoder weights
python train.py --mnist --dim $DIM --latent $LATENT --batch-size $BS --epochs $EPOCHS --resume logs/mnist-direct-$NOISE-$NUM --name mnistc-fs-freeze-$NUM --noise $NOISE --lr $LR --classify --freeze-encoder
# fspool + unfrozen encoder weights (finetune)
python train.py --mnist --dim $DIM --latent $LATENT --batch-size $BS --epochs $EPOCHS --resume logs/mnist-direct-$NOISE-$NUM --name mnistc-fs-nofreeze-$NUM --noise $NOISE --lr $LR --classify
# fspool from random init
python train.py --mnist --dim $DIM --latent $LATENT --batch-size $BS --epochs $EPOCHS --name mnistc-fs-rinit-$NUM --noise $NOISE --lr $LR --classify

# sum pool + frozen encoder weights
python train.py --mnist --dim $DIM --latent $LATENT --batch-size $BS --epochs $EPOCHS --resume logs/mnist-chamfer-$NOISE-$NUM --name mnistc-base-freeze-$NUM --noise $NOISE --lr $LR --classify --freeze-encoder --encoder SumEncoder --decoder MLPDecoder
# sum pool + unfrozen encoder weights (finetune)
python train.py --mnist --dim $DIM --latent $LATENT --batch-size $BS --epochs $EPOCHS --resume logs/mnist-chamfer-$NOISE-$NUM --name mnistc-base-nofreeze-$NUM --noise $NOISE --lr $LR --classify --encoder SumEncoder --decoder MLPDecoder
# sum pool from random init
python train.py --mnist --dim $DIM --latent $LATENT --batch-size $BS --epochs $EPOCHS --name mnistc-base-rinit-$NUM --noise $NOISE --lr $LR --classify --encoder SumEncoder --decoder MLPDecoder
