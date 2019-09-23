#!/bin/bash
set -x

LATENT=16
DIM=32
LR=0.001
BS=8
EPOCHS=100
# When classifying MNIST sets without noise, the FSEncoder approach with pre-trained weights does better with a pre-trained model on noisy digits
# Thus, if you set NOISE=0.00, it's worth it to still resume from logs/mnist-fs-0.05-$NUM
NOISE=0.00
NUM=$1

PARAMS="--mnist --classify --dim $DIM --latent $LATENT --batch-size $BS --epochs $EPOCHS --noise $NOISE --lr $LR"

# fspool + frozen encoder weights
python train.py $PARAMS --resume logs/mnist-fs-0.05-$NUM --name mnistnc-fs-freeze-$NUM --freeze-encoder --encoder FSEncoder --decoder FSDecoder
# fspool + unfrozen encoder weights (finetune)
python train.py $PARAMS --resume logs/mnist-fs-0.05-$NUM --name mnistnc-fs-nofreeze-$NUM --encoder FSEncoder --decoder FSDecoder
# fspool from random init
python train.py $PARAMS --name mnistnc-fs-rinit-$NUM --encoder FSEncoder --decoder FSDecoder

# sum pool + frozen encoder weights
python train.py $PARAMS --resume logs/mnist-sum-$NOISE-$NUM --name mnistnc-sum-freeze-$NUM --freeze-encoder --encoder SumEncoder --decoder MLPDecoder
# sum pool + unfrozen encoder weights (finetune)
python train.py $PARAMS --resume logs/mnist-sum-$NOISE-$NUM --name mnistnc-sum-nofreeze-$NUM --encoder SumEncoder --decoder MLPDecoder
# sum pool from random init
python train.py $PARAMS --name mnistnc-sum-rinit-$NUM --encoder SumEncoder --decoder MLPDecoder

# max pool + frozen encoder weights
python train.py $PARAMS --resume logs/mnist-max-$NOISE-$NUM --name mnistnc-max-freeze-$NUM --freeze-encoder --encoder MaxEncoder --decoder MLPDecoder
# max pool + unfrozen encoder weights (finetune)
python train.py $PARAMS --resume logs/mnist-max-$NOISE-$NUM --name mnistnc-max-nofreeze-$NUM --encoder MaxEncoder --decoder MLPDecoder
# max pool from random init
python train.py $PARAMS --name mnistnc-max-rinit-$NUM --encoder MaxEncoder --decoder MLPDecoder

# mean pool + frozen encoder weights
python train.py $PARAMS --resume logs/mnist-mean-$NOISE-$NUM --name mnistnc-mean-freeze-$NUM --freeze-encoder --encoder MeanEncoder --decoder MLPDecoder
# mean pool + unfrozen encoder weights (finetune)
python train.py $PARAMS --resume logs/mnist-mean-$NOISE-$NUM --name mnistnc-mean-nofreeze-$NUM --encoder MeanEncoder --decoder MLPDecoder
# mean pool from random init
python train.py $PARAMS --name mnistnc-mean-rinit-$NUM --encoder MeanEncoder --decoder MLPDecoder
