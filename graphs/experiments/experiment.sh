#!/bin/bash

NAME=$1
DATASET=$2

for i in {0..9};
do
    python enzymes_topk_pool.py --fold $i --dataset $DATASET --batch-size 32 >> $NAME-$DATASET.log
done
