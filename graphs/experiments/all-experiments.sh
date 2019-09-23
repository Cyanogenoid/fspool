#!/bin/bash

NAME=$1

CUDA_VISIBLE_DEVICES=1 ./experiment.sh $NAME MUTAG &
CUDA_VISIBLE_DEVICES=1 ./experiment.sh $NAME PROTEINS &
CUDA_VISIBLE_DEVICES=1 ./experiment.sh $NAME NCI1 &
