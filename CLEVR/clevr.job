#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=10

NAME=$1
SEED=$2

module load conda/py3-latest 
module load cuda/9.0
source activate py3venv
source activate ban

python train.py --clevr-dir clevr --model $NAME --seed $2 | tee $NAME-$SEED.log
