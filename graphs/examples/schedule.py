import argparse
import subprocess
import os
import time
import itertools

import GPUtil
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--type', choices=['bio', 'social'], required=True)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()


if args.type == 'bio':
    datasets = ['MUTAG','PTC_MR', 'PROTEINS', 'NCI1']
    dim = [32, 16]
elif args.type == 'social':
    datasets = ['IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'REDDIT-MULTI-5K', 'COLLAB']
    dim = [64]

models = ['fsort', 'sum' if args.type == 'bio' else 'mean']
batch_size = [128 if args.type == 'bio' else 100, 32]
seed = [args.seed]
dropout = [0]#[0] + ([0.5] if args.type == 'bio' else [])
fold = range(10)
epochs = 500 if args.type == 'bio' else 250

tasks = itertools.product(*reversed([models, datasets, fold, dim, batch_size, dropout, seed]))


def command_for_task(task, gpu):
    task = list(reversed(task))
    filename = '-'.join(map(str, task))
    model, dataset, fold, dim, bs, drop, seed = task
    command = f'python train.py --model {model} --dataset {dataset} --fold {fold} --dim {dim} --batch-size {bs} --drop {drop} --seed {seed} --validation --epochs {epochs} --gpu {gpu} > {filename}.log'
    return command


bar = tqdm(list(tasks))
for task in bar:
    memory_limit = 0.5
    dataset = task[-2]
    bs = task[-5]
    if int(bs) > 32:
        if 'REDDIT' in dataset:
            memory_limit = 0.05
        if 'COLLAB' in dataset:
            memory_limit = 0.2
    gpu_id = GPUtil.getFirstAvailable(maxLoad=0.8, maxMemory=memory_limit, interval=10, attempts=2**30, order='memory')[0]
    command = command_for_task(task, gpu_id)
    bar.set_postfix(last=' '.join(map(str, reversed(task))))
    subprocess.Popen(command, shell=True)
    time.sleep(10)
