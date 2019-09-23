import sys
import argparse
import itertools
from collections import defaultdict

import pandas as pd
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('paths', nargs='+')
parser.add_argument('--epoch', action='store_true')
parser.add_argument('--cheat', action='store_true')
parser.add_argument('--merge', action='store_true')
parser.add_argument('--momentum', type=float, default=0.0)
args = parser.parse_args()


if not args.cheat:
    criterion = lambda x: x[0]  # val acc
else:
    criterion = lambda x: x[1]  # test acc


def get_acc(path):
    accs = []
    with open(path) as fd:
        moving_val_acc = 0
        for line in fd:
            prefix = 'Epoch: '
            if line.startswith(prefix):
                tokens = line.split(' ')
                val_acc = float(tokens[-4][:-1])
                test_acc = float(tokens[-1])
                moving_val_acc = args.momentum * moving_val_acc + (1 - args.momentum) * val_acc
                accs.append((moving_val_acc, test_acc))
    if len(accs) == 0:
        return None
    return accs

data = defaultdict(list)

for path in sorted(args.paths, key=lambda x: x.split('-')[::-1]):
    tokens = path[:-4].split('-')
    *_, fold, dim, bs, dropout, seed = tokens
    model = tokens[0]
    dataset = '-'.join(tokens[1:-5])
    acc = get_acc(path)
    if acc is None:
        continue
    parts = model.split('/')
    if len(parts) == 1:
        run = 'default'
    else:
        run = parts[-2]
        model = parts[-1]
    for i, (val_acc, test_acc) in enumerate(acc):
        data['seed'].append(seed)
        data['run'].append(run)
        data['model'].append(model)
        data['dataset'].append(dataset)
        data['fold'].append(fold)
        data['dim'].append(dim)
        data['bs'].append(bs)
        data['dropout'].append(dropout)
        data['epoch'].append(i)
        data['val_acc'].append(val_acc)
        data['test_acc'].append(test_acc)


data = pd.DataFrame.from_dict(data)
if args.epoch:
    dgcnn_epochs = {
        'MUTAG': 300,
        'PTC_MR': 200,
        'NCI1': 200,
        'PROTEINS': 100,
    }
    for dataset, epoch in dgcnn_epochs.items():
        applies_to_dataset = data['dataset'] == dataset
        matches_epoch = data['epoch'] == epoch
        data = data[~applies_to_dataset | matches_epoch]

# average over folds
df = data.groupby(['seed', 'run', 'model', 'dataset', 'dim', 'bs', 'dropout', 'epoch']).mean()
if args.merge:
    df = df.mean(axis=1)
# group hyperparams away
df2 = df.groupby(['seed', 'run', 'model', 'dataset'])
if not args.cheat:
    df2 = df2['val_acc']
elif not args.merge:
    df2 = df2['test_acc']
idx = df2.idxmax()
best = df.loc[idx]
print(best.to_string())
df3 = best.groupby(['run', 'model', 'dataset'])
mean = df3.mean()
std = df3.std().rename(index=str, columns={'test_acc': 'test_std', 'val_acc': 'val_std'})
best_across_seeds = pd.concat([mean, std], axis=1)
print(best_across_seeds)
