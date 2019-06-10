import sys
import argparse
import itertools
from collections import defaultdict

import pandas as pd
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('paths', nargs='+')
parser.add_argument('--min-epoch', default=340, type=int)
args = parser.parse_args()


def get_acc(path):
    prefixes = [
        'Test Epoch',
        'exist',
        'number',
        'material',
        'size',
        'shape',
        'color',
    ]
    accs = []
    epoch = 0
    with open(path) as fd:
        for line in fd:
            is_new_epoch = False
            if line.startswith('Test Epoch'):
                epoch = int(line.split(':')[0].split(' ')[-1])
                is_new_epoch = True

            if any(line.startswith(p) for p in prefixes):
                if is_new_epoch:
                    category = 'all'
                else:
                    category = line.split(' ')[0]
                tokens = line.split('%')
                before_acc = tokens[0]
                acc = float(before_acc[-5:])
                accs.append([epoch, category, acc])
    return accs

data = defaultdict(list)

for path in sorted(args.paths, key=lambda x: x.split('-')[::-1]):
    tokens = path[:-4].split('-')
    model, _, seed = tokens
    accs = get_acc(path)
    parts = model.split('/')
    for epoch, category, acc in accs:
        data['seed'].append(seed)
        data['model'].append(model)
        data['epoch'].append(epoch)
        data['category'].append(category)
        data['acc'].append(acc)


data = pd.DataFrame.from_dict(data)

# only use values from after this epoch
df = data[data['epoch'] >= args.min_epoch]

# acc over seed and selected epochs
df = df.groupby(['model', 'category'])['acc']

# mean and std over seeds
#df = pd.concat([df.mean(), df.std()], axis=1)
df = df.mean()

# turn Series back into DataFrame
df = pd.DataFrame(df).reset_index()
df = df.pivot(index='model', columns='category', values='acc')

# round a bit
df = df.round(2)
print(df)


# find first epoch exceeding 98% acc overall

dfs = []
for acc_threshold in [98, 98.5, 99]:
    # remove entries that are less than 98% or in the wrong category
    df2 = data[data['category'] == 'all']
    df2 = df2[df2['acc'] >= acc_threshold]

    # don't care about acc and category now
    df2 = df2[['model', 'seed', 'epoch']]

    # first epoch that reached 98% 
    df2 = df2.groupby(['model', 'seed']).first()

    # some RNs didn't manage to reach 99%
    #print(df2.groupby('model').count())

    # average first epoch over seeds
    df2 = df2.groupby('model').mean()

    df2 = df2.rename(index=str, columns={'epoch': f'{acc_threshold:.1f}%'})
    dfs.append(df2)

df2 = pd.concat(dfs, axis=1)
df2 = df2.round().astype(int)

df = pd.concat([df, df2], axis=1)
print(df)
