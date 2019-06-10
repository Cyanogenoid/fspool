import os
import argparse

import torch


parser = argparse.ArgumentParser()
parser.add_argument('paths', nargs='+')
args = parser.parse_args()

paths = [p for p in args.paths if p.endswith('.pth')]
number_in_path = lambda x: int(x[:-4].split('_')[2])

nums = []
for i, path in enumerate(sorted(paths, key=number_in_path)):
    weights = torch.load(path)
#    pool = weights['rl.pool.weight']
    before_pool = weights['rl.g_layers.3.weight']
    nonzero_weights = before_pool.norm(dim=1) > 1e-8
    num_nonzero = nonzero_weights.sum().item()
    print(i, num_nonzero)
    nums.append(num_nonzero)

print(nums)
