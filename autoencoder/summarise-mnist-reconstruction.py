import os
import pandas as pd
import torch
import numpy as np


data = []
for path in os.listdir('logs'):
    if not 'mnist-' in path:
        continue

    path = path.replace('eval-', '')

    if 'rnn' not in path:
        model, cat, num = path.split('-')[1:]
    else:
        model, model2, cat, num = path.split('-')[1:]
        model = f'{model}-{model2}'

    log = torch.load(f'logs/{path}')
    test_cha = log['tracker']['test_cha']
    last_epoch = test_cha[-1]
    loss = np.mean(last_epoch) * 1000

    data.append((model, cat, num, loss))

data = pd.DataFrame(data)
#print(data)
data.columns = ['model', 'noise', 'num', 'loss']
data = data.sort_values(['model', 'noise', 'num'])
print(data.to_string())
grouped = data.groupby(['model', 'noise'])
#print(grouped.std().round(3))
mean = grouped.mean().round(2)
std = grouped.std().round(2)

# latexify
i = 0
for m, s in zip(mean['loss'], std['loss']):
    if i == 0:
        print('\\textsc{} ', end='')
    print(f'& {m:.2f}\\tiny$\\pm${s:.2f} ', end='')
    i += 1
    if i == 6:
        print(r'\\')
        i = 0
    

result = pd.concat([mean, std], axis=1)
print(result)
