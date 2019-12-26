import os
import pandas as pd
import torch
import numpy as np


data = []
for path in os.listdir('logs'):
    #if not 'mnistc' in path:
    if not 'mnistnc' in path:  # use this line for no noise
        continue

    _, model, cat, num = path.split('-')

    log = torch.load(f'logs/{path}')
    test_acc = log['tracker']['test_acc']
    last_epoch = test_acc[-1]
    #last_epoch = test_acc[9]  # uncomment this line to show 10th epoch
    #last_epoch = test_acc[0]  # uncomment this line to show first epoch instead of last epoch results
    acc = np.mean(last_epoch)

    data.append((model, cat, num, acc))

data = pd.DataFrame(data)
data.columns = ['model', 'category', 'num', 'acc']
print(data.to_string())
grouped = data.groupby(['model', 'category'])
print(100 * grouped.std().round(3))
print(100 * grouped.mean().round(3))
