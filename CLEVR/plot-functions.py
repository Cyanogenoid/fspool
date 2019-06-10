import matplotlib.pyplot as plt
import numpy as np

import torch


weights = torch.load('RN_epoch_350.pth')
W = weights['rl.pool.weight'].cpu()

# filter out virtually-zero weights
# 20 pieces is too much for 12 set size, W[:, 1] and W[:, -2] are always 0
# So, we don't want to plot these because they are always approximately 0
W = W[W.norm(p=1, dim=1) > 1e-12]
x = torch.linspace(0, 1, W.size(1))
idx = list(range(W.size(1)))
idx.remove(1)
idx.remove(W.size(1) - 2)
W = W[:, idx]
x = x[idx]

for i, w in enumerate(W):
    plt.subplot(10, 9, i + 1)
    plt.axhline(y=0.0, color='k', linestyle='-', alpha=0.2)
    plt.plot(x.numpy(), w.numpy())
    plt.xticks([])
    plt.yticks([])
    plt.ylim(-1, 1)
plt.show()
