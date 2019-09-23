import os
import argparse
from datetime import datetime

import torch


import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.multiprocessing as mp

try:
    mp.set_start_method("forkserver")
except RuntimeError:
    pass

import scipy.optimize
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import data
import track
from model import *


def per_sample_set_loss(sample_np):
    row_idx, col_idx = scipy.optimize.linear_sum_assignment(sample_np)
    return row_idx, col_idx


def main():
    global net
    global test_loader
    global scatter
    parser = argparse.ArgumentParser()
    # generic params
    parser.add_argument('--name', default=datetime.now().strftime('%Y-%m-%d_%H:%M:%S'), help='Name to store the log file as')
    parser.add_argument('--resume', nargs='+', help='Path to log file to resume from')

    parser.add_argument('--encoder', default='FSEncoder', help='Encoder')
    parser.add_argument('--decoder', default='FSDecoder', help='Decoder')
    parser.add_argument('--cardinality', type=int, default=20, help='Size of set')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train with')
    parser.add_argument('--latent', type=int, default=8, help='Dimensionality of latent space')
    parser.add_argument('--dim', type=int, default=64, help='Dimensionality of hidden layers')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate of model')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size to train with')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of threads for data loader')
    parser.add_argument('--samples', type=int, default=2**14, help='Dataset size')
    parser.add_argument('--decay', action='store_true', help='Decay sort temperature')
    parser.add_argument('--skip', action='store_true', help='Skip permutation use in decoder')
    parser.add_argument('--mnist', action='store_true', help='Use MNIST dataset')
    parser.add_argument('--no-cuda', action='store_true', help='Run on CPU instead of GPU (not recommended)')
    parser.add_argument('--train-only', action='store_true', help='Only run training, no evaluation')
    parser.add_argument('--eval-only', action='store_true', help='Only run evaluation, no training')
    parser.add_argument('--multi-gpu', action='store_true', help='Use multiple GPUs')
    parser.add_argument('--show', action='store_true', help='Show generated samples')

    parser.add_argument('--loss', choices=['direct', 'hungarian', 'chamfer'], default='direct', help='Type of loss used')

    parser.add_argument('--shift', action='store_true', help='')
    parser.add_argument('--rotate', action='store_true', help='')
    parser.add_argument('--scale', action='store_true', help='')
    parser.add_argument('--variable', action='store_true', help='')
    parser.add_argument('--noise', type=float, default=0, help='Standard deviation of noise')
    args = parser.parse_args()
    args.mnist = True
    args.eval_only = True
    args.show = True
    args.cardinality = 342
    args.batch_size = 1

    model_args = {
        'set_size': args.cardinality,
        'dim': args.dim,
        'skip': args.skip,
    }
    net_class = SAE
    net = [net_class(
        encoder=globals()[args.encoder if k == 0 else 'SumEncoder'],
        decoder=globals()[args.decoder if k == 0 else 'MLPDecoder'],
        latent_dim=args.latent,
        encoder_args=model_args,
        decoder_args=model_args,
    ) for k in range(2)]

    if not args.no_cuda:
        net = [n.cuda() for n in net]

    dataset_train = data.MNISTSet(train=True)
    dataset_test = data.MNISTSet(train=False)

    train_loader = data.get_loader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = data.get_loader(dataset_test, batch_size=args.batch_size, num_workers=args.num_workers)

    tracker = track.Tracker(
        train_mae=track.ExpMean(),
        train_cha=track.ExpMean(),
        train_loss=track.ExpMean(),

        test_mae=track.Mean(),
        test_cha=track.Mean(),
        test_loss=track.Mean(),
    )

    optimizer = None

    def run(net, loader, optimizer, train=False, epoch=0, pool=None):
        [n.eval() for n in net]

        preds = []
        ns = []
        for i, sample in enumerate(loader):
            points, labels, n_points = map(lambda x: x.cuda(async=True), sample)

            if args.noise > 0:
                noise = torch.randn_like(points) * args.noise
                input_points = points + noise
            else:
                input_points = points

            # pad to fixed size
            padding = torch.zeros(points.size(0), points.size(1), args.cardinality - points.size(2)).to(points.device)
            padded_points = torch.cat([input_points, padding], dim=2)
            points2 = [input_points, padded_points]

            pred = [points, input_points] + [n(p, n_points) for n, p in zip(net, points2)]
            pred = [p[0].detach().cpu().numpy() for p in pred]
            preds.append(pred)
            ns.append(n_points)
            if i == 1:
                return preds, ns


    def scatter(tensor, n_points, transpose=False, *args, **kwargs):
        x, y = tensor
        n = n_points
        if transpose:
            x, y = y, x
            y = 1-y
        plt.scatter(x[:n], y[:n], *args, **kwargs)


    # group same noise levels together
    d = {}
    for path in sorted(args.resume):
        name = path.split('/')[-1]
        model, noise, num = name.split('-')[1:]
        noise = float(noise)
        
        d.setdefault(noise, []).append((model, path))
    print(d)

    plt.figure(figsize=(16, 3.9))
    for i, (noise, ms) in enumerate(d.items()):
        print(i, noise, ms)
        for (_, path), n in zip(ms, net):
            weights = torch.load(path)['weights']
            print(path, type(n.encoder), type(n.decoder))
            n.load_state_dict(weights, strict=True)
        args.noise = noise

        points, n_points = run(net, test_loader, None)

        for j, (po, np) in enumerate(zip(points, n_points)):
            for p, row in zip(po, [0, 0, 1, 2]):
                ax = plt.subplot(3, 12, 12*row+1+2*i+j)
                if row == 2:
                    np = 342
                scatter(p, np, transpose=True, marker='o', s=8, alpha=0.5)
                plt.xlim(0, 1)
                plt.ylim(0, 1)
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    plt.title(r'$\sigma = {:.2f}$'.format(noise))
                if i == 0 and j == 0:
                    label = {
                        0: 'Input / Target',
                        1: 'Ours',
                        2: 'Baseline',
                    }[row]
                    plt.ylabel(label)

    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.savefig('mnist.pdf', bbox_inches='tight')

if __name__ == '__main__':
    main()
