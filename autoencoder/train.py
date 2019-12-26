import os
import argparse
from datetime import datetime

import torch


import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.multiprocessing as mp

#try:
#    mp.set_start_method("forkserver")
#except RuntimeError:
#    pass

import scipy.optimize
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import data
import track
from model import *


def per_sample_hungarian_loss(sample_np):
    row_idx, col_idx = scipy.optimize.linear_sum_assignment(sample_np)
    return row_idx, col_idx


def main():
    global net
    global test_loader
    global scatter
    parser = argparse.ArgumentParser()
    # generic params
    parser.add_argument('--name', default=datetime.now().strftime('%Y-%m-%d_%H:%M:%S'), help='Name to store the log file as')
    parser.add_argument('--resume', help='Path to log file to resume from')

    parser.add_argument('--encoder', default='FSEncoder', help='Encoder model')
    parser.add_argument('--decoder', default='FSDecoder', help='Decoder model')
    parser.add_argument('--cardinality', type=int, default=20, help='Size of set')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train with')
    parser.add_argument('--latent', type=int, default=8, help='Dimensionality of latent space')
    parser.add_argument('--dim', type=int, default=64, help='Dimensionality of hidden layers')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate of model')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size to train with')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of threads for data loader')
    parser.add_argument('--samples', type=int, default=2**14, help='Dataset size')
    parser.add_argument('--skip', action='store_true', help='Skip permutation use in decoder')
    parser.add_argument('--mnist', action='store_true', help='Use MNIST dataset')
    parser.add_argument('--masked', action='store_true', help='Use masked version of MNIST dataset')
    parser.add_argument('--no-cuda', action='store_true', help='Run on CPU instead of GPU (not recommended)')
    parser.add_argument('--train-only', action='store_true', help='Only run training, no evaluation')
    parser.add_argument('--eval-only', action='store_true', help='Only run evaluation, no training')
    parser.add_argument('--multi-gpu', action='store_true', help='Use multiple GPUs')
    parser.add_argument('--show', action='store_true', help='Show generated samples')
    parser.add_argument('--classify', action='store_true', help='Classifier version')
    parser.add_argument('--freeze-encoder', action='store_true', help='Freeze weights in encoder')

    parser.add_argument('--loss', choices=['direct', 'hungarian', 'chamfer'], default='direct', help='Type of loss used')

    parser.add_argument('--shift', action='store_true', help='')
    parser.add_argument('--rotate', action='store_true', help='')
    parser.add_argument('--scale', action='store_true', help='')
    parser.add_argument('--variable', action='store_true', help='')
    parser.add_argument('--noise', type=float, default=0, help='Standard deviation of noise')
    args = parser.parse_args()


    if args.mnist:
        args.cardinality = 342

    model_args = {
        'set_size': args.cardinality,
        'dim': args.dim,
        'skip': args.skip,
        'relaxed': not args.classify,  # usually relaxed, not relaxed when classifying
    }
    net_class = SAE
    net = net_class(
        encoder=globals()[args.encoder],
        decoder=globals()[args.decoder],
        latent_dim=args.latent,
        encoder_args=model_args,
        decoder_args=model_args,
        classify=args.classify,
        input_channels=3 if args.mnist and args.masked else 2,
    )

    if not args.no_cuda:
        net = net.cuda()

    if args.multi_gpu:
        net = torch.nn.DataParallel(net)

    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=args.lr)

    dataset_settings = {
        'cardinality': args.cardinality,
        'shift': args.shift,
        'rotate': args.rotate,
        'scale': args.scale,
        'variable': args.variable,
    }
    if not args.mnist:
        dataset_train = data.Polygons(size=args.samples, **dataset_settings)
        dataset_test = data.Polygons(size=2**14, **dataset_settings)
    else:
        if not args.masked:
            dataset_train = data.MNISTSet(train=True)
            dataset_test = data.MNISTSet(train=False)
        else:
            dataset_train = data.MNISTSetMasked(train=True)
            dataset_test = data.MNISTSetMasked(train=False)

    train_loader = data.get_loader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = data.get_loader(dataset_test, batch_size=args.batch_size, num_workers=args.num_workers)

    tracker = track.Tracker(
        train_mse=track.ExpMean(),
        train_cha=track.ExpMean(),
        train_loss=track.ExpMean(),
        train_acc=track.ExpMean(),

        test_mse=track.Mean(),
        test_cha=track.Mean(),
        test_loss=track.Mean(),
        test_acc=track.Mean(),
    )

    if args.resume:
        log = torch.load(args.resume)
        weights = log['weights']
        n = net
        if args.multi_gpu:
            n = n.module
        strict = not args.classify
        n.load_state_dict(weights, strict=strict)
        if args.freeze_encoder:
            for p in n.encoder.parameters():
                p.requires_grad = False


    def outer(a, b=None):
        if b is None:
            b = a
        size_a = tuple(a.size()) + (b.size()[-1],)
        size_b = tuple(b.size()) + (a.size()[-1],)
        a = a.unsqueeze(dim=-1).expand(*size_a)
        b = b.unsqueeze(dim=-2).expand(*size_b)
        return a, b


    def hungarian_loss(predictions, targets):
        # predictions and targets shape :: (n, c, s)
        predictions, targets = outer(predictions, targets)
        # squared_error shape :: (n, s, s)
        squared_error = (predictions - targets).pow(2).mean(1)

        squared_error_np = squared_error.detach().cpu().numpy()
        indices = pool.map(per_sample_hungarian_loss, squared_error_np)
        losses = [sample[row_idx, col_idx].mean() for sample, (row_idx, col_idx) in zip(squared_error, indices)]
        total_loss = torch.mean(torch.stack(list(losses)))
        return total_loss


    def chamfer_loss(predictions, targets):
        # predictions and targets shape :: (n, c, s)
        predictions, targets = outer(predictions, targets)
        # squared_error shape :: (n, s, s)
        squared_error = (predictions - targets).pow(2).mean(1)
        loss = squared_error.min(1)[0] + squared_error.min(2)[0]
        return loss.mean()

    def run(net, loader, optimizer, train=False, epoch=0, pool=None):
        if train:
            net.train()
            prefix = 'train'
        else:
            net.eval()
            prefix = 'test'

        total_train_steps = args.epochs * len(loader)

        loader = tqdm(loader, ncols=0, desc='{1} E{0:02d}'.format(epoch, 'train' if train else 'test '))
        for i, sample in enumerate(loader):
            points, labels, n_points = map(lambda x: x.cuda(), sample)

            if args.decoder != 'FSDecoder' and points.size(2) < args.cardinality:
                # pad to fixed size
                padding = torch.zeros(points.size(0), points.size(1), args.cardinality - points.size(2)).to(points.device)
                points = torch.cat([points, padding], dim=2)

            if args.noise > 0:
                noise = torch.randn_like(points) * args.noise
                input_points = points + noise
            else:
                input_points = points
            pred = net(input_points, n_points)

            mse, cha, acc = torch.FloatTensor([-1, -1, -1])
            if not args.classify:
                mse = (pred - points).pow(2).mean()
                cha = chamfer_loss(pred, points)
                if args.loss == 'direct':
                    loss = mse
                elif args.loss == 'chamfer':
                    loss = cha
                elif args.loss == 'hungarian':
                    loss = hungarian_loss(pred, points)
                else:
                    raise NotImplementedError
            else:
                loss = F.cross_entropy(pred, labels)
                acc = (pred.max(dim=1)[1] == labels).float().mean()

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            tracked_mse = tracker.update('{}_mse'.format(prefix), mse.item())
            tracked_cha = tracker.update('{}_cha'.format(prefix), cha.item())
            tracked_loss = tracker.update('{}_loss'.format(prefix), loss.item())
            tracked_acc = tracker.update('{}_acc'.format(prefix), acc.item())

            fmt = '{:.5f}'.format
            loader.set_postfix(
                mse=fmt(tracked_mse),
                cha=fmt(tracked_cha),
                loss=fmt(tracked_loss),
                acc=fmt(tracked_acc),
            )

            if args.show and not train:
                #scatter(input_points, n_points, marker='o', transpose=args.mnist)
                scatter(pred, n_points, marker='x', transpose=args.mnist)
                plt.axes().set_aspect('equal', 'datalim')
                plt.show()


    def scatter(tensor, n_points, transpose=False, *args, **kwargs):
        x, y = tensor[0].detach().cpu().numpy()
        n = n_points[0].detach().cpu().numpy()
        if transpose:
            x, y = y, x
            y = -y
        plt.scatter(x[:n], y[:n], *args, **kwargs)


    import subprocess
    git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'])

    #torch.backends.cudnn.benchmark = True

    for epoch in range(args.epochs):
        tracker.new_epoch()
        with mp.Pool(4) as pool:
            if not args.eval_only:
                run(net, train_loader, optimizer, train=True, epoch=epoch, pool=pool)
            if not args.train_only:
                run(net, test_loader, optimizer, train=False, epoch=epoch, pool=pool)

        results = {
            'name': args.name,
            'tracker': tracker.data,
            'weights': net.state_dict() if not args.multi_gpu else net.module.state_dict(),
            'args': vars(args),
            'hash': git_hash,
        }
        torch.save(results, os.path.join('logs', args.name))
        if args.eval_only:
            break

if __name__ == '__main__':
    main()

    # net = net.to('cpu')
    # inp = next(iter(test_loader)); pred = net(inp[0], inp[2])
