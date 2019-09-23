import torch
import torch.nn as nn
import torch.nn.functional as F


class FSPool(nn.Module):
    def __init__(self, in_channels, n_pieces, relaxed=False, softmax=False, mode='mean'):
        super().__init__()
        if mode == 'sum':
            assert softmax
        self.n_pieces = n_pieces
        self.weight = nn.Parameter(torch.zeros(in_channels, n_pieces + 1))
        self.relaxed = relaxed
        self.softmax = softmax
        assert mode == 'mean' or mode == 'sum'
        self.mode = mode

        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.fill_(1)

    def forward(self, x, n=None):
        assert x.size(1) == self.weight.size(0), 'incorrect number of input channels in weight'
        if n is None:
            n = x.new(x.size(0)).fill_(x.size(2)).long()
        sizes, mask = fill_sizes(n)
        mask = mask.expand_as(x)

        weight = self.determine_weight(sizes)
        if self.softmax:
            weight = masked_softmax(weight, mask, dim=2)

        # make sure that fill value isn't affecting sort result
        # sort is ascending, so put unreasonably large value in places to be masked away
        if self.relaxed:
            x = x + (1 - mask).float() * -99999
            x, perm = cont_sort(x)
        else:
            x = x + (1 - mask).float() * 99999
            x, perm = x.sort(dim=2)

        x = (x * weight * mask.float()).sum(dim=2)
        if self.mode == 'sum':
            x = x * n.unsqueeze(1).float()
        return x, perm

    def forward_transpose(self, x, perm, n=None):
        if n is None:
            n = x.new(x.size(0)).fill_(perm.size(2)).long()
        sizes, mask = fill_sizes(n)
        mask = mask.expand(mask.size(0), x.size(1), mask.size(2))

        weight = self.determine_weight(sizes)
        if self.softmax:
            weight = masked_softmax(weight, mask, dim=2)

        if self.mode == 'sum':
            x = x * n.unsqueeze(1).float()
        x = x.unsqueeze(2) * weight * mask.float()

        # invert permutation
        if self.relaxed:
            x, _ = cont_sort(x, perm)
        else:
            x = x.scatter(2, perm, x)
        return x

    def determine_weight(self, sizes):
        # share same sequence length within each sample, so copy across batch dim
        weight = self.weight.unsqueeze(0)
        weight = weight.expand(sizes.size(0), weight.size(1), weight.size(2))

        # linspace [0, 1] -> linspace [0, n_pieces]
        index = self.n_pieces * sizes
        index = index.unsqueeze(1)
        index = index.expand(index.size(0), weight.size(1), index.size(2))

        # points in the weight vector to the left and right
        idx = index.long()
        frac = index.frac()
        left = weight.gather(2, idx)
        right = weight.gather(2, (idx + 1).clamp(max=self.n_pieces))

        # interpolate between left and right point
        return (1 - frac) * left + frac * right


def fill_sizes(sizes):
    max_size = sizes.max()
    size_tensor = sizes.new(sizes.size(0), max_size).float().fill_(-1)

    size_tensor = torch.arange(end=max_size, device=sizes.device, dtype=torch.float32)
    size_tensor = size_tensor.unsqueeze(0) / (sizes.float() - 1).clamp(min=1).unsqueeze(1)

    mask = size_tensor <= 1
    mask = mask.unsqueeze(1)

    return size_tensor.clamp(max=1), mask


def masked_softmax(x, mask, dim):
    x = x - 99999 * (1 - mask).float()
    return F.softmax(x, dim=dim)


def deterministic_sort(s, tau):
    """
    s: input elements to be sorted. Shape: batch_size x n x 1
    tau: temperature for relaxation. Scalar.
    """
    n = s.size()[1]
    one = torch.ones((n, 1), dtype = torch.float32, device=s.device)
    A_s = torch.abs(s - s.permute(0, 2, 1))
    B = torch.matmul(A_s, torch.matmul(one, one.transpose(0, 1)))
    scaling = (n + 1 - 2 * (torch.arange(n, device=s.device) + 1)).type(torch.float32)
    C = torch.matmul(s, scaling.unsqueeze(0))
    P_max = (C - B).permute(0, 2, 1)
    sm = torch.nn.Softmax(-1)
    P_hat = sm(P_max / tau)
    return P_hat


def cont_sort(x, perm=None):
    original_size = x.size()
    x = x.view(-1, x.size(2), 1)
    if perm is None:
        perm = deterministic_sort(x, 1)
    else:
        perm = perm.transpose(1, 2)
    x = perm.matmul(x)
    x = x.view(original_size)
    return x, perm


if __name__ == '__main__':
    pool = FSort(2, 1)
    x = torch.arange(0, 2*3*4).view(3, 2, 4).float()
    print('x', x)
    y, perm = pool(x, torch.LongTensor([2,3,4]))
    print('perm')
    print(perm)
    print('result')
    print(y)
