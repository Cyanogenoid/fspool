import torch
from torch.nn import Parameter
from torch_scatter import scatter_add

from .fsort import FSPool
from ..inits import uniform
from ...utils.num_nodes import maybe_num_nodes
from ...utils.sparse import dense_to_sparse
from torch_geometric.nn.conv import GraphConv
from torch_geometric.utils import to_batch, degree


class FSPooling(torch.nn.Module):
    """Featurewise sort pooling layer

    Args:
        in_channels (int): Size of each input sample.
        num_pieces (int): Number of pieces to parametrise continuous weights with
        global_pool (bool): Pool all nodes instead of neighbourhood
    """

    def __init__(self, in_channels, num_pieces=5, global_pool=False):
        super(FSPooling, self).__init__()

        self.in_channels = in_channels
        self.num_pieces = num_pieces
        self.global_pool = global_pool
        self.pool = FSPool(in_channels, num_pieces)

    def forward(self, x, edge_index=None, batch=None):
        if not self.global_pool:
            row, col = edge_index
            dense_x, num_nodes = to_batch(x[col], row, dim_size=x.size(0))
        else:
            dense_x, num_nodes = to_batch(x, batch)
        dense_x = dense_x.transpose(1, 2)
        x, _ = self.pool(dense_x, num_nodes)
        return x

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.num_pieces)


class FSGraphConv(GraphConv):
    def __init__(self, *args, **kwargs):
        self.num_pieces = kwargs['num_pieces']
        del kwargs['num_pieces']
        super(FSGraphConv, self).__init__(*args, **kwargs)

        self.pool = FSPooling(self.out_channels, self.num_pieces)

    def forward(self, x, edge_index):
        row, col = edge_index

        out = torch.mm(x, self.weight)
        out = self.pool(out, edge_index)

        # Normalize by node degree (if wished).
        if self.norm:
            deg = degree(row, x.size(0), x.dtype)
            out = out / deg.unsqueeze(-1).clamp(min=1)

        # Weight root node separately (if wished).
        if self.root is not None:
            out = out + torch.mm(x, self.root)

        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
