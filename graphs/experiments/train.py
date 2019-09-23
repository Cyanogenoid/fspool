import os.path as osp
import argparse

import torch
from torch.nn import Sequential, Linear, ReLU
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GraphConv, FSPooling, GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import degree
import torch_geometric.transforms as T
from sklearn.model_selection import KFold


parser = argparse.ArgumentParser()
parser.add_argument('--dataset')
parser.add_argument('--model', choices=['fsort', 'sum', 'mean'])
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--validation', action='store_true')
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--dim', type=int, default=32)
parser.add_argument('--drop', type=float, default=0.5)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu', type=int)
args = parser.parse_args()

if args.gpu is not None:
    torch.cuda.set_device(args.gpu)

max_degrees = {
    'IMDB-BINARY': 135,
    'IMDB-MULTI': 88,
    'COLLAB': 491,
}

transforms = []
if 'REDDIT' in args.dataset or args.dataset in max_degrees:
    transforms.append(T.Constant(1))
if args.dataset in max_degrees:
    transforms.append(T.OneHotDegree(max_degrees[args.dataset]))
print('transforms:', transforms)

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', args.dataset)
dataset = TUDataset(path, name=args.dataset, transform=T.Compose(transforms))

# different seeds for different folds so that one particularly good or bad init doesn't affect the results for the whole seed
# multiply folds by 10 so that nets in different seeds are initialised with different seeds
seed = args.seed + 10 * args.fold
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

dataset = dataset.shuffle()
kfold = KFold(n_splits=10)
train_indices, test_indices = list(kfold.split(range(len(dataset))))[args.fold]


if args.dataset in max_degrees:
    # make sure that correct num is used
    max_degree = max(degree(d.edge_index[0]).max() for d in dataset).item()
    specified = max_degrees[args.dataset]
    assert specified == max_degree, f'max node degree is {max_degree} but was specified as {specified}'

train_dataset = dataset[torch.LongTensor(train_indices)]
test_dataset = dataset[torch.LongTensor(test_indices)]
n = (len(dataset) + 9) // 10
if args.validation:
    train_dataset, val_dataset = train_dataset[n:], train_dataset[:n]
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)


class GINBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        nn = Sequential(
            Linear(in_channels, out_channels),
            ReLU(),
            torch.nn.BatchNorm1d(out_channels),
            Linear(out_channels, out_channels),
            ReLU(),
            torch.nn.BatchNorm1d(out_channels),
        )
        self.conv = GINConv(nn, in_channels)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        num_features = dataset.num_features
        dim = args.dim
        blocks = 5

        convs = []
        in_channels = num_features
        for _ in range(blocks):
            convs.append(GINBlock(in_channels, dim))
            in_channels = dim

        self.convs = torch.nn.ModuleList(convs)

        self.pools = torch.nn.ModuleList(FSPooling(dim, 5, global_pool=True) for _ in range(blocks))

        self.classifier = Sequential(
            torch.nn.BatchNorm1d(blocks * dim),
            Linear(blocks * dim, dim),
            ReLU(inplace=True),
            torch.nn.Dropout(args.drop),
            Linear(dim, dataset.num_classes),
        )

        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        features = []
        for conv, pool in zip(self.convs, self.pools):
            x = conv(x, edge_index)
            if args.model == 'fsort':
                features.append(pool(x, batch=batch))
            elif args.model == 'sum':
                features.append(global_add_pool(x, batch))
            elif args.model == 'mean':
                features.append(global_mean_pool(x, batch))
            else:
                raise ValueError('invalid model variant')

        x = torch.cat(features, dim=1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=-1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(epoch):
    model.train()

    if epoch % 50 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.5 * param_group['lr']

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)


def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        pred = model(data).max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


for epoch in range(1, 1 + args.epochs):
    loss = train(epoch)
    if args.validation:
        val_acc = test(val_loader)
        test_acc = test(test_loader)
        print('Epoch: {:03d}, Train Loss: {:.5f}, Val Acc: {:.5f}, Test Acc: {:.5f}'.
              format(epoch, loss, val_acc, test_acc))
    else:
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        print('Epoch: {:03d}, Train Loss: {:.5f}, Train Acc: {:.5f}, Test Acc: {:.5f}'.
              format(epoch, loss, train_acc, test_acc))
