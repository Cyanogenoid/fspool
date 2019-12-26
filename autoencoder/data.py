import os
import math
import random

import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as T


def collate(batch):
    points, labels, n_points = zip(*batch)

    point_tensor = torch.zeros(len(points), points[0].size(0), max(n_points))
    for i, (point, length) in enumerate(zip(points, n_points)):
        point_tensor[i, :, :length] = point

    labels = torch.LongTensor(labels)
    n_points = torch.LongTensor(n_points)
    return point_tensor, labels, n_points


def get_loader(dataset, batch_size, num_workers=8, shuffle=True):
    return torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=collate,
    )


class Polygons(torch.utils.data.Dataset):
    def __init__(self, size=2**10, cardinality=10, shift=False, rotate=False, scale=False, variable=False):
        self.size = size
        self.shift = shift
        self.rotate = rotate
        self.scale = scale
        self.cardinality = cardinality
        self.variable = variable

    def __getitem__(self, item):
        if self.variable:
            cardinality = random.choice(range(3, self.cardinality))
        else:
            cardinality = self.cardinality
        rad = torch.linspace(0, 2 * math.pi, cardinality + 1)[:-1]

        centre = torch.zeros(2)
        if self.shift:
            centre += torch.rand(2) - 0.5

        radius = 1.0
        if self.scale:
            radius += torch.rand(1) - 0.5

        if self.rotate:
            rad += torch.rand(1) * 2 * math.pi

        x, y = torch.sin(rad), torch.cos(rad)
        points = radius * torch.stack([x, y]) + centre.unsqueeze(1)
        points = points[:, torch.randperm(cardinality)]
        return points, 0, cardinality

    def __len__(self):
        return self.size


class MNISTSet(torch.utils.data.Dataset):
    def __init__(self, threshold=0.0, train=True, root='mnist'):
        self.train = train
        self.root = root
        self.threshold = threshold
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        mnist = torchvision.datasets.MNIST(train=train, transform=transform, download=True, root=root)
        self.data = self.cache(mnist)

    def cache(self, dataset):
        cache_path = os.path.join(self.root, f'mnist_{self.train}_{self.threshold}.pth')
        if os.path.exists(cache_path):
            return torch.load(cache_path)

        print('Processing dataset...')
        data = []
        for datapoint in dataset:
            img, label = datapoint
            point_set, cardinality = self.image_to_set(img)
            data.append((point_set, label, cardinality))
        torch.save(data, cache_path)
        print('Done!')
        return data

    def image_to_set(self, img):
        idx = (img.squeeze(0) > self.threshold).nonzero().transpose(0, 1)
        cardinality = idx.size(1)
        return idx, cardinality

    def __getitem__(self, item):
        s, l, c = self.data[item]
        s = s[:, torch.randperm(c)]
        s = s.float() / 27  # put in range [0, 1]
        return s, l, c

    def __len__(self):
        return len(self.data)


class MNISTSetMasked(MNISTSet):
    def __getitem__(self, item):
        s, l, c = super().__getitem__(item)
        ones = torch.ones(1, s.size(1), device=s.device)
        s = torch.cat([s, ones], dim=0)
        return s, l, c


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataset = Circles()
    for i in range(2):
        points, centre, n_points = dataset[i]
        x, y = points[0], points[1]
        plt.scatter(x.numpy(), y.numpy())
        plt.scatter(centre[0], centre[1])
    plt.axes().set_aspect('equal', 'datalim')
    plt.show()
