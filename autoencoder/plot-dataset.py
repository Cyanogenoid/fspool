import matplotlib.pyplot as plt

import data


def scatter(tensor, n, transpose=False, *args, **kwargs):
    x, y = tensor.detach().cpu().numpy()
    if transpose:
        x, y = y, x
        y = -y
    plt.scatter(x[:n], y[:n], *args, **kwargs)
    plt.axes().set_aspect('equal', 'datalim')
    plt.xticks([])
    plt.yticks([])

# polygon
polygons = data.Polygons(cardinality=6, rotate=True)
points, centre, cardinality = polygons[0]

plt.figure(figsize=(1.7, 1.7))
scatter(points, cardinality)
plt.savefig('hexagon.pdf', bbox_inches='tight', pad_inches=0)


# mnist

mnist = data.MNISTSet()
points, label, cardinality = mnist[0]

plt.figure(figsize=(4, 4))
scatter(points, cardinality, transpose=True)
plt.savefig('mnist-0.pdf', bbox_inches='tight')
