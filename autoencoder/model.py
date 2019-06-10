import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision

from fspool import FSPool, cont_sort


class SAE(nn.Module):
    def __init__(self, encoder, decoder, latent_dim, latent_dim_encoder=None, encoder_args={}, decoder_args={}, classify=False):
        super().__init__()
        channels = 2
        latent_dim_encoder = latent_dim_encoder or latent_dim
        self.encoder = encoder(input_channels=channels, output_channels=latent_dim_encoder, **encoder_args)
        self.decoder = decoder(input_channels=latent_dim, output_channels=channels, **decoder_args)
        if classify:
            self.classifier = nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, 10),
            )
        else:
            self.classifier = None

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, n_points):
        # x :: (n, c, set_size)
        x_size = x.size()

        latent = self.encoder(x, n_points)
        if not isinstance(latent, tuple):
            latent = (latent,)
        self.x = latent

        if self.classifier is None:
            reconstruction = self.decoder(*latent, n_points)

            return reconstruction.view(x_size)
        else:
            return self.classifier(latent[0])


############
# Encoders #
############


class LinearEncoder(nn.Module):
    def __init__(self, *, input_channels, output_channels, set_size, **kwargs):
        super().__init__()
        self.lin = nn.Linear(input_channels * set_size, output_channels)

    def forward(self, x, *args):
        x = x.view(x.size(0), -1)
        return self.lin(x)


class MLPEncoder(nn.Module):
    def __init__(self, *, input_channels, output_channels, dim, set_size, **kwargs):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_channels * set_size, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, output_channels),
        )

    def forward(self, x, *args):
        x = x.view(x.size(0), -1)
        return self.model(x)


class FSEncoder(nn.Module):
    def __init__(self, *, input_channels, output_channels, dim, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim, dim, 1),
        )
        self.lin = nn.Sequential(
            nn.Linear(dim, dim, 1),
            nn.ReLU(inplace=True),
            nn.Linear(dim, output_channels, 1),
        )
        self.pool = FSPool(dim, 20, relaxed=kwargs.get('relaxed', True))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, n_points, *args):
        x = self.conv(x)
        x, perm = self.pool(x, n_points)
        x = self.lin(x)
        return x, perm


class SumEncoder(nn.Module):
    def __init__(self, *, input_channels, output_channels, dim, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim, dim, 1),
        )
        self.lin = nn.Sequential(
            nn.Linear(dim, dim, 1),
            nn.ReLU(inplace=True),
            nn.Linear(dim, output_channels, 1),
        )

    def forward(self, x, n_points, *args):
        x = self.conv(x)
        x = x.sum(2)
        x = self.lin(x)
        return x


############
# Decoders #
############


class LinearDecoder(nn.Module):
    def __init__(self, *, input_channels, output_channels, set_size, **kwargs):
        super().__init__()
        self.lin = nn.Linear(input_channels, output_channels * set_size)

    def forward(self, x, *args):
        x = x.view(x.size(0), -1)
        return self.lin(x)


class MLPDecoder(nn.Module):
    def __init__(self, *, input_channels, output_channels, set_size, dim, **kwargs):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_channels, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, output_channels * set_size),
        )

    def forward(self, x, *args):
        x = x.view(x.size(0), -1)
        return self.model(x)


class FSDecoder(nn.Module):
    def __init__(self, *, input_channels, output_channels, dim,  **kwargs):
        super().__init__()
        self.lin = nn.Sequential(
            nn.Linear(input_channels, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )
        self.unpool = FSPool(dim, 20, relaxed=True)
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim, output_channels, 1),
        )

    def forward(self, x, perm, n_points, *args):
        x = self.lin(x)
        x, mask = self.unpool.forward_transpose(x, perm, n=n_points)
        x = self.conv(x) * mask[:, :1, :]
        return x
