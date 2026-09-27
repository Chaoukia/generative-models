import torch
import torch.nn as nn
from collections import OrderedDict


class EncoderGaussVAE(nn.Module):

    def __init__(self, data_dim: int, latent_dim: int) -> None:
        """
        input_dim  : Input dimension.
        output_dim : Output dimension.
        """
        super().__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.backbone = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(data_dim, 512)),
            ("relu1", nn.ReLU()),
            ("linear2", nn.Linear(512, 256)),
            ("relu2", nn.ReLU()),
        ]))
        self.mu_logsigma = nn.Linear(256, latent_dim*2)

    def forward(self, x: torch.tensor) -> tuple[torch.tensor, torch.tensor]:
        """
        Forward propagation through the encoder.

        Parameters
        --------------------------
        x : input tensor representing a data point.

        Returns
        --------------------------
        mu    : output tensor, the mean of the latent normal distribution.
        sigma : output tensor, the diagonal of the covariance matrix of the latent normal distribution.
        """

        features = self.backbone(x)
        mu_logsigma = self.mu_logsigma(features)
        mu, logsigma = torch.chunk(mu_logsigma, chunks=2, dim=-1)
        return mu, logsigma


class DecoderGaussVAE(nn.Module):

    def __init__(self, data_dim: int, latent_dim: int) -> None:
        """
        input_dim  : Input dimension.
        output_dim : Output dimension.
        """
        super().__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.backbone = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(latent_dim, 256)),
            ("relu1", nn.ReLU()),
            ("linear2", nn.Linear(256, 512)),
            ("relu2", nn.ReLU()),
            ("linear3", nn.Linear(512, data_dim)),
        ]))

    def forward(self, z: torch.tensor) -> torch.tensor:
        """
        Forward propagation through the encoder.

        Parameters
        --------------------------
        z : input tensor representing a latent sample.

        Returns
        --------------------------
        mu : output tensor, the mean of the approximated data distribution.
        """

        mu = self.backbone(z)
        return mu