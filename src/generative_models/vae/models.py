import numpy as np
import torch
import torch.nn as nn
from core import GenerativeModel


class GaussVAE(GenerativeModel):

    def __init__(self, encoder: nn.Module, decoder: nn.Module, sigma: float) -> None:
        """
        encoder    : encoder network.
        decoder    : decoder network.
        sigma      : decoder's standard deviation.
        latent_dim : latent space dimension.
        """

        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sigma = sigma
        self.latent_dim = decoder.latent_dim

    def sample(self, n_samples: int) -> np.ndarray:
        """
        Description
        ----------------------------
        Sample n_samples from the approximated data distributions.

        Parameters
        ----------------------------
        n_samples : Int, number of samples to generate.

        Returns
        ----------------------------
        np.ndarray representing the generated sample.
        """

        with torch.no_grad():
            normal = torch.distributions.multivariate_normal.MultivariateNormal(
                torch.zeros(self.latent_dim), torch.eye(self.latent_dim)
            )
            z_e_batch = normal.sample(torch.Size((n_samples*2,)))
            z_batch, e_batch = torch.chunk(z_e_batch, chunks=2, dim=0)
            x_batch = self.decoder(z_batch) + self.sigma*e_batch
            return x_batch.numpy()
        