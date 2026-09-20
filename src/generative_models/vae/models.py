import numpy as np
import torch
import torch.nn as nn
from generative_models.core import GenerativeModel


class GaussVAE(GenerativeModel):

    def __init__(self, encoder: nn.Module, decoder: nn.Module, latent_dim: int) -> None:
        """
        encoder    : encoder network.
        decoder    : decoder network.
        latent_dim : latent space dimension.
        """

        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim

    # def sample(self, n_samples: int) -> np.ndarray:
    #     normal = torch.distributions.multivariate_normal.MultivariateNormal(0, torch.eye(self.latent_dim))

    






    