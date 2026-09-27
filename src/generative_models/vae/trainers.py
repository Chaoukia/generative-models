import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from pathlib import Path
from tqdm import tqdm
from core import Trainer
from vae.models import GaussVAE


class GaussVAETrainer(Trainer):

    def __init__(self, model: GaussVAE, dataloader: DataLoader, optimizer: Optimizer) -> None:
        super().__init__(model, dataloader, optimizer)
        self.latent_dim = model.latent_dim

    def train(self, n_epochs: int, n_latent: int, log_dir: str | Path, n_log: int) -> None:
        """
        Train a GaussVAE model.

        Parameters
        ---------------------------
        n_epochs : Int, number of epochs.
        n_latent : Int, number of samples to draw from the latent distribution.
        log_dir  : Path where to store the tensorboard logs.
        n_log    : Int, number of iterations between two consecutive tensorbaord logs.

        Returns
        ---------------------------
        """

        writer = SummaryWriter(log_dir)
        normal = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.zeros(self.latent_dim), torch.eye(self.latent_dim)
        )
        count = 0
        for i in range(n_epochs):
            for x in tqdm(self.dataloader, desc=f"epoch [{i}/{n_epochs}]", leave=False):
                # unsqueeze at dim=1 to use broadcasting with e.
                x = x.unsqueeze(1)
                e = normal.sample((x.shape[0], n_latent,))
                mu_theta, logsigma_theta = self.model.encoder(x)
                sigma_theta = torch.exp(logsigma_theta)
                z = mu_theta + sigma_theta*e
                loss_reconstruction = (torch.linalg.vector_norm(x - self.model.decoder(z), dim=-1)**2).mean()/(2*self.model.sigma**2)
                loss_regularization = (torch.linalg.vector_norm(sigma_theta, dim=-1)**2 + torch.linalg.vector_norm(mu_theta, dim=-1)**2).mean()/2 - logsigma_theta.sum(dim=-1).mean()
                loss = loss_reconstruction + loss_regularization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if count%n_log==0:
                    self.log(writer, loss_reconstruction.item(), loss_regularization.item(), loss.item(), count)

                count += 1

    def log(self, writer: SummaryWriter, loss_reconstruction: float, loss_regularization: float, loss: float, step: int) -> None:
        writer.add_scalar("Loss/loss_reconstruction", loss_reconstruction, step)
        writer.add_scalar("Loss/loss_regularization", loss_regularization, step)
        writer.add_scalar("Loss/loss", loss, step)