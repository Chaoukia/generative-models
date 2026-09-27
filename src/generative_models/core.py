import numpy as np
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter


class GenerativeModel:

    def __init__(self, *args, **kwargs) -> None:
        pass

    def sample(self, n_samples: int) -> np.ndarray:
        """
        Sample n_samples data points.
        """

        raise NotImplementedError
    

class Trainer:

    def __init__(self, model: GenerativeModel, dataloader: DataLoader, optimizer: Optimizer) -> None:
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer

    def train(self, *args, **kwargs) -> None:
        """
        Train a generative model.
        """

        raise NotImplementedError

    def log(self, *args, writer: SummaryWriter, **kwargs) -> None:
        """
        Log variables in tensorboard.
        """

        raise NotImplementedError