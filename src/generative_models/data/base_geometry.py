import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from data.utils import plot2D, plot3D


class BaseGeometry(Dataset):
    """
    Base class for geometric shapes.
    """

    def __init__(self, size: int=1000, sigma: float=0.05) -> None:
        self.size = size
        self.sigma = sigma
        self.X = self.reset()

    def reset(self) -> np.ndarray:
        """
        Generate data.
        """
        
        raise NotImplementedError

    def plot(self, fig_size: tuple[int, int]=(6, 6)) -> None:
        """
        Plot the generated data points.
        """

        raise NotImplementedError

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> np.ndarray:
        return self.X[index]


class BaseGeometry2D(BaseGeometry):
    """
    Base class for 2D shapes.
    """

    def __init__(self, size = 1000, sigma = 0.05) -> None:
        super().__init__(size, sigma)

    def plot(self, fig_size: tuple[int, int]=(6, 6)) -> None:
        plot2D(self.X, fig_size)


class BaseGeometry3D(BaseGeometry):
    """
    Base class for 2D shapes.
    """

    def __init__(self, size = 1000, sigma = 0.05) -> None:
        super().__init__(size, sigma)

    def plot(self, fig_size: tuple[int, int]=(6, 6)) -> None:
        plot3D(self.X, fig_size)