import numpy as np
from torch.utils.data import Dataset
from data.utils import plot2D, plot3D


class BaseGeometry(Dataset):
    """
    Base class for geometric shapes.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.X = None

    def sample(self, size: int, sigma: float) -> None:
        """
        Description
        ----------------------------
        Sample n data points with an additional Gaussian noise with variance sigma**2.

        Parameters
        ----------------------------
        n     : Int, number of data points to sample.
        sigma : Float > 0, standard deviation of the Gaussian noise.

        Returns
        ----------------------------
        """
        
        raise NotImplementedError

    def plot(self, fig_size: tuple[int, int]=(6, 6)) -> None:
        """
        Plot the generated data points.
        """

        raise NotImplementedError

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, index: int) -> np.ndarray:
        return self.X[index]


class BaseGeometry2D(BaseGeometry):
    """
    Base class for 2D shapes.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def plot(self, fig_size: tuple[int, int]=(6, 6)) -> None:
        if self.X is None:
            raise ValueError("X must not be None. Populate it using sample() method.")
        
        plot2D(self.X, fig_size)