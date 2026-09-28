import numpy as np
from scipy.stats import multivariate_normal
from torch.utils.data import Dataset
from data.base_geometry import BaseGeometry2D, BaseGeometry3D
from data.utils import plot2D, plot3D


class Circle(BaseGeometry2D):

    def __init__(self, size: int=1000, sigma: float=0.05) -> None:
        super().__init__(size=size, sigma=sigma)

    def reset(self) -> np.ndarray:
        normal = multivariate_normal([0, 0], np.eye(2))
        e = normal.rvs(self.size)
        theta = np.random.uniform(0, 2*np.pi, self.size)
        x = np.cos(theta) + self.sigma*e[:, 0]
        y = np.sin(theta) + self.sigma*e[:, 1]
        return np.vstack((x, y), dtype=np.float32).T


class Square(BaseGeometry2D):

    def __init__(self, size = 1000, sigma = 0.05) -> None:
        super().__init__(size, sigma)

    def reset(self) -> np.ndarray:
        normal = multivariate_normal([0, 0], np.eye(2))
        e = normal.rvs(self.size)
        values = np.random.uniform(-1, 1, self.size)
        X = np.empty((self.size, 2))
        X[:self.size//4, 0] = -1
        X[:self.size//4, 1] = values[:self.size//4]
        X[self.size//4:self.size//2, 0] = 1
        X[self.size//4:self.size//2, 1] = values[self.size//4:self.size//2]
        X[self.size//2:3*self.size//4, 0] = values[self.size//2:3*self.size//4]
        X[self.size//2:3*self.size//4, 1] = -1
        X[3*self.size//4:, 0] = values[3*self.size//4:]
        X[3*self.size//4:, 1] = 1
        return (X + self.sigma*e).astype(np.float32)


class Sphere(BaseGeometry3D):

    def __init__(self, size: int=1000, sigma: float=0.05) -> None:
        super().__init__(size=size, sigma=sigma)

    def reset(self) -> np.ndarray:
        normal = multivariate_normal([0, 0, 0], np.eye(3))
        e = normal.rvs(self.size)
        theta = np.random.uniform(0, np.pi, self.size)
        phi = np.random.uniform(0, 2*np.pi, self.size)
        x = np.cos(phi)*np.sin(theta) + self.sigma*e[:, 0]
        y = np.sin(phi)*np.sin(theta) + self.sigma*e[:, 1]
        z = np.cos(theta) + self.sigma*e[:, 2]
        return np.vstack((x, y, z), dtype=np.float32).T


class Superposition2D(Dataset):

    def __init__(self, shapes: list[BaseGeometry2D]) -> None:
        self.shapes = shapes
        self.X = np.vstack(tuple(shape.X for shape in shapes))
        self.size = 0
        for shape in shapes:
            self.size += shape.size

    def plot(self, fig_size: tuple[int, int]=(6, 6)) -> None:
            plot2D(self.X, fig_size)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> np.ndarray:
        return self.X[index]


class Superposition3D:

    def __init__(self, shapes: list[BaseGeometry3D]) -> None:
        self.shapes = shapes
        self.X = np.vstack((shape.X for shape in shapes))

    def plot(self, fig_size: tuple[int, int]=(6, 6)) -> None:
            plot3D(self.X, fig_size)