import numpy as np
from scipy.stats import multivariate_normal
from data.base_geometry import BaseGeometry2D
from data.transformations import BaseAffineTransformation, ScaledTranslation, Rotation
from typing import Unpack


class Superposition2D(BaseGeometry2D):

    def __init__(self, modules: list[tuple[BaseGeometry2D, Unpack[tuple[BaseAffineTransformation, ...]]]]) -> None:
        """
        modules : List of tuples. Each tuple contains a shape in the first index followed by a series of affine transformations.
                  A tuple can have only a shape.
        """

        self.modules = modules
        self.n_shapes = len(modules)

    def sample(self, size: int, sigma: int) -> None:
        self.X = np.empty((size, 2), dtype=np.float32)
        quotient, remainder = divmod(size, self.n_shapes)
        for i, module in enumerate(self.modules):
            shape = module[0]
            shape_size = quotient if i < self.n_shapes-1 else quotient + remainder
            shape.sample(shape_size, 0)
            for transformation in module[1:]:
                transformation.apply(shape)

            self.X[i*quotient : i*quotient + shape_size, :] = shape.X

        normal = multivariate_normal([0, 0], np.eye(2))
        e = normal.rvs(size)
        self.X += sigma*e


class Segment(BaseGeometry2D):
    """
    Class describing segment [0, 1].
    """

    def __init__(self) -> None:
        super().__init__()

    def sample(self, size: int, sigma: float) -> None:
        self.X = np.zeros((size, 2), dtype=np.float32)
        self.X[:, 0] = np.random.uniform(0, 1, size)
        normal = multivariate_normal([0, 0], np.eye(2))
        e = normal.rvs(size)
        self.X += sigma*e


class Arc(BaseGeometry2D):
    """
    Class describing an arc.
    """

    def __init__(self, theta: float) -> None:
        super().__init__()
        self.theta = theta

    def sample(self, size: int, sigma: float) -> None:
        theta_samples = np.random.uniform(0, self.theta, size)
        normal = multivariate_normal([0, 0], np.eye(2))
        e = normal.rvs(size)
        x = np.cos(theta_samples) + sigma*e[:, 0]
        y = np.sin(theta_samples) + sigma*e[:, 1]
        self.X = np.vstack((x, y), dtype=np.float32).T


class Circle(BaseGeometry2D):

    def __init__(self) -> None:
        super().__init__()
        self.shape = Arc(theta=2*np.pi)

    def sample(self, size: int, sigma: float) -> None:
        self.shape.sample(size, sigma)
        self.X = self.shape.X


class Square(BaseGeometry2D):

    def __init__(self) -> None:
        super().__init__()
        self.shape = Superposition2D([
            (Segment(), ScaledTranslation(alpha=1, beta=[-1/2, -1/2])), 
            (Segment(), ScaledTranslation(alpha=1, beta=[-1/2, 1/2])), 
            (Segment(), Rotation(np.pi/2), ScaledTranslation(alpha=1, beta=[-1/2, -1/2])), 
            (Segment(), Rotation(np.pi/2), ScaledTranslation(alpha=1, beta=[1/2, -1/2])),
        ])

    def sample(self, size: int, sigma: float) -> None:
        self.shape.sample(size, sigma)
        self.X = self.shape.X


class Triangle(BaseGeometry2D):

    def __init__(self) -> None:
        super().__init__()
        self.shape = Superposition2D([
            (Segment(), ScaledTranslation(alpha=1, beta=[-1/2, -1/(2*np.sqrt(3))])),
            (Segment(), Rotation(theta=np.pi/3), ScaledTranslation(alpha=1, beta=[-1/2, -1/(2*np.sqrt(3))])), 
            (Segment(), Rotation(theta=2*np.pi/3), ScaledTranslation(alpha=1, beta=[1/2, -1/(2*np.sqrt(3))]))
        ])

    def sample(self, size: int, sigma: float) -> None:
        self.shape.sample(size, sigma)
        self.X = self.shape.X


class Cross(BaseGeometry2D):

    def __init__(self) -> None:
        super().__init__()
        self.shape = Superposition2D([
            (Segment(), ScaledTranslation(alpha=2, beta=[-1, 0])), 
            (Segment(), Rotation(np.pi/2), ScaledTranslation(alpha=2, beta=[0, -1]))
        ])

    def sample(self, size: int, sigma: float) -> None:
        self.shape.sample(size, sigma)
        self.X = self.shape.X


class Triforce(BaseGeometry2D):

    def __init__(self) -> None:
        super().__init__()
        self.shape = Superposition2D([
            (Triangle(), ScaledTranslation(alpha=1, beta=[1/2, 1/(2*np.sqrt(3))])),
            (Triangle(), ScaledTranslation(alpha=1, beta=[0, -1/np.sqrt(3)])),
            (Triangle(), ScaledTranslation(alpha=1, beta=[1, -1/np.sqrt(3)])),
        ])

    def sample(self, size: int, sigma: float) -> None:
        self.shape.sample(size, sigma)
        self.X = self.shape.X


class Stickman(BaseGeometry2D):

    def __init__(self) -> None:
        super().__init__()
        phi = (1 + np.sqrt(5))/2
        self.shape = Superposition2D([
            (Circle(), ScaledTranslation(alpha=1/4, beta=[0, 1/4])), 
            (Segment(), Rotation(np.pi/2), ScaledTranslation(alpha=1, beta=[0, -1])), 
            (Segment(), Rotation(-np.pi/4), ScaledTranslation(alpha=1-1/phi, beta=[0, -(1-1/phi)])), 
            (Segment(), Rotation(-3*np.pi/4), ScaledTranslation(alpha=1-1/phi, beta=[0, -(1-1/phi)])), 
            (Segment(), Rotation(-np.pi/4), ScaledTranslation(alpha=1-1/phi, beta=[0, -1])), 
            (Segment(), Rotation(-3*np.pi/4), ScaledTranslation(alpha=1-1/phi, beta=[0, -1])), 
        ])

    def sample(self, size: int, sigma: float) -> None:
        self.shape.sample(size, sigma)
        self.X = self.shape.X


class DeathlyHallows(BaseGeometry2D):

    def __init__(self) -> None:
        super().__init__()
        self.shape = Superposition2D([
            (Triangle(),), 
            (Circle(), ScaledTranslation(alpha=1/(2*np.sqrt(3)), beta=[0, 0])), 
            (Segment(), Rotation(np.pi/2), ScaledTranslation(alpha=np.sqrt(3)/2, beta=[0, -1/(2*np.sqrt(3))])), 
        ])

    def sample(self, size: int, sigma: float) -> None:
        self.shape.sample(size, sigma)
        self.X = self.shape.X


class PlayStation(BaseGeometry2D):

    def __init__(self) -> None:
        super().__init__()
        self.shape = Superposition2D([
            (Circle(), ScaledTranslation(alpha=1, beta=[3, 0])), 
            (Square(), ScaledTranslation(alpha=np.sqrt(np.pi), beta=[-3, 0])), 
            (Triangle(), ScaledTranslation(alpha=1.75*np.sqrt(np.pi/np.sqrt(3)), beta=[0, 3])), 
            (Cross(), Rotation(np.pi/4), ScaledTranslation(alpha=1.25, beta=[0, -3])), 
        ])

    def sample(self, size: int, sigma: float) -> None:
        self.shape.sample(size, sigma)
        self.X = self.shape.X