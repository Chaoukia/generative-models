import numpy as np
from data.base_geometry import BaseGeometry


class BaseAffineTransformation:
    """
    Base class for linear transformations.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.weights, self.bias = self.set_transformation()

    def set_transformation(self) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def apply(self, shape: BaseGeometry) -> None:
        """
        Apply the linear transformation to a shape.
        """

        shape.X = (self.weights@shape.X.T).T + self.bias


class ScaledTranslation(BaseAffineTransformation):
    """
    Class for scaling shapes by a constant factor.
    """

    def __init__(self, alpha: float, beta: list) -> None:
        self.alpha = alpha
        self.beta = beta
        super().__init__()

    def set_transformation(self) -> tuple[np.ndarray, np.ndarray]:
        return self.alpha*np.eye(2, dtype=np.float32), self.beta


class Rotation(BaseAffineTransformation):
    """
    Class for rotating shapes by a constant angle.
    """

    def __init__(self, theta: float) -> None:
        self.theta = theta
        super().__init__()

    def set_transformation(self) -> None:
        weights = np.array([[np.cos(self.theta), -np.sin(self.theta)], [np.sin(self.theta), np.cos(self.theta)]], dtype=np.float32)
        bias = 0
        return weights, bias