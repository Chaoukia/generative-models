import numpy as np


class GenerativeModel:

    def __init__(self) -> None:
        pass

    def sample(self) -> np.ndarray:
        """
        Sample data.
        """

        raise NotImplementedError
    

class Trainer:

    def __init__(self, model: GenerativeModel) -> None:
        self.model = model

    def train(self) -> None:
        """
        Train a generative model.
        """

        raise NotImplementedError
    
    