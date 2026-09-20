import numpy as np
from generative_models.core import GenerativeModel


class GaussVAE(GenerativeModel):

    def __init__(self) -> None:
        super().__init__()