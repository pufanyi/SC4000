import datasets

from abc import ABC, abstractmethod
from PIL import Image


class Model(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def train(self, train_ds: datasets.Dataset, val_ds: datasets.Dataset, epochs: int):
        raise NotImplementedError
