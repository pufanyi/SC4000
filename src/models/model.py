import datasets

from abc import ABC, abstractmethod
from PIL import Image


class Model(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def train(self, inputs: datasets.Dataset):
        raise NotImplementedError

    @abstractmethod
    def inference(self, input: Image.Image):
        raise NotImplementedError
