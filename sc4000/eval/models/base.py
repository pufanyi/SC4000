from abc import ABC, abstractmethod
from PIL import Image

from sc4000.eval.utils.results import Result


class Model(object):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def predict(self, image: Image.Image) -> Result:
        raise NotImplementedError
