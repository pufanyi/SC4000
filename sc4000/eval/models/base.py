from typing import List

from abc import ABC, abstractmethod
from PIL import Image

from sc4000.eval.utils.results import Result


class Model(object):
    def __init__(self, name: str, device=None):
        self.name = name
        self.device = device

    @abstractmethod
    def predict(self, images: List[Image.Image]) -> List[Result]:
        raise NotImplementedError
