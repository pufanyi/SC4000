from sc4000.train.models.base import Model
from sc4000.train.models.vit import ViT
from sc4000.train.models.resnet import ResNet

models = {"vit": ViT, "resnet": ResNet}


def load_model(model_name: str, **kwargs):
    return models[model_name](**kwargs)
