from sc4000.eval.models.base import Model
from sc4000.eval.models.vit import ViT
from sc4000.eval.models.convnextv2 import ConvNeXtV2

models = {"vit": ViT, "convnextv2": ConvNeXtV2}


def load_model(model_name: str, **kwargs):
    return models[model_name](**kwargs)
