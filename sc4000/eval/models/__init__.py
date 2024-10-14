from sc4000.eval.models.base import Model
from sc4000.eval.models.vit import ViT

models = {"vit": ViT}


def load_model(model_name: str, **kwargs):
    return models[model_name](**kwargs)
