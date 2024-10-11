from sc4000.train.models.model import Model
from sc4000.train.models.vit import ViT

models = {"vit": ViT}


def load_model(model_name: str, **kwargs):
    return models[model_name](**kwargs)
