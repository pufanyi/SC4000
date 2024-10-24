from sc4000.eval.models.vit import ViT
from sc4000.eval.models.convnextv2 import ConvNeXtV2

import torch

models = {"vit": ViT, "convnextv2": ConvNeXtV2}


def load_model(model_name: str, **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return models[model_name](device = device, **kwargs)
