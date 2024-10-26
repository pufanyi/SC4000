from sc4000.train.models.base import Model
from sc4000.train.models.vit import ViT
from sc4000.train.models.resnet import ResNet
from sc4000.train.models.convnextv2 import ConvNeXtV2
from sc4000.train.models.mobilenetv3 import MobileNetV3

models = {
    "vit": ViT,
    "resnet": ResNet,
    "convnextv2": ConvNeXtV2,
    "mobilenetv3": MobileNetV3,
}


def load_model(model_name: str, **kwargs):
    return models[model_name](**kwargs)
