from sc4000.train.models.base import Model
from sc4000.utils.logger import setup_logger

import timm
import torch
from timm import utils

logger = setup_logger(__name__)


class ResNet(Model):
    def __init__(self, *, model_name="resnext101_32x8d", num_classes=5):
        super().__init__("ResNet")

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        self.model = timm.create_model(
            model_name=model_name, pretrained=True, num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)

    def train(self, train_ds, val_ds, epochs):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
