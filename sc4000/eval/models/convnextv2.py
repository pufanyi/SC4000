from sc4000.eval.models.base import Model
from sc4000.eval.utils.results import Result

import torch

from transformers import AutoImageProcessor, AutoModelForImageClassification
from typing import List
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


class ConvNeXtV2(Model):
    def __init__(
        self,
        *,
        pretrained="pufanyi/SC4000_ConvNeXtV2_base_full_9000",
        id2label=None,
        label2id=None,
        device=None,
    ):
        super().__init__("ConvNeXtV2", device=device)
        self.image_processor = AutoImageProcessor.from_pretrained(pretrained)
        self.model = AutoModelForImageClassification.from_pretrained(
            pretrained,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        ).to(self.device)

        self.image_mean, self.image_std = (
            self.image_processor.image_mean,
            self.image_processor.image_std,
        )
        size = self.image_processor.size["shortest_edge"]
        normalize = Normalize(mean=self.image_mean, std=self.image_std)
        self.test_transforms = Compose(
            [
                Resize(size),
                CenterCrop(size),
                ToTensor(),
                normalize,
            ]
        )

    def predict(self, images: List[Image.Image]) -> List[Result]:
        batch_inputs = torch.stack(
            [self.test_transforms(image.convert("RGB")) for image in images]
        ).to(self.device)
        outputs = self.model(batch_inputs)
        logits_list = outputs.logits.detach().cpu().numpy().tolist()
        predictions = outputs.logits.argmax(dim=-1).tolist()
        return [
            Result(prediction=pred, logs=logits)
            for pred, logits in zip(predictions, logits_list)
        ]
