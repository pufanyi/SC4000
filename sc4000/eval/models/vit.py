from sc4000.eval.models.base import Model
from sc4000.eval.utils.results import Result

from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
from torchvision.transforms import (
    Compose,
    Normalize,
    ToTensor,
    Resize,
    CenterCrop,
)


class ViT(Model):
    def __init__(
        self,
        *,
        pretrained="google/vit-large-patch16-224",
        id2label=None,
        label2id=None,
        device=None
    ):
        super().__init__("ViT", device=device)
        self.image_processor = ViTImageProcessor.from_pretrained(pretrained)
        self.model = ViTForImageClassification.from_pretrained(
            pretrained,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        ).to(self.device)

        self.image_mean, self.image_std = (
            self.image_processor.image_mean,
            self.image_processor.image_std,
        )
        size = self.image_processor.size["height"]
        normalize = Normalize(mean=self.image_mean, std=self.image_std)
        self.test_transforms = Compose(
            [
                Resize(size),
                CenterCrop(size),
                ToTensor(),
                normalize,
            ]
        )

    def predict(self, image: Image.Image) -> Result:
        inputs = self.test_transforms(image.convert("RGB")).unsqueeze(0).to(self.device)
        outputs = self.model(inputs)
        logits_list = outputs.logits[0].detach().cpu().numpy().tolist()
        return Result(prediction=outputs.logits.argmax(dim=-1).item(), logs=logits_list)
