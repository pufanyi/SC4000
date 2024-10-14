from sc4000.eval.models.base import Model
from sc4000.eval.utils.results import Result

from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
from torchvision.transforms import Normalize, ToTensor, Resize, CenterCrop, Compose


class ConvNeXtV2(Model):
    def __init__(
        self, *, pretrained="google/vit-large-patch16-224", id2label=None, label2id=None
    ):
        super().__init__("ViT")
        self.image_processor = AutoImageProcessor.from_pretrained(pretrained)
        self.model = AutoModelForImageClassification.from_pretrained(
            pretrained,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )

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

    def apply_test_transforms(self, examples):
        examples["pixel_values"] = [
            self.test_transforms(image.convert("RGB")) for image in examples["image"]
        ]
        return examples

    def predict(self, image: Image.Image) -> Result:
        image = self.test_transforms(image.convert("RGB"))
        inputs = self.image_processor(image, return_tensors="pt")
        outputs = self.model(**inputs)
        logists_list = list(map(float, outputs.logits[0].detach().numpy()))
        return Result(
            prediction=outputs.logits.argmax(dim=-1).item(), logs=logists_list
        )
